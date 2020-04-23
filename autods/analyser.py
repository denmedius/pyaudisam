# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Analyser: Run a bunch of DS analyses according to a user-friendly set of analysis specs
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment


import sys

import pathlib as pl
from packaging import version

import copy

from collections import OrderedDict as odict, namedtuple as ntuple
import numpy as np
import pandas as pd

import logging

logger = logging.getLogger('autods')

from autods.data import MonoCategoryDataSet, MCDSResultsSet
from autods.executor import Executor
from autods.engine import MCDSEngine
from autods.analysis import MCDSAnalysis, MCDSPreAnalysis


class DSAnalyser(object):

    """Run a bunch of DS analyses on samples extracted from an individualised sightings data set,
       according to a user-friendly set of analysis specs
       Abstract class.
    """

    # Generation of a table of implicit "partial variant" specification,
    # from a list of possible data selection criteria for each variable.
    # "Partial variant" because its only about a sub-set of variants
    # * dVariants : { target columns name: list of possibles criteria for data selection }
    @staticmethod
    def implicitPartialVariantSpecs(dVariants):
        
        def fixedLengthList(toComplete, length):
            return toComplete + [np.nan]*(length - len(toComplete))

        nRows = max(len(l) for l in dVariants.values())

        return pd.DataFrame({ colName : fixedLengthList(variants, nRows) for colName, variants in dVariants.items() })

    @staticmethod
    def explicitPartialVariantSpecs(implSpecs, convertCols=dict()):

        """Generation of a table of explicit "partial variant" specifications, from an implicit one
           (= generate all combinations of variants)
           :param:implSpecs: implicit partial specs object, as a DataFrame, taken as is,
              or a dict, preprocessed through implicitPartialVariantSpecs
           :param:convertCols Name and conversion method for explicit columns to convert 
             (each column is converted through :
              dfExplSpecs[colName] = dfExplSpecs[colName].apply(convertCol)
              for colName, convertCol in convertCols.items()) 
        """
        
        # Convert spec from dict to DataFrame if needed.
        if isinstance(implSpecs, dict):
            dfImplSpecs = DSAnalyser.implicitPartialVariantSpecs(implSpecs)
        else:
            assert isinstance(implSpecs, pd.DataFrame)
            dfImplSpecs = implSpecs
        
        # First columns kept as is.
        dfExplSpecs = dfImplSpecs[dfImplSpecs.columns[:1]].dropna()
        
        # For each implicit specs column (but the first)
        for col in dfImplSpecs.columns[1:]:
            
            # Get variants
            sVariants = dfImplSpecs[col].dropna()
            
            # Duplicate current explicit table as much as variants are many
            dfExplSpecs = dfExplSpecs.loc[np.repeat(dfExplSpecs.index.to_numpy(), len(sVariants))]
            
            # Add the new columns by tiling the variants along the whole index range
            dfExplSpecs[col] = np.tile(sVariants.to_numpy(), len(dfExplSpecs) // len(sVariants))
            
            # Reset index for easy next duplication
            dfExplSpecs.reset_index(inplace=True, drop=True)

        # Convert specified columns if any
        assert all(colName in dfExplSpecs.columns for colName in convertCols), \
               'Could not find some column to convert of {} in explicit table columns {}' \
               .format(list(convertCols.keys()), list(dfExplSpecs.columns))
        for colName, convertCol in convertCols.items():
            dfExplSpecs[colName] = dfExplSpecs[colName].apply(convertCol)
        
        # Done.
        return dfExplSpecs

    SupportedFileExts = ['.xlsx'] + (['.ods'] if version.parse(pd.__version__).release >= (0, 25) else [])
    
    @classmethod
    def _loadPartSpecsFromFile(cls, sourceFpn):
        
        if isinstance(sourceFpn, str):
            sourceFpn = pl.Path(sourceFpn)
    
        assert sourceFpn.exists(), 'Source file for partial analysis specs not found : {}'.format(sourceFpn)

        ext = sourceFpn.suffix.lower()
        assert ext in cls.SupportedFileExts, \
               'Unsupported source file type {}: not from {{{}}}' \
               .format(ext, ','.join(cls.SupportedFileExts))
        if ext in ['.xlsx']:
            dfData = pd.read_excel(sourceFpn, sheet_name=None)
        elif ext in ['.ods']:
            dfData = pd.read_excel(sourceFpn, sheet_name=None, engine='odf' if ext in ['.ods'] else 'openpyxml')
            
        return dfData
    
    @classmethod
    def explicitVariantSpecs(cls, partSpecs, varIndCol=None, convertCols=dict(), computedCols=dict()):
        
        """Generation of a table of explicit variant specifications,
           from a set of implicit and explicit partial variant specs objects 
           :param:odPartSpecs The ordered dict of name => partial specs objects,
              each as a DataFrame, taken as is, or a dict, preprocessed through implicitPartialVariantSpecs.
              Or: pathname of an Excel (.xlsx) or Open Doc. (.ods) worksheet file (1 sheet per specs table)
              Warning: implicit tables are only found by their name containing "_impl"
           :param:varIndCol Name of the autogenerated variant index column (defaut: None = no such column added)
           :param:convertCols Name and conversion method for explicit columns to convert 
             (each column is converted through :
              dfExplSpecs[colName] = dfExplSpecs[colName].apply(convertCol)
              for colName, convertCol in convertCols.items()) 
           :param:computedCols Name and computing method for explicit columns to add (after appending :param:varIndCol)
             (each column to add is computed through :
              dfExplSpecs[colName] = dfExplSpecs.apply(computeCol, axis='columns')
              for colName, computeCol in computedCols.items()) 

        """
        
        # Load partial variant specs from source (trivial if given as an odict).
        odPartSpecs = partSpecs if isinstance(partSpecs, odict) else cls._loadPartSpecsFromFile(partSpecs)
        assert len(odPartSpecs) > 0, "Error: Can't explicit variants with no partial variant"
        
        # Convert any implicit partial variant spec from dict to DataFrame if needed.
        oddfPartSpecs = odict()
        
        for psName, psValues in odPartSpecs.items():
            if isinstance(psValues, dict):
                oddfPartSpecs[psName] = DSAnalyser.implicitPartialVariantSpecs(psValues)
            else:
                assert isinstance(psValues, pd.DataFrame)
                oddfPartSpecs[psName] = psValues
        
        # Group partial specs tables with same column sets (according to column names)
        odSameColsPsNames = odict() # { sorted(cols): [table names] }
        
        for psName, dfPsValues in oddfPartSpecs.items():
            
            colSetId = ':'.join(sorted(dfPsValues.columns))
            if colSetId not in odSameColsPsNames:
                odSameColsPsNames[colSetId] = list()
                
            odSameColsPsNames[colSetId].append(psName)

        # For each group, concat. tables into one, after expliciting if needed
        ldfExplPartSpecs = list()

        for lPsNames in odSameColsPsNames.values():

            ldfSameColsPartSpecs= list()
            for psName in lPsNames:

                dfPartSpecs = oddfPartSpecs[psName]

                # Implicit specs case:
                if '_impl' in psName:

                    dfPartSpecs = DSAnalyser.explicitPartialVariantSpecs(dfPartSpecs)

                # Now, specs are explicit.
                ldfSameColsPartSpecs.append(dfPartSpecs)

            # Concat groups of same columns set explicit specs
            ldfExplPartSpecs.append(pd.concat(ldfSameColsPartSpecs, ignore_index=True))
        
        # Combinaison des specs explicites (dans l'ordre)
        dfExplSpecs = ldfExplPartSpecs[0]

        for dfExplPartSpecs in ldfExplPartSpecs[1:]:

            commCols = [col for col in dfExplSpecs.columns if col in dfExplPartSpecs.columns]

            if commCols: # Any column in common : left join each left row to each matching right row

                dfExplSpecs = dfExplSpecs.join(dfExplPartSpecs.set_index(commCols), on=commCols)

            else: # No columns in common : combine each left row with all right rows

                nInitSpecs = len(dfExplSpecs)

                dfExplSpecs = dfExplSpecs.loc[np.repeat(dfExplSpecs.index, len(dfExplPartSpecs))]
                dfExplSpecs.reset_index(drop=True, inplace=True)

                dfExplPartSpecs = pd.DataFrame(data=np.tile(dfExplPartSpecs, [nInitSpecs, 1]), columns=dfExplPartSpecs.columns)

                dfExplSpecs = pd.concat([dfExplSpecs, dfExplPartSpecs], axis='columns')

            dfExplSpecs.reset_index(drop=True, inplace=True)
        
        # Convert specified columns if any
        assert all(colName in dfExplSpecs.columns for colName in convertCols), \
               'Could not find some column to convert of {} in explicit table columns {}' \
               .format(list(convertCols.keys()), list(dfExplSpecs.columns))
        for colName, convertCol in convertCols.items():
            dfExplSpecs[colName] = dfExplSpecs[colName].apply(convertCol)
        
        # Generate explicit variant index column if specified
        if varIndCol:
            dfExplSpecs.reset_index(drop=False, inplace=True)
            dfExplSpecs.rename(columns=dict(index=varIndCol), inplace=True)
            
        # Compute and add supplementary columns if any
        for colName, computeCol in computedCols.items():
            dfExplSpecs[colName] = dfExplSpecs.apply(computeCol, axis='columns')
                
        # Done.
        return dfExplSpecs


    def __init__(self, dfMonoCatObs, dfTransects=None, effortConstVal=1, dSurveyArea=dict(), 
                       resultsHeadCols=dict(before=['AnlysNum', 'Sample'], after=['AnlysAbbrev'], 
                                            sample=['Species', 'Pass', 'Adult', 'Duration']),
                       abbrevCol='AnlysAbbrev',
                       transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                       sampleDecFields=['Effort', 'Distance'],
                       distanceUnit='Meter', areaUnit='Hectare',
                       workDir='.', **options):
                       
        """Ctor
            :param pd.DataFrame dfMonoCatObs: mono-category data from FieldDataSet.monoCategorise() or individualise()
            :param pd.DataFrame dfTransects: Transects infos with columns : transectPlaceCols (n), passIdCol (1),
                effortCol (1) ; if None, auto generated from input sightings
            :param effortConstVal: if dfTransects is None and effortCol not in source table, use this constant value
            :param resultsHeadCols: dict of list of column names (from dfMonoCatObs) to use in order
                to build results (right) header columns ; 'sample' columns are sample selection columns.
            :param options: See DSEngine
        """

        self.dfMonoCatObs = dfMonoCatObs

        self.resultsHeadCols = resultsHeadCols
        self.abbrevCol = abbrevCol
            
        self.workDir = workDir

        # Save specific options (as a named tuple for easier use through dot operator).
        options = copy.deepcopy(options)
        options.update(distanceUnit=distanceUnit, areaUnit=areaUnit)
        self.OptionsClass = ntuple('Options', options.keys())
        self.options = self.OptionsClass(**options) 
        
        # Individualised data (all samples)
        self._mcDataSet = \
            MonoCategoryDataSet(dfMonoCatObs, dfTransects=dfTransects, effortConstVal=effortConstVal,
                                dSurveyArea=dSurveyArea, transectPlaceCols=transectPlaceCols,
                                passIdCol=passIdCol, effortCol=effortCol, sampleDecFields=sampleDecFields)
                                
        # Analysis engine and executor.
        self._executor = None
        self._engine = None


# MCDSAnalyser: Run a bunch of MCDS analyses
class MCDSAnalyser(DSAnalyser):

    def __init__(self, dfMonoCatObs, dfTransects=None, effortConstVal=1, dSurveyArea=dict(), 
                       resultsHeadCols=dict(before=['AnlysNum', 'Sample'], after=['AnlysAbbrev'], 
                                            sample=['Species', 'Pass', 'Adult', 'Duration']),
                       abbrevCol='AnlysAbbrev',
                       transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                       sampleDecFields=['Effort', 'Distance'],
                       distanceUnit='Meter', areaUnit='Hectare',
                       surveyType='Point', distanceType='Radial', clustering=False,
                       workDir='.'):

        super().__init__(dfMonoCatObs=dfMonoCatObs, dfTransects=dfTransects, 
                         effortConstVal=effortConstVal, dSurveyArea=dSurveyArea, 
                         resultsHeadCols=resultsHeadCols, abbrevCol=abbrevCol,
                         transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                         sampleDecFields=sampleDecFields,
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         surveyType=surveyType, distanceType=distanceType, clustering=clustering,
                         workDir=workDir)
                         
    def setupResults(self):
    
        """Build an empty results objects.
        """
    
        # Results object construction
        # a. Sample multi-index columns
        sampleSelCols = self.resultsHeadCols['sample']
        sampMCols = [('header (sample)', col, 'Value') for col in sampleSelCols]
        miSampCols = pd.MultiIndex.from_tuples(sampMCols)

        # b. Full custom multi-index columns to prepend to raw analysis results
        beforeCols = self.resultsHeadCols['before']
        custMCols = [('header (head)', col, 'Value') for col in beforeCols]
        custMCols += sampMCols
        
        afterCols = self.resultsHeadCols['after']
        custMCols += [('header (tail)', col, 'Value') for col in afterCols]

        customCols = beforeCols + sampleSelCols + afterCols
        miCustCols = pd.MultiIndex.from_tuples(custMCols)

        # c. Translation for it (well, only one language forced for all ...)
        dfCustColTrans = pd.DataFrame(index=miCustCols, data={ lang: customCols for lang in ['fr', 'en'] })

        # d. And finally, the result object
        results = MCDSResultsSet(miCustomCols=miCustCols, 
                                 dfCustomColTrans=dfCustColTrans, miSampleCols=miSampCols)

        return results
    
    def _getResults(self, dAnlyses):
    
        """Wait for and gather dAnalyses (MCDSAnalysis futures) results into a MCDSResultsSet 
        """
    
        # Results object construction
        results = self.setupResults()

        # For each analysis as it gets completed (first completed => first yielded)
        for anlysFut in self._executor.asCompleted(dAnlyses):
            
            # Retrieve analysis object from its associated future object
            anlys = dAnlyses[anlysFut]
            
            # Get analysis results
            sResult = anlys.getResults()

            # Get custom header values, and set target index (= columns) for results
            sCustomHead = anlys.customData
            sCustomHead.index = results.miCustomCols

            # Save results
            results.append(sResult, sCustomHead=sCustomHead)

        # Terminate analysis executor
        self._executor.shutdown()

        # Terminate analysis engine
        self._engine.shutdown()

        return results

    def run(self, dfAnlysExplSpecs, dAnlysParamsSpecs=dict(), threads=1, processes=0):
    
        """Run specified analysis
           :param:dAnlysParamsSpecs MCDS analysis param name to dfAnlysExplSpecs column name
             (or const value) mapping ; for possible param. names, see MCDSAnalysis ctor ;
             missing ones won't be passed to MCDSAnalysis ctor ;
             dict. values can be a column name of dfAnlysExplSpecs or a const value replacment
           :param:threads, :param:processes Number of parallel threads / processes to use (default: no parallelism)
        """
    
        # Executor (parallel or séquential).
        self._executor = Executor(parallel=threads > 1 or processes > 1, threads=threads, processes=processes)

        # MCDS analysis engine
        self._engine = MCDSEngine(workDir=self.workDir, executor=self._executor, 
                                  distanceUnit=self.options.distanceUnit, areaUnit=self.options.areaUnit,
                                  surveyType=self.options.surveyType, distanceType=self.options.distanceType,
                                  clustering=self.options.clustering)

        # Custom columns for results.
        customCols = \
            self.resultsHeadCols['before'] + self.resultsHeadCols['sample'] + self.resultsHeadCols['after']
        
        # For each analysis to run :
        runHow = 'in sequence' if threads <= 1 and processes <= 1 \
                 else '{} parallel {}'.format(threads if threads > 1 \
                                              else processes, 'threads' if threads > 1 else 'processes')
        logger.info('Running {} MCDS analyses ({}) ...'.format(len(dfAnlysExplSpecs), runHow))
        dAnlyses = dict()
        for anInd, sAnSpec in dfAnlysExplSpecs.iterrows():
            
            logger.info('#{}/{} : {}'.format(anInd+1, len(dfAnlysExplSpecs), sAnSpec[self.abbrevCol]))

            # Select data sample to process
            sds = self._mcDataSet.sampleDataSet(sAnSpec[self.resultsHeadCols['sample']])
            if not sds:
                continue

            # Analysis parameters
            dAnlysParams = { parName: sAnSpec[colNameOrVal] \
                                      if isinstance(colNameOrVal, str) and colNameOrVal in sAnSpec.index \
                                      else colNameOrVal
                             for parName, colNameOrVal in dAnlysParamsSpecs.items() }

            # Analysis object
            anlys = MCDSAnalysis(engine=self._engine, sampleDataSet=sds, name=sAnSpec[self.abbrevCol],
                                 customData=sAnSpec[customCols].copy(), logData=False, **dAnlysParams)

            # Start running pre-analysis in parallel, but don't wait for it's finished, go on
            anlysFut = anlys.run()
            
            # Store pre-analysis object and associated "future" for later use (should be running soon or later).
            dAnlyses[anlysFut] = anlys
            
            # Next analysis (loop).

        logger.info('All analyses started ; now waiting for their end, and results ...')

        # Wait for and gaher results of all analyses.
        results = self._getResults(dAnlyses)
        
        # Done.
        logger.info('Analyses completed.')

        return results

    def shutdown(self):
    
        # Final clean-up in case not already done (some exception in run ?)
        self._engine.shutdown()
        self._executor.shutdown()
        
    def __del__(self):
    
        self.shutdown()

# Default strategy for model choice sequence (if one fails, take next in order, and so on)
ModelEstimCritDef = 'AIC'
ModelCVIntervalDef = 95
ModelStrategyDef = [dict(keyFn=kf, adjSr='COSINE', estCrit=ModelEstimCritDef, cvInt=ModelCVIntervalDef) \
                     for kf in['HNORMAL', 'HAZARD', 'UNIFORM', 'NEXPON']]

# MCDSPreAnalyser: Run a bunch of MCDS pre-analyses
class MCDSPreAnalyser(MCDSAnalyser):


    def __init__(self, dfMonoCatObs, dfTransects=None, effortConstVal=1, dSurveyArea=dict(), 
                       resultsHeadCols=dict(before=['SampleNum'], after=['SampleAbbrev'], 
                                            sample=['Species', 'Pass', 'Adult', 'Duration']),
                       transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort', abbrevCol='SampAbbrev',
                       sampleDecFields=['Effort', 'Distance'],
                       distanceUnit='Meter', areaUnit='Hectare',
                       surveyType='Point', distanceType='Radial', clustering=False,
                       workDir='.'):

        super().__init__(dfMonoCatObs=dfMonoCatObs, dfTransects=dfTransects, 
                         effortConstVal=effortConstVal, dSurveyArea=dSurveyArea, 
                         resultsHeadCols=resultsHeadCols, abbrevCol=abbrevCol,
                         transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                         sampleDecFields=sampleDecFields,
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         surveyType=surveyType, distanceType=distanceType, clustering=clustering,
                         workDir=workDir)

    def run(self, dfSamplesExplSpecs, dModelStrategy=ModelStrategyDef, threads=1, processes=0):
    
        """Run specified analysis
           :param threads:, :param processes: Number of parallel threads / processes to use (default: no parallelism)
        """
    
        # Executor (parallel or séquential).
        self._executor = Executor(parallel=threads > 1 or processes > 1, threads=threads, processes=processes)

        # MCDS analysis engine (a sequential one: 'cause MCDSPreAnalysis does the parallel stuff itself).
        self._engine = MCDSEngine(workDir=self.workDir,
                                  distanceUnit=self.options.distanceUnit, areaUnit=self.options.areaUnit,
                                  surveyType=self.options.surveyType, distanceType=self.options.distanceType,
                                  clustering=self.options.clustering)

        # Custom columns for results.
        customCols = \
            self.resultsHeadCols['before'] + self.resultsHeadCols['sample'] + self.resultsHeadCols['after']
        
        # For each sample to analyse :
        runHow = 'in sequence' if threads <= 1 and processes <= 1 \
                 else '{} parallel {}'.format(threads if threads > 1 \
                                              else processes, 'threads' if threads > 1 else 'processes')
        logger.info('Running {} MCDS pre-analyses ({}) ...'.format(len(dfSamplesExplSpecs), runHow))
        
        dAnlyses = dict()
        for anInd, sAnSpec in dfSamplesExplSpecs.iterrows():
            
            logger.info('#{}/{} : {}'.format(anInd+1, len(dfSamplesExplSpecs), sAnSpec[self.abbrevCol]))

            # Select data sample to process
            sds = self._mcDataSet.sampleDataSet(sAnSpec[self.resultsHeadCols['sample']])
            if not sds:
                continue # No data => no analysis.

            # Pre-analysis object
            anlys = MCDSPreAnalysis(engine=self._engine, executor=self._executor,
                                    sampleDataSet=sds, name=sAnSpec[self.abbrevCol],
                                    customData=sAnSpec[customCols].copy(),
                                    logData=False, modelStrategy=dModelStrategy)

            # Start running pre-analysis in parallel, but don't wait for it's finished, go on
            anlysFut = anlys.run()
            
            # Store pre-analysis object and associated "future" for later use (should be running soon or later).
            dAnlyses[anlysFut] = anlys
            
            # Next analysis (loop).

        logger.info('All analyses started ; now waiting for their end, and results ...')

        # Wait for and gaher results of all analyses.
        results = self._getResults(dAnlyses)
        
        # Done.
        logger.info('Analyses completed.')

        return results
        
        
if __name__ == '__main__':

    sys.exit(0)
