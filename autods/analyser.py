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

import re
import pathlib as pl
from packaging import version

import copy

from collections import OrderedDict as odict
from collections import namedtuple as ntuple

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger('autods')

from autods.data import MonoCategoryDataSet, MCDSAnalysisResultsSet
from autods.executor import Executor
from autods.engine import MCDSEngine
from autods.analysis import MCDSAnalysis, MCDSPreAnalysis


class DSAnalyser(object):

    """Run a bunch of DS analyses on samples extracted from an individualised sightings data set,
       according to a user-friendly set of analysis specs,
       + Tools for building analysis variants specifications and explicitating them.
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

        TODO: Translate to english
        
        odPartSpecs est donc un dictionnaire ordonné de tables de specs partielles.
        * chaque table donne un sous-ensemble (ou la totalité) des colonnes de variantes d'analyses,
        * chaque table peut être implicite ou explicite :
            * explicite : toutes les combinaisons désirées sont données pour les colonnes de la table,
            * implicite : dans chaque colonne, verticalement, on donne la liste des variantes possibles
              (chaque colonne n'aura donc pas la même longueur) ; l'explicitation consistera à générer
              automatiquement la totalité des combinaisons possibles des valeurs données dans les colonnes,
            * le type des tables est déterminé par leur nom : implicite s'il contient "_impl",
              explicite sinon.
        * plusieurs tables peuvent avoir les mêmes colonnes :
            * chacune peut être soit implicite, soit explicite, peu importe,
            * une fois explicitées, elles doivent cependant fournir des lignes ne se recouvrant pas,
            * cela permet de spécifier facilement des combinaisons différentes pour des sous-ensembles
              distincts de valeurs d'un sous-ensemble donné de colonnes,
            * avant de les combiner avec les autres tables, on les concatène verticalement en 1 seule
              après explicitation individuelle,
        * les tables qui n'ont aucune colonne en commun avec les autres produiront
          la combinatoire complète des combinaisons obtenues par explicitation des colonnes la précédant,
        * celles qui ont certaines (mais pas toutes) colonnes en communs avec celles qui la précèdent
          permettent de produire des variantes spécifiques pour ces colonnes : elles feront l'objet
          d'une jointure à gauche avec ces tables précédentes,
        * car l'algorithme d'explicitation - fusion des tables suit leur ordre dans le classeur,
          une fois faite l'explicitation - concaténation verticale des tables de même jeux de colonnes.
          
        N.B. Pas prévu, mais ça marche : pour imposer des valeurs de paramètres vides,
             il suffit de fournit une table vide, avec les entêtes correspondants (exemple avec ACDC).
        """
        
        # Load partial variant specs from source (trivial if given as a dict).
        odPartSpecs = partSpecs if isinstance(partSpecs, dict) else cls._loadPartSpecsFromFile(partSpecs)
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

    # Possible regexps (values) for auto-detection of analyser _internal_ parameter spec names (keys)
    # from explicit _user_ spec columns
    # (regexps are re.search'ed : any match _anywhere_inside_ the column name is OK;
    #  and case is ignored during searching).
    Int2UserSpecREs = odict([])

    @staticmethod
    def userSpec2ParamNames(userSpecCols, int2UserSpecREs):

        logger.debug('Matching user spec. columns:')

        parNames = list()
        for specName in userSpecCols:
            try:
                parName = next(iter(parName for parName in int2UserSpecREs \
                                    if any(re.search(pat, specName, flags=re.IGNORECASE) \
                                           for pat in int2UserSpecREs[parName])))
                logger.debug(' * "{}" => {}'.format(specName, parName))
                parNames.append(parName)
            except StopIteration:
                raise KeyError('Could not match user spec. column "{}" in sample data set columns [{}]' \
                                .format(specName, ', '.join(int2UserSpecREs.keys())))

        logger.debug('... success.')

        return parNames
    
    def checkUserSpecs(self, dfAnlysExplSpecs, anlysParamSpecCols):
    
        """Check user specified analyses params for usability.
        
        Use it before calling analyser.run(dfAnlysExplSpecs, anlysParamSpecCols, ...)
        
        Parameters:
           :param pd.DataFrame dfAnlysExplSpecs: analysis params table
           :param list anlysParamSpecCols: columns of dfAnlysExplSpecs for analysis specs
           :param threads:, :param processes: Number of parallel threads / processes to use
              (default: no parallelism)
              
        :raise: KeyError if any column could not be matched with some of the expected parameter names.
        """
    
         # Try and convert explicit. analysis spec. columns to the internal parameter names.
        _ = self.userSpec2ParamNames(anlysParamSpecCols, self.Int2UserSpecREs)
        
        return all(col in dfAnlysExplSpecs.columns for col in anlysParamSpecCols) 

class MCDSAnalyser(DSAnalyser):

    """Run a bunch of MCDS analyses"""

    def __init__(self, dfMonoCatObs, dfTransects=None, effortConstVal=1, dSurveyArea=dict(), 
                       resultsHeadCols=dict(before=['AnlysNum', 'SampleNum'], after=['AnlysAbbrev'], 
                                            sample=['Species', 'Pass', 'Adult', 'Duration']),
                       abbrevCol='AnlysAbbrev',
                       transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                       sampleDecFields=['Effort', 'Distance'],
                       distanceUnit='Meter', areaUnit='Hectare',
                       surveyType='Point', distanceType='Radial', clustering=False,
                       workDir='.', logData=False,
                       defEstimKeyFn=MCDSEngine.EstKeyFnDef, defEstimAdjustFn=MCDSEngine.EstAdjustFnDef,
                       defEstimCriterion=MCDSEngine.EstCriterionDef, defCVInterval=MCDSEngine.EstCVIntervalDef, defMinDist=MCDSEngine.DistMinDef, defMaxDist=MCDSEngine.DistMaxDef, 
                       defFitDistCuts=MCDSEngine.DistFitCutsDef, defDiscrDistCuts=MCDSEngine.DistDiscrCutsDef):

        super().__init__(dfMonoCatObs=dfMonoCatObs, dfTransects=dfTransects, 
                         effortConstVal=effortConstVal, dSurveyArea=dSurveyArea, 
                         resultsHeadCols=resultsHeadCols, abbrevCol=abbrevCol,
                         transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                         sampleDecFields=sampleDecFields,
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         surveyType=surveyType, distanceType=distanceType, clustering=clustering,
                         workDir=workDir)
                         
        self.logData = logData
        self.defEstimKeyFn = defEstimKeyFn
        self.defEstimAdjustFn = defEstimAdjustFn
        self.defEstimCriterion = defEstimCriterion
        self.defCVInterval = defCVInterval
        self.defMinDist = defMinDist
        self.defMaxDist = defMaxDist
        self.defFitDistCuts = defFitDistCuts
        self.defDiscrDistCuts = defDiscrDistCuts
                         
    # Analyser internal parameter spec names, for which a match should be found (when one is needed)
    # with user explicit optimisation specs used in run() calls.
    IntSpecEstimKeyFn = 'EstimKeyFn'
    IntSpecEstimAdjustFn = 'EstimAdjustFn'
    IntSpecEstimCriterion = 'EstimCriterion'
    IntSpecCVInterval = 'CvInterval'
    IntSpecMinDist = 'MinDist' # Left truncation distance
    IntSpecMaxDist = 'MaxDist' # Right truncation distance
    IntSpecFitDistCuts = 'FitDistCuts'
    IntSpecDiscrDistCuts = 'DiscrDistCuts'

    # Possible regexps (values) for auto-detection of analyser _internal_ parameter spec names (keys)
    # from explicit _user_ spec columns
    # (regexps are re.search'ed : any match _anywhere_inside_ the column name is OK;
    #  and case is ignored during searching).
    Int2UserSpecREs = \
      odict([(IntSpecEstimKeyFn,     ['ke[a-z]*[\.\-_ ]*f', 'f[o]?n[a-z]*[\.\-_ ]*cl']),
             (IntSpecEstimAdjustFn,  ['ad[a-z]*[\.\-_ ]*s', 's[éa-z]*[\.\-_ ]*aj']),
             (IntSpecEstimCriterion, ['crit[èa-z]*[\.\-_ ]*']),
             (IntSpecCVInterval,     ['conf[a-z]*[\.\-_ ]*[a-z]*[\.\-_ ]*int',
                                      'in[o]?n[a-z]*[\.\-_ ]*conf']),
             (IntSpecMinDist,        ['min[a-z]*[\.\-_ ]*d', 'd[a-z]*[\.\-_ ]*min',
                                      'tr[a-z]*[\.\-_ ]*g[ca]', 'le[a-z]*[\.\-_ ]*tr']),
             (IntSpecMaxDist,        ['max[a-z]*[\.\-_ ]*d', 'd[a-z]*[\.\-_ ]*max',
                                      'tr[a-z]*[\.\-_ ]*d[rt]', 'le[a-z]*[\.\-_ ]*tr']),
             (IntSpecFitDistCuts,    ['fit[a-z]*[\.\-_ ]*d', 'tr[a-z]*[\.\-_ ]*[a-z]*[\.\-_ ]*mod']),
             (IntSpecDiscrDistCuts,  ['dis[a-z]*[\.\-_ ]*d', 'tr[a-z]*[\.\-_ ]*[a-z]*[\.\-_ ]*dis'])])

    # Analysis object ctor parameter names (MUST match exactly: check in analysis submodule !).
    ParmEstimKeyFn = 'estimKeyFn'
    ParmEstimAdjustFn = 'estimAdjustFn'
    ParmEstimCriterion = 'estimCriterion'
    ParmCVInterval = 'cvInterval'
    ParmMinDist = 'minDist'
    ParmMaxDist = 'maxDist'
    ParmFitDistCuts = 'fitDistCuts'
    ParmDiscrDistCuts = 'discrDistCuts'

    def _getAnalysisParams(self, sAnIntSpec):
    
        """Retrieve analysis parameters, from user specs and default values
        
        :param sAnIntSpec: analysis parameter user specs with internal names (indexed with IntSpecXXX)
        
        :return: dict(estimKeyFn=, estimAdjustFn=, estimCriterion=, cvInterval=,
                      minDist=, maxDist=, fitDistCuts=, discrDistCuts=)
        """
        return { self.ParmEstimKeyFn: sAnIntSpec.get(self.IntSpecEstimKeyFn, self.defEstimKeyFn),
                 self.ParmEstimAdjustFn: sAnIntSpec.get(self.IntSpecEstimAdjustFn, self.defEstimAdjustFn),
                 self.ParmEstimCriterion: sAnIntSpec.get(self.IntSpecEstimCriterion, self.defEstimCriterion),
                 self.ParmCVInterval: sAnIntSpec.get(self.IntSpecCVInterval, self.defCVInterval),
                 self.ParmMinDist: sAnIntSpec.get(self.IntSpecMinDist, self.defMinDist),
                 self.ParmMaxDist: sAnIntSpec.get(self.IntSpecMaxDist, self.defMaxDist),
                 self.ParmFitDistCuts: sAnIntSpec.get(self.IntSpecFitDistCuts, self.defFitDistCuts),
                 self.ParmDiscrDistCuts: sAnIntSpec.get(self.IntSpecDiscrDistCuts, self.defDiscrDistCuts) }

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
        return MCDSAnalysisResultsSet(miCustomCols=miCustCols, 
                                      dfCustomColTrans=dfCustColTrans, miSampleCols=miSampCols)
    
    def _getResults(self, dAnlyses):
    
        """Wait for and gather dAnalyses (MCDSAnalysis futures) results into a MCDSAnalysisResultsSet 
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

    def run(self, dfAnlysExplSpecs, anlysParamSpecCols, threads=1, processes=0):
    
        """Run specified analyses
        
        Call checkUserSpecs(...) before this to make sure user specs are OK
        
        Parameters:
           :param:dAnlysParamsSpecs MCDS analysis param name to dfAnlysExplSpecs column name
             (or const value) mapping ; for possible param. names, see MCDSAnalysis ctor ;
             missing ones won't be passed to MCDSAnalysis ctor ;
             dict. values can be a column name of dfAnlysExplSpecs or a const value replacment
           :param list anlysParamSpecCols: columns of dfAnlysExplSpecs for analysis specs
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
        
        # Convert explicit. analysis spec. columns to the internal parameter names.
        anlysIntParmSpecCols = self.userSpec2ParamNames(anlysParamSpecCols, self.Int2UserSpecREs)
        
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

            # Build optimisation params specs series with parameters internal names.
            sAnIntSpec = sAnSpec[anlysParamSpecCols].set_axis(anlysIntParmSpecCols, inplace=False)
            
            # Get analysis parameters from user specs and default values.
            dAnlysParams = self._getAnalysisParams(sAnIntSpec)
            
            # Analysis object
            anlys = MCDSAnalysis(engine=self._engine, sampleDataSet=sds, name=sAnSpec[self.abbrevCol],
                                 customData=sAnSpec[customCols].copy(), logData=self.logData, **dAnlysParams)

            # Start running pre-analysis in parallel, but don't wait for it's finished, go on
            anlysFut = anlys.submit()
            
            # Store pre-analysis object and associated "future" for later use (should be running soon or later).
            dAnlyses[anlysFut] = anlys
            
            # Next analysis (loop).

        logger.info('All analyses started ; now waiting for their end, and results ...')

        # Wait for and gather results of all analyses.
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
    
        """Run specified analyses
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
            anlysFut = anlys.submit()
            
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
