# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Data: Input and output DS data manipulation tools
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment


import sys
import pathlib as pl
from packaging import version

from collections import OrderedDict as odict

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger('autods')

from autods.analysis import DSAnalysis, MCDSAnalysis


# An abstract tabular data set built from various input sources.
# Warning:
# * Input support provided for pandas.DataFrame, Excel .xlsx file, tab-separated .csv/.txt files,
#   and even OpenDoc .ods file with pandas >= 0.25 (needs odfpy module)
class DataSet(object):
    
    SupportedFileExts = ['.xlsx', '.csv', '.txt'] \
                        + (['.ods'] if version.parse(pd.__version__).release >= (0, 25) else [])
    
    # Wrapper around pd.read_csv for smart ./, decimal character management (pandas is not smart on this)
    # TODO: Make this more efficient
    @staticmethod
    def _csv2df(fileName, decCols, sep='\t'):
        df = pd.read_csv(fileName, sep=sep)
        allRight = True
        for col in decCols:
            if df[col].dropna().apply(lambda v: isinstance(v, str)).any():
                allRight = False
                break
        if not allRight:
            df = pd.read_csv(fileName, sep=sep, decimal=',')
        return df
    
    @classmethod
    def _fromDataFile(cls, sourceFpn, decimalFields):
        
        if isinstance(sourceFpn, str):
            sourceFpn = pl.Path(sourceFpn)
    
        assert sourceFpn.exists(), 'Source file for DataSet not found : ' + sourceFpn

        ext = sourceFpn.suffix.lower()
        assert ext in cls.SupportedFileExts, \
               'Unsupported source file type {}: not from {{{}}}' \
               .format(ext, ','.join(cls.SupportedFileExts))
        if ext in ['.xlsx']:
            dfData = pd.read_excel(sourceFpn)
        elif ext in ['.ods']:
            dfData = pd.read_excel(sourceFpn, engine='odf')
        elif ext in ['.csv', '.txt']:
            dfData = cls._csv2df(sourceFpn, decCols=decimalFields, sep='\t')
            
        return dfData
    
    @classmethod
    def _fromDataFrame(cls, sourceDf):
        
        return sourceDf.copy()
    
    def __init__(self, source, importDecFields=[]):
    
        if isinstance(source, str) or isinstance(source, pl.Path):
            self._dfData = self._fromDataFile(source, importDecFields)
        elif isinstance(source, pd.DataFrame):
            self._dfData = self._fromDataFrame(source)
        else:
            raise Exception('source for DataSet must be a pandas.DataFrame or an existing file')

        assert not self._dfData.empty, 'No data in source data set'

    def __len__(self):
        
        return len(self._dfData)
    
    @property
    def dfData(self):
        
        return self._dfData

    @dfData.setter
    def dfData(self, dfData):
        
        raise NotImplementedError('No change allowed to data ; create a new dataset !')


# A tabular data set for producing indivividuals data sets from "raw sightings data" aka "field data"
# (with possibly multiple category counts on each row)
# * Input support provided for pandas.DataFrame, Excel .xlsx file, tab-separated .csv/.txt files,
#   and even OpenDoc .ods file with pandas >= 0.25 (needs odfpy module)
class FieldDataSet(DataSet):

    # Ctor
    # Input support provided for pandas.DataFrame, Excel .xlsx file, tab-separated .csv/.txt files,
    # and even OpenDoc .ods file with pandas >= 0.25 (needs odfpy module)
    # * source: the input field data table
    # * countCols: the category columns (each of them holds counts of individuals for the category)
    # * addMonoCatCols: name and method of computing for columns to add after separating multi-categorical counts
    #   (each column to add is computed through :
    #      dfMonoCatSights[colName] = dfMonoCatSights[].apply(computeCol, axis='columns')
    #      for colName, computeCol in addMonoCatCols.items()) 
    def __init__(self, source, countCols, addMonoCatCols=dict(), importDecFields=[]):
        
        super().__init__(source, importDecFields)
        
        self.countCols = countCols
        self.dCompdMonoCatColSpecs = addMonoCatCols
        
        self.dfIndivSights = None # Not yet computed.
    
    # Transform a multi-categorical sightings set into an equivalent mono-categorical sightings set,
    # that is where no sightings has more that one category with positive count (keeping the same total counts).
    # Highly optimized version.
    # Ex: A sightings set with 2 categorical count columns nMales and nFemales
    #     * in the input set, you may have 1 sightings with nMales = 5 and nFemales = 2
    #     * in the output set, this sightings have been separated in 2 distinct ones (all other properties left untouched) :
    #       the 1st with nMales = 5 and nFemales = 0, the 2nd with nMales = 0 and nFemales = 2.
    @staticmethod
    def _separateMultiCategoricalCounts(dfInSights, countColumns):
        
        # For each count column
        ldfMonoCat = list()
        for col in countColumns:

            # Select rows with some individuals in this column
            dfOneCat = dfInSights[dfInSights[col] > 0].copy()

            # Set all other count cols to 0.
            otherCols = countColumns.copy()
            otherCols.remove(col)
            dfOneCat[otherCols] = 0

            # Store into a list for later
            ldfMonoCat.append(dfOneCat)

        # Concat all data frames into one.
        dfOutSights = pd.concat(ldfMonoCat)

        # Sort to "initial" order (easier to read), and reset index (for unique labels).
        dfOutSights.sort_index(inplace=True)
        dfOutSights.reset_index(inplace=True, drop=True)

        # Done.
        return dfOutSights

    # Transform a multi-individual mono-categorical sightings set into an equivalent mono-individual
    # mono-categorical sightings set, that is where no sightings has more that one individual
    # per category (keeping the same total counts).
    # Highly optimized version.
    # Ex: A sightings set with 2 mono-categorical count columns nMales and nFemales
    #     * in tyhe input set, you may have 1 sightings with nMales = 3 and nFemales = 0
    #       (but none with nMales and nFemales > 0)
    #     * in the output set, this sightings have been separated in 3 distinct ones
    #       (all other properties left untouched) : all with nMales = 1 and nFemales = 0.
    @staticmethod
    def _individualiseMonoCategoricalCounts(dfInSights, countColumns):
        
        # For each category column
        ldfIndiv = list()
        for col in countColumns:
            
            # Select rows with some individuals in this column
            dfOneCat = dfInSights[dfInSights[col] > 0]
            
            # Repeat each one by its count of individuals
            dfIndiv = dfInSights.loc[np.repeat(dfOneCat.index.values, dfOneCat[col].astype(int).values)]

            # Replace non-zero counts by 1.
            dfIndiv.loc[dfIndiv[col] > 0, col] = 1
            
            # Store into a list for later
            ldfIndiv.append(dfIndiv)
            
        # Concat all data frames into one.
        dfOutSights = pd.concat(ldfIndiv)

        # Sort to "initial" order (easier to read), and reset index (for unique labels).
        dfOutSights.sort_index(inplace=True)
        dfOutSights.reset_index(inplace=True, drop=True)
        
        # Done.
        return dfOutSights
        
    # Access to the resulting individuals data set
    def individualiseDataSet(self):
    
        # Compute only if not already done.
        if self.dfIndivSights is None:
        
            # Separate multi-categorical counts
            dfMonoCatSights = self._separateMultiCategoricalCounts(self._dfData, self.countCols)
            
            # Compute and add supplementary mono-categorical columns
            for colName, computeCol in self.dCompdMonoCatColSpecs.items():
                dfMonoCatSights[colName] = dfMonoCatSights[self.countCols].apply(computeCol, axis='columns')
            
            # Individualise mono-categorical counts
            self.dfIndivSights = self._individualiseMonoCategoricalCounts(dfMonoCatSights, self.countCols)

        return self.dfIndivSights
        

# A tabular data set for producing on-demand sample data sets from "individuals sightings data"
# * Input support provided for pandas.DataFrame, Excel .xlsx file, tab-separated .csv/.txt files,
#   and even OpenDoc .ods file with pandas >= 0.25 (needs odfpy module)
class IndividualsDataSet(DataSet):

    # Ctor
    # * dfTransects: Transects infos with columns : transectIdCols (n), passIdCol (1), effortCol (1)
    #                If None, auto generated from input sightings
    # * effortDefVal: if dfTransects is None and effortCol not in source table, use this constant value
    def __init__(self, source, dSurveyArea, importDecFields=[], dfTransects=None,
                       transectIdCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                       sampleDecFields=['Effort', 'Distance'], effortDefVal=1):
        
        super().__init__(source, importDecFields)
        
        self.dSurveyArea = dSurveyArea
        self.transectIdCols = transectIdCols
        self.passIdCol = passIdCol
        self.effortCol = effortCol
        self.sampleDecFields = sampleDecFields
    
        self.dfTransects = dfTransects
        if self.dfTransects is None or self.dfTransects.empty:
            self.dfTransects = \
                self._extractTransects(self._dfData, transectIdCols=self.transectIdCols, passIdCol=self.passIdCol)

    # Extract transect infos from individuals sightings
    # * effortDefVal: if effortCol not in dfIndivSightings, create one with this constant value
    @staticmethod
    def _extractTransects(dfIndivSightings, transectIdCols=['Transect'], passIdCol='Pass', 
                                            effortCol='Effort', effortDefVal=1):
    
        transSightCols = transectIdCols + [passIdCol]
        if effortCol in dfIndivSightings.columns:
            transSightCols.append(effortCol)
        
        dfTrans = dfIndivSightings[transSightCols]
        
        dfTrans = dfTrans.drop_duplicates()
        
        dfTrans.reset_index(drop=True, inplace=True)
        
        if effortCol not in dfTrans.columns:
            dfTrans[effortCol] = effortDefVal

        return dfTrans
    
    # Select sample sightings from an all-samples sightings table,
    # and compute the associated total sample effort.
    # * dSample : { key, value } selection criteria (with '+' support for 'or' operator in value),
    #             keys being columns of dfAllSights (dict protocol : dict, pd.Series, ...)
    # * dfAllSights : the all-samples (individual) sightings table to search into
    # * dfAllEffort : effort values for each transect x pass really done, for the all-sample survey
    # * transectIdCols : name of the input dfAllEffort and dSample columns to identify the transects (not passes)
    # * passIdCol : name of the input dfAllEffort and dSample column to identify the passes (not transects)
    # * effortCol : name of the input dfAllEffort and output effort column to add / replace
    @staticmethod
    def _selectSampleSightings(dSample, dfAllSights, dfAllEffort,
                               transectIdCols=['Transect'], passIdCol='Pass', effortCol='Effort'):
        
        # Select sightings
        dfSampSights = dfAllSights
        for key, values in dSample.items():
            values = str(values) # For ints as strings that get forced to int in io sometimes (ex. from_excel)
            if values and values not in ['nan', 'None']: # Empty value means "no selection criteria for this columns"
                values = values.split('+') if '+' in values else [values]
                dfSampSights = dfSampSights[dfSampSights[key].astype(str).isin(values)]

        # Compute sample effort
        passes = str(dSample[passIdCol]) # Same as above ...
        if passes and passes not in ['nan', 'None']: # Same as above ...
            passes = passes.split('+') if '+' in passes else [passes]
            dfSampEffort = dfAllEffort[dfAllEffort[passIdCol].astype(str).isin(passes)]
        else:
            dfSampEffort = dfAllEffort
        dfSampEffort = dfSampEffort[transectIdCols + [effortCol]].groupby(transectIdCols).sum()
        
        # Add effort column
        dfSampSights = dfSampSights.drop(columns=effortCol, errors='ignore').join(dfSampEffort, on=transectIdCols)

        return dfSampSights, dfSampEffort
        
    # Add "abscence" sightings to field data collected on transects for a given sample
    # * dfInSights : input data table
    # * sampleCols : the names of the sample identification columns
    # * dfExpdTransects : the expected transects, as a data frame indexed by the transectId,
    #     an index with same name as the corresponding column in dfInSights,
    #     and with other info columns to duplicate in absence sightings (at least the effort value)
    @staticmethod
    def _addAbsenceSightings(dfInSights, sampleCols, dfExpdTransects):
        
        assert not dfInSights.empty, 'Error : Empty sightings data to add absence ones to !'

        # Use the 1st sightings of the sample to build the absence sightings prototype
        # (all null columns except for the sample identification ones, lefts as is)
        dAbscSightTmpl = dfInSights.iloc[0].to_dict()
        dAbscSightTmpl.update({ k: None for k in dAbscSightTmpl.keys() if k not in sampleCols })
        dAbscSightTmpl

        # Determine missing transects for the sample
        transectIdCols = dfExpdTransects.index.name
        dfSampTransects = dfInSights.drop_duplicates(subset=transectIdCols)
        dfSampTransects.set_index(transectIdCols, inplace=True)
        dfSampTransects = dfSampTransects[dfExpdTransects.columns]

        dfMissgTransects = dfExpdTransects.loc[dfExpdTransects.index.difference(dfSampTransects.index)]

        # Generate the abscence sightings : 1 row per missing transect
        ldAbscSights = list()
        for _, sMissgTrans in dfMissgTransects.reset_index().iterrows():

            dAbscSight = dAbscSightTmpl.copy() # Copy template sightings
            dAbscSight.update(sMissgTrans.to_dict()) # Update transect columns only

            ldAbscSights.append(dAbscSight)

        # Add them to the input sightings
        return dfInSights.append(pd.DataFrame(pd.DataFrame(ldAbscSights)))

    # Add survey area information to sightings data (in-place modification)
    # * dfInSights : input data table
    # * dSurveyArea : a column name to scalar dictionary-like data to add to the table
    @staticmethod
    def _addSurveyAreaInfo(dfInSights, dSurveyArea):
        
        for col, values in dSurveyArea.items():
            dfInSights[col] = values
            
        return dfInSights
    
    # Sample individuals data for given sampling criteria, as a SampleDataSet.
    # * dSample : { key, value } selection criteria (with '+' support for 'or' operator in value),
    #             keys being columns of dfAllSights (dict protocol : dict, pd.Series, ...)
    def sampleDataSet(self, sSample):
        
        # Select sample data.
        dfSampIndivObs, dfSampTransInfo = \
            self._selectSampleSightings(dSample=sSample, dfAllSights=self._dfData,
                                        dfAllEffort=self.dfTransects, transectIdCols=self.transectIdCols,
                                        passIdCol=self.passIdCol, effortCol=self.effortCol)

        # Add absence sightings
        dfSampIndivObs = self._addAbsenceSightings(dfSampIndivObs, sampleCols=sSample.index, 
                                                   dfExpdTransects=dfSampTransInfo)

        # Add information about the studied geographical area
        dfSampIndivObs = self._addSurveyAreaInfo(dfSampIndivObs, dSurveyArea=self.dSurveyArea)

        # Create and return SampleDataSet instance (sort by transectIdCols : mandatory for MCDS analysis).
        return SampleDataSet(dfSampIndivObs, decimalFields=self.sampleDecFields, sortFields=self.transectIdCols)   

# A tabular input data set for multiple analyses on the same sample, with 1 or 0 individual per row
# Warning:
# * Only Point transect supported as for now
# * No change made afterwards on decimal precision : provide what you need !
# * Rows can be sorted if and as specified
# * Input support provided for pandas.DataFrame, Excel .xlsx file, tab-separated .csv/.txt files,
#   and even OpenDoc .ods file with pandas >= 0.25 (needs odfpy module)
class SampleDataSet(DataSet):
    
    def __init__(self, source, decimalFields=[], sortFields=[]):
        
        self.decimalFields = decimalFields

        super().__init__(source, importDecFields=decimalFields)
                
        assert not self._dfData.empty, 'No data in set'
        assert len(self._dfData.columns) >= 5, 'Not enough columns (should be at leat 5)'
        
        missCols = [decFld for decFld in self.decimalFields if decFld not in self._dfData.columns]
        assert not missCols, '{} declared decimal field(s) are not in source columns {} : {}' \
                             .format(len(missCols), ','.join(self._dfData.columns), ','.join(missCols))
                             
        # Sort / group sightings as specified
        if sortFields:
            self._dfData.sort_values(by=sortFields, inplace=True)


# A result set for multiple analyses from the same engine.
# With ability to prepend custom heading columns to the engine output stat ones.
# And to get a 3-level multi-index columned or a mono-indexed translated columned version of the data table.
class ResultsSet(object):
    
    def __init__(self, analysisClass, miCustomCols=None, dfCustomColTrans=None,
                       dComputedCols=None, dfComputedColTrans=None, sortCols=[], sortAscend=[]):
        
        assert issubclass(analysisClass, DSAnalysis), 'analysisClass must derive from DSAnalysis'
        assert miCustomCols is None or isinstance(miCustomCols, pd.MultiIndex), \
               'customCols must be None or a pd.MultiIndex'
        assert miCustomCols is None or len(miCustomCols.levels) == 3, \
               'customCols must have 3 levels if not None'
        if dComputedCols is None:
            dComputedCols = dict()
            dfComputedColTrans = pd.DataFrame()
        assert len(sortCols) == len(sortAscend), 'sortCols and sortAscend must have same length'
        
        self.analysisClass = analysisClass
        self.engineClass = analysisClass.EngineClass
    
        # 3-level multi-index columns (module, statistic, figure)
        self.miAnalysisCols = analysisClass.MIRunColumns.append(self.engineClass.statModCols())
        for col, ind in dComputedCols.items(): # Add post-computed columns at the right place
            self.miAnalysisCols = self.miAnalysisCols.insert(ind, col)
        self.computedCols = list(dComputedCols.keys())
        self.miCustomCols = miCustomCols
        
        # DataFrames for translating 3-level multi-index columns to 1 level lang-translated columns
        self.dfAnalysisColTrans = analysisClass.DfRunColumnTrans.append(self.engineClass.statModColTrans())
        self.dfAnalysisColTrans = self.dfAnalysisColTrans.append(dfComputedColTrans)
        self.dfCustomColTrans = dfCustomColTrans
        
        # Sorting parameters (after postComuting)
        self.sortCols = sortCols
        self.sortAscend = sortAscend
        
        # Non-constant data members
        self._dfData = pd.DataFrame() # The real data (frame).
        self.rightColOrder = False # self._dfData columns are assumed to be in a wrong order.
        self.postComputed = False # Post-computation not yet done.
    
    def __len__(self):
        
        return len(self._dfData)
    
    # sResult : Series for result cols values
    # sCustomHead : Series for custom cols values
    def append(self, sResult, sCustomHead=None):
        
        assert not self.postComputed, 'Can\'t append after columns post-computation'
        
        assert isinstance(sResult, pd.Series), 'sResult : Can only append a pd.Series'
        assert sCustomHead is None or isinstance(sCustomHead, pd.Series), 'sCustom : Can only append a pd.Series'
        
        if sCustomHead is not None:
            sResult = sCustomHead.append(sResult)
        
        self._dfData = self._dfData.append(sResult, ignore_index=True)
        
        # Appending (or concat'ing) often changes columns order
        self.rightColOrder = False
        
    # Post-computations.
    def postComputeColumns(self):
        
        # Derive class to really compute things (=> work directly on self._dfData),
        pass
        
    @property
    def dfData(self):
        
        # Do post-computation and sorting if not already done.
        if not(self._dfData.empty or self.postComputed):
            
            # Make sure we jave a MultiIndex for columns (append breaks this, not fromExcel)
            if not isinstance(self._dfData.columns, pd.MultiIndex):
                self._dfData.columns = pd.MultiIndex.from_tuples(self._dfData.columns)
            
            # Post-compute as specified (or not).
            self.postComputeColumns()
            self.postComputed = True # No need to do it again !
            
            # Sort as/if specified.
            if self.sortCols:
                self._dfData.sort_values(by=self.sortCols, ascending=self.sortAscend, inplace=True)
        
        # Enforce right columns order.
        if not(self._dfData.empty or self.rightColOrder):
            
            miTgtColumns = self.miAnalysisCols
            if self.miCustomCols is not None:
                miTgtColumns = self.miCustomCols.append(miTgtColumns)
            self._dfData = self._dfData.reindex(columns=miTgtColumns)
            self.rightColOrder = True # No need to do it again, until next append() !
        
        # This is also documentation line !
        if not self._dfData.empty:
            assert isinstance(self._dfData.columns, pd.MultiIndex)
        
        # Don't return columns with no relevant data.
        return self._dfData.dropna(how='all', axis='columns')

    @dfData.setter
    def dfData(self, dfData):
        
        assert isinstance(dfData, pd.DataFrame), 'dfData must be a pd.DataFrame'
        
        self._dfData = dfData.copy()
        
        # Let's assume that columns order is dirty.
        self.rightColOrder = False
        
        # Remove any computed column (will be recomputed later on).
        self._dfData.drop(columns=self.computedCols, inplace=True)
        
        # Post-computation not yet done.
        self.postComputed = False
    
    # Get translate names of custom columns
    def transCustomColumns(self, lang):
        
        return self.dfCustomColTrans[lang].to_list()
    
    # Build translation table for lang from custom one and analysis one
    def transTable(self):
        
        dfColTrans = self.dfAnalysisColTrans
        if self.dfCustomColTrans is not None:
            dfColTrans = self.dfCustomColTrans.append(dfColTrans)
            
        return dfColTrans
        
    # Access a mono-indexed translated columns version of the data table
    def dfTransData(self, lang='en', subset=None):
        
        assert lang in ['en', 'fr'], 'No support for "{}" language'.format(lang)
        assert subset is None or isinstance(subset, list) or isinstance(subset, pd.MultiIndex), \
               'subset columns must be specified as None (all), or as a list of tuples, or as a pandas.MultiIndex'
        
        # Make a copy of / extract selected columns of dfData.
        if subset is None:
            dfTrData = self.dfData
        else:
            if isinstance(subset, list):
                miSubset = pd.MultiIndex.from_tuples(subset)
            else: # pd.MultiIndex
                miSubset = subset
            dfTrData = self.dfData.reindex(columns=miSubset)
        dfTrData = dfTrData.copy()
        
        # Translate column names.
        dfTrData.columns = [self.transTable()[lang].get(col, str(col)) for col in dfTrData.columns]
        
        return dfTrData

    # Save data to an Excel worksheet (XLSX format).
    def toExcel(self, fileName, sheetName=None):
        
        self.dfData.to_excel(fileName, sheet_name=sheetName or 'AllResults')

    # Save data to an Open Document worksheet (ODS format).
    def toOpenDoc(self, fileName, sheetName=None):
        
        raise NotImplementedError('Can\'t export to OpenDoc yet')
        #self.dfData.to_excel(fileName, sheet_name=sheetName or 'AllResults', engine='odf')

    # Load (overwrite) data from an Excel worksheet (XLSX format), assuming ctor params match with Excel sheet
    # column names and list, which can well be ensured by using the same ctor params as used for saving !
    def fromExcel(self, fileName, sheetName=None):
        
        self.dfData = pd.read_excel(fileName, sheet_name=sheetName or 'AllResults', 
                                    header=[0, 1, 2], skiprows=[3], index_col=0)

    # Load (overwrite) data from an Open Document worksheet (ODS format), assuming ctor params match
    # with ODF sheet column names and list, hich can well be ensured by using the same ctor params as used for saving !
    # Notes: Needs odfpy module and pandas.version >= 0.25.1
    def fromOpenDoc(self, fileName, sheetName=None):
        
        self.dfData = pd.read_excel(fileName, sheet_name=sheetName or 'AllResults', 
                                    header=[0, 1, 2], skiprows=[3], index_col=0, engine='odf')

        
# A specialized results set for MCDS analyses,
# with extra. post-computed columns : Delta AIC, Chi2 P
class MCDSResultsSet(ResultsSet):
    
    DeltaAicColInd = ('detection probability', 'Delta AIC', 'Value')
    Chi2ColInd = ('detection probability', 'chi-square test probability determined', 'Value')

    # * miSampleCols : only for delta AIC computation ; defaults to miCustomCols if None
    #                  = the cols to use for grouping by sample
    def __init__(self, miCustomCols=None, dfCustomColTrans=None, miSampleCols=None,
                       sortCols=[], sortAscend=[]):
        
        # Setup computed columns specs.
        dCompCols = odict([(self.DeltaAicColInd, 22), # Right before AIC
                           (self.Chi2ColInd, 24)]) # Right before all Chi2 tests 
        dfCompColTrans = \
            pd.DataFrame(index=dCompCols.keys(),
                         data=dict(en=['Delta AIC', 'Chi2 P'], fr=['Delta AIC', 'Chi2 P']))

        # Initialise base.
        super().__init__(MCDSAnalysis, miCustomCols=miCustomCols, dfCustomColTrans=dfCustomColTrans,
                                       dComputedCols=dCompCols, dfComputedColTrans=dfCompColTrans,
                                       sortCols=sortCols, sortAscend=sortAscend)
        
        self.miSampleCols = miSampleCols if miSampleCols is not None else self.miCustomCols
        
    # Get translate names of custom columns
    def transSampleColumns(self, lang):
        
        return self.dfCustomColTrans.loc[self.miSampleCols, lang].to_list()
    
    # Post-computations.
    KMaxChi2 = 3 # TODO: Really a constant, or actually depends on some analysis params ?
    def postComputeColumns(self):
        
        #logger.debug('postComputeColumns: ...')
        
        # Compute Delta AIC (AIC - min(group)) per { species, periods, precision, duration } group.
        # a. Minimum AIC per group
        aicColInd = ('detection probability', 'AIC value', 'Value')
        aicGroupColInds = self.miSampleCols.to_list()
        df2Join = self._dfData.groupby(aicGroupColInds)[[aicColInd]].min()
        
        # b. Rename computed columns to target
        df2Join.columns = pd.MultiIndex.from_tuples([self.DeltaAicColInd])
        
        # c. Join the column to the target data-frame
        #logger.debug(str(self._dfData.columns) + ', ' + str(df2Join.columns))
        self._dfData = self._dfData.join(df2Join, on=aicGroupColInds)
        
        # d. Compute delta-AIC in-place
        self._dfData[self.DeltaAicColInd] = self._dfData[aicColInd] - self._dfData[self.DeltaAicColInd]

        # Compute determined Chi2 test probability (last value of all the tests done).
        def determineChi2(sChi2All):
            for chi2 in sChi2All:
                if not np.isnan(chi2):
                    return chi2
            return np.nan
        chi2AllColInds = [('detection probability', 'chi-square test probability (distance set {})'.format(i), 'Value') \
                          for i in range(self.KMaxChi2, 0, -1)]
        chi2AllColInds = [col for col in chi2AllColInds if col in self._dfData.columns]
        if chi2AllColInds:
            self._dfData[self.Chi2ColInd] = self._dfData[chi2AllColInds].apply(determineChi2, axis='columns')


if __name__ == '__main__':

    sys.exit(0)
