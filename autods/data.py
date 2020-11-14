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
from packaging import version as pkgver

import numpy as np
import pandas as pd

import autods.log as log

logger = log.logger('ads.dat')

from autods.analysis import DSAnalysis, MCDSAnalysis


class DataSet(object):

    """"A tabular data set built from various input sources (only 1 table supported)"""
    
    def __init__(self, source, importDecFields=[], separator='\t', sheet=None):
    
        """Ctor
        :param source: input support provided for pandas.DataFrame, Excel .xlsx file,
             tab-separated .csv/.txt files, and even OpenDoc .ods file with pandas >= 0.25 (needs odfpy module)
        :param decimalFields: for smart ./, decimal character management in CSV sources (pandas is not smart on this)
        :param separator: columns separator for CSV sources
        :param sheet: name of the sheet to read from, for multi-sheet data files
                                (like Excel or Open Doc. workbooks)
        """
    
        if isinstance(source, str) or isinstance(source, pl.Path):
            self._dfData = self._fromDataFile(source, importDecFields, separator, sheet)
        elif isinstance(source, pd.DataFrame):
            self._dfData = self._fromDataFrame(source)
        else:
            raise Exception('source for DataSet must be a pandas.DataFrame or an existing file')

        assert not self._dfData.empty, 'No data in source data set'

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
    
    SupportedFileExts = ['.xlsx', '.csv', '.txt'] \
                        + (['.ods'] if pkgver.parse(pd.__version__).release >= (0, 25) else [])
    
    @classmethod
    def _fromDataFile(cls, sourceFpn, decimalFields, separator='\t', sheet=None):
        
        if isinstance(sourceFpn, str):
            sourceFpn = pl.Path(sourceFpn)
    
        assert sourceFpn.exists(), 'Source file for DataSet not found : {}'.format(sourceFpn)

        ext = sourceFpn.suffix.lower()
        assert ext in cls.SupportedFileExts, \
               'Unsupported source file type {}: not from {{{}}}' \
               .format(ext, ','.join(cls.SupportedFileExts))
        if ext in ['.xlsx']:
            dfData = pd.read_excel(sourceFpn, sheet_name=sheet or 0)
        elif ext in ['.ods']:
            dfData = pd.read_excel(sourceFpn, sheet_name=sheet or 0, engine='odf')
        elif ext in ['.csv', '.txt']:
            dfData = cls._csv2df(sourceFpn, decCols=decimalFields, sep=separator)
            
        return dfData
    
    @classmethod
    def _fromDataFrame(cls, sourceDf):
        
        return sourceDf.copy()
    
    def __len__(self):
        
        return len(self._dfData)
    
    @property
    def dfData(self):
        
        return self._dfData

    @dfData.setter
    def dfData(self, dfData_):
        
        raise NotImplementedError('No change allowed to data ; create a new dataset !')


# A tabular data set for producing mono-category or even indivividuals data sets
# from "raw sightings data", aka "field data" (with possibly multiple category counts on each row)
# * Input support provided for pandas.DataFrame, Excel .xlsx file, tab-separated .csv/.txt files,
#   and even OpenDoc .ods file with pandas >= 0.25 (needs odfpy module)
class FieldDataSet(DataSet):

    # Ctor
    # Input support provided for pandas.DataFrame, Excel .xlsx file, tab-separated .csv/.txt files,
    # and even OpenDoc .ods file with pandas >= 0.25 (needs odfpy module)
    # * source: the input field data table
    # * countCols: the category columns (each of them holds counts of individuals for the category)
    # * addMonoCatCols: name and method of computing for columns to add after separating multi-category counts
    #   (each column to add is computed through :
    #      dfMonoCatSights[colName] = dfMonoCatSights[].apply(computeCol, axis='columns')
    #      for colName, computeCol in addMonoCatCols.items()) 
    def __init__(self, source, countCols, addMonoCatCols=dict(), importDecFields=[], separator='\t', sheet=None):
        
        super().__init__(source, importDecFields, separator, sheet)
        
        self.countCols = countCols
        self.dCompdMonoCatColSpecs = addMonoCatCols
        
        self.dfIndivSights = None # Not yet computed.
        self.dfMonoCatSights = None # Idem.
        
        logger.info(f'Field data : {len(self)} sightings')

    @property
    def dfData(self):
        
        return self._dfData

    @dfData.setter
    def dfData(self, dfData_):
        
        assert sorted(self._dfData.columns) == sorted(dfData_.columns), 'Can\'t set data with diffent columns'
        
        self._dfData = dfData_
        
        self.dfIndivSights = None # Not yet computed.
        self.dfMonoCatSights = None # Idem.
    
    # Transform a multi-category sightings set into an equivalent mono-category sightings set,
    # that is where no sightings has more that one category with positive count (keeping the same total counts).
    # Highly optimized version.
    # Ex: A sightings set with 2 category count columns nMales and nFemales
    #     * in the input set, you may have 1 sightings with nMales = 5 and nFemales = 2
    #     * in the output set, this sightings have been separated in 2 distinct ones (all other properties left untouched) :
    #       the 1st with nMales = 5 and nFemales = 0, the 2nd with nMales = 0 and nFemales = 2.
    @staticmethod
    def _separateMultiCategoryCounts(dfInSights, countColumns):
        
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

    # Transform a multi-individual mono-category sightings set into an equivalent mono-individual
    # mono-category sightings set, that is where no sightings has more that one individual
    # per category (keeping the same total counts).
    # Highly optimized version.
    # Ex: A sightings set with 2 mono-category count columns nMales and nFemales
    #     * in tyhe input set, you may have 1 sightings with nMales = 3 and nFemales = 0
    #       (but none with nMales and nFemales > 0)
    #     * in the output set, this sightings have been separated in 3 distinct ones
    #       (all other properties left untouched) : all with nMales = 1 and nFemales = 0.
    @staticmethod
    def _individualiseMonoCategoryCounts(dfInSights, countColumns):
        
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
        
    # Access to the resulting mono-category data set
    def monoCategorise(self, copy=True):
    
        # Compute only if not already done.
        if self.dfMonoCatSights is None:
        
            # Separate multi-category counts
            self.dfMonoCatSights = self._separateMultiCategoryCounts(self._dfData, self.countCols)
            
            # Compute and add supplementary mono-category columns
            for colName, computeCol in self.dCompdMonoCatColSpecs.items():
                self.dfMonoCatSights[colName] = self.dfMonoCatSights[self.countCols].apply(computeCol, axis='columns')
        
        return self.dfMonoCatSights if copy else self.dfMonoCatSights
    
    # Access to the resulting individuals data set 
    # = a mono-category data set with individualised data : 1 individuals per row
    def individualise(self, copy=True):
    
        # Compute only if not already done.
        if self.dfIndivSights is None:
        
            # Individualise mono-category counts
            self.dfIndivSights = \
                self._individualiseMonoCategoryCounts(self.monoCategorise(copy=False), self.countCols)

        return self.dfIndivSights.copy() if copy else self.dfIndivSights
        

# A tabular data set for producing on-demand sample data sets from "mono-category sightings data"
# * Input support provided for pandas.DataFrame (from FieldDataSet.indivulaise/monoCategorise),
#   Excel .xlsx file, tab-separated .csv/.txt files, and even OpenDoc .ods file with pandas >= 0.25 (needs odfpy module)
class MonoCategoryDataSet(DataSet):

    # Ctor
    # * dfTransects: Transects infos with columns : transectPlaceCols (n), passIdCol (1), effortCol (1)
    #                If None, auto generated from input sightings
    # * effortConstVal: if dfTransects is None and effortCol not in source table, use this constant value
    def __init__(self, source, dSurveyArea, importDecFields=[], dfTransects=None,
                       transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                       sampleDecFields=['Effort', 'Distance'], effortConstVal=1, 
                       separator='\t', sheet=None):
        
        super().__init__(source, importDecFields, separator, sheet)
        
        self.dSurveyArea = dSurveyArea
        self.transectPlaceCols = transectPlaceCols
        self.passIdCol = passIdCol
        self.effortCol = effortCol
        self.sampleDecFields = sampleDecFields
    
        self.dfTransects = dfTransects
        if self.dfTransects is None or self.dfTransects.empty:
            self.dfTransects = self._extractTransects(self._dfData, transectPlaceCols=self.transectPlaceCols,
                                                      passIdCol=self.passIdCol, effortConstVal=effortConstVal)
        elif effortCol not in self.dfTransects.columns:
            self.dfTransects[effortCol] = effortConstVal

        # A cache used by sampleDataSet to optimise consecutive calls with same sample specs (happens often).
        self._sdsSampleDataSetCache = None
        self._sSampleSpecsCache = None
        
        logger.info(f'Individuals data : {len(self)} sightings, {len(self.dfTransects)} transects')

    @property
    def dfData(self):
        
        return self._dfData

    @dfData.setter
    def dfData(self, dfData_):
        
        assert sorted(self._dfData.columns) == sorted(dfData_.columns), 'Can\'t set data with diffent columns'
        
        self._dfData = dfData_
    
    # Extract transect infos from individuals sightings
    # * effortConstVal: if effortCol not in dfIndivSightings, create one with this constant value
    @staticmethod
    def _extractTransects(dfIndivSightings, transectPlaceCols=['Transect'], passIdCol='Pass', 
                                            effortCol='Effort', effortConstVal=1):
    
        transCols = transectPlaceCols + [passIdCol]
        if effortCol in dfIndivSightings.columns:
            transCols.append(effortCol)
        
        dfTrans = dfIndivSightings[transCols]
        
        dfTrans = dfTrans.drop_duplicates()
        
        dfTrans.reset_index(drop=True, inplace=True)
        
        if effortCol not in dfTrans.columns:
            dfTrans[effortCol] = effortConstVal

        return dfTrans
    
    # Select sample sightings from an all-samples sightings table,
    # and compute the associated total sample effort.
    # * dSample : { key, value } selection criteria (with '+' support for 'or' operator in value),
    #             keys being columns of dfAllSights (dict protocol : dict, pd.Series, ...)
    # * dfAllSights : the all-samples (individual) sightings table to search into
    # * dfAllEffort : effort values for each transect x pass really done, for the all-sample survey
    # * transectPlaceCols : name of the input dfAllEffort and dSample columns to identify the transects (not passes)
    # * passIdCol : name of the input dfAllEffort and dSample column to identify the passes (not transects)
    # * effortCol : name of the input dfAllEffort and output effort column to add / replace
    @staticmethod
    def _selectSampleSightings(dSample, dfAllSights, dfAllEffort,
                               transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort'):
        
        # Select sightings
        dfSampSights = dfAllSights
        for key, values in dSample.items():
            values = str(values).strip() # For ints as strings that get forced to int in io sometimes (ex. from_excel)
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
        dfSampEffort = dfSampEffort[transectPlaceCols + [effortCol]].groupby(transectPlaceCols).sum()
        
        # Add effort column
        dfSampSights = dfSampSights.drop(columns=effortCol, errors='ignore').join(dfSampEffort, on=transectPlaceCols)

        return dfSampSights, dfSampEffort
        
    # Add "abscence" sightings to field data collected on transects for a given sample
    # * dfInSights : input data table
    # * sampleCols : the names of the sample identification columns
    # * dfExpdTransects : the expected transects, as a data frame indexed by the transectPlace,
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
        transectPlaceCols = dfExpdTransects.index.name
        dfSampTransects = dfInSights.drop_duplicates(subset=transectPlaceCols)
        dfSampTransects.set_index(transectPlaceCols, inplace=True)
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
    # * sSample : { key, value } selection criteria (with '+' support for 'or' operator in value, no separating space),
    #             keys being columns of dfAllSights (dict protocol : dict, pd.Series, ...)
    def sampleDataSet(self, sSampleSpecs):
        
        # Don't redo what have just been done.
        if self._sSampleSpecsCache is not None and self._sSampleSpecsCache.equals(sSampleSpecs):
            return self._sdsSampleDataSetCache
        
        # Select sample data.
        dfSampIndivObs, dfSampTransInfo = \
            self._selectSampleSightings(dSample=sSampleSpecs, dfAllSights=self._dfData,
                                        dfAllEffort=self.dfTransects, transectPlaceCols=self.transectPlaceCols,
                                        passIdCol=self.passIdCol, effortCol=self.effortCol)
        
        # Don't go on if no selected data.
        if dfSampIndivObs.empty:
            logger.warning(f'Not even a single individual sighting selected for these specs: {sSampleSpecs.to_dict()}')
            return None

        # Add absence sightings
        dfSampIndivObs = self._addAbsenceSightings(dfSampIndivObs, sampleCols=sSampleSpecs.index,
                                                   dfExpdTransects=dfSampTransInfo)

        # Add information about the studied geographical area
        dfSampIndivObs = self._addSurveyAreaInfo(dfSampIndivObs, dSurveyArea=self.dSurveyArea)

        # Create SampleDataSet instance (sort by transectPlaceCols : mandatory for MCDS analysis) and save it into cache.
        self._sdsSampleDataSetCache = \
            SampleDataSet(dfSampIndivObs, decimalFields=self.sampleDecFields, sortFields=self.transectPlaceCols)
        self._sSampleSpecsCache = sSampleSpecs
        
        # Done.
        return self._sdsSampleDataSetCache

# A tabular input data set for multiple analyses on the same sample, with 1 or 0 individual per row
# Warning:
# * Only Point transect supported as for now
# * No change made afterwards on decimal precision : provide what you need !
# * Rows can be sorted if and as specified
# * Input support provided for pandas.DataFrame, Excel .xlsx file, tab-separated .csv/.txt files,
#   and even OpenDoc .ods file with pandas >= 0.25 (needs odfpy module)
class SampleDataSet(DataSet):
    
    def __init__(self, source, decimalFields=[], sortFields=[], separator='\t', sheet=None):
        
        self.decimalFields = decimalFields

        super().__init__(source, importDecFields=decimalFields, separator=separator, sheet=sheet)
                
        assert not self._dfData.empty, 'No data in set'
        assert len(self._dfData.columns) >= 5, 'Not enough columns (should be at leat 5)'
        
        missCols = [decFld for decFld in self.decimalFields if decFld not in self._dfData.columns]
        assert not missCols, '{} declared decimal field(s) are not in source columns {} : {}' \
                             .format(len(missCols), ','.join(self._dfData.columns), ','.join(missCols))
                             
        # Sort / group sightings as specified
        if sortFields:
            self._dfData.sort_values(by=sortFields, inplace=True)

        # Report some basic stats.
        nAbscRows = self._dfData.isna().any(axis='columns').sum()
        logger.info('Sample data : {} sightings = {} individuals + {} absence rows'
                    .format(len(self), len(self) - nAbscRows, nAbscRows))


class ResultsSet(object):
    
    """
    A tabular result set for some computation process repeated multiple times with different input / results,
    each process result(s) being given as a pd.Series (1 row for the target table) or a pd.DataFrame (multiple rows).
    
    With ability to prepend custom heading columns to each process results ones
    (same value for each 1-process results rows).
    
    The columns of the internal pd.DataFrame, like those of each process result(s), can be a multi-index.
    
    Support for column translation is included (a mono-indexed).
    """

    def __init__(self, miCols, dfColTrans=None, miCustomCols=None, dfCustomColTrans=None,
                       dComputedCols=None, dfComputedColTrans=None, sortCols=[], sortAscend=[]):
                       
        """Ctor
        
        :param miCols: process results columns (MultiIndex or not)
        :param miCustomCols: custom columns to prepend (on the left) to process results columns (MultiIndex or not)
        """
        
        assert len(sortCols) == len(sortAscend), 'sortCols and sortAscend must have same length'

        # Columns stuff.
        if dComputedCols is None:
            dComputedCols = dict()
            dfComputedColTrans = pd.DataFrame()

        self.miCols = miCols
        for col, ind in dComputedCols.items(): # Add post-computed columns at the right place
            self.miCols = self.miCols.insert(ind, col)
        self.computedCols = list(dComputedCols.keys())
        self.miCustomCols = miCustomCols
        
        self.isMultiIndexedCols = isinstance(miCols, pd.MultiIndex)
        
        # DataFrames for translating 3-level multi-index columns to 1 level lang-translated columns
        self.dfColTrans = dfColTrans
        self.dfColTrans = self.dfColTrans.append(dfComputedColTrans)
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
    
    @property
    def columns(self):
    
        return self.dfData.columns
        
    def dropRows(self, sbSelRows):
    
        """Drop specific rows in-place, selected through boolean indexing on self.dfData
        
        Parameters:
        :param sbSelRows: boolean series with same index as self.dfData"""
    
        self._dfData.drop(self._dfData[sbSelRows].index, inplace=True)
        
    def copy(self, withData=True):
    
        """Clone function (shallow), with optional (deep) data copy"""
    
        # 1. Call ctor without computed columns stuff for now and here (we no more have initial data)
        clone = ResultsSet(miCustomCols=self.miCustomCols, dfCustomColTrans=self.dfCustomColTrans,
                           sortCols=[], sortAscend=[])
    
        # 2. Complete clone initialisation.
        clone.miCols = self.miCols
        clone.computedCols = self.computedCols
        
        # DataFrames for translating 3-level multi-index columns to 1 level lang-translated columns
        clone.dfColTrans = self.dfColTrans
        
        # Copy data if needed.
        if withData:
            clone._dfData = self._dfData.copy()
            clone.rightColOrder = self.rightColOrder
            clone.postComputed = self.postComputed

        return clone

    def append(self, sdfResult, sCustomHead=None):
        
        """Append row(s) of results to the all-results table
        :param sdfResult: the Series (1 row) or DataFrame (N rows) to append
        :param sCustomHead: Series holding custom cols values, to prepend (left) to each row of sdfResult,
            before appending to the internal table.
        
          |---- custom heading columns ----|---- real results ---------------------|
          |          |          |          |         |         |         |         |
          |               ...              |                  ...                  |
          |                ( data already there before append)                     |
          |               ...              |                  ...                  |
          |          |          |          |         |         |         |         |
          --------------------------------------------------------------------------
          |     x     |    y    |    z     |    a1   |    b1   |    c1   |    d1   |<- append(5 rows)
          |     x     |    y    |    z     |    a2   |    b2   |    c2   |    d2   |
          | (from sCustomHead, replicated) | (from sdfResult: here a 5-row table)  | 
          |     x     |    y    |    z     |    a4   |    b4   |    c4   |    d4   |
          |     x     |    y    |    z     |    a5   |    b5   |    c5   |    d5   |
          --------------------------------------------------------------------------
        """

        assert not self.postComputed, 'Can\'t append after columns post-computation'
        
        assert isinstance(sdfResult, (pd.Series, pd.DataFrame)), \
               'sdfResult : Can only append a pd.Series or pd.DataFrame'
        assert sCustomHead is None or isinstance(sCustomHead, pd.Series), \
               'sCustomHead : Can only append a pd.Series'
        
        if sCustomHead is not None:
            if isinstance(sdfResult, pd.Series):
                sdfResult = sCustomHead.append(sdfResult)
            else: # DataFrame
                dfCustomHead = pd.DataFrame([sCustomHead]*len(sdfResult)).reset_index(drop=True)
                sdfResult = pd.concat([dfCustomHead, sdfResult], axis='columns')
        
        if self._dfData.columns.empty:
            # In order to preserve types, we can't use self._dfData.append(sdfResult),
            # because it doesn't preserve original types (int => float)
            if isinstance(sdfResult, pd.Series):
                #self._dfData = sdfResult.to_frame().T  # This doesn't preserve types (=> all object)
                self._dfData = pd.DataFrame([sdfResult])
            else: # DataFrame
                self._dfData = sdfResult
        else:
            self._dfData = self._dfData.append(sdfResult, ignore_index=True)
        
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
        
            # Make sure we keep a MultiIndex for columns if it was the case (append breaks this, not fromExcel)
            if self.isMultiIndexedCols and not isinstance(self._dfData.columns, pd.MultiIndex):
                self._dfData.columns = pd.MultiIndex.from_tuples(self._dfData.columns)
            
            # Post-compute as specified (or not).
            self.postComputeColumns()
            self.postComputed = True # No need to do it again from now on !
            
            # Sort as/if specified.
            if self.sortCols:
                self._dfData.sort_values(by=self.sortCols, ascending=self.sortAscend, inplace=True)
        
        # Enforce right columns order.
        if not(self._dfData.empty or self.rightColOrder):
            
            miTgtColumns = self.miCols
            if self.miCustomCols is not None:
                miTgtColumns = self.miCustomCols.append(miTgtColumns)
            self._dfData = self._dfData.reindex(columns=miTgtColumns)
            self.rightColOrder = True # No need to do it again, until next append() !
        
        # This is also documentation line !
        if self.isMultiIndexedCols and not self._dfData.empty:
            assert isinstance(self._dfData.columns, pd.MultiIndex)
        
        # Don't return columns with no relevant results data (unless among custom cols).
        if not self._dfData.empty:
            miCols2Cleanup = self._dfData.columns
            if self.miCustomCols is not None:
                if self.isMultiIndexedCols:
                    miCols2Cleanup = miCols2Cleanup.drop(self.miCustomCols.to_list())
                else:
                    miCols2Cleanup = [col for col in miCols2Cleanup if col not in self.miCustomCols]
            cols2Drop = [col for col in miCols2Cleanup if self._dfData[col].isna().all()]
            self._dfData.drop(columns=cols2Drop, inplace=True)

        # Done.
        return self._dfData.copy()

    @dfData.setter
    def dfData(self, dfData):
        
        assert isinstance(dfData, pd.DataFrame), 'dfData must be a pd.DataFrame'
        
        self._dfData = dfData.copy()
        
        # Let's assume that columns order is dirty.
        self.rightColOrder = False
        
        # Remove any computed column: will be recomputed later on
        # (but ignore those which are not there : backward compat. / tolerance)
        self._dfData.drop(columns=self.computedCols, errors='ignore', inplace=True)
        
        # Post-computation not yet done.
        self.postComputed = False
    
    # Sort rows in place and overwrite initial sortCols / sortAscend values.
    def sortRows(self, by, ascending=True):
    
        self._dfData.sort_values(by=by, ascending=ascending, inplace=True)
        
        self.sortCols = by
        self.sortAscend = ascending
    
    # Build translation table for lang (from custom and other columns)
    def transTable(self):
        
        dfColTrans = self.dfColTrans
        if self.dfCustomColTrans is not None:
            dfColTrans = self.dfCustomColTrans.append(dfColTrans, sort=False)
            
        return dfColTrans
        
    # Get translated names of some columns (custom or not)
    def transColumns(self, columns, lang):
        
        return [self.transTable()[lang].get(col, str(col)) for col in columns]
    
    # Get translated names of custom columns
    def transCustomColumns(self, lang):
        
        return self.dfCustomColTrans[lang].to_list()
    
    def dfSubData(self, subset=None, copy=False):
    
        """Get a subset of the all-results table columns
        :param subset: columns to select, as a list(string) or pd.Index when mono-indexed columns
                                       or as a list(tuple(string*)) or pd.MultiIndex wehn multi-indexed.
        :param copy: if True, return a full copy of the data, not a "reference" to the internal table
        """
        
        assert subset is None or isinstance(subset, list) or isinstance(subset, (pd.Index, pd.MultiIndex)), \
               'subset columns must be specified as None (all), or as a list of tuples, or as a pandas.MultiIndex'

        # Make a copy of / extract selected columns of dfData.
        if subset is None:
            dfSbData = self.dfData
        else:
            if self.isMultiIndexedCols and isinstance(subset, list):
                iSubset = pd.MultiIndex.from_tuples(subset)
            else:
                iSubset = subset
            dfSbData = self.dfData.reindex(columns=iSubset)
        
        if copy:
            dfSbData = dfSbData.copy()
            
        return dfSbData

    # Access a mono-indexed translated columns version of the data table
    def dfTransData(self, lang='en', subset=None):
        
        assert lang in ['en', 'fr'], 'No support for "{}" language'.format(lang)
        
        # Extract and copy selected columns of dfData.
        dfTrData = self.dfSubData(subset=subset, copy=True)
        
        # Translate column names.
        dfTrData.columns = self.transColumns(dfTrData.columns, lang)
        
        return dfTrData

    # Save data to an Excel worksheet (XLSX format).
    def toExcel(self, fileName, sheetName=None, lang=None, subset=None, engine='openpyxl'):
        
        dfOutData = self.dfSubData(subset=subset) \
                    if lang is None else self.dfTransData(subset=subset, lang=lang)
        
        dfOutData.to_excel(fileName, sheet_name=sheetName or 'AllResults', engine=engine)

        logger.info(f'Results saved to {fileName} ({len(self)} rows)')

    # Save data to an Open Document worksheet (ODS format).
    def toOpenDoc(self, fileName, sheetName=None, lang=None, subset=None):
        
        assert pkgver.parse(pd.__version__).release >= (1, 1), \
               'Don\'t know how to write to OpenDoc format before Pandas 1.1'
        
        self.toExcel(fileName, sheetName, lang, subset, engine='odf')

    def fromExcel(self, fileName, sheetName=None, header=[0, 1, 2], skiprows=[3]):
        
        """Load (overwrite) data from an Excel worksheet (XLSX format),
        assuming ctor params match with Excel sheet column names and list,
        which can well be ensured by using the same ctor params as used for saving !
        """

        self.dfData = pd.read_excel(fileName, sheet_name=sheetName or 0, 
                                    header=header, skiprows=skiprows, index_col=0)

        logger.info(f'Loaded results from {fileName} ({len(self)} rows)')

    def fromOpenDoc(self, fileName, sheetName=None, header=[0, 1, 2], skiprows=[3]):
        
        """Load (overwrite) data from an Open Document worksheet (ODS format),
        assuming ctor params match with ODF sheet column names and list,
        which can well be ensured by using the same ctor params as used for saving !
        """

        assert pkgver.parse(pd.__version__).release >= (0, 25, 1), \
               'Don\'t know how to read from OpenDoc format before Pandas 0.25.1 (using odfpy module)'
        
        self.dfData = pd.read_excel(fileName, sheet_name=sheetName or 0, 
                                    header=header, skiprows=skiprows, index_col=0, engine='odf')

        logger.info(f'Loaded results from {fileName} ({len(self)} rows)')

    @staticmethod
    def _closeness(sLeftRight):

        """
        Relative closeness of 2 numbers : -round(log10((actual - reference) / max(abs(actual), abs(reference))), 1)
        = Compute the order of magnitude that separate the difference to the absolute max. of the two values.
        
        The greater it is, the lower the relative difference
           Ex: 3 = 10**3 ratio between max absolute difference of the two,
               +inf = NO difference at all,
               0 = bad, one of the two is 0, and the other not.
               
        See unitary test in unintests notebook.
        """

        x, y = sLeftRight.to_list()
        
        # Special cases with 1 NaN, or 1 or more inf => all different
        if np.isnan(x):
            if not np.isnan(y):
                return 0 # All different
        elif np.isnan(y):
            return 0 # All different
        
        if np.isinf(x) or np.isinf(y):
            return 0 # All different
        
        # Normal case
        c = abs(x - y)
        if not np.isnan(c) and c != 0:
            c = c / max(abs(x), abs(y))
        
        return np.inf if c == 0 else round(-np.log10(c), 1)

    # Make results cell values hashable (and especially analysis model params)
    # * needed for use in indexes (hashability)
    # * needed to cope with to_excel/read_excel unconsistent None management
    @staticmethod
    def _toHashable(value):
    
        if isinstance(value, list):
            hValue = str([float(v) for v in value])
        elif pd.isnull(value):
            hValue = 'None'
        elif isinstance(value, (int, float)):
            hValue = value
        elif isinstance(value, str):
            if ',' in value: # Assumed already somewhat stringified list
                hValue = str([float(v) for v in value.strip('[]').split(',')])
            else:
                hValue = str(value)
        else:
            hValue = str(value)
        
        return hValue
    
    @classmethod
    def compareDataFrames(cls, dfLeft, dfRight, subsetCols=[], indexCols=[], dropCloser=np.inf, dropNans=True):
    
        """
        Compare 2 DataFrames.
        
        Parameters:
        :param dfLeft: Left DataFrame
        :param dfRight: Right DataFrame
        :param list subsetCols: on a subset of columns,
        :param list indexCols: ignoring these columns, but keeping them as the index and sorting order,
        :param float dropCloser: with only rows with all cell closeness > dropCloser
                                 (default: np.inf => all cols and rows kept).
        :param bool dropNans: with only rows with all cell closeness > dropCloser or of NaN value ('cause NaN != Nan :-(.
        :returns: a diagnostic DataFrame with same columns and merged index, with a "closeness" value
                  for each cell (see _closeness method) ; rows with closeness > dropCloser are yet dropped.
        """
        
        # Make copies : we need to change the frames.
        dfLeft = dfLeft.copy()
        dfRight = dfRight.copy()
        
        # Check input columns
        dColsSets = { 'Subset column': subsetCols, 'Index column': indexCols }
        for colsSetName, colsSet in dColsSets.items():
            for col in colsSet:
                if col not in dfLeft.columns:
                    raise KeyError('{} {} not in left result set'.format(colsSetName, col))
                if col not in dfRight.columns:
                    raise KeyError('{} {} not in right result set'.format(colsSetName, col))
        
        # Set specified cols as the index (after making them hashable) and sort it.
        dfLeft[indexCols] = dfLeft[indexCols].applymap(cls._toHashable)
        dfLeft.set_index(indexCols, inplace=True)
        dfLeft = dfLeft.sort_index() # Not inplace: don't modify a copy/slice

        dfRight[indexCols] = dfRight[indexCols].applymap(cls._toHashable)
        dfRight.set_index(indexCols, inplace=True)
        dfRight = dfRight.sort_index() # Idem

        # Filter data to compare (subset of columns).
        if subsetCols:
            dfLeft = dfLeft[subsetCols]
            dfRight = dfRight[subsetCols]

        # Append mutually missing rows to the 2 tables => a complete and identical index.
        anyCol = dfLeft.columns[0] # Need one, whichever.
        dfLeft = dfLeft.join(dfRight[[anyCol]], rsuffix='_r', how='outer')
        dfLeft.drop(columns=dfLeft.columns[-1], inplace=True)
        dfRight = dfRight.join(dfLeft[[anyCol]], rsuffix='_l', how='outer')
        dfRight.drop(columns=dfRight.columns[-1], inplace=True)

        # Compare : Compute closeness 
        nColLevels = dfLeft.columns.nlevels
        KRightCol = 'tmp' if nColLevels == 1 else tuple('tmp{}'.format(i) for i in range(nColLevels))
        dfRelDiff = dfLeft.copy()
        exception = False
        for leftCol in dfLeft.columns:
            dfRelDiff[KRightCol] = dfRight[leftCol]
            try:
                dfRelDiff[leftCol] = dfRelDiff[[leftCol, KRightCol]].apply(cls._closeness, axis='columns')
            except TypeError as exc:
                logger.error(f'Column {leftCol} : {exc}')
                exception = True
            dfRelDiff.drop(columns=[KRightCol], inplace=True)
            
        if exception:
            raise TypeError('Stopping: Some columns could not be compared')
            
        # Complete comparison : rows with index not in both frames forced to all-0 closeness
        # (of course they should result so ... unless some NaNs here and there : fix this)
        dfRelDiff.loc[dfLeft[~dfLeft.index.isin(dfRight.index)].index, :] = 0
        dfRelDiff.loc[dfRight[~dfRight.index.isin(dfLeft.index)].index, :] = 0
        
        # Drop rows and columns with closeness over the threshold (or of NaN value if authorized)
        sbRows2Drop = dfRelDiff.applymap(lambda v: v > dropCloser or (dropNans and pd.isnull(v))).all(axis='columns')
        dfRelDiff.drop(dfRelDiff[sbRows2Drop].index, axis='index', inplace=True)
        
        return dfRelDiff

    def compare(self, rsOther, subsetCols=[], indexCols=[], dropCloser=np.inf, dropNans=True):
    
        """
        Compare 2 results sets.
        
        Parameters:
        :param rsOther: Right results object to comare
        :param list subsetCols: on a subset of columns,
        :param list indexCols: ignoring these columns, but keeping them as the index and sorting order,
        :param float dropCloser: with only rows with all cell closeness > dropCloser
                                 (default: np.inf => all cols and rows kept).
        :param bool dropNans: with only rows with all cell closeness > dropCloser or of NaN value ('cause NaN != Nan :-(.
        :returns: a diagnostic DataFrame with same columns and merged index, with a "closeness" value
                  for each cell (see _closeness method) ; rows with closeness > dropCloser are yet dropped.
        """
        
        return self.compareDataFrames(dfLeft=self.dfData, dfRight=rsOther.dfData,
                                      subsetCols=subsetCols, indexCols=indexCols,
                                      dropCloser=dropCloser, dropNans=dropNans)
        

if __name__ == '__main__':

    sys.exit(0)
