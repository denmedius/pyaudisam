# coding: utf-8

# PyAuDiSam: Automation of Distance Sampling analyses with Distance software (http://distancesampling.org/)

# Copyright (C) 2021 Jean-Philippe Meuret, Sylvain Sainnier

# This program is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see https://www.gnu.org/licenses/.

# Submodule "data": Input and output DS data manipulation tools

import sys
import pathlib as pl
from packaging import version as pkgver

import lzma
import pickle

import numpy as np
import pandas as pd

from . import log, runtime, utils

runtime.update(numpy=np.__version__, pandas=pd.__version__)

logger = log.logger('ads.dat')


class DataSet:

    """A tabular data set built by concatenating various-formatted source tables into one.
    
    Note: visionat module also defines this class: no reason it differs in any way => synchronise please !
          Why ? Just to keep the 2 modules independent ; might change in the future"""
    
    def __init__(self, sources, dRenameCols={}, dComputeCols={}, importDecFields=[],
                 sheet=None, skipRows=None, headerRows=0, indexCols=None, separator='\t', encoding='utf-8'):
    
        """Ctor
        :param sources: data sources to read from ; input support provided for:
             * pandas.DataFrame,
             * Excel .xlsx (through 'openpyxl' or 'xlrd' module)
               and .xls files (through 'openpyxl' module, no need for 'xlwt'),
             * tab-separated .csv/.txt files,
             * and even OpenDoc .ods file with pandas >= 0.25 (through 'odfpy' module) ;
             when multiple source provided, sources are supposed to have compatible columns names,
             and data rows from each source are appended 1 source after the previous.
        :param dRenameCols: dict for renaming input columns right after loading data
        :param dComputeCols: name and compute method for computed columns to be auto-added ;
                             as a dict { new col. name => constant, or function to apply
                             to each row to auto-compute the new sightings column } ;
                             note: these columns can also be renamed, through dRenameCols
        :param importDecFields: for smart ./, decimal character management in CSV sources (pandas is not smart on this)
        :param sheet: name of the sheet to read from, for multi-sheet data files (like Excel or Open Doc. workbooks)
        :param skipRows: list of indexes of initial rows to skip for file sources (before the column names row)
        :param headerRows: index (or list of) of rows holding columns names (default 0 => 1st row)
        :param indexCols: index (or list of) of first columns to use as (multi)index (default None => auto-gen index)
        :param separator: columns separator for CSV sources
        :param encoding: encoding for CSV sources
        """
    
        if isinstance(sources, (str, pl.Path)):
            self._dfData = self._fromDataFile(sources, sheet=sheet, decimalFields=importDecFields,
                                              skipRows=skipRows, headerRows=headerRows, indexCols=indexCols,
                                              separator=separator, encoding=encoding)
        elif isinstance(sources, pd.DataFrame):
            self._dfData = self._fromDataFrame(sources)
        elif isinstance(sources, list):
            ldfData = list()
            for source in sources:
                if isinstance(source, (str, pl.Path)):
                    dfData = self._fromDataFile(source, sheet=sheet, decimalFields=importDecFields,
                                                skipRows=skipRows, headerRows=headerRows, indexCols=indexCols,
                                                separator=separator, encoding=encoding)
                elif isinstance(source, pd.DataFrame):
                    dfData = self._fromDataFrame(source)
                    logger.info1('Loaded {} rows x {} columns from data frame'.format(len(dfData), len(dfData.columns)))
                else:
                    raise Exception('source for DataSet must be a pandas.DataFrame or an existing file')
                ldfData.append(dfData)
            self._dfData = pd.concat(ldfData, ignore_index=True)
        else:
            raise Exception('Source for DataSet must be a pandas.DataFrame or an existing file')

        if self._dfData.empty:
            logger.warning('No data in source data set')
            return
            
        logger.info(f'Loaded {len(self)} x {len(self.columns)} total rows x columns in data set ...')
        logger.info('... found columns: [{}]'.format('|'.join(str(c) for c in self.columns)))
        
        # Rename columns if requested.
        if dRenameCols:
            for key in dRenameCols.keys():
                if key in dComputeCols:
                    dComputeCols[dRenameCols[key]] = dComputeCols[key]
                    dComputeCols.pop(key)            
            self.renameColumns(dRenameCols)

        # Add auto-computed columns if any.
        if dComputeCols:

            self.addColumns(dComputeCols)

    # Wrapper around pd.read_csv for smart ./, decimal character management (pandas is not smart on this)
    # TODO: Make this more efficient
    @staticmethod
    def _csv2df(fileName, decCols, skipRows=None, headerRows=0, indexCols=None, sep='\t', encoding='utf-8'):
        df = pd.read_csv(fileName, sep=sep, skiprows=skipRows,
                         header=headerRows, index_col=indexCols)
        allRight = True
        for col in decCols:
            if df[col].dropna().apply(lambda v: isinstance(v, str)).any():
                allRight = False
                break
        if not allRight:
            df = pd.read_csv(fileName, sep=sep, skiprows=skipRows, encoding=encoding,
                             header=headerRows, index_col=indexCols, decimal=',')

        return df
    
    SupportedFileExts = \
        ['.xlsx', '.xls', '.csv', '.txt'] + (['.ods'] if pkgver.parse(pd.__version__).release >= (0, 25) else [])
    
    @classmethod
    def _fromDataFile(cls, sourceFpn, sheet=None, skipRows=None, headerRows=0, indexCols=None,
                      decimalFields=[], separator='\t', encoding='utf-8'):
        
        if isinstance(sourceFpn, str):
            sourceFpn = pl.Path(sourceFpn)
    
        assert sourceFpn.exists(), 'Source file for DataSet not found : {}'.format(sourceFpn)

        ext = sourceFpn.suffix.lower()
        assert ext in cls.SupportedFileExts, \
               'Unsupported source file type {}: not from {{{}}}'.format(ext, ','.join(cls.SupportedFileExts))

        logger.info1('Loading set from file {} ...'.format(sourceFpn.as_posix()))

        if ext in ['.xlsx', '.xls', '.ods']:
            dfData = pd.read_excel(sourceFpn, sheet_name=sheet or 0,
                                   skiprows=skipRows, header=headerRows, index_col=indexCols)
        elif ext in ['.csv', '.txt']:
            dfData = cls._csv2df(sourceFpn, decCols=decimalFields, sep=separator, encoding=encoding,
                                 skipRows=skipRows, headerRows=headerRows, indexCols=indexCols)
        else:
            raise NotImplementedError(f'Unsupported DataSet file input format {ext}')

        logger.info1('... loaded {} rows x {} columns'.format(len(dfData), len(dfData.columns)))

        return dfData
    
    @classmethod
    def _fromDataFrame(cls, sourceDf):
        
        return sourceDf.copy()
    
    def __len__(self):
        
        return len(self._dfData)
    
    @property
    def empty(self):
        
        return self._dfData.empty

    @property
    def columns(self):
        
        return self._dfData.columns

    @property
    def index(self):
        
        return self._dfData.index

    @property
    def dfData(self):
        
        return self._dfData

    @dfData.setter
    def dfData(self, dfData_):
        
        raise NotImplementedError('No change allowed to data ; create a new dataset !')

    def dfSubData(self, index=None, columns=None, copy=False):
    
        """Get a subset of the all-results table rows and columns

        Parameters:
        :param index: rows to select, as an iterable of a subset of self.dfData.index values
                      (anything suitable for pd.DataFrame.loc[...]) ; None = all rows.
        :param columns: columns to select, as a list(string) or pd.Index when mono-indexed columns
                       or as a list(tuple(string*)) or pd.MultiIndex when multi-indexed ; None = all columns.
        :param copy: if True, return a full copy of the selected data, not a "reference" to the internal table
        """
        
        assert columns is None or isinstance(columns, list) or isinstance(columns, (pd.Index, pd.MultiIndex)), \
               'columns must be specified as None (all), or as a list of tuples, or as a pandas.MultiIndex'

        # Make a copy of / extract selected columns of dfData.
        if columns is None or len(columns) == 0:
            dfSbData = self.dfData
        else:
            if isinstance(self._dfData.columns, pd.MultiIndex) and isinstance(columns, list):
                iColumns = pd.MultiIndex.from_tuples(columns)
            else:
                iColumns = columns
            dfSbData = self.dfData.reindex(columns=iColumns)
        
        if index is not None:
            dfSbData = dfSbData.loc[index]

        if copy:
            dfSbData = dfSbData.copy()
            
        return dfSbData

    def dropColumns(self, cols):
    
        self._dfData.drop(columns=cols, inplace=True)
        
    def dropRows(self, sbSelRows):
        
        self._dfData.drop(self._dfData[sbSelRows].index, inplace=True)
        
    @staticmethod
    def _addComputedColumns(dfData, dComputeCols):
    
        """Add computed columns to a DataFrame
        
        :param dfData: the DataFrame to update
        :param dComputeCols: dict new col. name => constant, or function to apply
                          to each row to compute its value
        """
        
        for colName, computeCol in dComputeCols.items():
            if callable(computeCol):
                dfData[colName] = dfData.apply(computeCol, axis='columns')
            else:
                dfData[colName] = computeCol
               
        return dfData  # Can be useful when chaining ...

    def addColumns(self, dComputeCols):
    
        """Add computed columns to the sightings data set
        
        :param dComputeCols: dict new col. name => constant, or function to apply
                          to each row to compute its value
        """
            
        self._addComputedColumns(self._dfData, dComputeCols)

    def renameColumns(self, dRenameCols):

        self._dfData.rename(columns=dRenameCols, inplace=True)

    def toPickle(self, fileName, subset=None, index=True):

        """Save data table to a pickle file (XZ compressed format if requested).

        Parameters:
        :param fileName: target file pathname ; file is auto-compressed to XZ format
                         through the lzma module if its extension is .xz or .lzma,
                         or else not compressed.
        :param subset: subset of columns to save
        :param index: if True, save index column as is (otherwise reset&drop it to a range(size)-like)
        """
        
        start = pd.Timestamp.now()

        dfOutData = self.dfSubData(columns=subset)
        if not index:
            dfOutData = dfOutData.reset_index(drop=True)

        compress = pl.Path(fileName).suffix in ['.xz', '.lzma']
        with lzma.open(fileName, 'wb') if compress else open(fileName, 'wb') as file:
            pickle.dump(dfOutData, file)

        logger.info('{} results rows saved to {} in {:.3f}s'
                    .format(len(self), fileName, (pd.Timestamp.now() - start).total_seconds()))

    def toExcel(self, fileName, sheetName=None, subset=None, index=True, engine=None):
        
        """Save data table to a worksheet format (Excel, ODF, ...) :
        * newer XLSX format for .xlsx extensions (through 'openpyxl' or 'xlrd' module)
        * .xls (through 'xlwt').
        * .ods (through 'odfpy'), if pandas >= 0.25.

        Parameters:
        :param fileName: target file pathname
        :param sheetName: for results data only
        :param subset: subset of columns to save
        :param index: if True, save index column
        :param engine: None => auto-selection from file extension ; otherwise, use xlrd, openpyxl or odf.
        """

        dfOutData = self.dfSubData(columns=subset)
        
        dfOutData.to_excel(fileName, sheet_name=sheetName or 'AllResults',
                           index=index, engine=engine)

    # Save data to an Open Document worksheet (through 'odfpy' module)
    def toOpenDoc(self, fileName, sheetName=None, subset=None, index=True):

        """Save data table to an Open Document worksheet, ODF format (through 'odfpy' module)

        Warning: Needs pandas >= 0.25

        Parameters:
        :param fileName: target file pathname
        :param sheetName: for results data only
        :param subset: subset of columns to save
        :param index: if True, save index column
        """
        
        assert pkgver.parse(pd.__version__).release >= (1, 1), \
               'Don\'t know how to write to OpenDoc format before Pandas 1.1'
        
        return self.toExcel(fileName, sheetName=sheetName, subset=subset,
                            index=index, engine='odf')  # Force engine in case not a .ods.

    @staticmethod
    def _closeness(sLeftRight):

        """
        Relative closeness of 2 numbers : -round(log10((actual - reference) / max(abs(actual), abs(reference))), 1)
        = Compute the order of magnitude that separates the difference to the absolute max. of the two values.
        
        The greater it is, the lower the relative difference
           Ex: 3 = 10**3 ratio between max absolute difference of the two,
               +inf = NO difference at all,
               0 = bad, one of the two is 0, and the other not.
        """

        x, y = sLeftRight.to_list()
        
        # Special cases with 1 NaN, or 1 or more inf => all different
        if np.isnan(x):
            if not np.isnan(y):
                return 0  # All different
        elif np.isnan(y):
            return 0  # All different
        
        if np.isinf(x) or np.isinf(y):
            return 0  # All different
        
        # Normal case
        c = abs(x - y)
        if not np.isnan(c) and c != 0:
            c = c / max(abs(x), abs(y))
        
        return np.inf if c == 0 else round(-np.log10(c), 1)

    # Make results cell values hashable (and especially analysis model params)
    # * needed for use in indexes (hashability)
    # * needed to cope with to_excel/read_excel Inconsistent None management
    @staticmethod
    def _toHashable(value):
    
        if isinstance(value, list):
            hValue = str([float(v) for v in value])
        elif pd.isnull(value):
            hValue = 'None'
        elif isinstance(value, (int, float)):
            hValue = value
        elif isinstance(value, str):
            if ',' in value:  # Assumed already somewhat stringified list
                hValue = str([float(v) for v in value.strip('[]').split(',')])
            else:
                hValue = str(value)
        else:
            hValue = str(value)
        
        return hValue
    
    @classmethod
    def compareDataFrames(cls, dfLeft, dfRight, subsetCols=[], indexCols=[],
                          noneIsNan=False, dropCloser=np.inf, dropNans=True, dropCloserCols=False):
    
        """
        Compare 2 DataFrames.

        The resulting diagnosis DataFrame will have the same columns and merged index,
        with a "closeness" value for each cell (see _closeness method) ;
        rows where all cells have closeness > dropCloser (or eventually NaN) are yet dropped.
        and columns where all cells have closeness > dropCloser (or eventually NaN) can also be dropped.
        
        Parameters:
        :param dfLeft: Left DataFrame
        :param dfRight: Right DataFrame
        :param list subsetCols: on a subset of columns,
        :param list indexCols: ignoring these columns, but keeping them as the index and sorting order,
        :param bool noneIsNan: if True, replace any None by a np.nan in left and right before comparing
        :param float dropCloser: result will only include rows with all cell closeness > dropCloser
                                 (default: np.inf => all cols and rows kept).
        :param bool dropNans: smoother condition for dropCloser : if True, NaN values are also considered > dropCloser
                              ('cause NaN != NaN :-( ).
        :param bool dropCloserCols: if True, also drop all "> dropCloser (or eventually NaN)"-all-cell columns,
                                    just as rows

        :return: the diagnostic DataFrame.
        """

        if dfLeft.empty and dfRight.empty:
            return pd.DataFrame()

        # Make copies : we need to change the frames.
        dfLeft = dfLeft.copy()
        dfRight = dfRight.copy()
        
        # Check input columns
        dColsSets = {'Subset column': subsetCols, 'Index column': indexCols}
        for colsSetName, colsSet in dColsSets.items():
            for col in colsSet:
                if col not in dfLeft.columns:
                    raise KeyError(f'{colsSetName} {col} not in left result set')
                if col not in dfRight.columns:
                    raise KeyError(f'{colsSetName} {col} not in right result set')
        
        # Set specified cols as the index (after making them hashable) and sort it.
        dfLeft[indexCols] = utils.mapDataFrame(dfLeft[indexCols], cls._toHashable)
        dfLeft.set_index(indexCols, inplace=True)
        dfLeft = dfLeft.sort_index()  # Not inplace: don't modify a copy/slice

        dfRight[indexCols] = utils.mapDataFrame(dfRight[indexCols], cls._toHashable)
        dfRight.set_index(indexCols, inplace=True)
        dfRight = dfRight.sort_index()  # Idem

        # Filter data to compare (subset of columns).
        if subsetCols:
            dfLeft = dfLeft[subsetCols]
            dfRight = dfRight[subsetCols]

        # Append mutually missing rows to the 2 tables => a complete and identical index.
        anyCol = dfLeft.columns[0]  # Need one, whichever.
        dfLeft = dfLeft.join(dfRight[[anyCol]], rsuffix='_r', how='outer')
        dfLeft.drop(columns=dfLeft.columns[-1], inplace=True)
        dfRight = dfRight.join(dfLeft[[anyCol]], rsuffix='_l', how='outer')
        dfRight.drop(columns=dfRight.columns[-1], inplace=True)

        # Replace None by NaNs if specified
        # Note: As for the option_context and infer_objects stuff ... it silents the warning
        #       with pandas 2.2, but not sure if it works when the new behaviour is enforced ...
        if noneIsNan:
            with pd.option_context("future.no_silent_downcasting", True):
                dfLeft = dfLeft.replace({None: np.nan}).infer_objects(copy=False)
                dfRight = dfRight.replace({None: np.nan}).infer_objects(copy=False)

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
                logger.error(f'_closeness failed for left column {leftCol}: {exc} ...')
                logger.error(f'* left: {dfRelDiff[leftCol].to_dict()}')
                logger.error(f'* right: {dfRelDiff[KRightCol].to_dict()}')
                exception = True
            dfRelDiff.drop(columns=[KRightCol], inplace=True)
            
        if exception:
            raise TypeError('Stopping: Some columns could not be compared (see errors logged above)')
            
        # Complete comparison : rows with index not in both frames forced to all-0 closeness
        # (of course they should result so ... unless some NaNs here and there : fix this)
        dfRelDiff.loc[dfLeft[~dfLeft.index.isin(dfRight.index)].index, :] = 0
        dfRelDiff.loc[dfRight[~dfRight.index.isin(dfLeft.index)].index, :] = 0
        
        # Drop rows (and maybe columns) with closeness over the threshold (or of NaN value if authorized)
        dfCells2Drop = utils.mapDataFrame(dfRelDiff,
                                          lambda v: v > dropCloser or (dropNans and pd.isnull(v)))
        dfRelDiff.drop(dfRelDiff[dfCells2Drop.all(axis='columns')].index, inplace=True)
        if dropCloserCols:
            # dfRelDiff.drop(columns=dfRelDiff.T[dfCells2Drop.all(axis='index')].index, axis='index', inplace=True)
            dfRelDiff.drop(columns=[col for col, drop in dfCells2Drop.all(axis='index').items() if drop],
                           inplace=True)  # 40% faster.
            
        return dfRelDiff

    def compare(self, other, subsetCols=[], indexCols=[],
                noneIsNan=False, dropCloser=np.inf, dropNans=True, dropCloserCols=False):
    
        """
        Compare 2 data sets.
        
        The resulting diagnosis DataFrame will have the same columns and merged index,
        with a "closeness" value for each cell (see _closeness method) ;
        rows where all cells have closeness > dropCloser (or eventually NaN) are yet dropped.
        and columns where all cells have closeness > dropCloser (or eventually NaN) can also be dropped.
        
        Parameters:
        :param other: Right data set or DataFrame object to compare
        :param list subsetCols: on a subset of columns,
        :param list indexCols: ignoring these columns, but keeping them as the index and sorting order,
        :param bool noneIsNan: if True, replace any None by a np.nan in left and right before comparing
        :param float dropCloser: result will only include rows with all cell closeness > dropCloser
                                 (default: np.inf => all cols and rows kept).
        :param bool dropNans: smoother condition for dropCloser : if True, NaN values are also considered > dropCloser
                              ('cause NaN != NaN :-( ).
        :param bool dropCloserCols: if True, also drop all "> dropCloser (or eventually NaN)" columns-all-cell,
                                    just as rows

        :return: the diagnostic DataFrame.
        """
        
        return self.compareDataFrames(dfLeft=self.dfData,
                                      dfRight=other if isinstance(other, pd.DataFrame) else other.dfData,
                                      subsetCols=subsetCols, indexCols=indexCols, noneIsNan=noneIsNan,
                                      dropCloser=dropCloser, dropNans=dropNans, dropCloserCols=dropCloserCols)


class FieldDataSet(DataSet):

    """A tabular data set for producing mono-category or even individuals data sets from "raw sightings data",
    aka "field data" (with possibly multiple category counts on each row)

    Input support provided for pandas.DataFrame, Excel .xlsx file, tab-separated .csv/.txt files,
      and even OpenDoc .ods file with pandas >= 0.25 (needs odfpy module)
    """
    def __init__(self, source, countCols, addMonoCatCols=dict(), importDecFields=[], sheet=None, separator='\t'):

        """Ctor

        Parameters:
        :param source: the input field-data table
        :param countCols: the category columns (each of them holds counts of individuals for the category)
        :param addMonoCatCols: name and method of computing for columns to add after separating multi category counts
             (each column to add is computed through :
                dfMonoCatSights[colName] = dfMonoCatSights[].apply(computeCol, axis='columns')
                for colName, computeCol in addMonoCatCols.items())
        :param importDecFields: for smart ./, decimal character management in CSV sources (pandas is not smart on this)
        :param sheet: name of the sheet to read from, for multi-sheet data input file
             (like Excel or Open Doc. workbooks)
        :param separator: columns separator for CSV input file
        """

        super().__init__(sources=source, importDecFields=importDecFields, sheet=sheet, separator=separator)
        
        self.countCols = countCols
        self.dCompdMonoCatColSpecs = addMonoCatCols
        
        self.dfIndivSights = None  # Not yet computed.
        self.dfMonoCatSights = None  # Idem.
        
        logger.info(f'Field data : {len(self)} sightings')

    @property
    def dfData(self):
        
        return self._dfData

    @dfData.setter
    def dfData(self, dfData_):
        
        assert sorted(self._dfData.columns) == sorted(dfData_.columns), "Can't set data with different columns"
        
        self._dfData = dfData_
        
        self.dfIndivSights = None  # Not yet computed.
        self.dfMonoCatSights = None  # Idem.
    
    # Transform a multi-category sightings set into an equivalent mono-category sightings set,
    # that is where no sighting has more than one category with positive count (keeping the same total counts).
    # Highly optimized version.
    # Ex: A sightings set with 2 category count columns nMales and nFemales
    #     * in the input set, you may have 1 sighting with nMales = 5 and nFemales = 2
    #     * in the output set, this sighting have been separated in 2 distinct ones
    #       (all other properties left untouched) :
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
    # mono-category sightings set, that is where no sighting has more than one individual
    # per category (keeping the same total counts).
    # Highly optimized version.
    # Ex: A sightings set with 2 mono-category count columns nMales and nFemales
    #     * in tyhe input set, you may have 1 sighting with nMales = 3 and nFemales = 0
    #       (but none with nMales and nFemales > 0)
    #     * in the output set, this sighting have been separated in 3 distinct ones
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
# * Input support provided for pandas.DataFrame (from FieldDataSet.individualise/monoCategorise),
#   Excel .xlsx file, tab-separated .csv/.txt files, and even OpenDoc .ods file with pandas >= 0.25 (needs odfpy module)
class MonoCategoryDataSet(DataSet):

    # Ctor
    # * dfTransects: Transects infos with columns : transectPlaceCols (n), passIdCol (1), effortCol (1)
    #                If None, auto generated from input sightings
    # * effortConstVal: if dfTransects is None and effortCol not in source table, use this constant value
    def __init__(self, source, dSurveyArea, importDecFields=[], dfTransects=None,
                 transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                 sampleDecFields=['Effort', 'Distance'], effortConstVal=1,
                 sheet=None, separator='\t'):
        
        super().__init__(sources=source, importDecFields=importDecFields, sheet=sheet, separator=separator)
        
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
        
        assert sorted(self._dfData.columns) == sorted(dfData_.columns), "Can't set data with different columns"
        
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
            values = str(values).strip()  # For ints as strings that get forced to int in io sometimes (ex. from_excel)
            if values and values not in ['nan', 'None']:  # Empty value means "no selection criteria for this column"
                values = values.split('+') if '+' in values else [values]
                dfSampSights = dfSampSights[dfSampSights[key].astype(str).isin(values)]

        # Compute sample effort
        passes = str(dSample[passIdCol])  # Same as above ...
        if passes and passes not in ['nan', 'None']:  # Same as above ...
            passes = passes.split('+') if '+' in passes else [passes]
            dfSampEffort = dfAllEffort[dfAllEffort[passIdCol].astype(str).isin(passes)]
        else:
            dfSampEffort = dfAllEffort
        dfSampEffort = dfSampEffort[transectPlaceCols + [effortCol]].groupby(transectPlaceCols).sum()
        
        # Add effort column
        dfSampSights = dfSampSights.drop(columns=effortCol, errors='ignore').join(dfSampEffort, on=transectPlaceCols)

        return dfSampSights, dfSampEffort
        
    # Add "absence" sightings to field data collected on transects for a given sample
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
        dAbscSightTmpl.update({k: None for k in dAbscSightTmpl.keys() if k not in sampleCols})

        # Determine missing transects for the sample
        transectPlaceCols = dfExpdTransects.index.name
        dfSampTransects = dfInSights.drop_duplicates(subset=transectPlaceCols)
        dfSampTransects.set_index(transectPlaceCols, inplace=True)
        dfSampTransects = dfSampTransects[dfExpdTransects.columns]

        dfMissgTransects = dfExpdTransects.loc[dfExpdTransects.index.difference(dfSampTransects.index)]

        # Generate the absence sightings : 1 row per missing transect
        ldAbscSights = list()
        for _, sMissgTrans in dfMissgTransects.reset_index().iterrows():

            dAbscSight = dAbscSightTmpl.copy()  # Copy template sightings
            dAbscSight.update(sMissgTrans.to_dict())  # Update transect columns only

            ldAbscSights.append(dAbscSight)

        # Add them to the input sightings
        return pd.concat([dfInSights, pd.DataFrame(ldAbscSights)])

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

        # Create SampleDataSet instance (sort by transectPlaceCols : mandatory for MCDS analysis)
        # and save it into cache.
        self._sdsSampleDataSetCache = \
            SampleDataSet(dfSampIndivObs, decimalFields=self.sampleDecFields, sortFields=self.transectPlaceCols)
        self._sSampleSpecsCache = sSampleSpecs
        
        # Done.
        return self._sdsSampleDataSetCache


class SampleDataSet(DataSet):

    """
    A tabular input data set for multiple analyses on the same sample, with 1 or 0 individual per row
    Warning:
    * Only Point transect supported as for now
    * No change made afterward on decimal precision : provide what you need !
    * Rows can be sorted if and as specified
    * Input support provided for pandas.DataFrame, Excel .xlsx file, tab-separated .csv/.txt files,
      and even OpenDoc .ods file with pandas >= 0.25 (needs odfpy module)
    """
    def __init__(self, source, decimalFields=[], sortFields=[], sheet=None, separator='\t'):
        
        self.decimalFields = decimalFields

        super().__init__(sources=source, importDecFields=decimalFields, sheet=sheet, separator=separator)
                
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


class ResultsSet:
    
    """
    A tabular result set for some computation process repeated multiple times with different input / results,
    each process result(s) being given as a pd.Series (1 row for the target table) or a pd.DataFrame (multiple rows).
    
    With ability to prepend custom heading columns to each process results ones
    (same value for each 1-process results rows).
    
    The columns of the internal pd.DataFrame, like those of each process result(s), can be a multi-index.
    
    Support for column translation is included (a mono-indexed).
    """

    def __init__(self, miCols, dfColTrans=None, miCustomCols=None, dfCustomColTrans=None,
                 dComputedCols=None, dfComputedColTrans=None, sortCols=[], sortAscend=[],
                 dropNACols=True, miExemptNACols=None):
    
        """Ctor
        
        Parameters:
        :param miCols: process results columns (MultiIndex or not)
        :param dfColTrans: translation DataFrame for results column labels,
                           * index=column labels,
                           * data=dict(<lang>=[<translation for each label> for all labels] for all <lang>)
        :param miCustomCols: custom columns to prepend (on the left) to process results columns (MultiIndex or not)
        :param dfCustomColTrans: translation DataFrame for custom column labels,
        :param dComputedCols: dict(label: position) for inserting computed columns in results table (None = after end)
        :param dfComputedColTrans: translation DataFrame for computed column labels,
        :param sortCols: if not empty, iterable of columns to sort values by in dfData()
        :param sortAscend: sorting order for all (bool), or each column (iterable of bool) in dfData()
        :param dropNACols: If True, dfData() won't return columns with no relevant results data
                           (except for custom cols and exempt cols).
        :param miExemptNACols: if not None, non-custom columns to keep even if
        dropNACols and all NaN
        """
        
        assert len(sortCols) == len(sortAscend), 'sortCols and sortAscend must have same length'

        # Columns stuff.
        if dComputedCols is None:
            dComputedCols = dict()
            dfComputedColTrans = pd.DataFrame()

        # Columns: Process results + computed results (at the specified position if any)
        self.miCols = miCols.copy()
        self.isMultiIndexedCols = isinstance(miCols, pd.MultiIndex)
        for col, ind in dComputedCols.items():
            if ind is not None:
                self.miCols = self.miCols.insert(ind, col)
        lastCompCols = [col for col, ind in dComputedCols.items() if ind is None]
        if lastCompCols:
            self.miCols = self.miCols.append(pd.MultiIndex.from_tuples(lastCompCols))

        # Columns: post-computed ones, and custom ones (to prepend to process results)
        self.computedCols = list(dComputedCols.keys())
        self.miCustomCols = miCustomCols.copy() if miCustomCols is not None else list()

        # DataFrames for translating 3-level multi-index columns to 1 level lang-translated columns
        self.dfColTrans = pd.concat([dfColTrans if dfColTrans is not None else pd.DataFrame(),
                                     dfComputedColTrans if dfComputedColTrans is not None else pd.DataFrame()])
        self.dfCustomColTrans = dfCustomColTrans.copy() if dfCustomColTrans is not None else pd.DataFrame()
        
        # Sorting and cleaning parameters (after postComputing)
        self.sortCols = sortCols
        self.sortAscend = sortAscend
        self.dropNACols = dropNACols
        self.miExemptNACols = miExemptNACols
        
        # Non-constant data members
        self._dfData = pd.DataFrame()  # The real data (frame).
        self.rightColOrder = False  # self._dfData columns are assumed to be in a wrong order.
        self.postComputed = False  # Post-computation not yet done.
    
        # Specifications of computations that led to the results
        self.specs = dict()

    def __len__(self):
        
        return len(self._dfData)
    
    @property
    def empty(self):
        
        return self._dfData.empty

    @property
    def columns(self):
        # Yes, using dfData triggers post-computation, we mean it !
        return self.dfData.columns
        
    @property
    def index(self):
        
        return self._dfData.index

    def dropRows(self, sbSelRows):
    
        """Drop specific rows in-place, selected through boolean indexing on self.dfData
        
        Parameters:
        :param sbSelRows: boolean series with same index as self.dfData"""
    
        self._dfData.drop(self._dfData[sbSelRows].index, inplace=True)
        
    def copy(self, withData=True):
    
        """Clone function (shallow), with optional (deep) data copy"""

        # 1. Call ctor without computed columns stuff for now and here (we no more have initial data)
        clone = ResultsSet(miCols=self.miCols,
                           miCustomCols=self.miCustomCols.copy(),
                           dfCustomColTrans=self.dfCustomColTrans.copy(),
                           sortCols=self.sortCols.copy(), sortAscend=self.sortAscend.copy(),
                           dropNACols=self.dropNACols, miExemptNACols=self.miExemptNACols.copy())
    
        # 2. Complete clone initialisation.
        clone.miCols = self.miCols.copy()
        clone.computedCols = self.computedCols.copy()
        
        # DataFrames for translating 3-level multi-index columns to 1 level lang-translated columns
        clone.dfColTrans = self.dfColTrans.copy()
        
        # Copy data if needed.
        if withData:
            clone._dfData = self._dfData.copy()
            clone.rightColOrder = self.rightColOrder
            clone.postComputed = self.postComputed

        return clone

    def _acceptNewColumns(self, newCols):

        """Update results columns list (self.miCols) with new columns if not already present"""

        logger.debug1(f'_acceptNewColumns: {newCols=}')
        logger.debug2(f'{self.miCols=}')
        logger.debug2(f'{self.miCustomCols=}')
        
        # Select columns to really append.
        newCols = [col for col in newCols if col not in self.miCols and col not in self.miCustomCols]
        logger.debug2(f'{newCols=}')

        if self.isMultiIndexedCols:
            self.miCols = self.miCols.append(newCols)
            newColTrans = [' '.join(col) for col in newCols]  # Quite rough, but what else ?
        else:
            self.miCols += newCols
            newColTrans = newCols
        logger.debug2(f'{self.miCols=}')

        # Update columns translation table also.
        dfNewColTrans = pd.DataFrame(index=newCols,
                                     data={lang: newColTrans for lang in self.dfColTrans.columns})
        self.dfColTrans = pd.concat([self.dfColTrans, dfNewColTrans])

    def append(self, sdfResult, sCustomHead=None, acceptNewCols=False):
        
        """Append row(s) of results to the all-results table
        :param sdfResult: the Series (1 row) or DataFrame (N rows) to append
        :param acceptNewCols: if True, append results columns list (self.miCols) dynamically 
            as unexpected columns appear in results to append
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

        assert not self.postComputed, "Can't append after columns post-computation"
        
        assert isinstance(sdfResult, (pd.Series, pd.DataFrame)), \
               'sdfResult : Can only append a pd.Series or pd.DataFrame'
        assert sCustomHead is None or isinstance(sCustomHead, pd.Series), \
               'sCustomHead : Can only append a pd.Series'

        # If specified, update results columns list (self.miCols) dynamically 
        # as unexpected columns appear in results to append.
        if acceptNewCols:
            resCols = list(sdfResult.index) if isinstance(sdfResult, pd.Series) else list(sdfResult.columns)
            self._acceptNewColumns(resCols)

        # Prepend header columns to results (on the left) if any.
        if sCustomHead is not None:
            if isinstance(sdfResult, pd.Series):
                sdfResult = pd.concat([sCustomHead, sdfResult])
            else:  # DataFrame
                dfCustomHead = pd.DataFrame([sCustomHead]*len(sdfResult)).reset_index(drop=True)
                sdfResult = pd.concat([dfCustomHead, sdfResult], axis='columns')
        
        # Append results rows to the already present ones (at the end)
        if self._dfData.columns.empty:
            # In order to preserve types, we can't use pd.concat([self._dfData, sdfResult]),
            # because it doesn't preserve original types (int => float)
            if isinstance(sdfResult, pd.Series):
                # self._dfData = sdfResult.to_frame().T  # This doesn't preserve types (=> all object)
                self._dfData = pd.DataFrame([sdfResult])
            else:  # DataFrame
                self._dfData = sdfResult
        else:
            if isinstance(sdfResult, pd.Series):
                # self._dfData = sdfResult.to_frame().T  # This doesn't preserve types (=> all object)
                sdfResult = pd.DataFrame([sdfResult])
            self._dfData = pd.concat([self._dfData, sdfResult], ignore_index=True)
        
        # Appending (or concat'ing) often changes columns order
        self.rightColOrder = False
        
    # Post-computations.
    def postComputeColumns(self):
        
        # Derive class to really compute things (=> work directly on self._dfData),
        pass
        
    @property
    def dfRawData(self):

        """Direct access to non post-processed data

        May have to remove computed columns and reset postCompute state.
        """

        # Post-computation is now "not yet done" (and remove any computed column: will be recomputed later on).
        self.setPostComputed(False)

        return self._dfData
        
    def getData(self, copy=True):
        
        # Do post-computation and sorting if not already done.
        if not (self._dfData.empty or self.postComputed):
        
            # Make sure we keep a MultiIndex for columns if it was the case (append breaks this, not fromExcel)
            if self.isMultiIndexedCols and not isinstance(self._dfData.columns, pd.MultiIndex):
                self._dfData.columns = pd.MultiIndex.from_tuples(self._dfData.columns)
            
            # Post-compute as specified (or not).
            self.postComputeColumns()
            self.postComputed = True  # No need to do it again from now on !
            
            # Sort as/if specified.
            if self.sortCols:
                self._dfData.sort_values(by=self.sortCols, ascending=self.sortAscend, inplace=True)
        
        # Enforce right columns order.
        if not (self._dfData.empty or self.rightColOrder):
            
            miTgtColumns = self.miCols
            if self.miCustomCols is not None:
                if self.isMultiIndexedCols:
                    miTgtColumns = self.miCustomCols.append(miTgtColumns)
                else:
                    miTgtColumns = self.miCustomCols + miTgtColumns
            self._dfData = self._dfData.reindex(columns=miTgtColumns)
            self.rightColOrder = True  # No need to do it again, until next append() !
        
        # These are also documentation lines !
        if self.isMultiIndexedCols and not self._dfData.empty:
            assert isinstance(self._dfData.columns, pd.MultiIndex)
        
        # If specified, don't return columns with no relevant results data,
        # unless among custom cols or explicitly exempt cols.
        if self.dropNACols and not self._dfData.empty:
            miCols2Cleanup = self._dfData.columns
            if self.isMultiIndexedCols:
                if self.miCustomCols is not None:
                    miCols2Cleanup = miCols2Cleanup.drop(self.miCustomCols.to_list())
                if self.miExemptNACols is not None:
                    miCols2Cleanup = miCols2Cleanup.drop(self.miExemptNACols.to_list())
            else:
                if self.miCustomCols is not None:
                    miCols2Cleanup = [col for col in miCols2Cleanup if col not in self.miCustomCols]
                if self.miExemptNACols is not None:
                    miCols2Cleanup = [col for col in miCols2Cleanup if col not in
                                      self.miExemptNACols]
            cols2Drop = [col for col in miCols2Cleanup if self._dfData[col].isna().all()]
            logger.debug(f'Dropping all-NaN result columns {cols2Drop}')
            self._dfData.drop(columns=cols2Drop, inplace=True)

        # Done.
        return self._dfData.copy() if copy else self._dfData

    @property
    def dfData(self):

        return self.getData(copy=True)
    
    def setPostComputed(self, on=True):

        # If not specified as already postComputed, prepare for re-computation later by removing any "computed" column.
        if not on:
            # ... but ignore those which are not there : backward compat. / tolerance ...
            self._dfData.drop(columns=self.computedCols, errors='ignore', inplace=True)
        
        # Post-computation not yet done (unless told it _is_: accept blindly).
        self.postComputed = on

    def setData(self, dfData, postComputed=False, acceptNewCols=False):

        # Prerequisites
        assert isinstance(dfData, pd.DataFrame), 'dfData must be a pd.DataFrame'
        assert dfData.index.nunique() == len(dfData), 'dfData index must be unique'

        # Take source data as ours now (after copying).
        self._dfData = dfData.copy()
        
        # If specified, update results columns list (self.miCols) dynamically 
        # as unexpected columns appear in results to append.
        if acceptNewCols:
            self._acceptNewColumns(dfData.columns)

        # Let's assume that columns order is dirty.
        self.rightColOrder = False
        
        # Post-computation as not yet done, unless told it _is_: accept blindly.
        self.setPostComputed(postComputed)
    
    @dfData.setter
    def dfData(self, dfData):
        
        self.setData(dfData)
    
    # Sort rows in place and overwrite initial sortCols / sortAscend values.
    def sortRows(self, by, ascending=True):
    
        self._dfData.sort_values(by=by, ascending=ascending, inplace=True)
        
        self.sortCols = by
        self.sortAscend = ascending
    
    # Add columns translations (update if already there)
    def addColumnsTrans(self, dColsTrans=dict()):

        for col, dTrans in dColsTrans.items():
            self.dfColTrans.loc[col] = dTrans

    # Build translation table for lang (from custom and other columns)
    def transTable(self):
        
        dfColTrans = self.dfColTrans
        if self.dfCustomColTrans is not None:
            dfColTrans = pd.concat([self.dfCustomColTrans, dfColTrans], sort=False)
            
        return dfColTrans
        
    # Get translated names of some columns (custom or not)
    def transColumns(self, columns, lang):
        
        dTransCols = self.transTable()[lang]
        return [dTransCols.get(col, str(col)) for col in columns]
    
    # Get translated names of custom columns
    def transCustomColumns(self, lang):
        
        return self.dfCustomColTrans[lang].to_list()
    
    # Get translated names of some specific column (custom or not)
    def transColumn(self, column, lang):
        
        return self.dfColTrans.loc[column, lang] if column in self.dfColTrans.index \
               else self.dfCustomColTrans.loc[column, lang]
    
    def dfSubData(self, index=None, columns=None, copy=False):
    
        """Get a subset of the all-results table rows and columns

        Parameters:
        :param index: rows to select, as an iterable of a subset of self.dfData.index values
                      (anything suitable for pd.DataFrame.loc[...]) ; None = all rows.
        :param columns: columns to select, as a list(string) or pd.Index when mono-indexed columns
        :param copy: if True, return a full copy of the selected data, not a "reference" to the internal table
        """
        
        assert columns is None or isinstance(columns, list) or isinstance(columns, (pd.Index, pd.MultiIndex)), \
               'columns must be specified as None/[] (all), or as a list of tuples, or as a pandas.[Multi]Index'

        # Make a copy of / extract selected columns of dfData.
        dfSbData = self.getData(copy=False)
        if columns is not None and len(columns) > 0:
            if self.isMultiIndexedCols and isinstance(columns, list):
                iColumns = pd.MultiIndex.from_tuples(columns)
            else:
                iColumns = columns
            dfSbData = dfSbData.reindex(columns=iColumns)
        
        if index is not None:
            dfSbData = dfSbData.loc[index]

        if copy:
            dfSbData = dfSbData.copy()
            
        return dfSbData

    # Access a mono-indexed translated columns version of the data table
    def dfTransData(self, lang='en', index=None, columns=None):
        
        """Get a subset of the all-results table rows and columns with translated column names (mono-index)

        Note: The resulting table holds a copy (of part) of the internal table ; this is needed because
              of the translation of the column names ; hence the absence of the expected "copy" parameter !

        Parameters:
        :param lang: target language for translation ('en' or 'fr')
        :param index: rows to select, as a list(int) or pd.Index (subset of self.dfData.index) ; None = all rows.
        :param columns: columns to select, as a list(string) or pd.Index when mono-indexed columns
                       or as a list(tuple(string*)) or pd.MultiIndex when multi-indexed ; None = all columns.
        """

        assert lang in ['en', 'fr'], f'No support for "{lang}" language'
        
        # Extract selected rows and columns of dfData.
        dfTrData = self.dfSubData(index=index, columns=columns, copy=True)
        
        # Translate column names.
        dfTrData.columns = self.transColumns(dfTrData.columns, lang)
        
        return dfTrData

    def updateSpecs(self, reset=False, overwrite=False, **specs):

        """Update specs as given

        Parameters:
        :param reset: if True, cleanup before updating => like a set !
        :param overwrite: if False, and at least 1 of the spec already exists,
                      raise an exception (refuse to update) ; otherwise, overwrite silently
        :param specs: named specs to add/update
        """
        if reset:
            self.specs.clear()

        if not overwrite:
            assert all(name not in self.specs for name in specs), \
                   "Unless explicitly specified, won't overwrite already present specs {}" \
                   .format(', '.join(name for name in specs if name in self.specs))

        self.specs.update(specs)

    def toPickle(self, fileName, specs=True, raw=False):

        """Save raw data and specs to a pickle file (XZ compressed format if requested).

        Parameters:
        :param fileName: target file pathname ; file is auto-compressed to XZ format
                         through the lzma module if its extension is .xz or .lzma,
                         or else not compressed.
        :param specs: if False, don't save specs (actually save empty specs)
        :param raw: if False, save post-processed data (through dfData) ;
                    if True, save raw data (through dfRawData)
        """
        
        start = pd.Timestamp.now()

        dfOutData = self.dfRawData if raw else self.dfData

        logger.debug(f'toPickle: {dfOutData.columns=}')

        compress = pl.Path(fileName).suffix in ['.xz', '.lzma']
        with lzma.open(fileName, 'wb') if compress else open(fileName, 'wb') as file:
            pickle.dump((dfOutData, self.specs if specs else dict()), file)

        logger.info('{}x{} results rows x columns and {} specs saved to {} in {:.3f}s'
                    .format(len(dfOutData), len(dfOutData.columns),
                            len(self.specs) if specs else 'no', fileName,
                            (pd.Timestamp.now() - start).total_seconds()))

    def specs2Tables(self):

        """Transform specs into tables

        :return: dict(name=DataFrame)
        """
        ddfSpecs = dict()

        for spName, spData in self.specs.items():
            if isinstance(spData, (dict, list, pd.Series)):
                if not isinstance(spData, pd.Series):
                    spData = pd.Series(spData)
                spData = spData.to_frame()
            elif not isinstance(spData, pd.DataFrame):
                raise NotImplementedError
            ddfSpecs[spName] = spData

        return ddfSpecs

    @staticmethod
    def specsFromTables(ddfTables):

        specs = dict()
        for spName, dfSpData in ddfTables.items():
            if len(dfSpData.columns) == 1:  # Output a Series, or dict or list
                if dfSpData.columns[0] == 0:
                    if dfSpData.index.equals(pd.RangeIndex(stop=len(dfSpData))):
                        specs[spName] = dfSpData.loc[:, 0].to_list()
                    else:
                        specs[spName] = dfSpData.loc[:, 0].to_dict()
                else:  # Output a Series
                    specs[spName] = dfSpData[dfSpData.columns[0]]
            else:  # Output a DataFrame
                specs[spName] = dfSpData

        return specs

    DefAllResultsSheetName = 'all-results'

    def toExcel(self, fileName, sheetName=None, lang=None, subset=None, index=True,
                specs=True, specSheetsPrfx='sp-', engine=None):

        """Save data and specs to a worksheet format (Excel, ODF, ...) :
        * newer XLSX format for .xlsx extensions (through 'openpyxl' or 'xlrd' module)
        * .xls (through 'xlwt').
        * .ods (through 'odfpy'), if pandas >= 0.25.

        Details:
        * data to the named sheet (sheetName)
        * specs to the sheets given spec name, prefixed as specified (specSheetsPrfx)

        Parameters:
        :param fileName: target file name
        :param sheetName: for results data only
        :param lang: if not None, save translated data columns names
        :param subset: subset of data columns to save
        :param index: if True, save data index column
        :param specs: if False, don't save specs
        :param specSheetsPrfx: prefix to spec names to use to build spec sheet names
        :param engine: None => auto-selection from file extension ; otherwise, use xlrd, openpyxl or odf.
        """
        
        assert sheetName is None or not specs or not sheetName.lower().startswith(specSheetsPrfx), \
               f"Results data sheet name can't start with reserved prefix {specSheetsPrfx} (whatever case)"
        assert not (sheetName is None
                    and specs and self.DefAllResultsSheetName.lower().startswith(specSheetsPrfx.lower())), \
               "Sheet prefix '{}' can't be a heading part of {} (whatever case)" \
               .format(specSheetsPrfx, self.DefAllResultsSheetName)

        start = pd.Timestamp.now()
        
        dfOutData = self.dfSubData(columns=subset) if lang is None else self.dfTransData(columns=subset, lang=lang)
        
        with pd.ExcelWriter(fileName, engine=engine) as xlWrtr:
            dfOutData.to_excel(xlWrtr, sheet_name=sheetName or self.DefAllResultsSheetName, index=index)
            for spName, dfSpData in self.specs2Tables().items():
                dfSpData.to_excel(xlWrtr, sheet_name=specSheetsPrfx + spName, index=True)

        logger.info('{}x{} results rows x columns and {} specs saved to {} in {:.3f}s'
                    .format(len(dfOutData), len(dfOutData.columns),
                            len(self.specs) if specs else 'no', fileName,
                            (pd.Timestamp.now() - start).total_seconds()))

    def toOpenDoc(self, fileName, sheetName=None, lang=None, subset=None, index=True,
                  specs=True, specSheetsPrfx='sp-'):
        
        """Save data and specs to ODF worksheet format
        
        Note: Needs pandas >= 0.25.

        Details:
        * data to the named sheet (sheetName)
        * specs to the sheets given spec name, prefixed as specified (specSheetsPrfx)

        Parameters:
        :param fileName: target file name
        :param sheetName: for results data only
        :param lang: if not None, save translated data columns names
        :param subset: subset of data columns to save
        :param index: if True, save data index column
        :param specs: if False, don't save specs
        :param specSheetsPrfx: prefix to spec names to use to build spec sheet names
        """

        assert pkgver.parse(pd.__version__).release >= (1, 1), \
               "Don't know how to write to OpenDoc format before Pandas 1.1"
        
        self.toExcel(fileName, sheetName=sheetName, lang=lang, subset=subset, index=index,
                     specs=specs, specSheetsPrfx=specSheetsPrfx, engine='odf')

    def fromPickle(self, fileName, specs=True, postComputed=False, acceptNewCols=False, dDefMissingCols=dict()):

        """Load (overwrite) data and optionally specs from a pickle file, possibly lzma-compressed,
        assuming ctor params match the results object used for prior toPickle(),
        which can well be ensured by using the same ctor params as used for saving !

        :param fileName: source file pathname ; file is auto-decompressed through the lzma module
                         if its extension is .xz or .lzma.
        :param specs: if False, don't load specs
        :param postComputed: if True, prevents next post-computation 
        :param acceptNewCols: if True, append results columns list (self.miCols) dynamically 
            if unexpected columns appear in loaded data to append
        :param dDefMissingCols: default row value to use for missing columns (as a dict/pd.Series)
            (Warning: only from self.miCols)
        """
        
        start = pd.Timestamp.now()

        # Load results data and spec from file
        compressed = pl.Path(fileName).suffix in ['.xz', '.lzma']
        with lzma.open(fileName, 'rb') if compressed else open(fileName, 'rb') as file:
            dfData, dSpecs = pickle.load(file)

        # Complete missing columns if any.
        for colName, colDefVal in dDefMissingCols.items():
            if colName not in dfData.columns:
                dfData[colName] = colDefVal

        # Set data.
        self.setData(dfData, postComputed=postComputed)

        # If specified, update results columns list (self.miCols) dynamically 
        # if unexpected columns appear in loaded data.
        if acceptNewCols:
            self._acceptNewColumns(dfData.columns)

        # Set specs if specified.
        if specs:
            self.specs = dSpecs

        logger.info('{}x{} results rows x columns and {} specs loaded from {} in {:.3f}s'
                    .format(len(dfData), len(dfData.columns),
                            len(self.specs) if specs else 'no', fileName,
                            (pd.Timestamp.now() - start).total_seconds()))

    def fromExcel(self, fileName, sheetName=None, header=[0, 1, 2], skipRows=[3], indexCol=0,
                  specs=True, specSheetsPrfx='sp-', postComputed=False,
                  acceptNewCols=False, dDefMissingCols=dict(), engine=None):
        
        """Load (overwrite) data from the first or named sheet of an Excel worksheet (XLSX or XLS format),
        assuming ctor params match with Excel sheet column names and list,
        which can well be ensured by using the same ctor params as used for saving !

        Also optionally load specs from other sheets named with given prefix, as dataframes
        (ignore others ; empty prefix => all others)

        Parameters:
        :param fileName: source file name
        :param sheetName: name of the sheet to load data from (default None => 1st sheet)
        :param header: list of source data row indexes to use for column index (1st sheet only)
        :param skipRows: list of source data row indexes to ignore (1st sheet only)
        :param indexCol: index of the source data column to use as index (1st sheet only)
                         (None => auto-generated, not read)
        :param specs: if False, don't load specs
        :param specSheetsPrfx: name prefix to use to detect spec sheets
        :param postComputed: if True, prevents next post-computation 
        :param acceptNewCols: if True, append results columns list (self.miCols) dynamically 
            if unexpected columns appear in loaded data to append
        :param dDefMissingCols: default row value to use for missing columns (as a dict/pd.Series)
            (Warning: only from self.miCols)
        :param engine: None => auto-selection from file extension ; otherwise, use xlrd, openpyxl or odf.
        """

        start = pd.Timestamp.now()
        
        # Load results data.
        logger.info1(f'Loading {fileName}:')
        with pd.ExcelFile(fileName, engine=engine) as xlReader:

            logger.info2(f'* {sheetName or xlReader.sheet_names[0]} ...')
            dfData = pd.read_excel(xlReader, sheet_name=sheetName or 0, header=header,
                                   skiprows=skipRows, index_col=indexCol)

            # Complete missing columns if any.
            for colName, colDefVal in dDefMissingCols.items():
                if colName not in dfData.columns:
                    dfData[colName] = colDefVal

            # Set data.
            self.setData(dfData, postComputed=postComputed)

            # If specified, update results columns list (self.miCols) dynamically
            # if unexpected columns appear in loaded data.
            if acceptNewCols:
                self._acceptNewColumns(dfData.columns)

            # Load specs
            self.specs = dict()
            if specs:
                ddfSpecs = dict()
                for shName in xlReader.sheet_names:
                    if shName.startswith(specSheetsPrfx):
                        logger.info2(f'* {shName} ...')
                        ddfSpecs[shName[len(specSheetsPrfx):]] = \
                            pd.read_excel(xlReader, sheet_name=shName, index_col=0)
                self.specs = self.specsFromTables(ddfSpecs)

            logger.info('{}x{} results rows x columns and {} specs loaded from {} in {:.3f}s'
                        .format(len(dfData), len(dfData.columns),
                                len(self.specs) if specs else 'no', fileName,
                                (pd.Timestamp.now() - start).total_seconds()))

    def fromOpenDoc(self, fileName, sheetName=None, header=[0, 1, 2], skipRows=[3], indexCol=0,
                    specs=True, specSheetsPrfx='sp-', postComputed=False,
                    acceptNewCols=False, dDefMissingCols=dict()):
        
        """Load (overwrite) data from the first or named sheet of an Open Document worksheet (ODS format),
        assuming ctor params match with ODF sheet column names and list,
        which can well be ensured by using the same ctor params as used for saving !

        Also, optionally load specs from other sheets with given prefix as dataframes
        (ignore others ; empty prefix => all others)

        Parameters:
        :param fileName: source file name
        :param sheetName: name of the sheet to load data from (default None => 1st)
        :param header: list of source data row indexes to use for column index
        :param skipRows: list of source data row indexes to ignore
        :param indexCol: index of the source data column to use as index (None => auto-generated, not read)
        :param specs: if False, don't load specs
        :param specSheetsPrfx: name prefix to use to detect spec sheets
        :param postComputed: if True, prevents next post-computation 
        :param acceptNewCols: if True, append results columns list (self.miCols) dynamically 
            if unexpected columns appear in loaded data to append
        :param dDefMissingCols: default row value to use for missing columns (as a dict/pd.Series)
            (Warning: only from self.miCols)
        """

        assert pkgver.parse(pd.__version__).release >= (0, 25, 1), \
               'Don\'t know how to read from OpenDoc format before Pandas 0.25.1 (using odfpy module)'
        
        self.fromExcel(fileName, sheetName=sheetName, header=header, skipRows=skipRows, indexCol=indexCol,
                       specs=specs, specSheetsPrfx=specSheetsPrfx, postComputed=postComputed,
                       acceptNewCols=acceptNewCols, dDefMissingCols=dDefMissingCols)

    def fromFile(self, fileName, sheetName=None, header=[0, 1, 2], skipRows=[3], indexCol=0,
                 specs=True, specSheetsPrfx='sp-', postComputed=False,
                 acceptNewCols=False, dDefMissingCols=dict()):
        
        """Load (overwrite) data and eventually specs from a given file,
        (supported formats are .pickle, .pickle.xz, .ods, .xlsx, .xls, auto-detected from file name)
        assuming ctor params match with the results object used to generate source file,
        which can well be ensured by using the same ctor params as used for saving !
        Notes: Needs odfpy module and pandas.version >= 0.25.1

        Parameters:
        :param fileName: source file name
        :param sheetName: name of the sheet to load data from (default None => 1st)
        :param header: list of source data row indexes to use for column index
        :param skipRows: list of source data row indexes to ignore
        :param indexCol: index of the source data column to use as index (None => auto-generated, not read)
        :param specs: if False, don't load specs
        :param specSheetsPrfx: name prefix to use to detect spec sheets
        :param postComputed: if True, prevents next post-computation 
        :param acceptNewCols: if True, append results columns list (self.miCols) dynamically 
            if unexpected columns appear in loaded data to append
        :param dDefMissingCols: default row value to use for missing columns (as a dict/pd.Series)
            (Warning: only from self.miCols)
        """

        fpn = pl.Path(fileName)
        if fpn.suffix in ['.xz', '.pickle']:
            self.fromPickle(fileName, specs=specs, postComputed=postComputed,
                            acceptNewCols=acceptNewCols, dDefMissingCols=dDefMissingCols)
        elif fpn.suffix in ['.ods']:
            self.fromOpenDoc(fileName, sheetName=sheetName, header=header, skipRows=skipRows,
                             indexCol=indexCol, specs=specs, specSheetsPrfx=specSheetsPrfx, postComputed=postComputed,
                             acceptNewCols=acceptNewCols, dDefMissingCols=dDefMissingCols)
        elif fpn.suffix in ['.xls', '.xlsx']:
            self.fromExcel(fileName, sheetName=sheetName, header=header, skipRows=skipRows,
                           indexCol=indexCol, specs=specs, specSheetsPrfx=specSheetsPrfx, postComputed=postComputed,
                           acceptNewCols=acceptNewCols, dDefMissingCols=dDefMissingCols)
        else:
            raise NotImplementedError(f'Unsupported ResultsSet input file format: {fileName}')

    def compare(self, other, subsetCols=[], indexCols=[],
                noneIsNan=False, dropCloser=np.inf, dropNans=True, dropCloserCols=False):
    
        """
        Compare 2 results sets.
        
        The resulting diagnosis DataFrame will have the same columns and merged index,
        with a "closeness" value for each cell (see _closeness method) ;
        rows where all cells have closeness > dropCloser (or eventually NaN) are yet dropped.
        and columns where all cells have closeness > dropCloser (or eventually NaN) can also be dropped.
        
        Parameters:
        :param other: Right results or DataFrame object to compare
        :param list subsetCols: on a subset of columns,
        :param list indexCols: ignoring these columns, but keeping them as the index and sorting order,
        :param bool noneIsNan: if True, replace any None by a np.nan in left and right before comparing
        :param float dropCloser: result will only include rows with all cell closeness > dropCloser
                                 (default: np.inf => all cols and rows kept).
        :param bool dropNans: smoother condition for dropCloser : if True, NaN values are also considered > dropCloser
                              ('cause NaN != NaN :-( ).
        :param bool dropCloserCols: if True, also drop "> dropCloser (or eventually NaN)"-all-cell columns,
                                    just as rows

        :return: the diagnostic DataFrame.
        """
        
        return DataSet.compareDataFrames(dfLeft=self.dfData,
                                         dfRight=other if isinstance(other, pd.DataFrame) else other.dfData,
                                         subsetCols=subsetCols, indexCols=indexCols, noneIsNan=noneIsNan,
                                         dropCloser=dropCloser, dropNans=dropNans, dropCloserCols=dropCloserCols)
        

if __name__ == '__main__':

    sys.exit(0)
