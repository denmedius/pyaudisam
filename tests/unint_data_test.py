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

# Automated unit and integration tests for "data" submodule

# To run : simply run "pytest" or "python <this file>" in current folder
#          and check standard output ; and ./tmp/unt-dat.{datetime}.log for details

import sys
import time

import pandas as pd
import numpy as np

import pytest

import pyaudisam as ads

import unintval_utils as uivu


# Setup local logger.
logger = uivu.setupLogger('unt.dat', level=ads.DEBUG,
                          otherLoggers={'ads.eng': ads.INFO2, 'ads.dat': ads.INFO})

what2Test = 'data'

###############################################################################
#                         Actions to be done after all tests                  #
###############################################################################
def testBegin():
    uivu.logBegin(what=what2Test)


###############################################################################
#                         Input Data Preparation                              #
###############################################################################
#   Generate DataFrame (returned) and other format files  from .ods source
#   and return a list of sources (4 files and 1 DataFrame)
def sources():

    dfPapAlaArv = pd.read_excel(uivu.pRefInDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.ods')

    dfPapAlaArv.to_csv(uivu.pTmpDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.csv', sep='\t', index=False)
    dfPapAlaArv.to_excel(uivu.pTmpDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.xls', index=False)

    # DataSet from multiple sources from various formats (same columns).
    # For test, list of require to contain 1 DataFrame, and one or several
    # source files (one for each different extension)
    # DataFrame and all files need to contain the same data
    srcs = [uivu.pRefInDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.ods',  # Need for module odfpy
            uivu.pRefInDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.xlsx',  # Need for module openpyxl (or xlrd)
            uivu.pTmpDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.xls',  # No need for xlwt(openpyxl seems OK)
            uivu.pTmpDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.csv',
            dfPapAlaArv]
    return srcs


# same as above => to allow using pytest
@pytest.fixture
def sources_fxt():
    return sources()


###############################################################################
#                          Miscellaneous Tools                                #
###############################################################################
def male2bool(s):
    return False if pd.isnull(s.MALE) or s.MALE.lower() != 'oui' else True


def sex2bool(s):
    return False if pd.isnull(s.SEXE) or s.SEXE.lower() != 'oui' else True


def newval2othercol(_):
    return "New Value"


###############################################################################
#                                Test Cases                                   #
###############################################################################
# test DataSet - method "Ctor", getter "dfData", and "__len__"
# from various sources: .ads, .xslx, .xls, .csv and from DataFrame
def testDataSetCtorLen(sources_fxt):

    # gather source DataFrame (used for checking size of DataSet)
    srcDf = [src for src in sources_fxt if isinstance(src, pd.DataFrame)][0]

    # list of awaited columns form input DataFrame with associated types
    dTypes = {'ZONE': 'object', 'HA': 'int', 'POINT': 'int', 'ESPECE': 'object',
              'DISTANCE': 'float', 'MALE': 'object', 'DATE': 'object', 'OBSERVATEUR': 'object',
              'PASSAGE': 'object', 'NOMBRE': 'float', 'EFFORT': 'int'}

    ds = ads.DataSet(sources_fxt, importDecFields=['EFFORT', 'DISTANCE', 'NOMBRE'],
                     sheet='Sheet1', skipRows=None, separator='\t')

    # checking size of DataFrame
    assert ds.dfData.size == srcDf.size * len(sources_fxt), \
        'Error: testDataSetCtorLen: DataSet Ctor: Size of DataFrame "DataFrame.dfData" inconsistent with sources'
    # checking list of columns of DataFrame
    assert sorted(ds.columns) == sorted(dTypes.keys()), \
        'Error: testDataSetCtorLen: Ctor: Columns list from "DataFrame.dfData" mismatched' \
        'with columns list of the source file'
    # checking column dtypes of DataFrame
    assert all(typ.name.startswith(dTypes[col]) for col, typ in ds.dfData.dtypes.items()), \
        'Error: testDataSetCtorLen: DataSet Ctor: types of data from the "DataFrame.dfData" mismatched ' \
        'awaited data types'

    logger.info0('PASS (testDataSetCtorLen) => DATASET => Constructor, getter "dfData"'
                 ' and method "__len__"\n(Includes _csv2df, _fromDataFrame, _fromDataFile)')


# test setter dfData
def testDataSetDfData(sources_fxt):

    ds = ads.DataSet(sources_fxt, sheet='Sheet1', skipRows=None, separator='\t')

    # Check Exception raising with dfData setter call
    with pytest.raises(NotImplementedError) as e:
        ds.dfData = pd.DataFrame()
    logger.info0(f'PASS (testDataSetDfData) => EXCEPTION RAISED AS AWAITED:\n{e}')


# test DataSet - method "empty"
def testDataSetEmpty():

    emptyDf = pd.DataFrame()
    emptyDs = ads.DataSet(emptyDf)
    assert emptyDs.empty, 'Error: testDataSetEmpty: DataFrame from DataSet should be empty'

    ds = pd.DataFrame(data={'col1': [1, 2]})
    assert not ds.empty, 'Error: testDataSetEmpty: DataFrame from DataSet should not be empty'

    logger.info0('PASS (testDataSetEmpty) => DATASET => method "empty"')


# test DataSet - method "columns"
def testDataSetColumns(sources_fxt):

    ds = ads.DataSet(sources_fxt, sheet='Sheet1', skipRows=None, separator='\t')

    assert (ds.columns == ds.dfData.columns).all(), \
        'Error: testDataSetColumns: columns: Issue occurred with method "DataSet.columns"'

    logger.info0('PASS (testDataSetColumns) => DATASET => method "columns"')


# test columns renaming
def testDataSetRenameColumns(sources_fxt):

    ds = ads.DataSet(sources_fxt, dRenameCols={'NOMBRE': 'INDIVIDUS', 'MALE': 'SEXE'},
                     importDecFields=['EFFORT', 'DISTANCE', 'NOMBRE'], sheet='Sheet1',
                     skipRows=None, separator='\t')

    assert all([(col not in ds.columns) for col in ['MALE', 'NOMBRE']]) \
           and all([(col in ds.columns) for col in ['SEXE', 'INDIVIDUS']]), \
        'Error: testDataSetRenameColumns: renameColumns: Issue occurred with renaming process: ' \
        'columns "NOMBRE" and "MALE" should have been renamed "INDIVIDUS" and "SEXE"'

    logger.info0('PASS (testDataSetRenameColumns) => DATASET => method "renameColumns"\n'
                 '(Includes _csv2df, _fromDataFrame, _fromDataFile)')


# test columns addition, recomputing, and combination addition/recomputing
# (_addComputedColumns) for existing or new column
def testDataSetAddComputedColumns(sources_fxt):

    ds = ads.DataSet(sources_fxt, sheet='Sheet1', skipRows=None, separator='\t')

    # Checking addition of a columns
    ds = ads.DataSet(ds.dfData, dComputeCols={'NEWCOL': ''})
    assert 'NEWCOL' in ds.columns, \
        'Error: testDataSetAddComputedColumns: Issue occurred' \
        ' with simple addition of a new column. Columns "NEWCOL" should have been added'

    # Checking re-computing a columns
    #   Checking initial data type of 'MALE' column values are not boolean
    assert ds.dfData['MALE'].dtype != bool, 'Error: testDataSetAddComputedColumns: Pre-test: Values from MALE \
    column should not be bool'
    #   Checking data type of 'MALE' column modified to boolean (dComputeCols)
    ds = ads.DataSet(ds.dfData, dComputeCols={'MALE': male2bool})
    assert ds.dfData['MALE'].dtype == bool, \
        'Error: testDataSetAddComputedColumns: Issue occurred with recomputing existing' \
        ' column process. Values from MALE column should be bool'

    # Checking adding a new column with computing process
    #   Checking data type of 'NEWCOL' colonne was added with boolean (dComputeCols)
    ds = ads.DataSet(ds.dfData, dComputeCols={'OTHERCOL': newval2othercol})
    assert ds.dfData.OTHERCOL[0] == 'New Value', \
        'Error: testDataSetAddComputedColumns: _addComputedColumns:' \
        ' Issue occurred with process of adding a new AND computed column.'

    # Checking adding a new column with computing process and rename it
    #   a new column also added to prepare following test step
    ds = ads.DataSet(sources_fxt, dRenameCols={'MALE': 'SEXE'},
                     dComputeCols={'MALE': sex2bool, 'NEWCOL': ''},
                     sheet='Sheet1', skipRows=None, separator='\t')
    print(ds.columns)
    print(ds.dfData.SEXE)
    assert ds.dfData['SEXE'].dtype == bool, \
        'Error: testDataSetAddComputedColumns: Issue occurred with process of adding' \
        ' and renaming a new AND computed column.'

    logger.info0('PASS (testDataSetAddComputedColumns) => DATASET => methods "addColumns" and "_addComputedColumns":\n'
                 + '\t'*5 + 'addition, recomputing, and combination addition/recomputing'
                 + ' for existing or new column checked\n'
                 + '\t'*6 + '(Includes _csv2df, _fromDataFrame, _fromDataFile)')


# test method "dfSubData"
def testDataSetDfSubData(sources_fxt):

    ds = ads.DataSet(sources_fxt, sheet='Sheet1', skipRows=None, separator='\t')

    # subset = None
    df = ds.dfSubData()
    assert df.compare(ds.dfData).empty, \
        'Error: testDataSetDfSubData: Issue occurred. With "None", df should be a copy of source DataSet.'
    assert id(df) == id(ds.dfData), \
        'Error: testDataSetDfSubData: Issue occurred. With "copy = False (default)",' \
        ' subset of DataFrame should be a reference to the subset of source DataFrame, not a copy of data.'

    df = ds.dfSubData(copy=True)
    assert df.compare(ds.dfData).empty, \
        'Error: testDataSetDfSubData: Issue occurred. With "None", df should be a copy of source DataSet.'
    assert id(df) != id(ds.dfData), \
        'Error: testDataSetDfSubData: Issue occurred. With "copy = True",' \
        ' subset of DataFrame should be a copy of data, not a reference to the subset of source DataFrame.'

    # columns = as an empty list
    df = ds.dfSubData(columns=[])
    assert df.compare(ds.dfData).empty, \
        'Error: testDataSetDfSubData: Issue occurred. With "columns=[]", df should be equal source DataSet.'

    # columns = as list
    df = ds.dfSubData(columns=['POINT', 'ESPECE', 'DISTANCE', 'INDIVIDUS', 'EFFORT'])
    assert df.compare(ds.dfData.reindex(columns=['POINT', 'ESPECE', 'DISTANCE', 'INDIVIDUS', 'EFFORT'])).empty, \
        'Error: testDataSetDfSubData: Issue occurred with "columns=list(...)".'

    # columns = as pandas.Index
    df = ds.dfSubData(columns=pd.Index(['POINT', 'ESPECE', 'DISTANCE', 'INDIVIDUS', 'EFFORT']))
    assert df.compare(ds.dfData.reindex(columns=['POINT', 'ESPECE', 'DISTANCE', 'INDIVIDUS', 'EFFORT'])).empty, \
        'Error: testDataSetDfSubData: Issue occurred with "columns=Index(...)".'

    # columns = as other (str)
    # Check Exception raising
    with pytest.raises(Exception) as e:
        _ = ds.dfSubData(columns='POINT')
    logger.info0(f'PASS (testDataSetDfSubData) => EXCEPTION RAISED AS AWAITED:\n{e}')

    # TODO: columns with MultiIndexed-columned dataset

    # columns = as an empty list
    df = ds.dfSubData(index=[])
    assert df.empty, \
        'Error: testDataSetDfSubData:Issue occurred. With "index=[]", df should be empty.'

    # index = as list
    df = ds.dfSubData(index=[1, 4, 7, 9, 38])
    assert df.compare(ds.dfData.loc[[1, 4, 7, 9, 38]]).empty, \
        'Error: testDataSetDfSubData: Issue occurred with "columns=list(...)".'

    # index = as pandas.Index
    df = ds.dfSubData(index=pd.Index([1, 4, 7, 9, 38]))
    assert df.compare(ds.dfData.loc[[1, 4, 7, 9, 38]]).empty, \
        'Error: testDataSetDfSubData: Issue occurred with "columns=Index(...)".'

    # index = as range
    df = ds.dfSubData(index=range(1, 300, 3))
    assert df.compare(ds.dfData.loc[range(1, 300, 3)]).empty, \
        'Error: testDataSetDfSubData: Issue occurred with "columns=range(...)".'

    logger.info0('PASS (testDataSetDfSubData) => DATASET => method "dfSubData"')


# test columns deletion
def testDataSetDropColumns(sources_fxt):

    ds = ads.DataSet(sources_fxt, sheet='Sheet1', skipRows=None, separator='\t')

    # gather source DataFrame (used for checking size of DataSet)
    srcDf = [src for src in sources_fxt if isinstance(src, pd.DataFrame)][0]

    # Checking deletion of columns
    dropCols = ['ZONE', 'HA', 'OBSERVATEUR']

    ds.dropColumns(dropCols)

    assert ds.columns.to_list() == ['POINT', 'ESPECE', 'DISTANCE', 'MALE', 'DATE', 'PASSAGE', 'NOMBRE', 'EFFORT'], \
        'Error: testDataSetDropColumns: Issue occurred with dropping columns: one or more \
        columns were not drop.'

    assert len(ds) == len(srcDf) * len(sources_fxt),  \
        'Error: testDataSetDropColumns: Issue occurred: inconsistent size of the DataFrame next to columns dropping.'

    logger.info0('PASS (testDataSetDropColumns) => DATASET => method "dropColumns"')


# test rows deletion
def testDataSetDropRows(sources_fxt):

    ds = ads.DataSet(sources_fxt, sheet='Sheet1', dComputeCols={'MALE': male2bool}, skipRows=None, separator='\t')

    # gather source DataFrame (used for checking size of DataSet)
    srcDf = [src for src in sources_fxt if isinstance(src, pd.DataFrame)][0]

    # deletion of rows with no distance noted
    ds.dropRows(ds.dfData.DISTANCE.isnull())

    # check that result DataFrame is of same size as entry with distance not null in source DataFrame
    assert len(ds) == len(srcDf[srcDf.DISTANCE.notnull()]) * len(sources_fxt), \
        'Error: testDataSetDropRows: Issue occurred. Size of resulted DataFrame' \
        ' not consistent after deletion process.'

    # check nb of rows are consistent with data before/after deletion (refer source files data)
    assert ds.dfData.MALE.value_counts()[True] == ds.dfData.NOMBRE.sum() == srcDf.NOMBRE.sum() * len(sources_fxt), \
        'Error: testDataSetDropRows: Issue occurred. Comparison of numbers of data in resulted DataFrame' \
        ' not consistent after deletion process.'

    logger.info0('PASS (testDataSetDropRows) => DATASET => method "dropRows"')


# DATASET TESTS
# test methods "toExcel", "toOpenDoc", "toPickle", "compareDataFrames"
def testDataSetToFiles(sources_fxt):

    ds = ads.DataSet(sources_fxt, importDecFields=['EFFORT', 'DISTANCE', 'NOMBRE'],
                     dRenameCols={'NOMBRE': 'INDIVIDUS'}, dComputeCols={'MALE': male2bool},
                     sheet='Sheet1', skipRows=None, separator='\t')
    ds.dropColumns(['ZONE', 'HA', 'OBSERVATEUR'])
    ds.dropRows(ds.dfData.DISTANCE.isnull())
    
    # => toExcel, toOpenDoc, toPickle, compareDataFrames
    closenessThreshold = 15  # => max relative delta = 1e-15
    subsetCols = ['POINT', 'ESPECE', 'DISTANCE', 'INDIVIDUS', 'EFFORT']
    filePathName = uivu.pTmpDir / 'dataset-uni.ods'
    dfRef = ds.dfSubData(columns=subsetCols).reset_index(drop=True)

    for fpn in [filePathName, filePathName.with_suffix('.xlsx'), filePathName.with_suffix('.xls'),
                filePathName.with_suffix('.pickle'), filePathName.with_suffix('.pickle.xz')]:

        print(fpn.as_posix(), end=' : ')
        if fpn.suffix == '.ods':
            ds.toOpenDoc(fpn, sheetName='utest', subset=subsetCols, index=False)
        elif fpn.suffix in ['.xlsx', '.xls']:
            ds.toExcel(fpn, sheetName='utest', subset=subsetCols, index=False)
        elif fpn.suffix in ['.pickle', '.xz']:
            ds.toPickle(fpn, subset=subsetCols, index=False)
        assert fpn.is_file(), 'Error: testDataSetToFiles'

        if fpn.suffix in ['.ods', '.xlsx', '.xls']:
            df = pd.read_excel(fpn, sheet_name='utest')
        elif fpn.suffix in ['.pickle', '.xz']:
            df = pd.read_pickle(fpn)
            df.reset_index(drop=True, inplace=True)
        else:
            raise Exception(f'No test for this input file format {fpn.as_posix()}')
        assert ds.compareDataFrames(df.reset_index(), dfRef.reset_index(),
                                    subsetCols=['POINT', 'DISTANCE', 'INDIVIDUS', 'EFFORT'],
                                    indexCols=['index'], dropCloser=closenessThreshold, dropNans=True).empty
        print('1e-{} comparison OK (df.equals(dfRef) is {}, df.compare(dfRef) {}empty)'
              .format(closenessThreshold, df.equals(dfRef), '' if df.compare(dfRef).empty else 'not')), \
            'Error: testDataSetToFiles'

    logger.info0('PASS (testDataSetToFiles) => DATASET => method "toExcel",'
                 ' "toOpenDoc", "toPickle", "compareDataFrames"')


# Test of the base function for comparison (test from static hard-coded data, not from loaded DataSets)
# method "_closeness" used for
def testDataSetCloseness():
    # creation of empty DataSet
    ds = ads.DataSet(pd.DataFrame())

    values = [np.nan, -np.inf,
              -1.0e12, -1.0e5, -1.0-1e-5, -1.0, -1.0+1e-5, -1.0e-8,
              0.0, 1.0e-8, 1.0, 1.0e5, 1.0e12, np.inf]
    aClose = np.ndarray(shape=(len(values), len(values)))

    for r in range(len(values)):
        for c in range(len(values)):
            try:
                aClose[r, c] = ds._closeness(pd.Series([values[r], values[c]]))
            except Exception as e:
                print(e, r, c, values[r], values[c])
                raise

    # Infinite closeness on the diagonal (except for nan and +/-inf)
    assert all(np.isnan(values[i]) or np.isinf(values[i]) or np.isinf(aClose[i, i]) for i in range(len(values))), \
           'Error: testDataSetCloseness: Inequality on the diagonal'

    # No infinite proximity elsewhere
    assert all(r == c or not np.isinf(aClose[r, c]) for r in range(len(values)) for c in range(len(values))), \
           'Error: testDataSetCloseness: No equality should be found outside the diagonal'

    # Good closeness around -1 only
    whereClose = [i for i in range(len(values)) if abs(values[i] + 1) <= 1.0e-5]
    assert all(aClose[r, c] > 4 for r in whereClose for c in whereClose), \
        'Error: test_closeness: Unexpectedly bad closeness around -1'

    logger.info0('PASS (testDataSetCloseness) => DATASET => method "_closeness"')


# Comparison (from other files data sources, the same as for ResultsSet.compare below, but through DataSet)
# => compare, compareDataFrames, _toHashable, _closeness
def testDataSetCompare():
    # a. Loading of Distance 7 and values to compare issued from PyAuDiSam
    dsDist = ads.DataSet(uivu.pRefInDir / 'ACDC2019-Papyrus-ALAARV-TURMER-comp-dist-auto.ods',
                         sheet='RefDist73', skipRows=[3], headerRows=[0, 1, 2], indexCols=0)

    dsAuto = ads.DataSet(uivu.pRefInDir / 'ACDC2019-Papyrus-ALAARV-TURMER-comp-dist-auto.ods',
                         sheet='ActAuto', skipRows=[3], headerRows=[0, 1, 2], indexCols=0)

    # b. Index columns to be compared
    indexCols = [('sample', 'AnlysNum', 'Value')] \
                + [('sample', col, 'Value') for col in ['Species', 'Periods', 'Prec.', 'Duration']] \
                + [('model', 'Model', 'Value')] \
                + [('parameters', 'left truncation distance', 'Value'),
                   ('parameters', 'right truncation distance', 'Value'),
                   ('parameters', 'model fitting distance cut points', 'Value'),
                   ('parameters', 'distance discretisation cut points', 'Value')]

    # # c. Columns to be compared (DeltaDCV and DeltaAIC were removed as results depend on a set of ran analyses,
    # #    different between reference and PyAuDiSam run).
    subsetCols = [col for col in dsDist.dfData.columns.to_list()
                  if col not in indexCols + [('run output', 'run time', 'Value'),
                                             ('density/abundance', 'density of animals', 'Delta Cv'),
                                             ('detection probability', 'Delta AIC', 'Value')]]

    # # d. "Exact" Comparison : no line pass (majority of epsilons due to IO ODS)
    dfRelDiff = dsDist.compare(dsAuto, subsetCols=subsetCols, indexCols=indexCols)
    assert len(dfRelDiff) == len(dsDist), \
        'Error: testCompare: compare: Issue occurred. Exact row by row comparison' \
        ' of both DataSets should fail for all rows.'

    # # e. Comparison with 10**-16 accuracy : almost all lines pass, but 3 (majority of epsilons due to IO ODS)
    dfRelDiff = dsDist.compare(dsAuto, subsetCols=subsetCols, indexCols=indexCols, dropCloser=16, dropNans=True)
    assert len(dfRelDiff) == 3, \
        'Error: testCompare: compare: Issue occurred. Row by row comparison of both DataSet' \
        ' with accuracy of 10**-16 should fail for all rows but 3.'
    print(dfRelDiff)

    # # e. Comparison with 10**-5 accuracy : all lines pass
    dfRelDiff = dsDist.compare(dsAuto, subsetCols=subsetCols, indexCols=indexCols, dropCloser=5, dropNans=True)
    assert len(dfRelDiff) == 2, \
        'Error: testCompare: compare: Issue occurred. Row by row comparison of both DataSet' \
        'with accuracy of 10**5 should pass for all rows.'

    logger.info0('PASS (testCompare) => DATASET => method "_closeness" and "compare"')


# SAMPLEDATASET TESTS
# test methods "toExcel", "toOpenDoc", "toPickle", "compareDataFrames"
def testSDSCtor():
    # test: creation of SDS from DataFrame source.
    dfData = pd.DataFrame(columns=['Date', 'TrucDec', 'Espece', 'Point', 'Effort', 'Distance'],
                          data=[('2019-05-13', 3.5, 'TURMER', 23, 2, 83),
                                ('2019-05-15', np.nan, 'TURMER', 23, 2, 27.355),
                                ('2019-05-13', 0, 'ALAARV', 29, 2, 56.85),
                                ('2019-04-03', 1.325, 'PRUMOD', 53, 1.3, 7.2),
                                ('2019-06-01', 2, 'PHICOL', 12, 1, np.nan),
                                ('2019-06-19', np.nan, 'PHICOL', 17, 0.5, np.nan),
                                ])
    dfData['Region'] = 'ACDC'
    dfData['Surface'] = '2400'

    sds = ads.SampleDataSet(source=dfData, decimalFields=['Effort', 'Distance', 'TrucDec'])

    assert not any(sds.dfData[col].dropna().apply(lambda v: isinstance(v, str)).any() for col in sds.decimalFields), \
        'Error: testSDSCtor: Some strings found in declared decimal fields ... any decimal format issue ?'

    assert sds.columns.equals(dfData.columns), \
        'Error: testSDSCtor: inconsistency with columns list of SDS vs source DataFrame'
    assert len(sds) == len(dfData),  'Error: testSDSCtor: inconsistency with size of SDS vs source DataFrame'
    assert sds.dfData.Distance.notnull().sum() == 4, \
        'Error: testSDSCtor: issue with NaN values that should be considered as null'

    # test: empty SDS raises exception
    emptyDf = pd.DataFrame()
    with pytest.raises(Exception) as e_info:
        ads.SampleDataSet(emptyDf)
    logger.info0(f'PASS (testSDSCtor) => EXCEPTION RAISED AS AWAITED:\n{e_info.value}')

    # test: exception raised if SDS created with DataFrame < 5 columns
    dfData = pd.DataFrame(columns=['Col1', 'Col2', 'Col3', 'Col4'], data=[(1, 2, 3, 4)])
    with pytest.raises(Exception) as e_info:
        _ = ads.SampleDataSet(source=dfData)
    logger.info0(f'PASS (testSDSCtor) => EXCEPTION RAISED AS AWAITED:\n{e_info.value}')

    # Excel source (path as simple string)
    sds = ads.SampleDataSet(source=uivu.pRefInDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.xlsx',
                            decimalFields=['EFFORT', 'DISTANCE', 'NOMBRE'])

    assert sds.columns.to_list() == ['ZONE', 'HA', 'POINT', 'ESPECE', 'DISTANCE', 'MALE', 'DATE',
                                     'OBSERVATEUR', 'PASSAGE', 'NOMBRE', 'EFFORT'], \
        'Error: testSDSCtor: inconsistency with columns list of SDS vs source file'
    assert len(sds) == 256, \
        'Error: testSDSCtor: inconsistency with size of the resulting SDS. It should contain 256 rows'
    assert sds.dfData.NOMBRE.sum() == 217, \
        'Error: testSDSCtor: inconsistency with data loaded in SDS: Sum of values from column "NOMBRE" should be 217'

    # Libre / Open Office source (path as simple string)
    sds = ads.SampleDataSet(source=uivu.pRefInDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.ods',
                            decimalFields=['EFFORT', 'DISTANCE', 'NOMBRE'])

    assert sds.columns.to_list() == ['ZONE', 'HA', 'POINT', 'ESPECE', 'DISTANCE', 'MALE', 'DATE',
                                     'OBSERVATEUR', 'PASSAGE', 'NOMBRE', 'EFFORT'], \
        'Error: testSDSCtor: inconsistency with columns list of SDS vs source file'
    assert len(sds) == 256, 'Error: inconsistency with size of the resulting SDS. It should contain 256 rows'
    assert sds.dfData.NOMBRE.sum() == 217, \
        'Error: testSDSCtor: inconsistency with data loaded in SDS: Sum of values from column "NOMBRE" should be 217'

    # CSV source with ',' as decimal point (path as pl.Path)
    sds = ads.SampleDataSet(source=uivu.pRefInDir / 'ACDC2019-Papyrus-TURMER-AB-5mn-1dec-dist.txt',
                            decimalFields=['Point transect*Survey effort', 'Observation*Radial distance'])

    assert not any(sds.dfData[col].dropna().apply(lambda v: isinstance(v, str)).any() for col in sds.decimalFields), \
        'Error: testSDSCtor: Some strings found in declared decimal fields ... any decimal format issue ?'

    assert sds.columns.to_list() == ['Region*Label', 'Region*Area', 'Point transect*Label',
                                     'Point transect*Survey effort', 'Observation*Radial distance'], \
        'Error: testSDSCtor: inconsistency with columns list of SDS vs source file'
    assert len(sds) == 330, \
        'Error: testSDSCtor: inconsistency with size of the resulting SDS. It should contain 330 rows'
    assert sds.dfData['Observation*Radial distance'].notnull().sum() == 324, \
        'Error: testSDSCtor: inconsistency with \
    data loaded in SDS: Sum of values from column "Observation*Radial distance" should be 324'

    # CSV source with '.' as decimal point
    sds = ads.SampleDataSet(source=uivu.pRefInDir / 'ACDC2019-Papyrus-ALAARV-AB-10mn-1dotdec-dist.txt',
                            decimalFields=['Point transect*Survey effort', 'Observation*Radial distance'])

    assert not any(sds.dfData[col].dropna().apply(lambda v: isinstance(v, str)).any() for col in sds.decimalFields), \
        'Error: testSDSCtor: Some strings found in declared decimal fields ... any decimal format issue ?'

    assert sds.columns.to_list() == ['Region*Label', 'Region*Area', 'Point transect*Label',
                                     'Point transect*Survey effort', 'Observation*Radial distance'], \
        'Error: testSDSCtor: inconsistency with columns list of SDS vs source file'
    assert len(sds) == 256, \
        'Error: testSDSCtor: inconsistency with size of the resulting SDS. It should contain 256 rows'
    assert sds.dfData['Observation*Radial distance'].notnull().sum() == 217, \
        'Error: testSDSCtor: inconsistency with data loaded in SDS:' \
        ' Sum of values from column "Observation*Radial distance" should be 217'

    logger.info0('PASS (testSDSCtor) => SampleDATASET => Constructor')


# ## 6. Class FieldDataSet (and base DataSet)
# Note: For real unit tests of DataSet, see `visionat` module:
#       it defines the same class (have to be the same: check it !)
def count2AdultCat(sCounts):
    return 'm' if 'Mal' in sCounts[sCounts > 0].index[0] else 'a'


def count2DurationCat(sCounts):
    return '5mn' if '5' in sCounts[sCounts > 0].index[0] else '10mn'


KFdsCountCols = ['nMalAd10', 'nAutAd10', 'nMalAd5', 'nAutAd5']


def testFieldDataSet():

    # ### a. Load data sample
    dfObs = pd.read_csv('refin/ACDC2019-Naturalist-ExtraitObsBrutesAvecDist.txt', sep='\t', decimal=',')
    dfObs.head()

    sCounts = dfObs[KFdsCountCols].sum()

    logger.info('Data sample: len={len(dfObs)}, sums={sCounts.to_dict()}')

    assert len(dfObs) == 724
    assert not any(sCounts - pd.Series({'nMalAd10': 613, 'nAutAd10': 192, 'nMalAd5': 326, 'nAutAd5': 102}))

    # ### b. FieldDataSet._separateMultiCategoryCounts
    dfObsMonoCat_ = ads.FieldDataSet._separateMultiCategoryCounts(dfObs, KFdsCountCols)
    len(dfObsMonoCat_), dfObsMonoCat_[KFdsCountCols].sum()

    s = dfObs[KFdsCountCols].apply(lambda s: len(s[s > 0]), axis='columns')
    logger.info(f'Data sample => apply(lambda s: len(s[s > 0]): len={len(s)}'
                f', value_counts={s.value_counts().to_dict()}')
    assert len(s) - len(s[s < 1]) + sum((i - 1) * len(s[s == i]) for i in range(1, s.max() + 1)) == len(dfObsMonoCat_)
    assert len(dfObsMonoCat_) == 1125
    assert not any(dfObsMonoCat_[KFdsCountCols].sum() - sCounts)

    logger.info('dfObsMonoCat_ head: ' + dfObsMonoCat_.head().to_string())
    logger.info0('PASS (testFieldDataSet) => _separateMultiCategoryCounts')

    # ### c. Categorise sightings
    #
    # Needed for adding absence data below
    #
    # (no more counts - by the way, all 0 or 1 - => only categories)
    # Should not see any sightings with all null counts
    assert dfObsMonoCat_[~dfObsMonoCat_[KFdsCountCols].any(axis='columns')].empty

    dfObsMonoCat_['Adulte'] = dfObsMonoCat_[KFdsCountCols].apply(count2AdultCat, axis='columns')

    dfObsMonoCat_['Durée'] = dfObsMonoCat_[KFdsCountCols].apply(count2DurationCat, axis='columns')
    logger.info('dfObsMonoCat_ tail: ' + dfObsMonoCat_.tail().to_string())

    # ### d. FieldDataSet._individualiseMonoCategoryCounts
    dfObsIndiv_ = ads.FieldDataSet._individualiseMonoCategoryCounts(dfObsMonoCat_, KFdsCountCols)
    logger.info(f'dfObsIndiv_: len={len(dfObsIndiv_)}, sums={dfObsIndiv_[KFdsCountCols].sum()}')

    logger.info('dfObsIndiv_ head: ' + dfObsIndiv_.head().to_string())
    assert len(dfObsIndiv_) == 1233
    assert not any(dfObsIndiv_[KFdsCountCols].sum() - sCounts)

    logger.info0('PASS (testFieldDataSet) => _individualiseMonoCategoryCounts')

    # ### e. FieldDataSet.monoCategorise
    #
    # (combines a, b, c and d above in one function : the one to use actually !)
    # First, load FieldDataSet from dfObs
    fds = ads.FieldDataSet(source=dfObs, countCols=KFdsCountCols,
                           addMonoCatCols={'Adulte': count2AdultCat, 'Durée': count2DurationCat})

    dfObsMonoCat = fds.monoCategorise()

    logger.info('dfObsMonoCat head: ' + dfObsMonoCat.head().to_string())
    assert (dfObsMonoCat == dfObsMonoCat_).all().all()

    logger.info0('PASS (testFieldDataSet) => ctor(DataFrame), monoCategorise')

    # ### f. FieldDataSet.individualise
    # (combines a, b, c and d above in one function : the one to use actually !)
    dfObsIndiv = fds.individualise()

    logger.info('dfObsIndiv head: ' + dfObsIndiv.head().to_string())
    assert (dfObsIndiv == dfObsIndiv_).all().all()

    logger.info0('PASS (testFieldDataSet) => individualise')

    # Second, try from source CSV file
    fds = ads.FieldDataSet(source='refin/ACDC2019-Naturalist-ExtraitObsBrutesAvecDist.txt',
                           importDecFields=['distMem'], countCols=KFdsCountCols,
                           addMonoCatCols={'Adulte': count2AdultCat, 'Durée': count2DurationCat})

    dfObsIndiv = fds.individualise()

    logger.info('dfObsIndiv head: ' + dfObsIndiv.head().to_string())

    assert (dfObsIndiv == dfObsIndiv_).all().all()

    logger.info0('PASS (testFieldDataSet) => ctor(CSV), individualise')


# ## 7. Class MonoCategoryDataSet (and base DataSet)
# Note: For real unit tests of DataSet, see `visionat` module:
# it defines the same class (have to be the same: check it !)
def testMonoCategoryDataSet():

    # Setup source FDS
    fds = ads.FieldDataSet(source='refin/ACDC2019-Naturalist-ExtraitObsBrutesAvecDist.txt',
                           importDecFields=['distMem'], countCols=KFdsCountCols,
                           addMonoCatCols={'Adulte': count2AdultCat, 'Durée': count2DurationCat})

    dfObsIndiv = fds.individualise()

    # Drop now unneeded count columns (only 0 or 1 inside + columns Adulte and Duree to explain what a 1 means)
    # No more need for count cols then (only 0 or 1 inside + columns Adulte and Duree to explain what 1 means)
    dfObsIndiv.drop(columns=KFdsCountCols, inplace=True)
    logger.info('dfObsIndiv tail: ' + dfObsIndiv.tail().to_string())

    # ### a. Extract transect info
    # (assuming that each transect x pass gave at least 1 sighting, otherwise the effort will be wrong)
    transectPlaceCol = 'Point'
    transectPlaceCols = [transectPlaceCol]
    passIdCol = 'Passage'
    effortCol = 'Effort'

    dfTransPassEffort = ads.MonoCategoryDataSet._extractTransects(dfObsIndiv, transectPlaceCols=transectPlaceCols,
                                                                  passIdCol=passIdCol,
                                                                  effortCol=effortCol, effortConstVal=1)
    logger.info('dfTransPassEffort: ' + dfTransPassEffort.to_string())

    assert len(dfTransPassEffort) == 41 \
           and len(dfTransPassEffort[dfTransPassEffort.Passage == 'a']) == 21 \
           and len(dfTransPassEffort[dfTransPassEffort.Passage == 'b']) == 20

    logger.info0('PASS (testMonoCategoryDataSet) => _extractTransects')

    # ### b. Select sighting from 1 sample
    logger.info('dfObsIndiv: ' + dfObsIndiv.head().to_string())

    # Select 1 sample
    espece = 'Sylvia atricapilla'
    passage = 'a'
    adulte = 'm'
    duree = '10mn'
    # dfObsIndivSmpl = dfObsIndiv[(dfObsIndiv.Passage == passage) & (dfObsIndiv.Adulte == adulte)
    #                            & (dfObsIndiv.Duree == duree) & (dfObsIndiv.Espece == espece)]
    dfObsIndivSmpl, dfTrPassEffSmpl = \
        ads.MonoCategoryDataSet._selectSampleSightings(dSample={'Passage': passage, 'Adulte': adulte,
                                                                'Durée': duree, 'Espèce': espece},
                                                       dfAllSights=dfObsIndiv, dfAllEffort=dfTransPassEffort,
                                                       transectPlaceCols=['Point'], passIdCol='Passage',
                                                       effortCol='Effort')

    assert len(dfObsIndivSmpl) == 36 and dfObsIndivSmpl[transectPlaceCol].nunique() == 18
    assert len(dfTrPassEffSmpl) == 21 \
           and dfTrPassEffSmpl.reset_index()[transectPlaceCol].nunique() == len(dfTrPassEffSmpl)
    # 1 seul passage, et sur tous les points sans exception
    assert len(dfTrPassEffSmpl[dfTrPassEffSmpl.Effort != 1]) == 0

    logger.info0('PASS (testMonoCategoryDataSet) => _selectSampleSightings')

    # ### c. Add abscence sightings
    logger.info('dfObsIndivSmpl: ' + dfObsIndivSmpl.to_string())

    sampleCols = ['Passage', 'Adulte', 'Durée', 'Espèce']
    dfObsIndivAbscSmpl = ads.MonoCategoryDataSet._addAbsenceSightings(dfObsIndivSmpl, sampleCols, dfTrPassEffSmpl)

    dAbscSightTmpl = dfObsIndivSmpl.iloc[0].to_dict()
    dAbscSightTmpl.update({k: None for k in dAbscSightTmpl.keys() if k not in sampleCols})
    logger.info(f'dAbscSightTmpl: {dAbscSightTmpl}')

    logger.info(f'dfObsIndivAbscSmpl: len={len(dfObsIndivAbscSmpl)}')

    logger.info('Espèces: ' + str(sorted(dfObsIndiv['Espèce'].unique())))

    # Check for no change in sample columns
    assert list(dfObsIndivAbscSmpl.columns) == list(dfObsIndivSmpl.columns)
    # Check for number of added rows
    assert len(dfObsIndivAbscSmpl) == 39  # 36 sightings + 3 missings transects
    # Check for final number of transects
    assert dfObsIndivAbscSmpl[dfTrPassEffSmpl.index.name].nunique() == 21
    # Check for no change in sample identification
    assert list(dfObsIndivAbscSmpl['Espèce'].unique()) == [espece]
    assert list(dfObsIndivAbscSmpl.Passage.unique()) == [passage]
    assert list(dfObsIndivAbscSmpl.Adulte.unique()) == [adulte]
    assert list(dfObsIndivAbscSmpl['Durée'].unique()) == [duree]

    logger.info0('PASS (testMonoCategoryDataSet) => _addAbsenceSightings')

    # ### d. ads.MonoCategoryDataSet._addSurveyAreaInfo
    dSurveyArea = dict(Zone='ACDC', Surface='2400')
    dfObsIndivAbscSmpl = ads.MonoCategoryDataSet._addSurveyAreaInfo(dfObsIndivAbscSmpl, dSurveyArea=dSurveyArea)

    dfObsIndivAbscSmpl.head()

    # ### e. MonoCategoryDataSet.sampleDataSet
    # (combines a, b, c and d above in one function : the one to use actually, of course !)
    mds = ads.MonoCategoryDataSet(dfObsIndiv, dSurveyArea=dSurveyArea, sampleDecFields=['Effort', 'distMem'],
                                  transectPlaceCols=transectPlaceCols, passIdCol=passIdCol,
                                  effortCol=effortCol, effortConstVal=1)

    sds = mds.sampleDataSet(sSampleSpecs=pd.Series({'Passage': passage, 'Adulte': adulte,
                                                    'Durée': duree, 'Espèce': espece}))
    # Compare result to above combination one.
    dfRes2Comp = sds.dfData.sort_values(by=['Point', 'DateHeure', 'distMem']).reset_index(drop=True)
    dfComb2Comp = dfObsIndivAbscSmpl.sort_values(by=['Point', 'DateHeure', 'distMem']).reset_index(drop=True)
    assert dfRes2Comp.compare(dfComb2Comp).empty
    logger.info0('PASS (testMonoCategoryDataSet) => sampleDataSet')

    # ### f. Performance test (should always pass, no real check)
    logger.info('Performance tests: ...')
    logger.info('Should give around 1.0s on a Core i7 8850H (6 HT cores, 2.6-4.3GHz, cache 9Mb) + NVME SSD')
    logger.info('Should give around 1.0s on a Core i5 8365U (4 HT cores, 1.6-4.1GHz, cache 6Mb) + NVME SSD')
    logger.info('Should give around 0.7s on a Core i7 10850H CPU (6 HT cores, 2.7-5.1GHz, cache 12Mb) + NVME SSD')
    logger.info('Espèce      Passage  Adulte Durée NbDonnées')
    start = time.perf_counter()
    for esp in ['Alauda arvensis', 'Sylvia communis', 'Phylloscopus collybita', 'Sylvia atricapilla']:
        for pas in ['a', 'b', 'a+b']:
            for ad in ['m', 'a', 'm+a']:
                for dur in ['5mn', '10mn']:
                    # passages = passage.split('+')
                    # adultes = adulte.split('+')
                    # dfObsIndivSmpl = dfObsIndiv[dfObsIndiv.Passage.isin(passages) & dfObsIndiv.Adulte.isin(adultes) \
                    #                            & (dfObsIndiv.Duree == dur) & (dfObsIndiv.Espece == esp)]
                    dfObsIndivSmpl, dfTrPassEffSmpl = \
                        ads.MonoCategoryDataSet._selectSampleSightings(dSample={'Passage': pas, 'Adulte': ad,
                                                                                'Durée': dur, 'Espèce': esp},
                                                                       dfAllSights=dfObsIndiv,
                                                                       dfAllEffort=dfTransPassEffort,
                                                                       transectPlaceCols=['Point'], passIdCol='Passage',
                                                                       effortCol='Effort')
                    try:
                        dfObsIndivAbscSmpl_ = \
                            ads.MonoCategoryDataSet._addAbsenceSightings(dfObsIndivSmpl, sampleCols, dfTrPassEffSmpl)
                        logger.info(f'{esp}, {pas}, {ad}, {dur}: {len(dfObsIndivSmpl)}'
                                    f' => {len(dfObsIndivAbscSmpl_)}')
                    except Exception as e:
                        print(e)
    end = time.perf_counter()
    logger.info(f'Elapsed time: {end - start}')

    logger.info0('PASS (testMonoCategoryDataSet) => performance: _selectSampleSightings + _addAbsenceSightings')


###############################################################################
#                         Actions to be done after all tests                  #
###############################################################################
def testEnd():

    uivu.logEnd(what=what2Test)


# This pytest-compatible module can also be run as a simple python script.
if __name__ == '__main__':

    run = True

    # Run auto-tests (exit(0) if OK, 1 if not).
    rc = -1

    uivu.logBegin(what=what2Test)

    if run:
        try:
            # Let's go.
            testBegin()

            # Tests for DataSet
            testDataSetCtorLen(sources())
            testDataSetDfData(sources())
            testDataSetEmpty()
            testDataSetColumns(sources())
            testDataSetRenameColumns(sources())
            testDataSetAddComputedColumns(sources())
            testDataSetDropColumns(sources())
            testDataSetDfSubData(sources())
            testDataSetDropRows(sources())
            testDataSetToFiles(sources())
            testDataSetCloseness()
            testDataSetCompare()

            # Tests for other XxxDataSet
            testSDSCtor()
            testFieldDataSet()
            testMonoCategoryDataSet()

            # Tests for ResultsSet
            # => See unint_analyser_results_test and unint_optanalyser_results_test

            # Done.
            testEnd()

            # Success !
            rc = 0

        except Exception as exc:
            logger.exception(f'Exception: {exc}')
            rc = 1

    uivu.logEnd(what=what2Test, rc=rc)

    sys.exit(rc)
