# Unit integration automated tests
#
# Application Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# Current test file author:Jean-Philippe Meuret & Sylvain Sainnier
# License: GPL 3

# To run : simply run "pytest" or "python <this file>" in current folder
#          and check standard output ; and tmp/unit-int-test.{datetime}.log for details

import sys
import pathlib as pl
import pandas as pd
import numpy as np
import logging

sys.path.insert(0, '..')
import autods as ads

import pytest


###############################################################################
#   LOGGERS CONFIG & RUN ENVIRONMENT RECORDING
#
#   Located there instead of in '__main__' in order that loggers could be used
#   by running '__main__' OR pytest
#
# INFORMATION: I kept this function to set handlers for multiple children.
# May be implemented as method from the 'log' class ???
###############################################################################
def configureLoggers(loggers=[dict(name='child', level=logging.ERROR)],
                     handlers=[sys.stdout], fileMode='w',
                     format='%(asctime)s %(name)s %(levelname)s\t%(message)s'):
    """Configure loggers (levels, handlers, formatter, ...)

    Note: Setting handlers for multiple children rather than once and for all for root ...
           gives bad things on FileHandlers, with many missing / intermixed / unsorted lines ...
           => unusable. Whereas it seems to work well with StreamHandlers
    """

    # Configure root logger (assuming children have propagate=on).
    root = logging.getLogger()
    root.debug('ROOT')
    formatter = logging.Formatter(format)
    for hdlr in handlers:
        if isinstance(hdlr, str):
            handler = logging.FileHandler(hdlr, mode=fileMode)
        else:
            handler = logging.StreamHandler(stream=hdlr)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    def handlerId(hdlr):
        return 'File({})'.format(hdlr) if isinstance(hdlr, str) else 'Stream({})'.format(hdlr.name)
    root.setLevel(logging.INFO)

    # Configure children loggers.
    msg = 'Logging to {}'.format(', '.join(handlerId(hdlr) for hdlr in handlers))
    for logrCfg in loggers:
        logr = logging.getLogger(logrCfg['name'])
        logr.info(msg)
        if 'level' in logrCfg:
            logr.setLevel(logrCfg['level'])


def describeRunEnv():

    logger.info('Python environment:')
    logger.info('*  {}: {}'.format(sys.implementation.name, sys.version))
    logger.info('* platform: {}'.format(sys.platform))
    for module in ['pytest', 'pandas', 'numpy', 'logging']:  # 'lxml', 'scipy', 'pyproj', 'shapely']:
        logger.info('* {:>8s}: {}'.format(module, sys.modules[module].__version__))
    logger.info('')


# logger = logging.getLogger('unint_data_test')
logger = ads.logger('unint_data_test')  # works with both lines. This ones seems more consistent to me

# List of logger/sub_loggers
l_loggers = [dict(name='unint_data_test', level=ads.DEBUG),
             dict(name='ads', level=ads.INFO),
             dict(name='matplotlib', level=ads.WARNING),
             dict(name='ads.eng', level=ads.INFO2),
             dict(name='ads.opn', level=ads.INFO1),
             dict(name='ads.opr', level=ads.INFO1),
             dict(name='ads.anr', level=ads.INFO1)]
#          line below added to limit log ouput
#            dict(name='ads.dat', level=ads.WARNING)]

# Configure logging.
configureLoggers(l_loggers, handlers=['tmp/unit-int-test.{}.log'.format(pd.Timestamp.now().strftime('%Y%m%d'))])
logger.info('')

# Describe environment.
describeRunEnv()

logger.info(f'Testing ads {ads.__version__} ...')


###############################################################################
#                         Input Data Preparation                              #
###############################################################################
#   Generate DataFrame (returned) and other format files  from .ods source
#   and return a list of sources (4 files and 1 DataFrame)
def sources():
    dfPapAlaArv = pd.read_excel('refin/ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.ods')

    dfPapAlaArv.to_csv('tmp/ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.csv', sep='\t', index=False)
    dfPapAlaArv.to_excel('tmp/ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.xls', index=False)

    # DataSet from multiple sources from various formats (same columns).
    # For test, list of require to contain 1 DataFrame, and one or several
    # source files (one for each different extension)
    # DataFrame and all files need to contain the same data
    sources = ['refin/ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.ods',  # Need for module odfpy
               'refin/ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.xlsx',  # Need for module openpyxl (or xlrd)
               'tmp/ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.xls',  # No need module xlwt(openpyxl seems to just do it)
               'tmp/ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.csv',
               dfPapAlaArv]
    return sources


# same as above => to allow using pytest
@pytest.fixture
def sources_fxt():
    return sources()


###############################################################################
#                          Miscellaneous Tools                                #
###############################################################################
def male2bool(s):
    return False if pd.isnull(s.MALE) or s.MALE.lower() != 'oui' else True


def sexe2bool(s):
    return False if pd.isnull(s.SEXE) or s.SEXE.lower() != 'oui' else True


def newval2othercol(s):
    return "New Value"


###############################################################################
#                                Test Cases                                   #
###############################################################################
# test DataSet - method "Ctor", getter "dfData", and "__len__"
    # from various sources: .ads, .xslx, .xls, .csv and from DataFrame
def test_DataSet_Ctor_len(sources_fxt):

    # gather source DataFrame (used for checking size of DataSet)
    srcDf = [src for src in sources_fxt if isinstance(src, pd.core.frame.DataFrame)][0]

    # list of awaited columns form input DataFrame with associated types
    dTypes = {'ZONE': 'object', 'HA': 'int', 'POINT': 'int', 'ESPECE': 'object',
              'DISTANCE': 'float', 'MALE': 'object', 'DATE': 'object', 'OBSERVATEUR': 'object',
              'PASSAGE': 'object', 'NOMBRE': 'float', 'EFFORT': 'int'}

    ds = ads.DataSet(sources_fxt, importDecFields=['EFFORT', 'DISTANCE', 'NOMBRE'],
                     sheet='Sheet1', skipRows=None, separator='\t')

    # checking size of DataFrame
    assert ds.dfData.size == srcDf.size * len(sources_fxt), \
        'Error: test_DataSet_Ctor_len: DataSet Ctor: Size of DataFrame "DataFrame.dfData" inconsistent with sources'
    # checking list of columns of DataFrame
    assert sorted(ds.columns) == sorted(dTypes.keys()), \
        'Error: test_DataSet_Ctor_len: Ctor: Columns list from "DataFrame.dfData" mismatched with columns list of \
        the source file'
    # checking column dtypes of DataFrame
    assert all(typ.name.startswith(dTypes[col]) for col, typ in ds.dfData.dtypes.items()), \
        'Error: test_DataSet_Ctor_len: DataSet Ctor: types of data from the "DataFrame.dfData" mismatched awaited \
        data types'

    logger.info0('PASS (test_DataSet_Ctor_len) => DATASET => Constructor, getter "dfData" and \
    method "__len__"\n(Includes _csv2df, _fromDataFrame, _fromDataFile)')


# test setter dfData
def test_DataSet_dfData_getter_setter(sources_fxt):

    ds = ads.DataSet(sources_fxt, sheet='Sheet1', skipRows=None, separator='\t')

    # Check Exception raising with dfData setter call
    with pytest.raises(NotImplementedError) as e:
        ds.dfData = pd.DataFrame()
    logger.info0('PASS (test_DataSet_dfData_getter_setter) => EXCEPTION RAISED AS AWAITED with the following \
    Exception message:\n{}'.format(e))


# test DataSet - methoc "empty"
def test_DataSet_empty():

    emptyDf = pd.DataFrame()
    emptyDs = ads.DataSet(emptyDf)
    assert emptyDs.empty, 'Error: test_DataSet_empty: DataFrame from DataSet should be empty'

    ds = pd.DataFrame(data={'col1': [1, 2]})
    assert not ds.empty, 'Error: test_DataSet_empty: DataFrame from DataSet should not be empty'

    logger.info0('PASS (test_DataSet_empty) => DATASET => method "empty"')


# test DataSet - method "columns"
def test_DataSet_columns(sources_fxt):

    ds = ads.DataSet(sources_fxt, sheet='Sheet1', skipRows=None, separator='\t')

    assert (ds.columns == ds.dfData.columns).all(), 'Error: test_DataSet_columns: columns: Issue occurred with \
    method "DataSet.columns"'

    logger.info0('PASS (test_DataSet_columns) => DATASET => method "columns"')


# test columns renaming
def test_DataSet_renameColumns(sources_fxt):

    ds = ads.DataSet(sources_fxt, dRenameCols={'NOMBRE': 'INDIVIDUS', 'MALE': 'SEXE'},
                     importDecFields=['EFFORT', 'DISTANCE', 'NOMBRE'], sheet='Sheet1', skipRows=None, separator='\t')

    assert all([(col not in ds.columns) for col in ['MALE', 'NOMBRE']]) \
        and all([(col in ds.columns) for col in ['SEXE', 'INDIVIDUS']]), 'Error: test_DataSet_renameColumns: \
        renameColumns: Issue occurred with renaming process: columns "NOMBRE" and "MALE" should have been \
        renamed "INDIVIDUS" and "SEXE"'

    logger.info0('PASS (test_DataSet_renameColumns) => DATASET => method "renameColumns"\n(Includes _csv2df, \
    _fromDataFrame, _fromDataFile)')


# test columns addition, recomputing, and combination addition/recomputing
# (_addComputedColumns) for existing or new column
def test_DataSet_addComputedColumns(sources_fxt):

    ds = ads.DataSet(sources_fxt, sheet='Sheet1', skipRows=None, separator='\t')

    # Checking addition of a columns
    ds = ads.DataSet(ds.dfData, dComputeCols={'NEWCOL': ''})
    assert 'NEWCOL' in ds.columns, 'Error: test_DataSet_addComputedColumns: _addComputedColumns: Issue occurred with \
    simple addition of a new column. Columns "NEWCOL" should have been added'

    # Checking re-computing a columns
    #   Checking initial data type of 'MALE' column values are not boolean
    assert ds.dfData['MALE'].dtype != bool, 'Error: test_DataSet_addComputedColumns: Pre-test: Values from MALE \
    column should not be bool'
    #   Checking data type of 'MALE' column modified to boolean (dComputeCols)
    ds = ads.DataSet(ds.dfData, dComputeCols={'MALE': male2bool})
    assert ds.dfData['MALE'].dtype == bool, \
        'Error: test_DataSet_addComputedColumns: _addComputedColumns: Issue occurred with recomputing existing \
        column process. Values from MALE column should be bool'

    # Checking adding a new column with computing process
    #   Checking data type of 'NEWCOL' colonne was added with boolean (dComputeCols)
    ds = ads.DataSet(ds.dfData, dComputeCols={'OTHERCOL': newval2othercol})
    assert ds.dfData.OTHERCOL[0] == 'New Value', 'Error: test_DataSet_addComputedColumns: _addComputedColumns: \
    Issue occurred with process of adding a new AND computed column.'

    # Checking adding a new column with computing process and rename it
    #   a new column also added to prepare following test step
    ds = ads.DataSet(sources_fxt, dRenameCols={'MALE': 'SEXE'}, dComputeCols={'MALE': sexe2bool, 'NEWCOL': ''},
                     sheet='Sheet1', skipRows=None, separator='\t')
    print(ds.columns)
    print(ds.dfData.SEXE)
    assert ds.dfData['SEXE'].dtype == bool, \
        'Error: test_DataSet_addComputedColumns: _addComputedColumns: Issue occurred with process of adding and \
        renaming a new AND computed column.'

    logger.info0('PASS (test_DataSet_renameColumns) => DATASET => methods "addColumns" and "_addComputedColumns": \
    \n\t\t\t\t\t\taddition, recomputing, and combination addition/recomputing for existing or new column checked \
    \n\t\t\t\t\t\t(Includes _csv2df, _fromDataFrame, _fromDataFile)')


# test method "dfSubData"
def test_DataSet_dfSubData(sources_fxt):

    ds = ads.DataSet(sources_fxt, sheet='Sheet1', skipRows=None, separator='\t')

    # subset = None
    df = ds.dfSubData()
    assert df.compare(ds.dfData).empty, 'Error: test_DataSet_dfSubData: dfSubData: Issue occurred. With "None", \
    df should be a copy of source DataSet.'
    assert id(df) == id(ds.dfData), 'Error: test_DataSet_dfSubData: dfSubData: Issue occurred. With "copy = False \
    (default)", subset of DataFrame should be a reference to the subset of source DataFrame, not a copy of data.'

    df = ds.dfSubData(copy=True)
    assert df.compare(ds.dfData).empty, 'Error: test_DataSet_dfSubData: dfSubData: Issue occurred. With "None", \
    df should be a copy of source DataSet.'
    assert id(df) != id(ds.dfData), 'Error: test_DataSet_dfSubData: dfSubData: Issue occurred. With "copy = True", \
    subset of DataFrame should be a copy of data, not a reference to the subset of source DataFrame.'

    # subset = as list
    df = ds.dfSubData(subset=['POINT', 'ESPECE', 'DISTANCE', 'INDIVIDUS', 'EFFORT'])
    assert df.compare(ds.dfData.reindex(columns=['POINT', 'ESPECE', 'DISTANCE', 'INDIVIDUS', 'EFFORT'])).empty, \
        'Error: test_DataSet_dfSubData: dfSubData: Issue occurred. With "None", df should be a copy of source DataSet.'

    # subset = as pandas.index
    df = ds.dfSubData(subset=pd.Index(['POINT', 'ESPECE', 'DISTANCE', 'INDIVIDUS', 'EFFORT']))
    assert df.compare(ds.dfSubData(subset=pd.Index(['POINT', 'ESPECE', 'DISTANCE', 'INDIVIDUS', 'EFFORT']))).empty, \
        'Error: test_DataSet_dfSubData: dfSubData: Issue occurred. With "None", df should be a copy of source DataSet.'

    # subset = as other (str)
    # Check Exception raising
    with pytest.raises(Exception) as e:
        df = ds.dfSubData(subset='POINT')
    logger.info0('PASS (test_DataSet_dfSubData) => EXCEPTION RAISED AS AWAITED with the \
    following Exception message:\n{}'.format(e))

    logger.info0('PASS (test_DataSet_dfSubData) => DATASET => method "dfSubData"')


# test columns deletion
def test_DataSet_dropColumns(sources_fxt):

    ds = ads.DataSet(sources_fxt, sheet='Sheet1', skipRows=None, separator='\t')

    # gather source DataFrame (used for checking size of DataSet)
    srcDf = [src for src in sources_fxt if isinstance(src, pd.core.frame.DataFrame)][0]

    # Checking deletion of columns
    dropCols = ['ZONE', 'HA', 'OBSERVATEUR']

    ds.dropColumns(dropCols)

    assert ds.columns.to_list() == ['POINT', 'ESPECE', 'DISTANCE', 'MALE', 'DATE', 'PASSAGE', 'NOMBRE', 'EFFORT'], \
        'Error: test_DataSet_dropColumns: dropColumns: Issue occurred with dropping columns: one or more \
        columns were not drop.'

    assert len(ds) == len(srcDf) * len(sources_fxt),  \
        'Error: test_DataSet_dropColumns: dropColumns: Issue occurred: inconsistent size of the DataFrame next \
        to columns dropping.'

    logger.info0('PASS (test_DataSet_dropColumns) => DATASET => method "dropColumns"')


# test rows deletion
def test_DataSet_dropRows(sources_fxt):

    ds = ads.DataSet(sources_fxt, sheet='Sheet1', dComputeCols={'MALE': male2bool}, skipRows=None, separator='\t')

    # gather source DataFrame (used for checking size of DataSet)
    srcDf = [src for src in sources_fxt if isinstance(src, pd.core.frame.DataFrame)][0]

    # deletion of rows with no distance noted
    ds.dropRows(ds.dfData.DISTANCE.isnull())

    # check result DataFrame is same size than entry with distance not null in source DataFrame
    assert len(ds) == len(srcDf[srcDf.DISTANCE.notnull()]) * len(sources_fxt), \
        'Error: test_DataSet_dropRows: dropRows: Issue occurred. Size of resulted DataFrame \
        not consistent after deletion process.'

    # check nb of rows are consistent with data before/after deletion (refer source files data)
    assert ds.dfData.MALE.value_counts()[True] == ds.dfData.NOMBRE.sum() == srcDf.NOMBRE.sum() * len(sources_fxt), \
        'Error: test_DataSet_dropRows: dropRows: Issue occurred. Comparison of numbers of data in resulted DataFrame \
        not consistentafter deletion process.'

    logger.info0('PASS (test_DataSet_dropRows) => DATASET => method "dropRows"')


# DATASET TESTS
# test methods "toExcel", "toOpenDoc", "toPickle", "compareDataFrames"
def test_toFiles(sources_fxt):

    ds = ads.DataSet(sources_fxt, importDecFields=['EFFORT', 'DISTANCE', 'NOMBRE'],
                     dRenameCols={'NOMBRE': 'INDIVIDUS'}, dComputeCols={'MALE': male2bool},
                     sheet='Sheet1', skipRows=None, separator='\t')
    ds.dropColumns(['ZONE', 'HA', 'OBSERVATEUR'])
    ds.dropRows(ds.dfData.DISTANCE.isnull())
    # => toExcel, toOpenDoc, toPickle, compareDataFrames
    closenessThreshold = 15  # => max relative delta = 1e-15
    subsetCols = ['POINT', 'ESPECE', 'DISTANCE', 'INDIVIDUS', 'EFFORT']
    filePathName = pl.Path('tmp') / 'dataset-uni.ods'
    dfRef = ds.dfSubData(subsetCols).reset_index(drop=True)

    for fpn in [filePathName, filePathName.with_suffix('.xlsx'), filePathName.with_suffix('.xls'),
                filePathName.with_suffix('.pickle'), filePathName.with_suffix('.pickle.xz')]:

        print(fpn.as_posix(), end=' : ')
        if fpn.suffix == '.ods':
            ds.toOpenDoc(fpn, sheetName='utest', subset=subsetCols, index=False)
        elif fpn.suffix in ['.xlsx', '.xls']:
            ds.toExcel(fpn, sheetName='utest', subset=subsetCols, index=False)
        elif fpn.suffix in ['.pickle', '.xz']:
            ds.toPickle(fpn, subset=subsetCols, index=False)
        assert fpn.is_file(), 'Error: test_toFiles'

        if fpn.suffix in ['.ods', '.xlsx', '.xls']:
            df = pd.read_excel(fpn, sheet_name='utest')
        elif fpn.suffix in ['.pickle', '.xz']:
            df = pd.read_pickle(fpn)
            df.reset_index(drop=True, inplace=True)
        assert ds.compareDataFrames(df.reset_index(), dfRef.reset_index(),
                                    subsetCols=['POINT', 'DISTANCE', 'INDIVIDUS', 'EFFORT'],
                                    indexCols=['index'], dropCloser=closenessThreshold, dropNans=True).empty
        print('1e-{} comparison OK (df.equals(dfRef) is {}, df.compare(dfRef) {}empty)'.
              format(closenessThreshold, df.equals(dfRef), '' if df.compare(dfRef).empty else 'not')), \
            'Error: test_toFiles'

    logger.info0('PASS (test_toFiles) => DATASET => method "toExcel", "toOpenDoc", "toPickle", "compareDataFrames"')


# Test of the base function for comparison (test from static hard-coded data, not from loaded DataSets)
# method "_closeness" used for
def test_closeness():
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
            except Exception as exc:
                print(exc, r, c, values[r], values[c])
                raise

    # Infinite closeness on the diagonal (except for nan and +/-inf)
    assert all(np.isnan(values[i]) or np.isinf(values[i]) or np.isinf(aClose[i, i]) for i in range(len(values))), \
           'Error: test_closeness: Inequality on the diagonal'

    # No infinite proximity elsewhere
    assert all(r == c or not np.isinf(aClose[r, c]) for r in range(len(values)) for c in range(len(values))), \
           'Error: test_closeness: No equality should be found outside the diagonal'

    # Good closeness around -1 only
    whereClose = [i for i in range(len(values)) if abs(values[i] + 1) <= 1.0e-5]
    assert all(aClose[r, c] > 4 for r in whereClose for c in whereClose), \
        'Error: test_closeness: Unexpectedly bad closeness around -1'

    logger.info0('PASS (test_closeness) => DATASET => method "_closeness"')


# Comparison (from other files data sources, the same as for ResultsSet.compare below, but through DataSet)
# => compare, compareDataFrames, _toHashable, _closeness
def test_compare():
    # a. Loading of Distance 7 and values to compare issued from AutoDS
    dsDist = ads.DataSet('refin/ACDC2019-Papyrus-ALAARV-TURMER-comp-dist-auto.ods',
                         sheet='RefDist73', skipRows=[3], headerRows=[0, 1, 2], indexCols=0)

    dsAuto = ads.DataSet('refin/ACDC2019-Papyrus-ALAARV-TURMER-comp-dist-auto.ods',
                         sheet='ActAuto', skipRows=[3], headerRows=[0, 1, 2], indexCols=0)

    # b. Index columns to be compared
    indexCols = [('sample', 'AnlysNum', 'Value')] \
              + [('sample', col, 'Value') for col in ['Species', 'Periods', 'Prec.', 'Duration']] \
              + [('model', 'Model', 'Value')] \
              + [('parameters', 'left truncation distance', 'Value'),
                 ('parameters', 'right truncation distance', 'Value'),
                 ('parameters', 'model fitting distance cut points', 'Value'),
                 ('parameters', 'distance discretisation cut points', 'Value')]

    # # c. Columns to be compared (DeltaDCV and DeltaAIC were removed as results are dependant of a set of ran analyses,
    # #    different between reference and AutoDS run.
    subsetCols = [col for col in dsDist.dfData.columns.to_list()
                  if col not in indexCols + [('run output', 'run time', 'Value'),
                                             ('density/abundance', 'density of animals', 'Delta Cv'),
                                             ('detection probability', 'Delta AIC', 'Value')]]

    # # d. "Exact" Comparison : no line pass (majority of epsilons due to IO ODS)
    dfRelDiff = dsDist.compare(dsAuto, subsetCols=subsetCols, indexCols=indexCols)
    assert len(dfRelDiff) == len(dsDist), 'Error: test_compare: compare: Issue occurred. Exact row by row comparison \
    of both DataSets should fail for all rows.'

    # # e. Comparison with 10**-16 accuracy : almost all lines pass, but 3 (majority of epsilons due to IO ODS)
    dfRelDiff = dsDist.compare(dsAuto, subsetCols=subsetCols, indexCols=indexCols, dropCloser=16, dropNans=True)
    assert len(dfRelDiff) == 3, 'Error: test_compare: compare: Issue occurred. Row by row comparison of both DataSet \
    with accuracy of 10**-16 should fail for all rows but 3.'
    print(dfRelDiff)

    # # e. Comparison with 10**-5 accuracy : all lines pass
    dfRelDiff = dsDist.compare(dsAuto, subsetCols=subsetCols, indexCols=indexCols, dropCloser=5, dropNans=True)
    assert len(dfRelDiff) == 2,  'Error: test_compare: compare: Issue occurred. Row by row comparison of both DataSet \
    with accuracy of 10**5 should pass for all rows.'

    logger.info0('PASS (test_compare) => DATASET => method "_closeness" and "compare"')


# SAMPLEDATASET TESTS
# test methods "toExcel", "toOpenDoc", "toPickle", "compareDataFrames"
def test_SDS_Ctor():
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
        'Error: test_SDS_Ctor: Some strings found in declared decimal fields ... any decimal format issue ?'

    assert sds.columns.equals(dfData.columns), 'Error: test_SDS_Ctor: inconsistency with columns list \
    of SDS vs source DataFrame'
    assert len(sds) == len(dfData),  'Error: test_SDS_Ctor: inconsistency with size of SDS vs source DataFrame'
    assert sds.dfData.Distance.notnull().sum() == 4, 'Error: test_SDS_Ctor: issue with NaN values \
    that should be considered a null'

    # test: empty SDS raises exception
    emptyDf = pd.DataFrame()
    with pytest.raises(Exception) as e_info:
        ads.SampleDataSet(emptyDf)
    logger.info0('PASS (test_SDS_Ctor) => EXCEPTION RAISED AS AWAITED with the following Exception message:\n{}'.
                 format(e_info.value))

    # test: exception raised if SDS created with DataFrame < 5 columns
    dfData = pd.DataFrame(columns=['Col1', 'Col2', 'Col3', 'Col4'], data=[(1, 2, 3, 4)])
    with pytest.raises(Exception) as e_info:
        sds = ads.SampleDataSet(source=dfData)
    logger.info0('PASS (test_SDS_Ctor) => EXCEPTION RAISED AS AWAITED with the following Exception message:\n{}'.
                 format(e_info.value))

    # Excel source (path as simple string)
    sds = ads.SampleDataSet(source='refin/ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.xlsx',
                            decimalFields=['EFFORT', 'DISTANCE', 'NOMBRE'])

    assert sds.columns.to_list() == ['ZONE', 'HA', 'POINT', 'ESPECE', 'DISTANCE', 'MALE', 'DATE',
                                     'OBSERVATEUR', 'PASSAGE', 'NOMBRE', 'EFFORT'], \
        'Error: test_SDS_Ctor: inconsistency with columns list of SDS vs source file'
    assert len(sds) == 256, 'Error: test_SDS_Ctor: inconsistency with size of the resulting SDS. \
    It should contain 256 rows'
    assert sds.dfData.NOMBRE.sum() == 217, 'Error: test_SDS_Ctor: inconsistency with data loaded in SDS: \
    Sum of values from column "NOMBRE" should be 217'

    # Libre / Open Office source (path as simple string)
    sds = ads.SampleDataSet(source='refin/ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.ods',
                            decimalFields=['EFFORT', 'DISTANCE', 'NOMBRE'])

    assert sds.columns.to_list() == ['ZONE', 'HA', 'POINT', 'ESPECE', 'DISTANCE', 'MALE', 'DATE',
                                     'OBSERVATEUR', 'PASSAGE', 'NOMBRE', 'EFFORT'], \
        'Error: test_SDS_Ctor: inconsistency with columns list of SDS vs source file'
    assert len(sds) == 256, 'Error: inconsistency with size of the resulting SDS. It should contain 256 rows'
    assert sds.dfData.NOMBRE.sum() == 217, 'Error: test_SDS_Ctor: inconsistency with data loaded in SDS: Sum of values \
    from column "NOMBRE" should be 217'

    # CSV source with ',' as decimal point (path as pl.Path)
    sds = ads.SampleDataSet(source=pl.Path('refin/ACDC2019-Papyrus-TURMER-AB-5mn-1dec-dist.txt'),
                            decimalFields=['Point transect*Survey effort', 'Observation*Radial distance'])

    assert not any(sds.dfData[col].dropna().apply(lambda v: isinstance(v, str)).any() for col in sds.decimalFields), \
        'Error: test_SDS_Ctor: Some strings found in declared decimal fields ... any decimal format issue ?'

    assert sds.columns.to_list() == ['Region*Label', 'Region*Area', 'Point transect*Label',
                                     'Point transect*Survey effort', 'Observation*Radial distance'], \
        'Error: test_SDS_Ctor: inconsistency with columns list of SDS vs source file'
    assert len(sds) == 330, 'Error: test_SDS_Ctor: inconsistency with size of the resulting SDS. \
    It should contain 330 rows'
    assert sds.dfData['Observation*Radial distance'].notnull().sum() == 324, 'Error: test_SDS_Ctor: inconsistency with \
    data loaded in SDS: Sum of values from column "Observation*Radial distance" should be 324'

    # CSV source with '.' as decimal point
    sds = ads.SampleDataSet(source=pl.Path('refin/ACDC2019-Papyrus-ALAARV-AB-10mn-1dotdec-dist.txt'),
                            decimalFields=['Point transect*Survey effort', 'Observation*Radial distance'])

    assert not any(sds.dfData[col].dropna().apply(lambda v: isinstance(v, str)).any() for col in sds.decimalFields), \
        'Error: test_SDS_Ctor: Some strings found in declared decimal fields ... any decimal format issue ?'

    assert sds.columns.to_list() == ['Region*Label', 'Region*Area', 'Point transect*Label',
                                     'Point transect*Survey effort', 'Observation*Radial distance'], \
        'Error: test_SDS_Ctor: inconsistency with columns list of SDS vs source file'
    assert len(sds) == 256,  'Error: test_SDS_Ctor: inconsistency with size of the resulting SDS. \
    It should contain 256 rows'
    assert sds.dfData['Observation*Radial distance'].notnull().sum() == 217, 'Error: test_SDS_Ctor: inconsistency with \
    data loaded in SDS: Sum of values from column "Observation*Radial distance" should be 217'

    logger.info0('PASS (test_SDS_Ctor) => SampleDATASET => Constructor')


if __name__ == '__main__':

    run = True
    # Run auto-tests (exit(0) if OK, 1 if not).
    rc = -1

    if run:
        try:
            # Tests for DataSet
            test_DataSet_Ctor_len(sources())
            test_DataSet_dfData_getter_setter(sources())
            test_DataSet_empty()
            test_DataSet_columns(sources())
            test_DataSet_renameColumns(sources())
            test_DataSet_addComputedColumns(sources())
            test_DataSet_dropColumns(sources())
            test_DataSet_dfSubData(sources())
            test_DataSet_dropRows(sources())
            test_toFiles(sources())
            test_closeness()
            test_compare()

            # Tests for SampleDataSet
            test_SDS_Ctor()

            # Success !
            rc = 0

        except Exception as exc:
            logger.exception('Exception: ' + str(exc))
            rc = 1

    logger.info('Done unit integration testing autods.data: {} (code: {})'
                .format({-1: 'Not run', 0: 'Success'}.get(rc, 'Error'), rc))
    sys.exit(rc)
