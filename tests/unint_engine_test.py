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

# Automated unit and integration tests for "engine" submodule

# To run : simply run "pytest" or "python <this file>" in current folder
#          and check standard output ; and tmp/unit-int-test.{datetime}.log for details

import sys
import re
import shutil
import time
import pathlib as pl
import pandas as pd
import numpy as np
import logging

import pyaudisam as ads

import pytest


# The "tests" folder.
KTestSrcPath = pl.Path(__file__).parent

# Temporary work folder (created once in conftest.py if needed).
tmpDir = KTestSrcPath / 'tmp'

# Logger for this module.
logger = ads.logger('unt.eng')

###############################################################################
#                         Actions to be done before any test                  #
###############################################################################
def test_init():

    # Configure logging.
    logFilePathName = tmpDir / 'unt-eng.{}.log'.format(pd.Timestamp.now().strftime('%Y%m%d%H%M'))
    ads.log.configure(loggers=[dict(name='matplotlib', level=ads.WARNING),
                               dict(name='ads', level=ads.INFO),
                               #dict(name='ads.dat', level=ads.WARNING),  # Uncomment to limit log ouput
                               dict(name='ads.eng', level=ads.INFO2),
                               dict(name='unt.eng', level=ads.DEBUG)],
                      handlers=[logFilePathName], reset=True)

    # Show testing configuration (traçability).
    logger.info('PyAuDiSam {} from {}'
                .format(ads.__version__, pl.Path(ads.__path__[0]).resolve().as_posix()))
    logger.info('Python environment:')
    logger.info('*  {}: {}'.format(sys.implementation.name, sys.version))
    logger.info('* platform: {}'.format(sys.platform))
    for module in ['pytest', 'pandas', 'numpy']:  # 'lxml', 'scipy', 'pyproj', 'shapely']:
        logger.info('* {:>8s}: {}'.format(module, sys.modules[module].__version__))

    logger.info('Testing pyaudisam.engine ...')

###############################################################################
#                         Input Data Preparation                              #
###############################################################################
#   Generate a short DataFrame (returned) for test purpose
#   and return a list of sources (4 files and 1 DataFrame)
def shortDF():

    shortDF = pd.DataFrame(columns=['Date', 'TrucDec', 'Espece', 'Point', 'Effort', 'Distance'],
                           data=[('2019-05-13', 3.5, 'TURMER', 23, 2, 83),
                                 ('2019-05-15', np.nan, 'TURMER', 23, 2, 27.355),
                                 ('2019-05-13', 0, 'ALAARV', 29, 2, 56.85),
                                 ('2019-04-03', 1.325, 'PRUMOD', 53, 1.3, 7.2),
                                 ('2019-06-01', 2, 'PHICOL', 12, 1, np.nan),
                                 ('2019-06-19', np.nan, 'PHICOL', 17, 0.5, np.nan),
                                 ])
    shortDF['Region'] = 'ACDC'
    shortDF['Surface'] = '2400'

    return shortDF


# same as above => to allow using pytest
@pytest.fixture
def shortDF_fxt():

    return shortDF()


###############################################################################
#                          Miscellaneous Tools                                #
###############################################################################
###############################################################################
#                                Test Cases                                   #
###############################################################################
def test_executableNotFound():

    with pytest.raises(Exception) as e_info:
        ads.MCDSEngine().findExecutable('WrongName')
    logger.info0('PASS (test_executableNotFound) => EXCEPTION RAISED AS AWAITED with the following \
    Exception message:\n{}'.format(e_info.value))


def test_MCDS_Ctor():

    # test Exception raising with one of unsupported characters in workdir string (space) - first way
    with pytest.raises(Exception) as e_info:
        ads.MCDSEngine(workDir=tmpDir.as_posix() + '/test out')  # Simple string path
    logger.info0('PASS (test_MCDS_Ctor) => EXCEPTION RAISED AS AWAITED with the following Exception message:\n{}'
                 .format(e_info.value))

    # test Exception raising with one of unsupported characters in workdir string (space) - second way
    with pytest.raises(Exception) as e_info:
        ads.MCDSEngine(workDir=tmpDir / 'test out')  # pl.Path path
    logger.info0('PASS (test_MCDS_Ctor) => EXCEPTION RAISED AS AWAITED with the following Exception message:\n{}'
                 .format(e_info.value))

    # test preferred method to initiate MCDSEngine
    assert ads.MCDSEngine(workDir=tmpDir / 'mcds-out', runMethod='os.system'), 'MCDS Engine: \
    Non identified issue at engine initiation'

    # test previous (older) method to initiate MCDSEngine
    assert ads.MCDSEngine(workDir=tmpDir / 'mcds-out'), 'MCDS Engine: Non identified issue at engine initiation'

    # test Specs DataFrames not empty
    # TODO: improve for deeper test ???
    assert not any([ads.MCDSEngine().statRowSpecs().empty, ads.MCDSEngine().statModSpecs().empty,
                    ads.MCDSEngine().statModCols().empty, ads.MCDSEngine().statModNotes().empty,
                    ads.MCDSEngine().statModColTrans().empty]), 'Specs DataFrames: issue occurred with initialization \
                    from external files'

    logger.info0('PASS => MCDSEngine => Constructor and methods "_run", "_runThroughOSSystem", \
    "_runThroughSubProcessRun", "loadStatSpecs" and setters for related class variables ')


def test_MCDS_setupRunFolder():

    runDir = ads.MCDSEngine().setupRunFolder(runPrefix=' t,e.s:t; ( )_setupRunFolder')

    assert not re.search('[ ,.:;()/]', str(runDir)), 'Error: test_MCDS_setupRunFolde: Setted up directory: unsupported \
    caracters should have been cleaned up'
    assert runDir.exists(), 'Error: test_MCDS_setupRunFolde: temporary directory not created'
    # clean-up: tmp directory deleted
    runDir.rmdir()

    logger.info0('PASS => MCDSEngine => method "setupRunFolder"')


def test_buildExportTable(shortDF_fxt):

    eng = ads.MCDSEngine()
    dfData = shortDF_fxt  # load source test data
    dfData.drop(['Surface'], axis=1, inplace=True)  # to create missing field

    # test exception raised if at least one field missing vs DSEngine.ImportFieldAliasREs
    with pytest.raises(Exception) as e_info:
        dfExport, extraFields = eng.buildExportTable(ads.SampleDataSet(dfData),
                                                     withExtraFields=True, decPoint='.')
    logger.info0('PASS (test_buildExportTable) => EXCEPTION RAISED AS AWAITED with the following Exception message:\n{}'
                 .format(e_info.value))

    # missing field added
    dfData['Surface'] = '2400'

    dfExport, extraFields = ads.MCDSEngine().buildExportTable(ads.SampleDataSet(dfData),
                                                              withExtraFields=False, decPoint='.')

    # withExtraFields=False, number/order of columns same as for DSEngine.ImportFieldAliasREs, and
    # comparison of source and resulting DataFrames'size
    assert dfExport.columns.to_list() == ['Region', 'Surface', 'Point', 'Effort', 'Distance'] and \
           dfExport.size == dfData.loc[:, ['Region', 'Surface', 'Point', 'Effort', 'Distance']].size, 'Error: \
           test_buildExportTable: issue with resulting exported DataFrame (size, columns list or columns order). \
           With option "withExtraFields=False"'

    dfExport, extraFields = ads.MCDSEngine().buildExportTable(ads.SampleDataSet(dfData, decimalFields=['TrucDec']),
                                                              withExtraFields=True, decPoint='.')

    # withExtraFields=True, number/order of columns same as for DSEngine.ImportFieldAliasREs + extraFields columns, and
    # comparison of source and resulting DataFrames'size
    assert dfExport.columns.to_list() == \
           ['Region', 'Surface', 'Point', 'Effort', 'Distance', 'Date', 'TrucDec', 'Espece'] and \
           dfExport.size == dfData.size, 'Error: test_buildExportTable: issue with resulting exported DataFrame \
           (size, columns list or columns order). With option "withExtraFields=True"'

    # test all decimal fields (those defined by MCDSEngine and by SampleDataSet) where changed to string type,
    # with '' instead of 'NaN'
    for col in ['Effort', 'Distance', 'TrucDec']:
        assert dfExport[col].compare(dfData[col].apply(lambda x: '' if np.isnan(x) else str(x))).empty, 'Error: \
       test_buildExportTable: issue with decimal fields: values of exported DataFrame should have same values than \
       source, but as string type (and NaN changed to \'\')'

    logger.info0('PASS => MCDSEngine => methods "buildExportTable", "matchDataFields" and "safeFloat2Str"')


def test_buildDataFile(shortDF_fxt):

    eng = ads.MCDSEngine(workDir=tmpDir / 'mcds-out')
    runDir = eng.setupRunFolder(runPrefix='uni')
    dfData = shortDF_fxt  # load source test data

    # export data to file
    dataFileName = eng.buildDataFile(runDir, ads.SampleDataSet(dfData))
    # gather exported data, with columns indexed
    dfFiled = pd.read_csv(dataFileName, sep='\t', header=None)
    dfFiled.set_axis(['Region', 'Surface', 'Point', 'Effort', 'Distance'], axis=1, inplace=True)
    # prepare dfData for comparison - change type of 'Surface' to integer (due to data type as read by pd.read_csv)
    dfData.Surface = dfData.Surface.apply(int)
    # test exported data match to source
    assert dfFiled.compare(dfData[['Region', 'Surface', 'Point', 'Effort', 'Distance']]).empty, 'Error: \
    test_buildDataFile: data exported to file do not match to source data'

    # clean-up: 'mcds-out' directory and content deleted
    shutil.rmtree(tmpDir / 'mcds-out')

    logger.info0('PASS => MCDSEngine => methods "buildDataFile"')


def test_buildCmdFile():

    # values selected for the test (t_values)
    t_estimKeyFn = 'HNORMAL'
    t_estimAdjustFn = 'COSINE'
    t_estimCriterion = 'AIC'
    t_cvInterval = 95
    eng = ads.MCDSEngine(workDir=tmpDir / 'mcds-out')
    runDir = eng.setupRunFolder(runPrefix='uni')
    cmdFileName = eng.buildCmdFile(estimKeyFn=t_estimKeyFn, estimAdjustFn=t_estimAdjustFn,
                                   estimCriterion=t_estimCriterion, cvInterval=t_cvInterval, runDir=runDir)
    # read cmd.txt file
    with open(cmdFileName, 'r') as cmdFile:
        lines = cmdFile.readlines()

    # check recorded param match with selected param in buildCmdFile method
    # TODO: default parameter may ne checked
    for line in lines:
        if re.match('^Estimator', line):
            val = re.search('/Key=(.+) /Adjust=(.+) /Criterion=(.+);$', line)
            assert val.group(1) == t_estimKeyFn and val.group(2) == t_estimAdjustFn and \
                   val.group(3) == t_estimCriterion, 'Error: test_buildCmdFile: issue with parameter \
                   recorded in cmd.txt (estimKeyFn, estimAdjustFn and estimCriterion)'
        elif re.match('^Confidence', line):
            val = re.search('=(.+)(;)$', line)
            assert val.group(1) == str(t_cvInterval), 'Error: test_buildCmdFile: issue with parameter \
                   recorded in cmd.txt (cvInterval)'

    # clean-up: 'mcds-out' directory and content deleted
    shutil.rmtree(tmpDir / 'mcds-out')

    logger.info0('PASS => MCDSEngine => methods "test_buildCmdFile"')


def test__run(shortDF_fxt):

    # init MCDSEngine
    eng = ads.MCDSEngine(workDir=tmpDir / 'mcds-out', runMethod='os.system')
    # Prepare temporary working folder
    runDir = eng.setupRunFolder(runPrefix='uni')
    # Prepare SampleDataSet and data.txt file
    sds = ads.SampleDataSet(source=shortDF_fxt, decimalFields=['Effort', 'Distance', 'TrucDec'])
    eng.buildDataFile(sampleDataSet=sds, runDir=runDir)
    # Prepare cmd.txt file
    cmdFileName = eng.buildCmdFile(estimKeyFn='HNORMAL', estimAdjustFn='COSINE', estimCriterion='AIC', cvInterval=95,
                                   runDir=runDir)

    # Debug mode - os.system method
    runStatus, startTime, elapsedTime = eng._run(eng.ExeFilePathName, cmdFileName, forReal=False, method=eng.runMethod)
    # test appropriate outputs
    assert runStatus == 0 and startTime is pd.NaT and elapsedTime == 0, 'Error: test__run: issue occured with \
    debug mode (forReal=False ; runMethod=os.system)'

    # Debug mode - subprocess.run method
    runStatus, startTime, elapsedTime = eng._run(eng.ExeFilePathName, cmdFileName, forReal=False,
                                                 method='subprocess.run')
    # test appropriate outputs
    assert runStatus == 0 and startTime is pd.NaT and elapsedTime == 0, 'Error: test__run: issue occured with \
    debug mode (forReal=False ; runMethod=subprocess.run)'

    # JUST TESTING No Exceptions raised (no specific tests)
    # run with Warning Status (2) as for JPM test in Nootebook "unintests.ipynb"
    # Real mode -  os.system method
    runStatus, startTime, engElapsedTime = eng._run(eng.ExeFilePathName, cmdFileName, forReal=True,
                                                    method=eng.runMethod)

    # Real mode -  subprocess.run method
    runStatus, startTime, engElapsedTime = eng._run(eng.ExeFilePathName, cmdFileName, forReal=True,
                                                    method='subprocess.run')

    # Timeout
    runStatus, startTime, engElapsedTime = \
        eng._run(eng.ExeFilePathName, cmdFileName, forReal=True, method='subprocess.run', timeOut=0.01)
    assert runStatus == 555, 'Error: runStatus should be 555 (refer MCDSEngine doc)'

    # Measure of performances (low level analysis execution)
    # BE CAREFULL: time.process_time() uses relative time for comparison only of codes among the same environment
    # NOT A REAL TIME reference
    timePerf = pd.DataFrame(columns=['OSS', 'SPR'], index=list('Cycle' + str(i) for i in range(1, 11)))

    i = 0
    while i < 10:
        j = 0
        start = time.perf_counter()
        while j < 5:
            runStatus, startTime, engElapsedTime = eng._run(eng.ExeFilePathName, cmdFileName, forReal=True,
                                                            method=eng.runMethod)
            j += 1
        end = time.perf_counter()

        timePerf.iloc[i, 0] = end - start
        i += 1

    i = 0
    while i < 10:
        j = 0
        start = time.perf_counter()
        while j < 5:
            runStatus, startTime, engElapsedTime = eng._run(eng.ExeFilePathName, cmdFileName, forReal=True,
                                                            method='subprocess.run')
            j += 1
        end = time.perf_counter()

        timePerf.iloc[i, 1] = end - start
        i += 1

    timePerf['OSS-faster'] = timePerf['OSS'] < timePerf['SPR']
    timePerf['%_vs_OSS'] = ((timePerf['SPR'] - timePerf['OSS']) / timePerf['OSS']) * 100

    logger.info0('\n\nPerformance: 10 loops of 5 runs (OSS = "os.system" ; SPR = "subprocess.run")\n')
    logger.info0(f'\n{timePerf.to_markdown(floatfmt=".2f")}\n')
    logger.info0(f'For "os.system": Mean +/- Std Dev = {timePerf["OSS"].mean():.2f} +/- {timePerf["OSS"].std():.2f}')
    logger.info0(
        f'For "subprocess.run": Mean +/- Std Dev = {timePerf["SPR"].mean():.2f} +/- {timePerf["SPR"].std():.2f}\n\n')

    logger.info0('PASS => MCDSEngine => methods "_run", "_runThroughOSSystem" and "_runThroughSubProcessRun"')


###############################################################################
#                         Actions to be done after all tests                  #
###############################################################################
def test_final():

    logger.info('Done testing pyaudisam.engine.')


if __name__ == '__main__':

    run = True
    # Run auto-tests (exit(0) if OK, 1 if not).
    rc = -1

    if run:
        try:
            test_init()

            test_executableNotFound()
            test_MCDS_Ctor()
            test_MCDS_setupRunFolder()
            test_buildExportTable(shortDF())
            test_buildDataFile(shortDF())
            test_buildCmdFile()
            test__run(shortDF())

            # Success !
            rc = 0

        except Exception as exc:
            logger.exception('Exception: ' + str(exc))
            rc = 1

    logger.info('Done unit integration testing ads: {} (code: {})'
                .format({-1: 'Not run', 0: 'Success'}.get(rc, 'Error'), rc))
    sys.exit(rc)
