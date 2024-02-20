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
#          and check standard output ; and tmp/unt-eng.{datetime}.log for details

import sys
import io
import re
import time
import concurrent.futures as cofu
import shutil
import pathlib as pl

import numpy as np
import pandas as pd

import pytest

import pyaudisam as ads

import unintval_utils as uivu


# Setup local logger.
logger = uivu.setupLogger('unt.eng', level=ads.DEBUG,
                          otherLoggers={'ads.dat': ads.INFO})

what2Test = 'engine'

###############################################################################
#                         Actions to be done before any test                  #
###############################################################################
def testBegin():
    uivu.logBegin(what=what2Test)


###############################################################################
#                Test function and related tooling functions                  #
###############################################################################
def testDsExecutableNotFound():

    with pytest.raises(Exception) as e_info:
        ads.MCDSEngine().findExecutable('WrongName')
    logger.info0(f'PASS (test_executableNotFound) => EXCEPTION RAISED AS AWAITED:\n{e_info.value}')


def testMcdsCtor():

    # test Exception raising with one of unsupported characters in workdir string (space) - first way
    with pytest.raises(Exception) as e_info:
        ads.MCDSEngine(workDir=uivu.pTmpDir.as_posix() + '/test out')  # Simple string path
    logger.info0(f'PASS (test_MCDS_Ctor) => EXCEPTION RAISED AS AWAITED:\n{e_info.value}')

    # test Exception raising with one of unsupported characters in workdir string (space) - second way
    with pytest.raises(Exception) as e_info:
        ads.MCDSEngine(workDir=uivu.pTmpDir / 'test out')  # pl.Path path
    logger.info0(f'PASS (test_MCDS_Ctor) => EXCEPTION RAISED AS AWAITED:\n{e_info.value}')

    # test preferred method to initiate MCDSEngine
    assert ads.MCDSEngine(workDir=uivu.pTmpDir / 'mcds-out', runMethod='os.system'), \
           'MCDS Engine: Non identified issue at engine initiation'

    # test previous (older) method to initiate MCDSEngine
    assert ads.MCDSEngine(workDir=uivu.pTmpDir / 'mcds-out'), 'MCDS Engine: Non identified issue at engine initiation'

    # test Specs DataFrames not empty
    # TODO: improve for deeper test ???
    assert not any([ads.MCDSEngine().statRowSpecs().empty, ads.MCDSEngine().statModSpecs().empty,
                    ads.MCDSEngine().statModCols().empty, ads.MCDSEngine().statModNotes().empty,
                    ads.MCDSEngine().statModColTrans().empty]), \
        'Specs DataFrames: issue occurred with initialization from external files'

    logger.info0('PASS => MCDSEngine => Constructor and methods "_run", "_runThroughOSSystem",'
                 ' "_runThroughSubProcessRun", "loadStatSpecs" and setters for related class variables')


def testDsSetupRunFolder():

    runDir = ads.MCDSEngine().setupRunFolder(runPrefix=' t,e.s:t; ( )_setupRunFolder')

    assert not re.search('[ ,.:;()/]', str(runDir)), \
        'Error: test_MCDS_setupRunFolder: Setup directory: unsupported chars should have been cleaned up'
    assert runDir.exists(), 'Error: test_MCDS_setupRunFolder: temporary directory not created'
    # clean-up: tmp directory deleted
    runDir.rmdir()

    logger.info0('PASS => MCDSEngine => method "setupRunFolder"')


# Generate a DataFrame (returned) suite bale for creating a short SampleDataSet
def dfShortSdsData():

    dfData = pd.DataFrame(columns=['Date', 'TrucDec', 'Espece', 'Point', 'Effort', 'Distance'],
                          data=[('2019-05-13', 3.5, 'TURMER', 23, 2, 83),
                                ('2019-05-15', np.nan, 'TURMER', 23, 2, 27.355),
                                ('2019-05-13', 0, 'ALAARV', 29, 2, 56.85),
                                ('2019-04-03', 1.325, 'PRUMOD', 53, 1.3, 7.2),
                                ('2019-06-01', 2, 'PHICOL', 12, 1, np.nan),
                                ('2019-06-19', np.nan, 'PHICOL', 17, 0.5, np.nan)])
    dfData['Region'] = 'ACDC'
    dfData['Surface'] = '2400'

    return dfData


# same as above => to allow using pytest
@pytest.fixture
def dfShortSdsData_fxt():
    return dfShortSdsData()


def testMcdsBuildExportTable(dfShortSdsData_fxt):

    eng = ads.MCDSEngine()
    dfData = dfShortSdsData_fxt  # load source test data
    dfData.drop(['Surface'], axis=1, inplace=True)  # to create missing field

    # test exception raised if at least one field missing vs DSEngine.ImportFieldAliasREs
    with pytest.raises(Exception) as e_info:
        _, _ = eng.buildExportTable(ads.SampleDataSet(dfData), withExtraFields=True, decPoint='.')
    logger.info0(f'PASS (test_buildExportTable) => EXCEPTION RAISED AS AWAITED:\n{e_info.value}')

    # missing field added
    dfData['Surface'] = '2400'

    dfExport, extraFields = ads.MCDSEngine().buildExportTable(ads.SampleDataSet(dfData),
                                                              withExtraFields=False, decPoint='.')

    # withExtraFields=False, number/order of columns same as for DSEngine.ImportFieldAliasREs, and
    # comparison of source and resulting DataFrames'size
    assert dfExport.columns.to_list() == ['Region', 'Surface', 'Point', 'Effort', 'Distance'] \
           and dfExport.size == dfData.loc[:, ['Region', 'Surface', 'Point', 'Effort', 'Distance']].size, \
        'Error: test_buildExportTable: issue with resulting exported DataFrame (size, columns list or columns order).' \
        ' With option "withExtraFields=False"'

    dfExport, extraFields = ads.MCDSEngine().buildExportTable(ads.SampleDataSet(dfData, decimalFields=['TrucDec']),
                                                              withExtraFields=True, decPoint='.')

    # withExtraFields=True, number/order of columns same as for DSEngine.ImportFieldAliasREs + extraFields columns, and
    # comparison of source and resulting DataFrames'size
    assert dfExport.columns.to_list() \
           == ['Region', 'Surface', 'Point', 'Effort', 'Distance', 'Date', 'TrucDec', 'Espece'] \
           and dfExport.size == dfData.size, \
        'Error: test_buildExportTable: issue with resulting exported DataFrame (size, columns list or columns order).' \
        'With option "withExtraFields=True"'

    # test all decimal fields (those defined by MCDSEngine and by SampleDataSet) where changed to string type,
    # with '' instead of 'NaN'
    for col in ['Effort', 'Distance', 'TrucDec']:
        assert dfExport[col].compare(dfData[col].apply(lambda x: '' if np.isnan(x) else str(x))).empty, \
            'Error: test_buildExportTable: issue with decimal fields: values of exported DataFrame' \
            "should have same values than source, but as string type (and NaN changed to '')"

    logger.info0('PASS => MCDSEngine => methods "buildExportTable", "matchDataFields" and "safeFloat2Str"')


def testMcdsBuildDataFile(dfShortSdsData_fxt):

    eng = ads.MCDSEngine(workDir=uivu.pTmpDir / 'mcds-out')
    runDir = eng.setupRunFolder(runPrefix='uni')
    dfData = dfShortSdsData_fxt  # load source test data

    # export data to file
    dataFileName = eng.buildDataFile(runDir, ads.SampleDataSet(dfData))
    # gather exported data, with columns indexed
    dfFiled = pd.read_csv(dataFileName, sep='\t', header=None)
    dfFiled.set_axis(['Region', 'Surface', 'Point', 'Effort', 'Distance'], axis=1, inplace=True)
    # prepare dfData for comparison - change type of 'Surface' to integer (due to data type as read by pd.read_csv)
    dfData.Surface = dfData.Surface.apply(int)
    # test exported data match to source
    assert dfFiled.compare(dfData[['Region', 'Surface', 'Point', 'Effort', 'Distance']]).empty, \
        'Error: test_buildDataFile: data exported to file do not match to source data'

    # clean-up: 'mcds-out' directory and content deleted
    shutil.rmtree(uivu.pTmpDir / 'mcds-out')

    logger.info0('PASS => MCDSEngine => methods "buildDataFile"')


def testMcdsBuildCmdFile():

    # values selected for the test (t_values)
    t_estimKeyFn = 'HNORMAL'
    t_estimAdjustFn = 'COSINE'
    t_estimCriterion = 'AIC'
    t_cvInterval = 95
    eng = ads.MCDSEngine(workDir=uivu.pTmpDir / 'mcds-out')
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
    shutil.rmtree(uivu.pTmpDir / 'mcds-out')

    logger.info0('PASS => MCDSEngine => methods "buildCmdFile"')


def testMcdsComputeSampleStats(dfShortSdsData_fxt):

    # init MCDSEngine
    eng = ads.MCDSEngine(workDir=uivu.pTmpDir / 'mcds-out', runMethod='os.system')
    # Prepare temporary working folder
    runDir = eng.setupRunFolder(runPrefix='uni')
    # Prepare SampleDataSet and data.txt file
    sds = ads.SampleDataSet(source=dfShortSdsData_fxt, decimalFields=['Effort', 'Distance', 'TrucDec'])
    eng.buildDataFile(sampleDataSet=sds, runDir=runDir)

    # Compute sample stats
    sSmpStats = eng.computeSampleStats(sds)

    # Check results
    assert (sSmpStats.index == eng.MIStatSampCols).all()
    assert (sSmpStats.values == [4, 7.2, 83.0]).all()

    logger.info0('PASS => MCDSEngine => methods "computeSampleStats"')


def testMcdsRun(dfShortSdsData_fxt):

    # init MCDSEngine
    eng = ads.MCDSEngine(workDir=uivu.pTmpDir / 'mcds-out', runMethod='os.system')
    # Prepare temporary working folder
    runDir = eng.setupRunFolder(runPrefix='uni')
    # Prepare SampleDataSet and data.txt file
    sds = ads.SampleDataSet(source=dfShortSdsData_fxt, decimalFields=['Effort', 'Distance', 'TrucDec'])
    eng.buildDataFile(sampleDataSet=sds, runDir=runDir)
    # Prepare cmd.txt file
    cmdFileName = eng.buildCmdFile(estimKeyFn='HNORMAL', estimAdjustFn='COSINE', estimCriterion='AIC', cvInterval=95,
                                   runDir=runDir)

    # Debug mode - os.system method
    runStatus, startTime, elapsedTime = eng._run(eng.ExeFilePathName, cmdFileName, forReal=False, method=eng.runMethod)
    # test appropriate outputs
    assert runStatus == 0 and startTime is pd.NaT and elapsedTime == 0, \
        'Error: test__run: issue occurred with debug mode (forReal=False ; runMethod=os.system)'

    # Debug mode - subprocess.run method
    runStatus, startTime, elapsedTime = eng._run(eng.ExeFilePathName, cmdFileName, forReal=False,
                                                 method='subprocess.run')
    # test appropriate outputs
    assert runStatus == 0 and startTime is pd.NaT and elapsedTime == 0, 'Error: test__run: issue occured with \
    debug mode (forReal=False ; runMethod=subprocess.run)'

    # JUST TESTING that no exception is raised (no specific tests)
    # run with Warning Status (2) as for JPM test in notebook "unintests.ipynb"
    # Real mode -  os.system method
    _, _, _ = eng._run(eng.ExeFilePathName, cmdFileName, forReal=True, method=eng.runMethod)

    # Real mode -  subprocess.run method
    _, _, _ = eng._run(eng.ExeFilePathName, cmdFileName, forReal=True, method='subprocess.run')

    # Timeout
    runStatus, startTime, engElapsedTime = \
        eng._run(eng.ExeFilePathName, cmdFileName, forReal=True, method='subprocess.run', timeOut=0.01)
    assert runStatus == 555, 'Error: runStatus should be 555 (refer MCDSEngine doc)'

    # Measure of performances (low level analysis execution)
    # BE CAREFUL: time.process_time() uses relative time for comparison only of codes among the same environment
    # NOT A REAL TIME reference
    timePerf = pd.DataFrame(columns=['OSS', 'SPR'], index=list('Cycle' + str(i) for i in range(1, 11)))

    i = 0
    while i < 10:
        j = 0
        start = time.perf_counter()
        while j < 5:
            _, _, _ = eng._run(eng.ExeFilePathName, cmdFileName, forReal=True, method=eng.runMethod)
            j += 1
        end = time.perf_counter()

        timePerf.iloc[i, 0] = end - start
        i += 1

    i = 0
    while i < 10:
        j = 0
        start = time.perf_counter()
        while j < 5:
            _, _, _ = eng._run(eng.ExeFilePathName, cmdFileName, forReal=True, method='subprocess.run')
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


#   Generate a reduced real-life SampleDataSet
def sdsRealReduced():
    return ads.SampleDataSet(source=uivu.pRefInDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.xlsx',
                             decimalFields=['EFFORT', 'DISTANCE', 'NOMBRE'])


# same as above => to allow using pytest
@pytest.fixture
def sdsRealReduced_fxt():
    return sdsRealReduced()


# ### f. High level analysis execution  (via executor), debug mode
# (generate cmd and data input files, but no call to executable)
def testMcdsSubmitAnalysisDebug(sdsRealReduced_fxt):

    sds = sdsRealReduced_fxt

    # init MCDSEngine
    eng = ads.MCDSEngine(workDir=uivu.pTmpDir / 'mcds-out')

    # Prepare temporary working folder
    runDir = eng.setupRunFolder(runPrefix='uni')
    logger.info0(f'Debug run {runDir=}')

    # Asynchronous model, even if no parallelism involved : submitAnalysis() returns a "future" object
    # (see module concurrent)
    futRun = eng.submitAnalysis(sds, realRun=False, runPrefix='int',
                                estimKeyFn='UNIFORM', estimAdjustFn='POLY',
                                estimCriterion='AIC', cvInterval=95)

    # Get run output from future object
    runCode, startTime, elapsedTime, runDir, sResults = futRun.result()
    assert runCode == ads.MCDSEngine.RCNotRun, 'Should have NOT run (run code = 0)'
    logger.info('Debug run:\n' + str(dict(runCode=runCode, runDir=runDir,
                                          startTime=startTime, elapsedTime=elapsedTime, sResults=sResults)))

    runCode, startTime, elapsedTime, runDir, sResults = \
        eng.submitAnalysis(sds, realRun=False, runPrefix='int',
                           estimKeyFn='UNIFORM', estimAdjustFn='POLY',
                           estimCriterion='AIC', cvInterval=95).result()
    assert runCode == ads.MCDSEngine.RCNotRun, 'Should have NOT run (run code = 0)'
    logger.info('Debug run:\n' + str(dict(runCode=runCode, runDir=runDir,
                                          startTime=startTime, elapsedTime=elapsedTime, sResults=sResults)))


# ### g. High level analysis execution  (via executor), real mode
KRunCheckErrMsg = {ads.MCDSEngine.RCOK: 'Oh, oh, should have run smoothly and successfully !',
                   ads.MCDSEngine.RCWarnings: 'Oh, oh, should have run smoothly (even if with warnings) !',
                   ads.MCDSEngine.RCTimedOut: 'Oh, oh, should have timed-out !'}


def checkEngineAnalysisRun(sampleDataSet, estimKeyFn='UNIFORM', estimAdjustFn='POLY', estimCriterion='AIC',
                           cvInterval=95,
                           minDist=None, maxDist=None, fitDistCuts=None, discrDistCuts=None,
                           runMethod='os.system', timeOut=None, expectRunCode=ads.MCDSEngine.RCOK):
    # Need for an async. executor for time limit checking with os.system run method.
    exor = None if runMethod != 'os.system' or timeOut is None else ads.Executor(threads=1)

    # Engine
    eng = ads.MCDSEngine(executor=exor, workDir=uivu.pTmpDir / 'mcds-out',
                         runMethod=runMethod, timeOut=timeOut)

    # Run analysis and get results
    fut = eng.submitAnalysis(sampleDataSet, realRun=True, runPrefix='int',
                             estimKeyFn=estimKeyFn, estimAdjustFn=estimAdjustFn,
                             estimCriterion=estimCriterion, cvInterval=cvInterval,
                             minDist=minDist, maxDist=maxDist,
                             fitDistCuts=fitDistCuts, discrDistCuts=discrDistCuts)

    try:
        if timeOut is not None:
            startTime = pd.Timestamp.now()  # In case of cofu.TimeoutError
        runCode, startTime, elapsedTime, runDir, sResults = fut.result(timeout=timeOut)
    except cofu.TimeoutError:
        logger.info('MCDS Analysis run timed-out after {}s'.format(timeOut))
        runCode, startTime, elapsedTime, runDir, sResults = eng.RCTimedOut, startTime, timeOut, None, None

    # Check status
    assert runCode == expectRunCode, KRunCheckErrMsg.get(expectRunCode, 'Oh, oh, unexpected expected run code ;-)')

    # Done
    eng.shutdown()
    if exor:
        exor.shutdown()

    logger.info('Real run:\n' + str(dict(runCode=runCode, runDir=runDir,
                                         startTime=startTime, elapsedTime=elapsedTime, sResults=sResults)))

    return runCode, startTime, elapsedTime, runDir, sResults


def testMcdsSubmitAnalysisReal(sdsRealReduced_fxt):

    sds = sdsRealReduced_fxt

    # init MCDSEngine
    eng = ads.MCDSEngine(workDir=uivu.pTmpDir / 'mcds-out')

    # Prepare temporary working folder
    runDir = eng.setupRunFolder(runPrefix='uni')
    logger.info0(f'Real run: {runDir=}')

    # No time limit
    checkEngineAnalysisRun(sds, estimKeyFn='NEXPON', estimAdjustFn='COSINE', estimCriterion='AIC', cvInterval=95,
                           runMethod='os.system', timeOut=None, expectRunCode=ads.MCDSEngine.RCWarnings)

    # Some time limit, but too long to stop analysis.
    checkEngineAnalysisRun(sds, estimKeyFn='HNORMAL', estimAdjustFn='COSINE', estimCriterion='AIC', cvInterval=95,
                           runMethod='os.system', timeOut=3, expectRunCode=ads.MCDSEngine.RCWarnings)

    # Too short time limit => analysis time-out (but MCDS goes on to its end : no kill done by executor)
    checkEngineAnalysisRun(sds, estimKeyFn='UNIFORM', estimAdjustFn='POLY', estimCriterion='AIC', cvInterval=95,
                           runMethod='os.system', timeOut=0.1, expectRunCode=ads.MCDSEngine.RCTimedOut)

    logger.info('Look: MCDS was not killed, it has gone to its end, whereas the analysis has timed-out')

    # No time limit
    checkEngineAnalysisRun(sds, estimKeyFn='NEXPON', estimAdjustFn='COSINE', estimCriterion='AIC', cvInterval=95,
                           runMethod='subprocess.run', timeOut=None, expectRunCode=ads.MCDSEngine.RCWarnings)

    # Some time limit, but too long to stop analysis.
    checkEngineAnalysisRun(sds, estimKeyFn='HNORMAL', estimAdjustFn='POLY', estimCriterion='AIC', cvInterval=95,
                           runMethod='subprocess.run', timeOut=3, expectRunCode=ads.MCDSEngine.RCErrors)

    # Too short time limit => analysis time-out (but MCDS goes on to its end : no kill done by executor)
    checkEngineAnalysisRun(sds, estimKeyFn='UNIFORM', estimAdjustFn='POLY', estimCriterion='AIC', cvInterval=95,
                           runMethod='subprocess.run', timeOut=0.05, expectRunCode=ads.MCDSEngine.RCTimedOut)

    logger.info('Look: MCDS was actually killed on time-out')


# h. Generate input data files for interactive Distance software
#   ('point transect' mode only as for now)
def testMcdsBuildDistanceDataFile(sdsRealReduced_fxt, dfShortSdsData_fxt):

    sds = sdsRealReduced_fxt

    # init MCDSEngine
    eng = ads.MCDSEngine(workDir=uivu.pTmpDir / 'mcds-out')

    # Prepare target folder
    tgtDir = pl.Path(eng.workDir, 'distance-in')
    tgtDir.mkdir(exist_ok=True)

    # Case 1: Point transect with radial distance, no extra fields, no clustering.
    distDataFileName = \
        eng.buildDistanceDataFile(sds, tgtFilePathName=tgtDir / 'import-data-noextra.txt')
    assert distDataFileName.as_posix() == (tgtDir / 'import-data-noextra.txt').as_posix()
    refDistDataFileName = uivu.pRefOutDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols-import-data-noextra.txt'
    with io.open(distDataFileName) as tst_f, io.open(refDistDataFileName) as ref_f:
        assert list(tst_f) == list(ref_f)

    # Point transect with radial distance, with extra fields, no clustering.
    distDataFileName = \
        eng.buildDistanceDataFile(sds, tgtFilePathName=tgtDir / 'import-data-withextra.txt',
                                  withExtraFields=True)
    assert distDataFileName.as_posix() == (tgtDir / 'import-data-withextra.txt').as_posix()
    refDistDataFileName = uivu.pRefOutDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols-import-data-withextra.txt'
    with io.open(distDataFileName) as tst_f, io.open(refDistDataFileName) as ref_f:
        assert list(tst_f) == list(ref_f)

    eng.shutdown()

    # Case 2: Point transect with radial distance, no extra fields, with clustering.
    eng = ads.MCDSEngine(workDir=uivu.pTmpDir / 'mcds-out', clustering=True)
    # Add cluster data to the data set
    dfData = dfShortSdsData_fxt
    dfData['Nombre'] = [1, 2, 1, 1, 2, 3]
    sds = ads.SampleDataSet(source=dfData, decimalFields=['Effort', 'Distance', 'TrucDec'])
    # Generate distance file
    tgtDir = pl.Path(eng.workDir, 'distance-in')
    tgtDir.mkdir(exist_ok=True)
    distDataFileName = \
        eng.buildDistanceDataFile(sds, tgtFilePathName=tgtDir / 'import-data-clusters.txt')
    refDistDataFileName = uivu.pRefOutDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols-import-data-clusters.txt'
    with io.open(distDataFileName) as tst_f, io.open(refDistDataFileName) as ref_f:
        assert list(tst_f) == list(ref_f)

    eng.shutdown()


# i. TODO:  Test lower level code
# * loadDataFile
# * buildExportTable
# * decodeStats
# * decodeLog
# * decodePlots
# * decodeOutput


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
            testBegin()

            testDsExecutableNotFound()
            testMcdsCtor()
            testDsSetupRunFolder()
            testMcdsBuildExportTable(dfShortSdsData())
            testMcdsBuildDataFile(dfShortSdsData())
            testMcdsBuildCmdFile()
            testMcdsComputeSampleStats(dfShortSdsData())
            testMcdsRun(dfShortSdsData())

            testMcdsSubmitAnalysisDebug(sdsRealReduced())
            testMcdsSubmitAnalysisReal(sdsRealReduced())
            testMcdsBuildDistanceDataFile(sdsRealReduced(), dfShortSdsData())

            testEnd()

            # Success !
            rc = 0

        except Exception as exc:
            logger.exception(f'Exception: {exc}')
            rc = 1

    uivu.logEnd(what=what2Test, rc=rc)

    sys.exit(rc)
