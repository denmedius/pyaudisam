# coding: utf-8
# PyAuDiSam: Automation of Distance Sampling analyses with Distance software (http://distancesampling.org/)

# Copyright (C) 2021 Jean-Philippe Meuret

# This program is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see https://www.gnu.org/licenses/.

# Automated unit and integration tests for "analysis" submodule

# To run : simply run "pytest" or "python <this file>" in current folder
#          and check standard output ; and ./tmp/unt-ars.{datetime}.log for details

import sys
import time

import pandas as pd
import pytest

import pyaudisam as ads

import unintval_utils as uivu


# Setup local logger.
logger = uivu.setupLogger('unt.ans', level=ads.DEBUG,
                          otherLoggers={'ads.eng': ads.INFO2, 'ads.dat': ads.INFO, 'ads.ans': ads.INFO2})

what2Test = 'analysis'


###############################################################################
#                         Actions to be done before any test                  #
###############################################################################
def testBegin():
    uivu.logBegin(what=what2Test)


###############################################################################
#                                Test Cases                                   #
###############################################################################

# 4. MCDSAnalysis class

KRunCheckErrMsg = {ads.MCDSEngine.RCOK: 'Oh, oh, should have run smoothly and successfully !',
                   ads.MCDSEngine.RCWarnings: 'Oh, oh, should have run smoothly (even if with warnings) !',
                   ads.MCDSEngine.RCTimedOut: 'Oh, oh, should have timed-out !'}


def checkAnalysisRun(sds, name=None, estimKeyFn='UNIFORM', estimAdjustFn='POLY',
                     estimCriterion='AIC', cvInterval=95,
                     minDist=None, maxDist=None, fitDistCuts=None, discrDistCuts=None,
                     runMethod='os.system', timeOut=None, expectStatus=ads.MCDSEngine.RCOK):

    # Need for a parallel executor for time limit checking with os.system run method.
    exor = None if runMethod != 'os.system' or timeOut is None else ads.Executor(threads=1)

    # Engine
    eng = ads.MCDSEngine(executor=exor, workDir=uivu.pTmpDir / 'mcds-out',
                         runMethod=runMethod, timeOut=timeOut)

    # Analysis
    anlys = ads.MCDSAnalysis(engine=eng, sampleDataSet=sds, name=name, logData=True,
                             estimKeyFn=estimKeyFn, estimAdjustFn=estimAdjustFn,
                             estimCriterion=estimCriterion, cvInterval=cvInterval,
                             minDist=minDist, maxDist=maxDist, fitDistCuts=fitDistCuts, discrDistCuts=discrDistCuts)
    # Run
    anlys.submit()

    # Get result
    sResult = anlys.getResults()
    logger.info('Results:\n' + sResult.to_string())

    # Check status
    sts = sResult[('run output', 'run status', 'Value')]
    assert sts == expectStatus, KRunCheckErrMsg.get(expectStatus, 'Oh, oh, unexpected expected status ;-)')

    # Check some results (no real validation: only check that the analysis & MCDS integrate correctly)
    if ads.MCDSEngine.wasRun(sts) and not ads.MCDSEngine.errors(sts):
        assert sResult[('sample stats', 'total number of observations', 'Value')] == sds.dfData.NOMBRE.sum()
        assert sResult[('sample stats', 'minimal observation distance', 'Value')] == sds.dfData.DISTANCE.min()
        assert sResult[('sample stats', 'maximal observation distance', 'Value')] == sds.dfData.DISTANCE.max()
        assert sResult[('parameters', 'estimator key function', 'Value')] == estimKeyFn
        assert sResult[('parameters', 'estimator adjustment series', 'Value')] == estimAdjustFn
        assert sResult[('parameters', 'estimator selection criterion', 'Value')] == estimCriterion
        assert sResult[('parameters', 'CV interval', 'Value')] == cvInterval
        assert sResult[('parameters', 'left truncation distance', 'Value')] == minDist
        assert sResult[('parameters', 'right truncation distance', 'Value')] == maxDist
        assert sResult[('parameters', 'model fitting distance cut points', 'Value')] == fitDistCuts
        assert sResult[('parameters', 'distance discretisation cut points', 'Value')] == discrDistCuts
        nExpObs = sds.dfData.loc[(sds.dfData.DISTANCE >= (minDist or 0))
                                 & (sds.dfData.DISTANCE <= (maxDist or sds.dfData.DISTANCE.max())), 'NOMBRE'].sum()
        assert sResult[('encounter rate', 'number of observations (n)', 'Value')] == nExpObs
        assert sResult[('encounter rate', 'number of samples (k)', 'Value')] == sds.dfData.POINT.nunique()
        assert sResult[('encounter rate', 'effort (L or K or T)', 'Value')] \
               == sds.dfData[['POINT', 'EFFORT']].drop_duplicates(subset=['POINT']).EFFORT.sum()
        assert sResult[('encounter rate', 'left truncation distance', 'Value')] == (minDist or 0)
        assert sResult[('encounter rate', 'right truncation distance (w)', 'Value')] \
               == (maxDist or sds.dfData.DISTANCE.max())

    # Done
    eng.shutdown()
    if exor:
        exor.shutdown()

    return sResult


# ### a. Dataset to work with : a real life (reduced) one
def sampleDataSet():
    sds = ads.SampleDataSet(source=uivu.pRefInDir / 'ACDC2019-Papyrus-ALAARV-saisie-ttes-cols.xlsx',
                            decimalFields=['EFFORT', 'DISTANCE', 'NOMBRE'])
    logger.info('SampleDataSet:\n' + sds.dfData.to_string(min_rows=20, max_rows=20))
    logger.info(f'Some stats: size={len(sds)}, individuals={sds.dfData.NOMBRE.sum()},'
                f' minDist={sds.dfData.DISTANCE.min()}, maxDist={sds.dfData.DISTANCE.max()}')
    logger.info(f'Other stats: points={sds.dfData.POINT.nunique()},'
                f" totalEffort={sds.dfData[['POINT', 'EFFORT']].drop_duplicates(subset=['POINT']).EFFORT.sum()}")
    return sds


@pytest.fixture
def sampleDataSet_fxt():
    return sampleDataSet()


def testMcdsAnlysCtorSubmitGetResults(sampleDataSet_fxt):

    sds = sampleDataSet_fxt

    # ### b. Engine 'os.system' RunMethod and run time limit management
    # No time limit
    sResult = checkAnalysisRun(sds, estimKeyFn='UNIFORM', estimAdjustFn='COSINE', estimCriterion='AIC', cvInterval=95,
                               minDist=None, maxDist=None, fitDistCuts=None, discrDistCuts=None,
                               runMethod='os.system', timeOut=None, name='anlys', expectStatus=ads.MCDSEngine.RCWarnings)

    # Some time limit, but too long to stop analysis.
    sResult = checkAnalysisRun(sds, estimKeyFn='HAZARD', estimAdjustFn='POLY', estimCriterion='AIC', cvInterval=95,
                               minDist=50, maxDist=300, fitDistCuts=[60, 70, 80, 100, 120, 180, 250], discrDistCuts=None,
                               runMethod='os.system', timeOut=5, name=None, expectStatus=ads.MCDSEngine.RCOK)

    # Too short time limit => analysis time-out
    sResult = checkAnalysisRun(sds, estimKeyFn='UNIFORM', estimAdjustFn='POLY', estimCriterion='AIC', cvInterval=95,
                               minDist=None, maxDist=None, fitDistCuts=None, discrDistCuts=None,
                               runMethod='os.system', timeOut=0.01, name='anlys', expectStatus=ads.MCDSEngine.RCTimedOut)

    # ### c. Engine 'subprocess.run' RunMethod and run time limit management
    # No time limit
    sResult = checkAnalysisRun(sds, estimKeyFn='HAZARD', estimAdjustFn='COSINE', estimCriterion='AIC', cvInterval=95,
                               minDist=None, maxDist=200, fitDistCuts=None, discrDistCuts=12,
                               runMethod='os.system', timeOut=None, name=None, expectStatus=ads.MCDSEngine.RCWarnings)

    # Some time limit, but too long to stop analysis.
    sResult = checkAnalysisRun(sds, estimKeyFn='UNIFORM', estimAdjustFn='POLY', estimCriterion='AIC', cvInterval=95,
                               minDist=40, maxDist=250, fitDistCuts=7, discrDistCuts=[60, 80, 100, 120, 160, 200],
                               runMethod='os.system', timeOut=5, name='anlys', expectStatus=ads.MCDSEngine.RCWarnings)

    # Too short time limit => analysis time-out
    sResult = checkAnalysisRun(sds, estimKeyFn='UNIFORM', estimAdjustFn='POLY', estimCriterion='AIC', cvInterval=95,
                               minDist=None, maxDist=None, fitDistCuts=None, discrDistCuts=None,
                               runMethod='os.system', timeOut=0.01, name=None, expectStatus=ads.MCDSEngine.RCTimedOut)

    logger.info0('PASS testMcdsAnlysCtorSubmitGetResults')


def testMcdsAnlysPerformances(sampleDataSet_fxt):

    sds = sampleDataSet_fxt

    # ### d. Performance tests
    # BE CAREFUL: time.process_time() uses relative time for comparison only of codes among the same environment
    # NOT A REAL TIME reference
    dfTimePerf = pd.DataFrame(columns=['OSS', 'SPR'])

    # i. RunMethod='subprocess.run' (sequential synchronous executor)
    eng = ads.MCDSEngine(workDir=uivu.pTmpDir / 'mcds-out', runMethod='subprocess.run')

    # timeit', '-r 5 -n 10',
    # Core i5  8365U (4 HT cores, 1.6-4.1GHz, cache  6Mb, bus 4GT/s) + SSD 256Gb NVME + RAM 16Gb "Optimal performance power scheme"
    # * 2020-01-06: 347 ms ± 8.71 ms per loop (mean ± std. dev. of 5 runs, 10 loops each)
    # * 2021-10-02: 326 ms ± 2.71 ms per loop (mean ± std. dev. of 5 runs, 10 loops each)
    # Core i7 10850H (6 HT cores, 2.7-5.1GHz, cache 12Mb, bus 8GT/s) + SSD 512Gb NVME + RAM 32Gb "Optimal performance power scheme"
    # * 20213-11-02: 169 ms ± 1.94 ms per loop (mean ± std. dev. of 5 runs, 10 loops each)
    for cycle in range(10):
        start = time.perf_counter()
        for _ in range(5):
            _, _, _, _, _ = eng.submitAnalysis(sds, realRun=True, runPrefix='int',
                                               estimKeyFn='UNIFORM', estimAdjustFn='POLY',
                                               estimCriterion='AIC', cvInterval=95).result()
        end = time.perf_counter()
        dfTimePerf.at[cycle + 1, 'OSS'] = end - start

    eng.shutdown()

    # ii. RunMethod='os.system' (sequential synchronous executor)
    eng = ads.MCDSEngine(workDir=uivu.pTmpDir / 'mcds-out', runMethod='os.system')

    # timeit', '-r 5 -n 10',
    # Python 3.8 + Core i5  8365U (4 HT cores, 1.6-4.1GHz, cache  6Mb, bus 4GT/s) + SSD 256Gb NVME + RAM 16Gb "Optimal performance power scheme"
    # * 2020-01-06: 272 ms ± 7.57 ms per loop (mean ± std. dev. of 5 runs, 10 loops each)
    # * 2021-10-02: 268 ms ± 20.4 ms per loop (mean ± std. dev. of 5 runs, 10 loops each)
    # Python 3.8 + Core i7 10850H (6 HT cores, 2.7-5.1GHz, cache 12Mb, bus 8GT/s) + SSD 512Gb NVME + RAM 32Gb "Optimal performance power scheme"
    # * 20213-11-02: 171 ms ± 5.89 ms per loop (mean ± std. dev. of 5 runs, 10 loops each)

    for cycle in range(10):
        start = time.perf_counter()
        for _ in range(5):
            _, _, _, _, _ = eng.submitAnalysis(sds, realRun=True, runPrefix='int',
                                               estimKeyFn='UNIFORM', estimAdjustFn='POLY',
                                               estimCriterion='AIC', cvInterval=95).result()
        end = time.perf_counter()
        dfTimePerf.at[cycle + 1, 'SPR'] = end - start

    eng.shutdown()

    # iii. Report
    dfTimePerf.index.name = 'Cycle'
    dfTimePerf['OSS-faster'] = dfTimePerf.OSS < dfTimePerf.SPR
    dfTimePerf['%_vs_OSS'] = ((dfTimePerf.SPR - dfTimePerf.OSS) / dfTimePerf.OSS) * 100

    logger.info0('Performance: 10 cycles of 5 runs (OSS = "os.system" ; SPR = "subprocess.run")')
    logger.info0('\n' + dfTimePerf.to_string(float_format=lambda f: f'{f:.2f}'))
    logger.info0(f'For "os.system": Mean +/- Std Dev = {dfTimePerf.OSS.mean():.2f}s +/- {dfTimePerf.OSS.std():.2f}')
    logger.info0(
        f'For "subprocess.run": Mean +/- Std Dev = {dfTimePerf.SPR.mean():.2f}s +/- {dfTimePerf.SPR.std():.2f}')
    logger.info0(
        f"%_vs_OSS: Mean +/- Std Dev = {dfTimePerf['%_vs_OSS'].mean():.1f}% +/- {dfTimePerf['%_vs_OSS'].std():.1f}")

    logger.info0('PASS testMcdsAnlysPerformances')


# 4bis. MCDSPreAnalysis class : TODO
def testMcdsPreAnlysCtorSubmitGetResults(sampleDataSet_fxt):

    raise NotImplementedError('testMcdsPreAnlysCtorSubmitGetResults: TODO !')



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

            # Tests for MCDSAnalysis
            testMcdsAnlysCtorSubmitGetResults(sampleDataSet())
            testMcdsAnlysPerformances(sampleDataSet())

            # Tests for MCDSPreAnalysis
            testMcdsPreAnlysCtorSubmitGetResults(sampleDataSet())

            # Done.
            testEnd()

            # Success !
            rc = 0

        except Exception as exc:
            logger.exception(f'Exception: {exc}')
            rc = 1

    uivu.logEnd(what=what2Test, rc=rc)

    sys.exit(rc)
