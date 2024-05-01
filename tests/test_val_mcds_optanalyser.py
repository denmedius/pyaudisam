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

# Automated validation tests for "optanalyser" submodule: MCDSAnalyser class,
# and "reports" submodule: MCDSResultsFilterSortReport class and bases (adapted from valtest.ipynb notebook)

# To run : simply run "pytest" and see ./tmp/val.onr.{datetime}.log for detailed test log

import time
import pathlib as pl
import shutil

import numpy as np
import pandas as pd

import pytest

import pyaudisam as ads

import unintval_utils as uivu


# Setup local logger.
logger = uivu.setupLogger('val.onr', level=ads.DEBUG)


class TestMcdsTruncOptAnalyser:

    # Set to False to skip final cleanup (useful for debugging)
    KFinalCleanup = False

    # Class and test function initializers / finalizers ###########################
    @pytest.fixture(autouse=True, scope='class')
    def _inifinalizeClass(self):

        KWhat2Test = 'truncation opt-analyser'

        uivu.logBegin(what=KWhat2Test)

        # Set up a clear ground before starting
        uivu.setupWorkDir('val-oanlr', cleanup=self.KFinalCleanup)

        # The code before yield is run before the first test function in this class
        yield
        # The code after yield is run after the last test function in this class

        # Let the ground clear after passing there
        if self.KFinalCleanup:
            uivu.cleanupWorkDir()
            (uivu.pTmpDir / 'ACDC2019-Naturalist-extrait-SpecsAnalyses.xlsx').unlink()

        uivu.logEnd(what=KWhat2Test)

    @pytest.fixture(autouse=True, scope='function')
    def _inifinalizeFunction(self):

        # The code before yield is run before every test function
        yield
        # The code after yield is run after every test function

    # Test functions #############################################################

    # IV. Run truncation opt-analyses with same real life field data
    @pytest.fixture()
    def inputDataSet_fxt(self):

        logger.info(f'Preparing individual. sightings ...')

        # ## 1. Individuals data set
        dfObsIndiv = ads.DataSet(uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-ObsIndiv.ods',
                                 sheet='DonnéesIndiv').dfData

        logger.info(f'Invidual. sightings: n={len(dfObsIndiv)} =>\n'
                    + dfObsIndiv.to_string(min_rows=30, max_rows=30))
        indObsDesc = {col: dfObsIndiv[col].unique()
                      for col in ['Observateur', 'Point', 'Passage', 'Adulte', 'Durée', 'Espèce']}
        logger.info(f'... {indObsDesc}')

        # ## 2. Actual transects
        # (can't deduce them from data, some points are missing because of data selection)
        dfTransects = ads.DataSet(uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-ObsIndiv.ods',
                                  sheet='Inventaires').dfData

        logger.info(f'Individual. sightings: n={len(dfObsIndiv)} =>\n' + dfTransects.to_string(min_rows=30, max_rows=30))

        logger.info(f'Done preparing individual. sightings.\n')

        return dfObsIndiv, dfTransects

    @pytest.fixture()
    def optAnalyser_fxt(self, inputDataSet_fxt):

        logger.info(f'Preparing analysis specs ...')

        # 0. Data description
        # 0.a. Source / Results data
        transectPlaceCols = ['Point']
        passIdCol = 'Passage'
        effortCol = 'Effort'

        sampleDistCol = 'Distance'
        sampleDecCols = [effortCol, sampleDistCol]

        sampleNumCol = 'NumEchant'
        sampleSelCols = ['Espèce', passIdCol, 'Adulte', 'Durée']

        dSurveyArea = dict(Zone='ACDC', Surface='2400')

        # 0.b. General DS analysis parameters
        varIndCol = 'NumAnlys'
        anlysAbbrevCol = 'AbrevAnlys'
        anlysParamCols = ['FonctionClé', 'SérieAjust', 'TrGche', 'TrDrte', 'NbTrchMod']

        distanceUnit = 'Meter'
        areaUnit = 'Hectare'
        surveyType = 'Point'
        distanceType = 'Radial'
        clustering = False

        # 0.c. Default optimisation parameters.
        defEstimKeyFn = 'UNIFORM'
        defEstimAdjustFn = 'POLY'
        defEstimCriterion = 'AIC'
        defCVInterval = 95

        defExpr2Optimise = 'balq3'
        defMinimiseExpr = False
        defOutliersMethod = 'tucquant'
        defOutliersQuantCutPct = 7
        defFitDistCutsFctr = ads.Interval(min=0.6, max=1.4)
        defDiscrDistCutsFctr = ads.Interval(min=0.5, max=1.2)

        defSubmitTimes = 1
        defSubmitOnlyBest = None

        defCoreEngine = 'zoopt'
        defCoreMaxIters = 100
        defCoreTermExprValue = None
        defCoreAlgorithm = 'racos'
        defCoreMaxRetries = 0

        dDefSubmitOtherParams = dict()

        # 0.d. Results post-computation parameters
        ldTruncIntrvSpecs = [dict(col='left', minDist=5.0, maxLen=5.0),
                             dict(col='right', minDist=25.0, maxLen=25.0)]
        truncIntrvEpsilon = 1e-6

        # 0.e. Les analyses à faire (avec specs d'optimisation dedans si nécessaire)
        optAnlysSpecFile = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-SpecsOptanalyses.xlsx'

        # 1. Individuals data set
        # 2. Actual transects
        # See inputDataSet_fxt

        # ## 3A. Really run opt-analyses
        # ### a. MCDSTruncationOptanalyser object
        logger.info(f'Preparing opt-analyser ...')

        dfObsIndiv, dfTransects = inputDataSet_fxt
        optanlr = \
            ads.MCDSTruncationOptanalyser(dfObsIndiv, dfTransects=dfTransects, dSurveyArea=dSurveyArea,
                                          transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                                          sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols,
                                          sampleDistCol=sampleDistCol,
                                          abbrevCol=anlysAbbrevCol, abbrevBuilder=uivu.analysisAbbrev,
                                          anlysIndCol=varIndCol, sampleIndCol=sampleNumCol,
                                          distanceUnit=distanceUnit, areaUnit=areaUnit,
                                          surveyType=surveyType, distanceType=distanceType, clustering=clustering,
                                          resultsHeadCols=dict(before=[varIndCol, sampleNumCol], sample=sampleSelCols,
                                                               after=anlysParamCols + [anlysAbbrevCol]),
                                          ldTruncIntrvSpecs=ldTruncIntrvSpecs, truncIntrvEpsilon=truncIntrvEpsilon,
                                          workDir=uivu.pWorkDir,
                                          logAnlysProgressEvery=5, logOptimProgressEvery=3, backupOptimEvery=5,
                                          defEstimKeyFn=defEstimKeyFn, defEstimAdjustFn=defEstimAdjustFn,
                                          defEstimCriterion=defEstimCriterion, defCVInterval=defCVInterval,
                                          defExpr2Optimise=defExpr2Optimise, defMinimiseExpr=defMinimiseExpr,
                                          defOutliersMethod=defOutliersMethod,
                                          defOutliersQuantCutPct=defOutliersQuantCutPct,
                                          defFitDistCutsFctr=defFitDistCutsFctr,
                                          defDiscrDistCutsFctr=defDiscrDistCutsFctr,
                                          defSubmitTimes=defSubmitTimes, defSubmitOnlyBest=defSubmitOnlyBest,
                                          dDefSubmitOtherParams=dDefSubmitOtherParams,
                                          dDefOptimCoreParams=dict(core=defCoreEngine, maxIters=defCoreMaxIters,
                                                                   termExprValue=defCoreTermExprValue,
                                                                   algorithm=defCoreAlgorithm,
                                                                   maxRetries=defCoreMaxRetries))
        logger.info(f'opt-analyser specs:\n{optanlr.specs}')

        assert optanlr.specs == dict(Zone='ACDC',
                                     Surface='2400',
                                     distanceUnit='Meter',
                                     areaUnit='Hectare',
                                     runMethod='subprocess.run',
                                     runTimeOut=300,
                                     surveyType='Point',
                                     distanceType='Radial',
                                     clustering=False,
                                     defEstimKeyFn='UNIFORM',
                                     defEstimAdjustFn='POLY',
                                     defEstimCriterion='AIC',
                                     defCVInterval=95,
                                     defMinDist=None,
                                     defMaxDist=None,
                                     defFitDistCuts=None,
                                     defDiscrDistCuts=None,
                                     defExpr2Optimise='balq3',
                                     defMinimiseExpr=False,
                                     dDefOptimCoreParams=dict(core='zoopt', maxIters=100, termExprValue=None,
                                                              algorithm='racos', maxRetries=0),
                                     defSubmitTimes=1,
                                     defSubmitOnlyBest=None,
                                     dDefSubmitOtherParams={},
                                     defOutliersMethod='tucquant',
                                     defOutliersQuantCutPct=7,
                                     defFitDistCutsFctr=[0.6, 1.4],
                                     defDiscrDistCutsFctr=[0.5, 1.2])

        logger.info(f'Done preparing opt-analyser.\n')

        # b. Check opt-analyses specs
        logger.info(f'Checking opt-analysis specs ...')

        dfOptAnlysSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols, verdict, reasons = \
            optanlr.explicitParamSpecs(implParamSpecs=optAnlysSpecFile, dropDupes=True, check=True)

        logger.info('Opt-analysis spec. explicitations:')
        logger.info(f'* verdict: {verdict}')
        logger.info(f'* reasons: {reasons}')
        logger.info(f'* userParamSpecCols: n={len(userParamSpecCols)} => {userParamSpecCols}')
        logger.info(f'* intParamSpecCols: n={len(intParamSpecCols)} => {intParamSpecCols}')
        logger.info(f'* unmUserParamSpecCols: n={len(unmUserParamSpecCols)} => {unmUserParamSpecCols}')

        logger.info(f'Explicitated opt-analysis specs: n={len(dfOptAnlysSpecs)} =>\n'
                    + dfOptAnlysSpecs.to_string(min_rows=30, max_rows=30))

        assert len(dfOptAnlysSpecs) == 60
        assert userParamSpecCols == ['FonctionClé', 'SérieAjust', 'TrGche', 'TrDrte', 'NbTrchMod', 'MultiOpt']
        assert intParamSpecCols == ['EstimKeyFn', 'EstimAdjustFn', 'MinDist', 'MaxDist', 'FitDistCuts', 'SubmitParams']
        assert unmUserParamSpecCols == []
        assert verdict
        assert not reasons

        # Done.
        logger.info(f'Done checking opt-analysis specs.\n')

        return optanlr, dfOptAnlysSpecs, optAnlysSpecFile

    @staticmethod
    def loadResults(anlysr, filePath, postComputed=False):

        logger.info(f'Loading results from {filePath.as_posix()} ...')

        rsRes = anlysr.setupResults()
        rsRes.fromFile(filePath, postComputed=postComputed)

        return rsRes

    @pytest.fixture()
    def refResults_fxt(self, optAnalyser_fxt):

        logger.info(f'Preparing reference results ...')

        # Prevent re-postComputation as this ref. file is old, with now missing computed cols
        anlysr, _, _ = optAnalyser_fxt
        rsRef = self.loadResults(anlysr, uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-OptResultats.ods',
                                 postComputed=True)

        logger.info(f'Reference results: n={len(rsRef)} =>\n' + rsRef.dfData.to_string(min_rows=30, max_rows=30))

        return rsRef

    @classmethod
    def compareUnoptResults(cls, rsRef, rsAct):
        """Prerequisite: Reference results generated with either MCDS 7.4 or 6.2"""

        optTruncFlagCol = ads.MCDSTruncationOptanalyser.OptimTruncFlagCol

        # a. Actual "unoptimised" analysis results ("all-results" sheet)
        logger.info('* non truncation-optimised analyses results:')
        rsRefUnopt = rsRef.copy()
        rsRefUnopt.dropRows(rsRef.dfData[('header (tail)', optTruncFlagCol, 'Value')] == 1)
        logger.info(f'  - found {len(rsRefUnopt)} reference unopt. result rows')
        rsActUnopt = rsAct.copy()
        rsActUnopt.dropRows(rsAct.dfData[('header (tail)', optTruncFlagCol, 'Value')] == 1)
        logger.info(f'  - found {len(rsActUnopt)} actual unopt. result rows')

        # b. Comparison index = analysis "Id": sample Id columns, analysis abbrev. and indexes,
        #     model and truncation parameters.
        indexCols = [col for col in rsAct.miCustomCols.to_list() if '(sample)' in col[0]] \
                    + [('parameters', 'estimator key function', 'Value'),
                       ('parameters', 'estimator adjustment series', 'Value'),
                       ('parameters', 'left truncation distance', 'Value'),
                       ('parameters', 'right truncation distance', 'Value'),
                       ('parameters', 'model fitting distance cut points', 'Value'),
                       ('header (tail)', 'AbrevAnlys', 'Value')]
        indexColsStr = '\n    . '.join(str(t) for t in indexCols)
        logger.info(f'  - index columns for comparison:\n    . ' + indexColsStr)
        customColsStr = '\n    . '.join(str(t) for t in rsAct.miCustomCols.to_list())
        logger.info(f'  - given rsAct.miCustomCols:\n    . ' + customColsStr)

        # c. Comparison columns: ignore ...
        #  - sample Id columns and analysis indexes (used as comparison index = analysis "Id"),
        #  - "Order" columns, difficult to compare, because a very small difference in one or more
        #    computations parameters is quickly amplified by the exponential formula,
        #    thus changing the row order by an amount of at least 1, which would imply another comparison method,
        #  - 'run output' chapter results (start time, elapsed time, run folder ... always different),
        #  - text columns (not supported by ResultsSet.compare),
        #  - a few computed columns.
        compCols = [col for col in rsAct.dfData.columns.to_list()
                    if col in rsRef.columns
                    and col not in (indexCols + [col for col in rsAct.columns.to_list() if col[2] == 'Order']
                                    + [col for col in rsAct.miCustomCols.to_list() if '(sample)' not in col[0]]
                                    + [('parameters', 'estimator selection criterion', 'Value'),
                                       ('run output', 'start time', 'Value'),
                                       ('run output', 'elapsed time', 'Value'),
                                       ('run output', 'run folder', 'Value'),
                                       ('detection probability', 'key function type', 'Value'),
                                       ('detection probability', 'adjustment series type', 'Value'),
                                       ('detection probability', 'Delta AIC', 'Value')])]

        # d. Compare
        dfDiff = rsRefUnopt.compare(rsActUnopt, indexCols=indexCols, subsetCols=compCols,
                                    noneIsNan=True, dropCloser=14, dropNans=True)

        logger.info(f'  - diff. to unopt. reference (relative): n={len(dfDiff)} =>\n'
                    + dfDiff.to_string(min_rows=30, max_rows=30))
        if not dfDiff.empty:
            dfDiff.to_excel(uivu.pWorkDir / 'unopt-res-comp-14.xlsx')

        assert dfDiff.empty, 'Oh oh ... some unexpected differences !'

        # e. To be perfectly honest ... there may be some 10**-14/-16 glitches (due to worksheet I/O ?) ... or not.
        dfComp = rsRefUnopt.compare(rsActUnopt, indexCols=indexCols, subsetCols=compCols,
                                    noneIsNan=True, dropNans=True)
        dfComp = dfComp[(dfComp != np.inf).all(axis='columns')]

        logger.info(f'  - diff. to unopt. reference (absolute): n={len(dfComp)} =>\n'
                    + dfComp.to_string(min_rows=30, max_rows=30))

        logger.info('  ... done with non truncation-optimised analyses results.')

    @classmethod
    def compareOptResults(cls, rsRef, rsAct):
        """Prerequisite: Reference results generated with either MCDS 7.4 or 6.2"""

        optTruncFlagCol = ads.MCDSTruncationOptanalyser.OptimTruncFlagCol

        # a. Actual "optimised" analysis results ("all-results" sheet)
        logger.info('* truncation-optimised analyses results:')
        rsRefOpt = rsRef.copy()
        rsRefOpt.dropRows(rsRef.dfData[('header (tail)', optTruncFlagCol, 'Value')] == 0)
        logger.info(f'  - found {len(rsRefOpt)} reference opt. result rows')
        rsActOpt = rsAct.copy()
        rsActOpt.dropRows(rsAct.dfData[('header (tail)', optTruncFlagCol, 'Value')] == 0)
        logger.info(f'  - found {len(rsActOpt)} actual opt. result rows')

        assert len(rsRefOpt) == len(rsActOpt)

        # b. Sort the results by optimisation parameter specs, and then by left truncation distance,
        #    the right one, and finally be fit distance cut point number
        miSortCols = [('header (tail)', 'AbrevAnlys', 'Value'),
                      ('parameters', 'left truncation distance', 'Value'),
                      ('parameters', 'right truncation distance', 'Value'),
                      ('parameters', 'model fitting distance cut points', 'Value')]
        rsRefOpt.sortRows(by=miSortCols)
        rsActOpt.sortRows(by=miSortCols)

        # b. Check that optimisation param. specs are the same as for the ref. results
        dfActOpt = rsActOpt.dfTransData('en')
        dfRefOpt = rsRefOpt.dfTransData('en')

        # Save to disk after "merging" ref and actual results, for visual checks
        dfRefOpt.insert(0, 'Source', 'ref')
        dfActOpt.insert(0, 'Source', 'act')
        dfMrgdOpt = pd.concat([dfRefOpt, dfActOpt], sort=False)
        dfMrgdOpt.sort_values(by=['NumAnlys', 'Source'], inplace=True)
        dfMrgdOpt.to_excel(uivu.pWorkDir / 'valtests-optanalyses-opt-results-raw-comp.xlsx')

        anlysAbrevCol = 'AbrevAnlys'
        logger.info(f'  - comparing optimisation param. specs ...')
        assert dfActOpt[anlysAbrevCol].to_list() == dfRefOpt[anlysAbrevCol].to_list()

        # c. Compare a small and simple subset of analyses results columns
        #    (not in search of equality, but rather under-control differences)
        anlysNumCol = 'NumAnlys'
        dfActOpt[anlysNumCol] = [i for i in range(len(dfActOpt))]
        dfRefOpt[anlysNumCol] = [i for i in range(len(dfRefOpt))]

        indexCols = [anlysNumCol, anlysAbrevCol]
        compCols = ['AIC', 'Qual Bal 3', 'PDetec', 'EDR/ESW', 'Density']
        dfDiff = ads.DataSet.compareDataFrames(dfActOpt, dfRefOpt, indexCols=indexCols, subsetCols=compCols,
                                               dropNans=True, noneIsNan=True)

        dfDiffStats = pd.DataFrame(data=[dfDiff.min(), dfDiff.max(), dfDiff.replace(np.inf, 16).mean()],
                                   index=['min', 'max', 'mean'])
        dfDiffStats.to_excel(uivu.pWorkDir / 'valtests-optanalyses-opt-results-diff-stats.xlsx')
        logger.info(f'  - stats on relative diff. between ref. and actual opt. results:\n'
                    + dfDiffStats.to_string())

        # Not too bad if less that 10% mean difference (100 / 10**1 = 10%) !
        minMeanLogRelDiff = dfDiffStats.loc['mean'].min()
        minMeanPctRelDiff = 100 / 10 ** minMeanLogRelDiff
        logger.info(f"  - minimum relative diff. for {', '.join(compCols)}:"
                    f" {minMeanLogRelDiff:.3f} <=> {minMeanPctRelDiff:.2f}%")
        assert minMeanPctRelDiff < 10

        logger.info('  ... done with truncation-optimised analyses results.')

    @classmethod
    def compareResultSpecs(cls, rsRef, rsAct):

        optTruncFlagCol = ads.MCDSTruncationOptanalyser.OptimTruncFlagCol

        # a. Analyses
        logger.info('* Specs: unopt. analyses ...')

        dfRefAnlSpecs = rsRef.specs['analyses']
        dfRefAnlSpecs = dfRefAnlSpecs[dfRefAnlSpecs[optTruncFlagCol] == 0]
        logger.info(f'  - ref. unopt. analyses specs: n={len(dfRefAnlSpecs)} =>\n'
                    + dfRefAnlSpecs.to_string(min_rows=30, max_rows=30))
        dfActAnlSpecs = rsAct.specs['analyses']
        dfActAnlSpecs = dfActAnlSpecs[dfActAnlSpecs[optTruncFlagCol] == 0]
        logger.info(f'  - actual unopt. analyses specs: n={len(dfActAnlSpecs)} =>\n'
                    + dfActAnlSpecs.to_string(min_rows=30, max_rows=30))
        dfComp = dfRefAnlSpecs.compare(dfActAnlSpecs)
        logger.info(f'  - unopt. analyses specs comparison: n={len(dfComp)} =>\n'
                    + dfComp.to_string(min_rows=30, max_rows=30))
        assert dfComp.empty

        logger.info('* Specs: opt. analyses ...')
        # Note: See test_unint_mcds_optimiser.testMcdsO0TruncOpter for deeper tests
        #       about the conformity of resulting (optimised) truncation parameters
        #       (left, right distance + nb of fitting and discretisation distance cut points,
        #        i.e. TrGche, TrDrte, NbTrchMod columns)
        #       and actual number of run optimisations kept at the end for each analysis (MultiOpt column).

        dfRefAnlSpecs = rsRef.specs['analyses']
        dfRefAnlSpecs = dfRefAnlSpecs[dfRefAnlSpecs[optTruncFlagCol] == 1]
        logger.info(f'  - ref. opt. analyses specs: n={len(dfRefAnlSpecs)} =>\n'
                    + dfRefAnlSpecs.to_string(min_rows=30, max_rows=30))
        dfActAnlSpecs = rsAct.specs['analyses']
        dfActAnlSpecs = dfActAnlSpecs[dfActAnlSpecs[optTruncFlagCol] == 1]
        logger.info(f'  - actual opt. analyses specs: n={len(dfActAnlSpecs)} =>\n'
                    + dfActAnlSpecs.to_string(min_rows=30, max_rows=30))

        optCols = ['TrGche', 'TrDrte', 'NbTrchMod']
        unoptCols = [col for col in dfRefAnlSpecs if col not in optCols]
        dfComp = dfRefAnlSpecs[unoptCols].compare(dfActAnlSpecs[unoptCols])
        logger.info(f'  - compared non-optimised columns: n={len(dfComp)} =>\n'
                    + dfComp.to_string(min_rows=30, max_rows=30))
        assert dfComp.empty

        logger.info(f'  - compared optimised columns: not here, see test_unint_mcds_optimiser')

        # b. Analyser
        # Note: Need to fix some actual analyser specs, of some type when produced
        #       through the API, but of str type when read from result file.
        logger.info('*  Specs: analyser ...')

        sRefAnrSpecs = rsRef.specs['analyser']
        logger.info(f'  - ref. analyser specs: n={len(sRefAnrSpecs)} =>\n'
                    + sRefAnrSpecs.to_string(min_rows=30, max_rows=30))

        sActAnrSpecs = rsAct.specs['analyser']
        logger.info(f'  - actual analyser specs (raw): n={len(sActAnrSpecs)} =>\n'
                    + sActAnrSpecs.to_string(min_rows=30, max_rows=30))

        for anrSpecName in ['defFitDistCutsFctr', 'defDiscrDistCutsFctr']:
            if not isinstance(sActAnrSpecs[anrSpecName], str):
                sActAnrSpecs[anrSpecName] = str(sActAnrSpecs[anrSpecName])

        logger.info(f'  - actual analyser specs (fixed types): n={len(sActAnrSpecs)} =>\n'
                    + sActAnrSpecs.to_string(min_rows=30, max_rows=30))
        dfComp = sRefAnrSpecs.compare(sActAnrSpecs)

        logger.info(f'  - analyser specs comparison: n={len(dfComp)} =>\n'
                    + dfComp.to_string(min_rows=30, max_rows=30))
        assert dfComp.empty

        # c. Run-time : whatever ref, expect a specific up-to-date list of item names, but nothing more
        # (values may vary, 'cause they are mostly software versions: it's OK)
        logger.info('* Specs: run platform ...')

        sRefRunSpecs = rsRef.specs['runtime']
        logger.info(f'  - ref. runtime specs: n={len(sRefRunSpecs)} =>\n' + sRefRunSpecs.to_string())
        sActRunSpecs = rsAct.specs['runtime']
        logger.info(f'  - actual runtime specs: n={len(sActRunSpecs)} =>\n' + sActRunSpecs.to_string())
        assert set(sActRunSpecs.index) \
               == {'os', 'processor', 'python', 'numpy', 'pandas', 'zoopt', 'matplotlib',
                   'jinja2', 'pyaudisam', 'MCDS engine', 'MCDS engine version'}

        logger.info('  ... done with opt-analyses result specs.')

    @classmethod
    def compareResults(cls, rsRef, rsAct):

        logger.info('Comparing reference to actual opt-analysis results ...')

        cls.compareUnoptResults(rsRef, rsAct)
        cls.compareOptResults(rsRef, rsAct)
        cls.compareResultSpecs(rsRef, rsAct)

        logger.info('Done comparing reference to actual opt-analysis results.')

    def testRun(self, optAnalyser_fxt, refResults_fxt):

        cleanup = False  # Debug only: Set to False to prevent cleaning at the end

        # c. Run opt-analyses through pyaudisam API
        # c.i. Cleanup test folder (Note: avoid any Ruindows shell or explorer inside this folder !)
        optanlr, dfOptAnlysSpecs, optAnlysSpecFile = optAnalyser_fxt
        if optanlr.workDir.exists():
            shutil.rmtree(optanlr.workDir, ignore_errors=True)

        # c.ii. Run and measure performance
        # Figures on a 4-HT-core i5-8365U Ruindows 10 laptop with PCI-e SSD,
        # "optimal performance power scheme", 12 threads, Python 3.8 :
        # * 2021-01-05
        #   * OptAnalyserspecs: Zone=ACDC, Surface=2400, distanceUnit=Meter, areaUnit=Hectare, surveyType=Point,
        #     distanceType=Radial, clustering=False, defEstimKeyFn=HNORMAL, defEstimAdjustFn=COSINE,
        #     defEstimCriterion=AIC, defCVInterval=95, defMinDist=None, defMaxDist=None, defFitDistCuts=None,
        #     defDiscrDistCuts=None, defExpr2Optimise=chi2, defMinimiseExpr=False,
        #     dDefOptimCoreParams={'core': 'zoopt', 'maxIters': 100, 'termExprValue': None,
        #                          'algorithm': 'racos', 'maxRetries': 0},
        #     defSubmitTimes=1, defSubmitOnlyBest=None, dDefSubmitOtherParams={},
        #     defOutliersMethod=tucquant, defOutliersQuantCutPct=7,
        #     defFitDistCutsFctr=[0.6, 1.4], defDiscrDistCutsFctr=[0.5, 1.2]
        #   * OptAnalyses specs: 60 optimisations, from refin/ACDC2019-Naturalist-extrait-SpecsOptanalyses.xlsx
        #     => 70 results,
        #   * runMethod: subprocess.run => 4mn40, 4mn52, 4mn38, 4mn23, 4mn40, 5mn00, 4mn41, 4mn35, 4mn47 (mean 4mn42)
        #   * runMethod: os.system      => 4mn35, 4mn24, 4mn20, 4mn30 (mean 4mn27)
        # * 2021-08-22, 2021-10-02
        #   * same OptAnalyserspecs, OptAnalyses specs
        #   * runMethod: subprocess.run => 4mn35 (n >= 2)
        # * 2021-10-06
        #   * same OptAnalyserspecs, OptAnalyses specs
        #   * runMethod: subprocess.run => 4mn08 (n = 1)
        # * 2021-11-19 After adding quality indicators computation in analysis results post-processing
        #   * same OptAnalyserspecs, OptAnalyses specs
        #   * runMethod: subprocess.run => 6mn21 (n = 1)

        # Figures on a 6-core (HT off) i7-10850H Ruindows 10 laptop with PCI-e SSD,
        # "optimal performance power scheme", Python 3.8 :
        # * 2021-11-28 After optimizing quality indicators computation in analysis results post-processing
        #   * same OptAnalyserspecs, OptAnalyses specs as on 2021-01-05
        #   * 12 threads, runMethod: subprocess.run => 4mn12 (n = 1)
        #   * 18 threads, runMethod: subprocess.run => 3mn20 (n = 1)
        #   * 24 threads, runMethod: subprocess.run => 3mn30 (n = 1)
        # * 2022-01-01,02 (no change)
        #   * 24 threads, runMethod: subprocess.run => 3mn16 to 3mn28 (n = 2)
        # * 2022-01-17 (no change)
        #   * 24 threads, runMethod: subprocess.run => 3mn03 (n = 1)

        # Figures on a 6-core (HT on) i7-10850H Ruindows 10 laptop with PCI-e SSD,
        # "optimal performance power scheme", Python 3.8 :
        # * 2023-11-02 (no change)
        #   * 24 threads, runMethod: subprocess.run => 2mn58 (n = 1)

        threads = 12
        logger.info(f'Running opt-analyses: {threads} parallel threads ...')
        logger.info('* OptAnalyser specs: ' + ', '.join(f'{k}={v}' for k, v in optanlr.specs.items()))
        logger.info(f'* OptAnalyses specs: {len(dfOptAnlysSpecs)} optimisations from {optAnlysSpecFile}')

        # BE CAREFUL: time.process_time() uses relative time for comparison only of codes among the same environment
        # NOT A REAL TIME reference
        start = time.perf_counter()

        rsAct = optanlr.run(implParamSpecs=optAnlysSpecFile, threads=threads)

        end = time.perf_counter()

        logger.info(f'* elapsed time={end - start:.2f}s')

        # d. Export results
        # Note: Broken multi-index columns with toOpenDoc !? => use toExcel.
        # rsAct.toOpenDoc(optanlr.workDir / 'valtests-optanalyses-results-api.ods')
        rsAct.toExcel(optanlr.workDir / 'valtests-optanalyses-results-api.xlsx')
        rsAct.toExcel(optanlr.workDir / 'valtests-optanalyses-results-api-en.xlsx', lang='en')

        # e. Check results: Compare to reference
        rsRef = refResults_fxt
        self.compareResults(rsRef, rsAct)

        # f. Minimal check of opt-analysis folders
        uivu.checkAnalysisFolders(rsAct.dfTransData('en').RunFolder, anlysKind='opt-analysis')

        # g. Restart from last backup + export and compare results
        logger.info(f'Restarting opt-analyses from last backup ...')
        rsAct = optanlr.run(implParamSpecs=optAnlysSpecFile, recoverOptims=True, threads=12)
        rsAct.toExcel(optanlr.workDir / 'valtests-optanalyses-restart-results-api.xlsx')
        self.compareResults(rsRef, rsAct)

        # h. Cleanup analyser (analysis folders, not results)
        if cleanup:
            optanlr.cleanup()

        # i. Done.
        logger.info(f'PASS testRun: run (first, restart), cleanup')

    # Run analyses through pyaudisam command line interface
    def testRunCli(self, optAnalyser_fxt, refResults_fxt):

        optanlr, _, _ = optAnalyser_fxt

        logger.info(f'Running opt-analyses (command line) ...')

        # a. Cleanup test folder (Note: avoid any Ruindows shell or explorer inside this folder !)
        if optanlr.workDir.exists():
            shutil.rmtree(optanlr.workDir)

        # b. Run "through the commande line"
        argv = f'-p {uivu.pTestDir.as_posix()}/valtests-ds-params.py -w {optanlr.workDir.as_posix()}' \
               ' -n --optanalyses -u'.split()
        rc = ads.main(argv, standaloneLogConfig=False)
        logger.info(f'CLI run(first): rc={rc}')

        # c. Load results
        rsAct = self.loadResults(optanlr, optanlr.workDir / 'valtests-optanalyses-results.xlsx')
        logger.info(f'Actual results(first): n={len(rsAct)} =>\n' + rsAct.dfData.to_string(min_rows=30, max_rows=30))

        # d. Compare to reference.
        rsRef = refResults_fxt
        self.compareResults(rsRef, rsAct)

        # e. Minimal check of analysis folders
        uivu.checkAnalysisFolders(rsAct.dfTransData('en').RunFolder, anlysKind='opt-analysis')

        # f. Restart from last backup + compare results
        logger.info(f'Restarting opt-analyses from last backup ...')
        argv = f'-p {uivu.pTestDir.as_posix()}/valtests-ds-params.py -w {optanlr.workDir.as_posix()}' \
               ' -n --optanalyses -u -c'.split()
        rc = ads.main(argv, standaloneLogConfig=False)
        logger.info(f'CLI run(restart): rc={rc}')

        rsAct = self.loadResults(optanlr, optanlr.workDir / 'valtests-optanalyses-results.xlsx')
        rsAct.toExcel(optanlr.workDir / 'valtests-optanalyses-results-cli-en.xlsx', lang='en')
        logger.info(f'Actual results(restart): n={len(rsAct)} =>\n' + rsAct.dfData.to_string(min_rows=30, max_rows=30))
        self.compareResults(rsRef, rsAct)

        # g. Don't clean up work folder / analysis folders : needed for report generations below

        # h. Done.
        logger.info(f'PASS testRunCli: main, run (command line mode)')

    @pytest.fixture()
    def excelRefReport_fxt(self):

        return pd.read_excel(uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-Rapport.ods',
                             sheet_name=None, index_col=0)

    @staticmethod
    def compareExcelReports(ddfRefReport, ddfActReport):
        """Prerequisite: Reference report generated with either MCDS 7.4 or 6.2"""

        logger.info('Comparing reference to actual workbook reports ...')

        # Compare "Synthesis" sheet
        dfRef = ddfRefReport['Synthesis'].drop(columns=['RunFolder']).set_index('NumAnlys')
        dfAct = ddfActReport['Synthesis'].drop(columns=['RunFolder']).set_index('NumAnlys')
        assert dfRef.compare(dfAct).empty

        # Compare "Details" sheet : not that simple ...
        # * 11 more "No Doc" columns with MCDS 7.4 (the ref) compared to MCDS 6.2,
        # * very small differences in "Qua" indicators between MCDS 7.4 compared to MCDS 6.2
        dfRef = ddfRefReport['Details'].drop(columns=['StartTime', 'ElapsedTime', 'RunFolder'])
        dfAct = ddfActReport['Details'].drop(columns=['StartTime', 'ElapsedTime', 'RunFolder'])
        # a. Compare all the string columns and a few "no precision issue" more.
        idCols = ['NumAnlys', 'NumEchant', 'Espèce', 'Passage', 'Adulte', 'Durée', 'AbrevAnlys']
        simpleCompCols = idCols
        simpleCompCols += ['NTot Obs', 'Mod Key Fn', 'Mod Adj Ser', 'Mod Chc Crit', 'Conf Interv', 'Key Fn', 'Adj Ser']
        assert dfRef[simpleCompCols].set_index('NumAnlys').compare(dfAct[simpleCompCols].set_index('NumAnlys')).empty

        # b. Compare other (all numerical) columns with a small margin (1e-14 relative diff)
        otherCompCols = [col for col in dfRef if col not in simpleCompCols]
        if len(dfAct.columns) != len(dfRef.columns):  # Not the same version of MCDS as for the ref. report
            spe74CompCols = [col for col in otherCompCols if col.startswith('SansDoc #')]  # 7.4-specifics
            assert len(spe74CompCols) == 11
            otherCompCols = [col for col in otherCompCols if col not in spe74CompCols]  # Remove 7.4-specifics

        logger.info(f'* {otherCompCols=}')
        dfDiff = ads.DataSet.compareDataFrames(dfLeft=dfRef, dfRight=dfAct,
                                               subsetCols=otherCompCols, indexCols=idCols,
                                               noneIsNan=True, dropCloserCols=True,
                                               dropCloser=14, dropNans=True)
        logger.info(f'* diff. to reference (relative): n={len(dfDiff)} =>\n'
                    + dfDiff.to_string(min_rows=30, max_rows=30))
        dfDiff.reset_index().to_excel(uivu.pWorkDir / 'rep_comp-14.xlsx')
        assert dfDiff.empty

        # Compare "Samples" sheet
        dfRef = ddfRefReport['Samples']
        dfAct = ddfActReport['Samples']
        assert dfRef.compare(dfAct).empty

        # Compare "Analyses" sheet
        dfRef = ddfRefReport['Analyses'].set_index('NumAnlys', drop=True)
        dfAct = ddfActReport['Analyses'].set_index('NumAnlys', drop=True)
        assert dfRef.compare(dfAct).empty

        # Compare "Analyser" sheet
        dfRef = ddfRefReport['Analyser']
        dfAct = ddfActReport['Analyser']
        assert dfRef.compare(dfAct).empty

        logger.info('Done comparing reference to actual workbook reports.')

    @pytest.fixture()
    def htmlRefReportLines_fxt(self):

        with open(uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-Rapport.html') as file:
            repLines = file.readlines()

        return repLines

    @staticmethod
    def compareHtmlReports(refReportLines, actReportLines):
        """Prerequisite: Reference report generated with either MCDS 7.4 or 6.2"""

        logger.info('Preprocessing HTML reports for comparison ...')

        # Pre-process actual report lines
        remRefLines = remActLines = 0

        # * list unique cell ids (keeping the original order) in both reports
        KRETableId = r'#T(_[0-9a-f]{5}_)'
        refTableIds = uivu.listUniqueStrings(KRETableId, refReportLines)
        actTableIds = uivu.listUniqueStrings(KRETableId, actReportLines)
        assert len(refTableIds) == 2
        assert len(actTableIds) == len(refTableIds)

        refDetTableId = refTableIds[1]  # "Details" table Id

        logger.info(f'* found table Ids: ref={refTableIds}, act={actTableIds}')

        # * replace each cell id in the actual report by the corresponding ref. report one
        #   (note: the heading and trailing '_' of the Id make this replacement safe ;
        #    without them, there are chances to also replace decimal figures in results !)
        repIdLines = uivu.replaceStrings(froms=actTableIds, tos=refTableIds, lines=actReportLines)
        logger.info(f'* replaced by analysis Id by ref. one in {repIdLines} act. lines')

        # * list unique analysis folders (keeping the original order) in both reports
        KREAnlysDir = r'="./([a-zA-Z0-9-_]+)/'
        refAnlysDirs = uivu.listUniqueStrings(KREAnlysDir, refReportLines)
        actAnlysDirs = uivu.listUniqueStrings(KREAnlysDir, actReportLines)
        assert len(refAnlysDirs) == len(actAnlysDirs)

        # * replace each analysis folder in the actual report by the corresponding ref. report one
        repAnlysDirLines = uivu.replaceStrings(froms=actAnlysDirs, tos=refAnlysDirs, lines=actReportLines)
        logger.info(f'* replaced analysis folder by ref. one in {repAnlysDirLines} act. lines')

        # * remove specific lines in both reports:
        #   - header meta "DateTime"
        KREDateTime = r'[0-9]{2}/[0-9]{2}/[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}'
        KREMetaDateTime = rf'<meta name="datetime" content="{KREDateTime}"/>'
        remRefLines += uivu.removeLines(KREMetaDateTime, refReportLines)
        remActLines += uivu.removeLines(KREMetaDateTime, actReportLines)

        #   - run time and duration table cells
        KRERunTimeDuration = r'<td id="T_.+_row.+_col(19|20)" class="data row.+ col(19|20)" >'
        remRefLines += uivu.removeLines(KRERunTimeDuration, refReportLines)
        remActLines += uivu.removeLines(KRERunTimeDuration, actReportLines)

        #   - footer "Generated on <date+time>" => not needed, 'cause in final ignored blocks (see below)
        # KREGenDateTime = rf'Generated on {KREDateTime}'
        # remRefLines += uivu.removeLines(KREGenDateTime, refReportLines)
        # remActLines += uivu.removeLines(KREGenDateTime, actReportLines)

        #   - new undocumented MCDS 7.4 result columns in ref "Details" table
        #     => ignore them in ref. report when running MCDS 6.2
        KREDetailsMcds74LastCol = rf'<td id="T{refDetTableId}row.+_col122"'  # No such column prior with MCDS 6.2
        if not uivu.selectLines(KREDetailsMcds74LastCol, actReportLines):
            KREDetailsUndocCol = rf'<td id="T{refDetTableId}row.+_col(66|67|68|69|70|71|72|73|74|75|76)"'
            remRefLines += uivu.removeLines(KREDetailsUndocCol, refReportLines)

        #   - header and style lines in ref "Details" table (simpler report comparison, at the price of not checking
        #     cell styling ... but this is not really useful / much examined after all,
        #     and column headers, which is a bit more annoying ...)
        KREDetailsLine2Drop = rf'^#T{refDetTableId}row0_col0,'
        remRefLines += uivu.removeLines(KREDetailsLine2Drop, refReportLines)
        remActLines += uivu.removeLines(KREDetailsLine2Drop, actReportLines)

        KREDetailsLine2Drop = rf'^ +}}#T{refDetTableId}row'
        remRefLines += uivu.removeLines(KREDetailsLine2Drop, refReportLines)
        remActLines += uivu.removeLines(KREDetailsLine2Drop, actReportLines)

        KREDetailsLine2Drop = rf'<table id="T{refDetTableId}" ><thead>'
        remRefLines += uivu.removeLines(KREDetailsLine2Drop, refReportLines)
        remActLines += uivu.removeLines(KREDetailsLine2Drop, actReportLines)

        logger.info(f'* removed {remRefLines} ref. and {remActLines} act. lines')

        # * remove cell id/class in "Details" table (simpler report comparison, at the price of not checking
        #   cell styling ... but this is not really useful / much examined after all)
        KDetailsConstCellIdClass = '<td>'
        KREDetailsCellIdClass = rf'<td id="T{refDetTableId}row.+_col.+" class=".+" >'
        repRefLines = uivu.replaceRegExps(re2Search=KREDetailsCellIdClass, repl=KDetailsConstCellIdClass,
                                          lines=refReportLines)
        repActLines = uivu.replaceRegExps(re2Search=KREDetailsCellIdClass, repl=KDetailsConstCellIdClass,
                                          lines=actReportLines)

        KDetailsConstCellIdClass = '<th>'
        KREDetailsCellIdClass = rf'<th id="T{refDetTableId}level.+_row.+" class=".+" >'
        repRefLines += uivu.replaceRegExps(re2Search=KREDetailsCellIdClass, repl=KDetailsConstCellIdClass,
                                           lines=refReportLines)
        repActLines += uivu.replaceRegExps(re2Search=KREDetailsCellIdClass, repl=KDetailsConstCellIdClass,
                                           lines=actReportLines)

        logger.info(f'* cleaned up "Details" table: {repRefLines} ref. and {repActLines} act. lines')

        # with open(uivu.pTmpDir / 'report-ref-after.html', 'w') as file:
        #     file.writelines(refReportLines)
        # with open(uivu.pTmpDir / 'report-act-after.html', 'w') as file:
        #     file.writelines(actReportLines)

        # Build the list of unified diff blocks
        blocks = uivu.unifiedDiff(refReportLines, actReportLines, logger=logger, subject='HTML reports')

        # Filter diff blocks to check (ignore some that are expected to change without any issue:
        # * computation platform table, with component versions,
        # * generation date, credits to components with versions, sources)
        blocks_to_check = []
        for block in blocks:
            if block.startLines.expected >= 10274 - remRefLines:  # <h3>Computing platform</h3><table ...<tbody>
                logger.info(f'Ignoring block @ -{block.startLines.expected} +{block.startLines.real} @')
                continue
            blocks_to_check.append(block)

        # Check filtered blocks: none should remain.
        assert len(blocks_to_check) == 0

        logger.info('HTML reports comparison: success !')

    # # ## 7. Generate HTML and Excel analyses reports through pyaudisam API
    # def testReports(self, optAnalyser_fxt, excelRefReport_fxt, htmlRefReportLines_fxt):
    #
    #     build = True  # Debug only: Set to False to avoid rebuilding the report, and only check it
    #     cleanup = True  # Debug only: Set to False to prevent cleaning at the end
    #
    #     # Pre-requisites : uncleaned analyser work dir (we need the results file and analysis folders).
    #     optanlr, _, _ = optAnalyser_fxt
    #     if build:
    #         logger.info('Checking analyser results presence ...')
    #         anlysrResFilePath = optanlr.workDir / 'valtests-optanalyses-results.xlsx'
    #         assert optanlr.workDir.is_dir() and anlysrResFilePath.is_file()
    #         anlysFolders = [path for path in optanlr.workDir.iterdir() if path.is_dir()]
    #         assert len(anlysFolders) == 48
    #         logger.info('Done checking analyser results presence: OK.')
    #
    #         # a. Load results
    #         rsAct = self.loadResults(optanlr, anlysrResFilePath)
    #         logger.info(f'Actual results: n={len(rsAct)} =>\n' + rsAct.dfData.to_string(min_rows=30, max_rows=30))
    #
    #         # b. Generate Excel and HTML reports
    #         R = rsAct.__class__
    #         # b.i. Super-synthesis sub-report : Selected analysis results columns for the 3 textual columns of the table
    #         sampleRepCols = [
    #             ('header (head)', 'NumEchant', 'Value'),
    #             ('header (sample)', 'Espèce', 'Value'),
    #             ('header (sample)', 'Passage', 'Value'),
    #             ('header (sample)', 'Adulte', 'Value'),
    #             ('header (sample)', 'Durée', 'Value'),
    #             R.CLNTotObs, R.CLMinObsDist, R.CLMaxObsDist
    #         ]
    #
    #         paramRepCols = [
    #             R.CLParEstKeyFn, R.CLParEstAdjSer,
    #             # R.CLParEstSelCrit, R.CLParEstCVInt,
    #             R.CLParTruncLeft, R.CLParTruncRight, R.CLParModFitDistCuts
    #         ]
    #
    #         resultRepCols = [
    #             ('header (head)', 'NumAnlys', 'Value'),
    #             R.CLRunStatus,
    #             R.CLNObs, R.CLEffort, R.CLSightRate, R.CLNAdjPars,
    #             R.CLAic, R.CLChi2, R.CLKS, R.CLDCv,
    #             R.CLCmbQuaBal3, R.CLCmbQuaBal2, R.CLCmbQuaBal1,
    #             R.CLDensity, R.CLDensityMin, R.CLDensityMax,
    #             R.CLNumber, R.CLNumberMin, R.CLNumberMax,
    #             R.CLEswEdr, R.CLPDetec
    #         ]
    #
    #         # b.ii. Synthesis sub-report : Selected analysis results columns for the table
    #         synthRepCols = [
    #             ('header (head)', 'NumEchant', 'Value'),
    #             ('header (sample)', 'Espèce', 'Value'),
    #             ('header (sample)', 'Passage', 'Value'),
    #             ('header (sample)', 'Adulte', 'Value'),
    #             ('header (sample)', 'Durée', 'Value'),
    #             ('header (head)', 'NumAnlys', 'Value'),
    #
    #             R.CLParEstKeyFn, R.CLParEstAdjSer,
    #             # R.CLParEstSelCrit, R.CLParEstCVInt,
    #             R.CLParTruncLeft, R.CLParTruncRight, R.CLParModFitDistCuts,
    #
    #             R.CLNTotObs, R.CLNObs, R.CLNTotPars, R.CLEffort, R.CLDeltaAic,
    #             R.CLChi2, R.CLKS, R.CLCvMUw, R.CLCvMCw, R.CLDCv,
    #             R.CLPDetec, R.CLPDetecMin, R.CLPDetecMax, R.CLDensity, R.CLDensityMin, R.CLDensityMax,
    #
    #             R.CLSightRate,
    #             R.CLCmbQuaBal1, R.CLCmbQuaBal2, R.CLCmbQuaBal3,
    #             R.CLCmbQuaChi2, R.CLCmbQuaKS, R.CLCmbQuaDCv,
    #
    #             R.CLGrpOrdSmTrAic,
    #             R.CLGrpOrdClTrChi2KSDCv,  # R.CLGrpOrdClTrChi2,
    #             R.CLGrpOrdClTrDCv,
    #             R.CLGrpOrdClTrQuaBal1, R.CLGrpOrdClTrQuaBal2, R.CLGrpOrdClTrQuaBal3, R.CLGrpOrdClTrQuaChi2,
    #             R.CLGrpOrdClTrQuaKS, R.CLGrpOrdClTrQuaDCv,
    #             R.CLGblOrdChi2KSDCv, R.CLGblOrdQuaBal1, R.CLGblOrdQuaBal2, R.CLGblOrdQuaBal3,
    #             R.CLGblOrdQuaChi2, R.CLGblOrdQuaKS, R.CLGblOrdQuaDCv,
    #             R.CLGblOrdDAicChi2KSDCv,
    #             R.CLRunFolder,
    #         ]
    #
    #         # b.iii. Sorting columns for all the sub-reports
    #         sortRepCols = \
    #             [('header (head)', 'NumEchant', 'Value')] \
    #             + [R.CLParTruncLeft, R.CLParTruncRight,
    #                R.CLDeltaAic,
    #                R.CLCmbQuaBal3]
    #
    #         sortRepAscend = [True] * (len(sortRepCols) - 1) + [False]
    #
    #         # b.iv. Report object
    #         report = ads.MCDSResultsFullReport(resultsSet=rsAct,
    #                                            sampleCols=sampleRepCols, paramCols=paramRepCols,
    #                                            resultCols=resultRepCols, synthCols=synthRepCols,
    #                                            sortCols=sortRepCols, sortAscend=sortRepAscend,
    #                                            title='PyAuDiSam Validation: Analyses',
    #                                            subTitle='Global analysis full report',
    #                                            anlysSubTitle='Detailed report',
    #                                            description='Easy and parallel run through MCDSAnalyser',
    #                                            keywords='pyaudisam, validation, analysis',
    #                                            lang='en', superSynthPlotsHeight=288,
    #                                            tgtFolder=anlysr.workDir, tgtPrefix='valtests-optanalyses-report')
    #
    #         # b.iv. Excel report
    #         xlsxRep = report.toExcel()
    #         logger.info('Excel report: ' + pl.Path(xlsxRep).resolve().as_posix())
    #
    #         # b.v. HTML report
    #         htmlRep = report.toHtml()
    #         logger.info('HTML report: ' + pl.Path(htmlRep).resolve().as_posix())
    #
    #     else:
    #         xlsxRep = optanlr.workDir / 'valtests-optanalyses-report.xlsx'
    #         htmlRep = optanlr.workDir / 'valtests-optanalyses-report.html'
    #
    #     # c. Load generated Excel report and compare it to reference one
    #     ddfRefRep = excelRefReport_fxt
    #
    #     ddfActRep = pd.read_excel(xlsxRep, sheet_name=None, index_col=0)
    #
    #     self.compareExcelReports(ddfRefRep, ddfActRep)
    #
    #     # c. Load generated HTML report and compare it to reference one
    #     htmlRefRepLines = htmlRefReportLines_fxt
    #
    #     with open(htmlRep) as file:
    #         htmlActRepLines = file.readlines()
    #
    #     self.compareHtmlReports(htmlRefRepLines, htmlActRepLines)
    #
    #     # e. Cleanup generated report (well ... partially at least)
    #     #    for clearing next function's ground
    #     if cleanup:
    #         pl.Path(xlsxRep).unlink()
    #         pl.Path(htmlRep).unlink()
    #
    #     # f. Done.
    #     logger.info(f'PASS testReports: MCDSResultsFullReport ctor, toExcel, toHtml')
    #
    # ## 7. Generate HTML and Excel analyses reports through pyaudisam command line
    # def testReportsCli(self, excelRefReport_fxt, htmlRefReportLines_fxt):
    #
    #     build = True  # Debug only: Set to False to avoid rebuilding the report
    #
    #     # a. Report "through the commande line"
    #     if build:
    #         argv = f'-p {uivu.pTestDir.as_posix()}/valtests-ds-params.py -w {uivu.pWorkDir.as_posix()}' \
    #                ' -n --optreports excel,html:mqua-r92 -u'.split()
    #         rc = ads.main(argv, standaloneLogConfig=False)
    #         logger.info(f'CLI run: rc={rc}')
    #
    #     # b. Load generated Excel report and compare it to reference one
    #     ddfActRep = pd.read_excel(uivu.pWorkDir / 'valtests-optanalyses-report.xlsx', sheet_name=None, index_col=0)
    #
    #     ddfRefRep = excelRefReport_fxt
    #     self.compareExcelReports(ddfRefRep, ddfActRep)
    #
    #     # c. Load generated HTML report and compare it to reference one
    #     with open(uivu.pWorkDir / 'valtests-optanalyses-report.html') as file:
    #         htmlActRepLines = file.readlines()
    #
    #     htmlRefRepLines = htmlRefReportLines_fxt
    #     self.compareHtmlReports(htmlRefRepLines, htmlActRepLines)
    #
    #     # d. No cleanup: let the final test class cleaner operate: _inifinalizeClass()
    #
    #     # e. Done.
    #     logger.info(f'PASS testReports: main, MCDSResultsFullReport ctor, toExcel, toHtml (command line mode)')
