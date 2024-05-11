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

# Some constants
RS = ads.MCDSTruncOptanalysisResultsSet
KOptTruncFlagCol = RS.OptimTruncFlagCol  # Not translated actually
KResLogCols = [
    ('header (head)', 'NumEchant', 'Value'),
    ('header (head)', 'NumAnlys', 'Value'),
    ('header (tail)', 'AbrevAnlys', 'Value'),
    RS.CLNTotObs,
    RS.CLParTruncLeft, RS.CLParTruncRight, RS.CLParModFitDistCuts,
    RS.CLNObs,
    RS.CLRunStatus,
    RS.CLCmbQuaBal3,
    RS.CLDensity, RS.CLDensityMin, RS.CLDensityMax,
]


@pytest.mark.valtests
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

        defSubmitTimes = 2
        defSubmitOnlyBest = None

        defCoreEngine = 'zoopt'
        defCoreMaxIters = 150
        defCoreTermExprValue = None
        defCoreAlgorithm = 'racos'
        defCoreMaxRetries = 0

        dDefSubmitOtherParams = dict()

        # 0.d. Results post-computation parameters
        ldTruncIntrvSpecs = [dict(col='left', minDist=5.0, maxLen=5.0),
                             dict(col='right', minDist=25.0, maxLen=25.0)]
        truncIntrvEpsilon = 1e-6

        # 0.e. Les analyses à faire (avec specs d'optimisation dedans si nécessaire)
        optAnlysSpecFile = (uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-SpecsOptAnalyses.ods')

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
                                          workDir=uivu.pWorkDir, runTimeOut=120, runMethod='subprocess.run',
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
                                     runTimeOut=120,
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
                                     dDefOptimCoreParams=dict(core='zoopt', maxIters=150, termExprValue=None,
                                                              algorithm='racos', maxRetries=0),
                                     defSubmitTimes=2,
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

        assert len(dfOptAnlysSpecs) == 74
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

        assert isinstance(rsRes, ads.MCDSTruncOptanalysisResultsSet)

        return rsRes

    @pytest.fixture()
    def refResults_fxt(self, optAnalyser_fxt):

        logger.info(f'Preparing reference results ...')

        # Prevent re-postComputation as this ref. file is old, with now missing computed cols
        anlysr, _, _ = optAnalyser_fxt
        rsRef = self.loadResults(anlysr, uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-OptResultats.ods',
                                 postComputed=True)

        logger.info(f'Reference results: n={len(rsRef)} =>\n'
                    + rsRef.dfTransData(columns=KResLogCols, lang='en').to_string(min_rows=99, max_rows=99))

        return rsRef

    @staticmethod
    def compareUnoptResultTables(dfRef, dfAct, mode='api', source='results', table='all'):
        """Prerequisite: Reference results generated with either MCDS 7.4 or 6.2
        Note: Used for result sheets comparison, but also for some report sheets comparison"""

        assert source in {'results', 'report'}
        assert mode in {'api', 'cli'}
        table = table.lower()

        # a. Actual "unoptimised" analysis results ("all-results" sheet)
        logger.info(f'* non truncation-optimised analyses {source} ({table}):')
        dfRefUnopt = dfRef[dfRef[KOptTruncFlagCol] == 0].copy()
        logger.info(f'  - found {len(dfRefUnopt)} reference unopt. result rows')
        dfActUnopt = dfAct[dfAct[KOptTruncFlagCol] == 0].copy()
        logger.info(f'  - found {len(dfActUnopt)} actual unopt. result rows')

        assert len(dfRefUnopt) == len(dfActUnopt)

        # b. Comparison index = analysis "Id": sample Id columns, analysis number, and model parameters.
        sampleNumCol = 'NumEchant'
        sampleCols = [sampleNumCol, 'Espèce', 'Passage', 'Adulte', 'Durée']
        anlysNumCol = 'NumAnlys'  # Actually identifies the analysis optim specs (truncation optimisation ones included)
        modelCols = ['Mod Key Fn', 'Mod Adj Ser']
        indexCols = sampleCols + modelCols + [anlysNumCol]
        indexColsStr = '\n    . '.join(str(t) for t in indexCols)
        logger.info(f'  - index columns for comparison:\n    . ' + indexColsStr)
        sampleColsStr = '\n    . '.join(str(t) for t in sampleCols)
        logger.info(f'  - given sample columns:\n    . ' + sampleColsStr)

        # c. Comparison columns: use reference columns, except for ...
        #  - analysis Id columns (the comparison index),
        #  - 'run output' chapter results (start time, elapsed time, run folder ... specific to each run),
        #  - all other text columns (not supported by ResultsSet.compare, anlysAbrevCol included),
        #  - "Order" columns, difficult to compare, because a very small difference in one or more
        #    computations parameters is quickly amplified by the exponential formula,
        #    thus changing the row order by an amount of at least 1, which would imply another comparison method,
        #  - "pre-selection" columns in filter & sort reports, for the same reason as the "Order" columns
        #    (they hold an integer index that can slightly change with optimised results,
        #     but slight changes for integers are at least of 1 unit, which is not suitable for the diff tool used).
        anlysAbrevCol = 'AbrevAnlys'
        compCols = [col for col in dfActUnopt.columns
                    if col not in indexCols
                       and col not in [anlysAbrevCol, 'Mod Chc Crit', 'StartTime', 'ElapsedTime', 'RunFolder',
                                       'FonctionClé', 'SérieAjust', 'Key Fn', 'Adj Ser']  # , 'Delta AIC']
                       and not col.startswith('Order')
                       and not col.startswith('Pre-selection')]
        compColsStr = '\n    . '.join(str(t) for t in compCols)
        logger.info(f'  - data columns for comparison:\n    . ' + compColsStr)

        with pd.ExcelWriter(uivu.pWorkDir / f'valtst-optanlys-{mode}-{source}-{table}-unopt-for-diff.xlsx') as xlWrtr:
            dfRefUnopt.set_index(indexCols)[compCols].to_excel(xlWrtr, sheet_name='ref')
            dfActUnopt.set_index(indexCols)[compCols].to_excel(xlWrtr, sheet_name='act')

        # d. Compare
        dfDiff = ads.DataSet.compareDataFrames(dfRefUnopt, dfActUnopt, indexCols=indexCols, subsetCols=compCols,
                                               noneIsNan=True, dropCloser=14, dropNans=True)

        logger.info(f'  - diff. to unopt. reference (relative: 1e-14): n={len(dfDiff)} =>\n'
                    + dfDiff.to_string(min_rows=30, max_rows=30))
        if not dfDiff.empty:
            dfDiff.to_excel(uivu.pWorkDir / f'valtst-optanlys-{mode}-{source}-{table}-unopt-diff-14.xlsx')

        assert dfDiff.empty, 'Oh oh ... some unexpected differences !'

        # e. To be perfectly honest ... there may be some 10**-15/-16 glitches (due to worksheet I/O ?) ... or not.
        dfDiff = ads.DataSet.compareDataFrames(dfRefUnopt, dfActUnopt, indexCols=indexCols, subsetCols=compCols,
                                               noneIsNan=True, dropNans=True)
        dfDiff = dfDiff[(dfDiff != np.inf).all(axis='columns')]

        logger.info(f'  - diff. to unopt. reference (relative): n={len(dfDiff)} =>\n'
                    + dfDiff.to_string(min_rows=30, max_rows=30))
        if not dfDiff.empty:
            dfDiff.to_excel(uivu.pWorkDir / f'valtst-optanlys-{mode}-{source}-{table}-unopt-diff.xlsx')

        logger.info(f'  ... done with non truncation-optimised analyses {source} ({table}).')

    @staticmethod
    def compareOptResultTables(dfRef, dfAct, mode='api', source='results', table='all'):
        """Prerequisite: Reference results generated with either MCDS 7.4 or 6.2
        Note: Used for result sheets comparison, but also for some report sheets comparison"""

        assert source in {'results', 'report'}
        assert mode in {'api', 'cli'}
        table = table.lower()

        # a. Number of ref. and actual results (may show some very limited difference because of duplicated opt. specs)
        logger.info(f'* truncation-optimised analyses {source} ({table}):')
        dfRefOpt = dfRef[dfRef[KOptTruncFlagCol] == 1].copy()
        logger.info(f'  - found {len(dfRefOpt)} reference opt. result rows')
        dfActOpt = dfAct[dfAct[KOptTruncFlagCol] == 1].copy()
        logger.info(f'  - found {len(dfActOpt)} actual opt. result rows')

        KMaxTotalRowDelta = 10
        assert abs(len(dfRefOpt) - len(dfActOpt)) <= KMaxTotalRowDelta

        # b. Number of ref. and actual results per sample (may show some very limited difference)
        sampleNumCol = 'NumEchant'
        sampleCols = [sampleNumCol, 'Espèce', 'Passage', 'Adulte', 'Durée']
        dfRefSampCnts = dfRefOpt[sampleCols + ['Density']].groupby(sampleCols).count()
        dfRefSampCnts.rename(columns=dict(Density='number'), inplace=True)
        logger.info(f'  - number of selected ref. result rows per sample: n={len(dfRefSampCnts)} =>\n'
                    + dfRefSampCnts.to_string())
        dfActSampCnts = dfActOpt[sampleCols + ['Density']].groupby(sampleCols).count()
        dfActSampCnts.rename(columns=dict(Density='number'), inplace=True)
        logger.info(f'  - number of selected ref. result rows per sample: n={len(dfActSampCnts)} =>\n'
                    + dfActSampCnts.to_string())

        KMaxPerSampleRowDelta = 3
        assert dfRefSampCnts.subtract(dfActSampCnts).number.abs().le(KMaxPerSampleRowDelta).all()

        # c. If possible, check that optimisation param. specs are the same as for the ref. results
        # c.i. Sort the results by analysis index (and so also by sample), by optimisation parameter specs,
        #      and then by left truncation distance, the right one, and finally the fit distance cut point number
        anlysNumCol = 'NumAnlys'
        distCols = ['Left Trunc Dist', 'Right Trunc Dist', 'Fit Dist Cuts']
        sortCols = [anlysNumCol] + distCols
        dfRefOpt.sort_values(by=sortCols, inplace=True)
        dfActOpt.sort_values(by=sortCols, inplace=True)

        # c.ii. Save to disk, after "merging" ref and actual results, for visual checks
        dfRefOpt.insert(0, 'Source', 'ref')
        dfActOpt.insert(0, 'Source', 'act')
        dfMrgdOpt = pd.concat([dfRefOpt, dfActOpt], sort=False)
        dfMrgdOpt.sort_values(by=[anlysNumCol, 'Source'], inplace=True)
        dfMrgdOpt.to_excel(uivu.pWorkDir / f'valtst-optanlys-{mode}-{source}-{table}-opt-raw-comp.xlsx')

        anlysAbrevCol = 'AbrevAnlys'
        if anlysAbrevCol in dfRefOpt:
            logger.info(f'  - comparing optimisation param. specs ...')
            assert dfActOpt[anlysAbrevCol].drop_duplicates().to_list() \
                   == dfRefOpt[anlysAbrevCol].drop_duplicates().to_list()

        # d. Compare a small and simple subset of analyses results columns = fitting statistics + functional results
        #    (not in search of equality, but rather under-control differences)
        # d.i. Need to add a unique analysis index for compareDataFrames ;
        #      otherwise, 'ValueError: cannot handle a non-unique multi-index' in pd.DataFrame.reindex ...
        #      because anlysNumCol is not unique, due to optimisation repetitions = 'times' !
        anlysIndCol = 'IndAnlys'
        dfActOpt[anlysIndCol] = [i for i in range(len(dfActOpt))]
        dfRefOpt[anlysIndCol] = [i for i in range(len(dfRefOpt))]

        # d.ii. Compare
        modelCols = ['Mod Key Fn', 'Mod Adj Ser']
        indexCols = [anlysIndCol, anlysNumCol] + sampleCols + modelCols
        compCols = ['Chi2 P', 'KS P', 'CoefVar Density', 'Qual Bal 3', 'PDetec', 'Density']
        compCols += [col for col in ['AIC', 'EDR/ESW'] if col in dfRefOpt]
        with pd.ExcelWriter(uivu.pWorkDir / f'valtst-optanlys-{mode}-{source}-{table}-opt-for-diff.xlsx') as xlWrtr:
            dfRefOpt.set_index(indexCols)[compCols].to_excel(xlWrtr, sheet_name='ref')
            dfActOpt.set_index(indexCols)[compCols].to_excel(xlWrtr, sheet_name='act')

        dfDiff = ads.DataSet.compareDataFrames(dfActOpt, dfRefOpt, indexCols=indexCols, subsetCols=compCols,
                                               dropNans=True, noneIsNan=True)

        dfDiffStats = pd.DataFrame(data=[dfDiff.min(), dfDiff.max(), dfDiff.replace(np.inf, 16).mean()],
                                   index=['min', 'max', 'mean'])
        with pd.ExcelWriter(uivu.pWorkDir / f'valtst-optanlys-{mode}-{source}-{table}-opt-stats.xlsx') as xlWrtr:
            dfDiff.to_excel(xlWrtr, sheet_name='diff')
            dfDiffStats.to_excel(xlWrtr, sheet_name='diff-stats')
        logger.info(f'  - stats on relative diff. between ref. and actual opt. results:\n' + dfDiffStats.to_string())

        # d.iii. Save dfDiffStats.loc['mean'] for each comparison column, to later tweak the thresholds below.
        dNewDiffStats = dict(source=source, table=table)
        dNewDiffStats.update(dfDiffStats.loc['mean'].to_dict())

        fpnDiffStatHistory = uivu.pTmpDir / f'dont-remove-valtst-optanlys-optres-diff-stats-history.xlsx'
        dfDiffStatHistory = pd.read_excel(fpnDiffStatHistory, index_col=0) if fpnDiffStatHistory.is_file() \
                            else pd.DataFrame()
        dfDiffStatHistory = pd.concat([dfDiffStatHistory,
                                       pd.DataFrame(data=[dNewDiffStats],
                                                    index=pd.Index([pd.Timestamp.now()], name='timestamp'))])
        dfDiffStatHistory.to_excel(fpnDiffStatHistory, index=True)

        # e. Expect less that N% mean difference (after watching stat. history over any runs)
        KMeanPctRelDiffThresh = 15
        minMeanLogRelDiff = dfDiffStats.loc['mean'].min()
        maxMeanPctRelDiff = 100 / 10 ** minMeanLogRelDiff
        logger.info(f"  - minimum relative diff. for {', '.join(compCols)}:"
                    f" 10**-{minMeanLogRelDiff:.3f} <=> {maxMeanPctRelDiff:.2f}%")
        assert maxMeanPctRelDiff < KMeanPctRelDiffThresh

        logger.info(f'  ... done with truncation-optimised analyses {source} ({table}).')

    @classmethod
    def compareResultSpecs(cls, dsdfRefSpecs, dsdfActSpecs, source='results'):
        """dsdfXxx : dict(DataFrame | Series)
        => expecting Series for 'analyser' and 'runtime' keys, DataFrames otherwise"""

        # a. Samples (DataFrames)
        if source == 'report':

            logger.info(f'* {source} specs: samples ...')

            dfRefSmpSpecs = dsdfRefSpecs['analyser']
            logger.info(f'  - ref. sample specs: n={len(dfRefSmpSpecs)} =>\n' + dfRefSmpSpecs.to_string())

            dfActSmpSpecs = dsdfActSpecs['analyser']
            logger.info(f'  - actual sample specs: n={len(dfActSmpSpecs)} =>\n' + dfActSmpSpecs.to_string())

            dfComp = dfRefSmpSpecs.compare(dfActSmpSpecs)

            logger.info(f'  - analyser specs comparison: n={len(dfComp)} =>\n' + dfComp.to_string())
            assert dfComp.empty

        # a. Analyses (DataFrames)
        dfRefAnlSpecs = dsdfRefSpecs['analyses'].set_index('NumAnlys')
        dfActAnlSpecs = dsdfActSpecs['analyses'].set_index('NumAnlys')

        logger.info(f'* {source} specs: unopt. analyses ...')

        dfRefUnoptAnlSpecs = dfRefAnlSpecs[dfRefAnlSpecs[KOptTruncFlagCol] == 0]
        logger.info(f'  - ref. unopt. analyses specs: n={len(dfRefUnoptAnlSpecs)} =>\n'
                    + dfRefUnoptAnlSpecs.to_string())
        dfActUnoptAnlSpecs = dfActAnlSpecs[dfActAnlSpecs[KOptTruncFlagCol] == 0]
        logger.info(f'  - actual unopt. analyses specs: n={len(dfActUnoptAnlSpecs)} =>\n'
                    + dfActUnoptAnlSpecs.to_string())

        dfComp = dfRefUnoptAnlSpecs.compare(dfActUnoptAnlSpecs)
        logger.info(f'  - unopt. analyses specs comparison: n={len(dfComp)} =>\n' + dfComp.to_string())
        assert dfComp.empty

        logger.info(f'* {source} specs: opt. analyses ...')
        # Note: See test_unint_mcds_optimiser.testMcdsO0TruncOpter for deeper tests
        #       about the conformity of resulting (optimised) truncation parameters
        #       (left, right distance + nb of fitting and discretisation distance cut points,
        #        i.e. TrGche, TrDrte, NbTrchMod columns)
        #       and actual number of run optimisations kept at the end for each analysis (MultiOpt column).

        optCols = ['TrGche', 'TrDrte', 'NbTrchMod']
        logger.info(f'  - removing optimised columns: n={len(optCols)} => ' + ', '.join(optCols))
        unoptCols = [col for col in dfRefAnlSpecs if col not in optCols]
        logger.info(f'  - left (non-optimised) columns: n={len(unoptCols)} => ' + ', '.join(unoptCols))

        dfRefOptAnlSpecs = dfRefAnlSpecs.loc[dfRefAnlSpecs[KOptTruncFlagCol] == 1, unoptCols].drop_duplicates()
        logger.info(f'  - ref. unique opt. analyses specs: n={len(dfRefOptAnlSpecs)} =>\n'
                    + dfRefOptAnlSpecs.to_string())
        dfActOptAnlSpecs = dfActAnlSpecs.loc[dfActAnlSpecs[KOptTruncFlagCol] == 1, unoptCols].drop_duplicates()
        logger.info(f'  - actual unique opt. analyses specs: n={len(dfActOptAnlSpecs)} =>\n'
                    + dfActOptAnlSpecs.to_string())

        dfComp = dfRefOptAnlSpecs.compare(dfActOptAnlSpecs)
        logger.info(f'  - comparison of left unique rows and non-optimised columns: n={len(dfComp)} =>\n'
                    + dfComp.to_string())
        assert dfComp.empty

        logger.info(f'  - comparison of optimised columns: not here, see test_unint_mcds_optimiser')

        # b. Analyser (Series)
        # Note: Need to fix some actual analyser specs, of some type when produced
        #       through the API, but of str type when read from result file.
        logger.info(f'* {source} specs: analyser ...')

        sRefAnrSpecs = dsdfRefSpecs['analyser']
        logger.info(f'  - ref. analyser specs: n={len(sRefAnrSpecs)} =>\n' + sRefAnrSpecs.to_string())
        sActAnrSpecs = dsdfActSpecs['analyser']
        logger.info(f'  - actual analyser specs (raw): n={len(sActAnrSpecs)} =>\n' + sActAnrSpecs.to_string())

        for anrSpecName in ['defFitDistCutsFctr', 'defDiscrDistCutsFctr']:
            if not isinstance(sActAnrSpecs[anrSpecName], str):
                sActAnrSpecs[anrSpecName] = str(sActAnrSpecs[anrSpecName])

        logger.info(f'  - actual analyser specs (fixed types): n={len(sActAnrSpecs)} =>\n' + sActAnrSpecs.to_string())
        dfComp = sRefAnrSpecs.compare(sActAnrSpecs)

        logger.info(f'  - analyser specs comparison: n={len(dfComp)} =>\n' + dfComp.to_string())
        assert dfComp.empty

        # c. Run-time (Series) : whatever ref, expect a specific up-to-date list of item names, but nothing more
        # (values may vary, 'cause they are mostly software versions: it's OK)
        logger.info(f'* {source} specs: run platform ...')

        sRefRunSpecs = dsdfRefSpecs['runtime']
        logger.info(f'  - ref. runtime specs: n={len(sRefRunSpecs)} =>\n' + sRefRunSpecs.to_string())
        sActRunSpecs = dsdfActSpecs['runtime']
        logger.info(f'  - actual runtime specs: n={len(sActRunSpecs)} =>\n' + sActRunSpecs.to_string())
        assert set(sActRunSpecs.index) \
               == {'os', 'processor', 'python', 'numpy', 'pandas', 'zoopt', 'matplotlib',
                   'jinja2', 'pyaudisam', 'MCDS engine', 'MCDS engine version'}

        logger.info(f'  ... done with opt-analyses {source} specs.')

    @classmethod
    def compareResults(cls, rsRef, rsAct, mode='api'):

        logger.info('Comparing reference to actual opt-analysis results ...')

        dfAct = rsAct.dfTransData('en')
        dfRef = rsRef.dfTransData('en')
        cls.compareUnoptResultTables(dfRef, dfAct, mode=mode, source='results', table='all')
        cls.compareOptResultTables(dfRef, dfAct, mode=mode, source='results', table='all')

        cls.compareResultSpecs(rsRef.specs, rsAct.specs, source='results')

        logger.info('Done comparing reference to actual opt-analysis results.')

    # Run opt-analyses through pyaudisam API
    #
    # Performance reference figures on a 4-HT-core i5-8365U Ruindows 10 laptop with PCI-e SSD,
    # "optimal performance power scheme", 12 threads, Python 3.8 :
    # * 2021-01-05
    #   * OptAnalyser specs: Zone=ACDC, Surface=2400, distanceUnit=Meter, areaUnit=Hectare, surveyType=Point,
    #     distanceType=Radial, clustering=False, defEstimKeyFn=HNORMAL, defEstimAdjustFn=COSINE,
    #     defEstimCriterion=AIC, defCVInterval=95, defMinDist=None, defMaxDist=None, defFitDistCuts=None,
    #     defDiscrDistCuts=None, defExpr2Optimise=chi2, defMinimiseExpr=False,
    #     dDefOptimCoreParams={'core': 'zoopt', 'maxIters': 100, 'termExprValue': None,
    #                          'algorithm': 'racos', 'maxRetries': 0},
    #     defSubmitTimes=1, defSubmitOnlyBest=None, dDefSubmitOtherParams={},
    #     defOutliersMethod=tucquant, defOutliersQuantCutPct=7,
    #     defFitDistCutsFctr=[0.6, 1.4], defDiscrDistCutsFctr=[0.5, 1.2]
    #   * OptAnalyses specs: 24 optimisations x maxIters ~ 2400 optimisation analyses
    #   * 70 final analyses = results (22 with distance optimisations, 48 with fixed analysis specs),
    #   * runMethod: subprocess.run => mean 4mn40 (n=9)
    #   * runMethod: os.system      => mean 4mn30 (n=4)
    #   * same OptAnalyserspecs, OptAnalyses specs
    #   * runMethod: subprocess.run => 4mn35 (n >= 2)
    # * 2021-10-06
    #   * same OptAnalyserspecs, OptAnalyses specs
    #   * runMethod: subprocess.run => 4mn08 (n=1)
    # * 2021-11-19 After adding quality indicators computation in analysis results post-processing
    #   * same OptAnalyserspecs, OptAnalyses specs
    #   * runMethod: subprocess.run => 6mn21 (n=1)
    #
    # Figures on a 6-core (HT off) i7-10850H Ruindows 10 laptop with PCI-e SSD,
    # "optimal performance power scheme", Python 3.8 :
    # * 2021-11-28 After optimizing quality indicators computation in analysis results post-processing
    #   * same OptAnalyser specs, OptAnalyses specs as on 2021-01-05
    #   * 12 threads, runMethod: subprocess.run => 4mn12 (n=1)
    #   * 18 threads, runMethod: subprocess.run => 3mn20 (n=1)
    #   * 24 threads, runMethod: subprocess.run => 3mn30 (n=1)
    # * 2022-01-01,02 (no change)
    #   * 24 threads, runMethod: subprocess.run => 3mn16 to 3mn28 (n=2)
    # * 2022-01-17 (no change)
    #   * 24 threads, runMethod: subprocess.run => 3mn03 (n=1)
    #
    # Figures on a 6-core (HT on) i7-10850H Ruindows 10 laptop with PCI-e SSD,
    # "optimal performance power scheme", Python 3.8 :
    # * 2023-11-02 (no change)
    #   * 24 threads, runMethod: subprocess.run => 2mn58 (n=1)
    #
    # Figures on a 6-core (HT on) i7-10850H Ruindows 11 laptop with PCI-e SSD,
    # "high performance power scheme", Python 3.8 :
    # * 2024-05-06 (more iters, more repeats, for more results to filter and sort at the end)
    #   * OptAnalyser specs: same, except for defExpr2Optimise=balq3,
    #     dDefOptimCoreParams={'maxIters': 150, otherwise same ...}, defSubmitTimes=2,
    #   * OptAnalyses specs: 38 optimisations x maxIters ~ 5700 optimisation analyses
    #   * 82 final analyses = results (34 with distance optimisations, 48 with fixed analysis specs),
    #   * 12 threads, runMethod: subprocess.run => 7mn40s (n=4)
    # * 2024-05-08 (more iters, more repeats, for even more results to filter and sort at the end)
    #   * OptAnalyser specs: same,
    #   * OptAnalyses specs: 120 optimisations x maxIters ~ 19000 optimisation analyses
    #   * max 168 final analyses = results, but often around 10 less
    #     (max of 120 with distance optimisations, but often around 10 dupl. specs + 48 with fixed analysis specs),
    #   * 18 threads, runMethod: subprocess.run => mean 24mn (n=7)
    def testRun(self, optAnalyser_fxt, refResults_fxt):

        cleanup = True  # Debug only: Set to False to prevent cleaning at the end
        restart = False  # Debug only: Set to False to prevent restart

        # a. Cleanup test folder (Note: avoid any Ruindows shell or explorer inside this folder !)
        optanlr, dfOptAnlysSpecs, optAnlysSpecFile = optAnalyser_fxt
        if optanlr.workDir.exists():
            shutil.rmtree(optanlr.workDir, ignore_errors=True)

        # b. Run and measure performance
        threads = 18
        logger.info(f'Running opt-analyses: {threads} parallel threads ...')
        logger.info('* opt-analyser specs: ' + ', '.join(f'{k}={v}' for k, v in optanlr.specs.items()))
        logger.info(f'* opt-analyses specs: {len(dfOptAnlysSpecs)} optimisations from {optAnlysSpecFile}')

        # BE CAREFUL: time.process_time() uses relative time for comparison only of codes among the same environment
        # NOT A REAL TIME reference
        start = time.perf_counter()

        rsAct = optanlr.run(implParamSpecs=optAnlysSpecFile, threads=threads)

        end = time.perf_counter()

        logger.info(f'* opt-analysis elapsed time: {end - start:.2f}s')

        logger.info(f'Actual results(first): n={len(rsAct)} =>\n'
                    + rsAct.dfTransData(columns=KResLogCols, lang='en').to_string(min_rows=99, max_rows=99))

        # c. Export results
        # Note: Broken multi-index columns with toOpenDoc !? => use toExcel.
        # rsAct.toOpenDoc(optanlr.workDir / 'valtests-optanalyses-results-api.ods')
        rsAct.toExcel(optanlr.workDir / 'valtests-optanalyses-results.xlsx')
        rsAct.toExcel(optanlr.workDir / 'valtests-optanalyses-results-api-en.xlsx', lang='en')

        # d. Check results: Compare to reference
        rsRef = refResults_fxt
        self.compareResults(rsRef, rsAct, mode='api')

        # e. Minimal check of opt-analysis folders
        KExptdMaxAnlysFolders = 120 + 48
        KExptdMinAnlysFolders = KExptdMaxAnlysFolders - 20  # Note: No real theoretical min, less may occur !
        uivu.checkAnalysisFolders(rsAct.dfTransData('en').RunFolder,
                                  expectedCount=(KExptdMinAnlysFolders, KExptdMaxAnlysFolders),
                                  anlysKind='opt-analysis')

        if restart:

            # f. Cleanup analyser
            #    (remove analysis folders, not workbook results files or backup files ; also reset results set)
            optanlr.cleanup()

            # f. Restart from last backup + export and compare results
            logger.info(f'Restarting opt-analyses from last backup ...')
            rsAct = optanlr.run(implParamSpecs=optAnlysSpecFile, recoverOptims=True, threads=12)
            logger.info(f'Actual results(restart): n={len(rsAct)} =>\n'
                        + rsAct.dfTransData(columns=KResLogCols, lang='en').to_string(min_rows=99, max_rows=99))
            rsAct.toExcel(optanlr.workDir / 'valtests-optanalyses-results.xlsx')
            rsAct.toExcel(optanlr.workDir / 'valtests-optanalyses-restart-results-api-en.xlsx', lang='en')
            self.compareResults(rsRef, rsAct, mode='api')

        # g. Cleanup analyser (analysis folders, not workbook results files)
        if cleanup:
            optanlr.cleanup()

        # h. Done.
        logger.info(f'PASS testRun: run (first, restart), cleanup')

    # Run analyses through pyaudisam command line interface
    def testRunCli(self, optAnalyser_fxt, refResults_fxt):

        restart = False  # Debug only: Set to False to prevent restart

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
        logger.info(f'Actual results(first): n={len(rsAct)} =>\n'
                    + rsAct.dfTransData(columns=KResLogCols, lang='en').to_string(min_rows=99, max_rows=99))

        # d. Compare to reference.
        rsRef = refResults_fxt
        self.compareResults(rsRef, rsAct, mode='cli')

        # e. Minimal check of analysis folders
        anlysFolders = rsAct.dfTransData('en').RunFolder
        KExptdMaxAnlysFolders = 120 + 48
        KExptdMinAnlysFolders = KExptdMaxAnlysFolders - 20  # Note: No real theoritical min, less may occur !
        uivu.checkAnalysisFolders(anlysFolders,
                                  expectedCount=(KExptdMinAnlysFolders, KExptdMaxAnlysFolders),
                                  anlysKind='opt-analysis')

        # f. Cleanup analysis folders and then restart from last backup + compare results
        if restart:

            for anlysFolder in anlysFolders:
                dpn = pl.Path(anlysFolder)
                if dpn.is_dir():
                    shutil.rmtree(dpn)

            logger.info(f'Restarting opt-analyses from last backup ...')
            argv = f'-p {uivu.pTestDir.as_posix()}/valtests-ds-params.py -w {optanlr.workDir.as_posix()}' \
                   ' -n --optanalyses -u -c'.split()
            rc = ads.main(argv, standaloneLogConfig=False)
            logger.info(f'CLI run(restart): rc={rc}')

            rsAct = self.loadResults(optanlr, optanlr.workDir / 'valtests-optanalyses-results.xlsx')
            rsAct.toExcel(optanlr.workDir / 'valtests-optanalyses-results-cli-en.xlsx', lang='en')
            logger.info(f'Actual results(restart): n={len(rsAct)} =>\n'
                        + rsAct.dfTransData(columns=KResLogCols, lang='en').to_string(min_rows=99, max_rows=99))
            self.compareResults(rsRef, rsAct, mode='cli')

            anlysFolders = rsAct.dfTransData('en').RunFolder
            KExptdMaxAnlysFolders = 120 + 48
            KExptdMinAnlysFolders = KExptdMaxAnlysFolders - 20  # Note: No real theoretical min, less may occur !
            uivu.checkAnalysisFolders(anlysFolders,
                                      expectedCount=(KExptdMinAnlysFolders, KExptdMaxAnlysFolders),
                                      anlysKind='opt-analysis')

        # g. Don't clean up work folder / analysis folders : needed for report generations below

        # h. Done.
        logger.info(f'PASS testRunCli: main, run (command line mode)')

    @pytest.fixture()
    def excelRefReport_fxt(self):

        return pd.read_excel(uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-OptRapport.ods',
                             sheet_name=None, index_col=0)

    KRep2SpecNames = {'Samples': 'samples', 'Analyses': 'analyses', 'Analyser': 'analyser',
                      'Computing platform': 'runtime'}

    @staticmethod
    def compareAFSStepsTables(dfRef, dfAct, mode='api'):

        # Remove timestamps (always changing)
        dfRef = dfRef[dfRef.Property != 'datetime'].reset_index(drop=True)
        dfAct = dfAct[dfAct.Property != 'datetime'].reset_index(drop=True)

        # Comparing parameters: should not change
        indexCols = ['Scheme', 'Step', 'Property']
        logger.info(f'* auto filter & sort report steps (afs-steps): parameters ...')
        dfRefPars = dfRef[dfRef.Property != 'results'].set_index(indexCols)
        logger.info(f'  - found {len(dfRefPars)} ref. rows')
        dfActPars = dfAct[dfAct.Property != 'results'].set_index(indexCols)
        logger.info(f'  - found {len(dfActPars)} actual rows')

        assert len(dfRefPars) == len(dfActPars), 'Unexpected actual number of step parameters'

        dfComp = dfRefPars.compare(dfActPars)
        logger.info(f'  - differences (compare): n={len(dfComp)} =>\n' + dfComp.to_string())

        if not dfComp.empty:
            fpn = uivu.pWorkDir / f'valtst-optanlys-{mode}-side-by-side-steps.xlsx'
            logger.info(f'  - side by side visual comparison => {fpn.as_posix()}')
            dfComp.to_excel(fpn)

        assert dfComp.empty

        # Comparing results: some differences always possible => use delta thresholds
        logger.info(f'* auto filter and sort report steps (afs-steps): results ...')
        dfRefRes = dfRef[dfRef.Property == 'results'].set_index(indexCols)
        logger.info(f'  - found {len(dfRefRes)} ref. rows')
        dfActRes = dfAct[dfAct.Property == 'results'].set_index(indexCols)
        logger.info(f'  - found {len(dfActRes)} actual rows')
        assert len(dfRefRes) == len(dfActRes), 'Unexpected actual number of step results'

        dfCompRes = dfRefRes.join(dfActRes, lsuffix=' Ref', rsuffix=' Act')
        dfCompRes['Delta'] = dfCompRes['Value Ref'].subtract(dfCompRes['Value Act']).abs()

        KSDeltaResThresh = pd.Series({
            'before': 10,
            'run status': 10,
            'duplicates on params': 5,
            'best AIC': 5,
            'best results for >= 1 indicator': 5,
            'non-outlier sightings': 10,
            'final best results': 5,
            'after': 5,
        }, name='Threshold')
        dfCompRes = dfCompRes.reset_index().join(KSDeltaResThresh, on='Step').set_index(indexCols)
        dfCompRes['IsCorrect'] = dfCompRes.Delta <= dfCompRes.Threshold

        logger.info(f'  - side by side comparison and verdict =>\n' + dfCompRes.to_string())

        assert dfCompRes.IsCorrect.all(), 'Some step results are over the thresholds'

    @staticmethod
    def compareAFSMTables(dfRef, dfAct, mode='api', table='all'):
        """Prerequisite: Reference results generated with either MCDS 7.4 or 6.2
        Note: Used for result sheets comparison, but also for some report sheets comparison"""

        source = 'report'
        assert mode in {'api', 'cli'}
        table = table.lower()

        # a. Number of ref. and actual rows (may show some limited difference)
        logger.info(f'* auto filter & sort {source} ({table}):')
        dfRef = dfRef.copy()
        logger.info(f'  - found {len(dfRef)} reference result rows')
        dfAct = dfAct.copy()
        logger.info(f'  - found {len(dfAct)} actual result rows')

        KMaxTotalRowDelta = 10
        assert abs(len(dfRef) - len(dfAct)) <= KMaxTotalRowDelta

        # b. Number of ref. and actual rows per sample (may show some very limited difference)
        sampleNumCol = 'NumEchant'
        sampleCols = [sampleNumCol, 'Espèce', 'Passage', 'Adulte', 'Durée']
        dfRefSampCnts = dfRef[sampleCols + ['Density']].groupby(sampleCols).count()
        dfRefSampCnts.rename(columns=dict(Density='number'), inplace=True)
        logger.info(f'  - number of selected ref. result rows per sample: n={len(dfRefSampCnts)} =>\n'
                    + dfRefSampCnts.to_string())
        dfActSampCnts = dfAct[sampleCols + ['Density']].groupby(sampleCols).count()
        dfActSampCnts.rename(columns=dict(Density='number'), inplace=True)
        logger.info(f'  - number of selected ref. result rows per sample: n={len(dfActSampCnts)} =>\n'
                    + dfActSampCnts.to_string())

        KMaxPerSampleRowDelta = 3
        assert dfRefSampCnts.subtract(dfActSampCnts).number.abs().le(KMaxPerSampleRowDelta).all()

        # c. Export a table to manually check that optimisation param. specs are the same as for the ref. results,
        #    at least for the "common" rows
        anlysNumCol = 'NumAnlys'
        distCols = ['Left Trunc Dist', 'Right Trunc Dist', 'Fit Dist Cuts']
        sortCols = [anlysNumCol] + distCols
        dfRef.sort_values(by=sortCols, inplace=True)
        dfAct.sort_values(by=sortCols, inplace=True)

        dfRef.insert(0, 'Source', 'ref')
        dfAct.insert(0, 'Source', 'act')
        dfMrgdOpt = pd.concat([dfRef, dfAct], sort=False)
        dfMrgdOpt.sort_values(by=sortCols, inplace=True)
        dfMrgdOpt.to_excel(uivu.pWorkDir / f'valtst-optanlys-{mode}-{source}-{table}-opt-raw-comp.xlsx')

        # d. Compare a small and simple subset of analyses results columns,
        #    via simple statistics on fitting statistics + functional results, by sample
        #    (not in search of equality - impossible -, but rather under-control differences)
        statCols = ['Qual Bal 3', 'Qual Bal 2', 'Qual Bal 1', 'PDetec', 'Density']

        dfRefStats = dfRef[sampleCols + statCols].groupby(sampleCols).agg(['mean', 'std'])
        dfRefStats.columns = [lvl0 + ' ' + lvl1 for lvl0, lvl1 in dfRefStats.columns]

        dfActStats = dfAct[sampleCols + statCols].groupby(sampleCols).agg(['mean', 'std'])
        dfActStats.columns = [lvl0 + ' ' + lvl1 for lvl0, lvl1 in dfActStats.columns]

        dfDiffStats = ads.DataSet.compareDataFrames(dfRefStats.reset_index(), dfActStats.reset_index(),
                                                    indexCols=sampleCols, subsetCols=dfRefStats.columns.to_list())
        with pd.ExcelWriter(uivu.pWorkDir / f'valtst-optanlys-{mode}-{source}-{table}-stats-and-diff.xlsx') as xlWrtr:
            dfRefStats.to_excel(xlWrtr, sheet_name='ref')
            dfActStats.to_excel(xlWrtr, sheet_name='act')
            dfDiffStats.to_excel(xlWrtr, sheet_name='diff')

        logger.info(f'  - relative diff. between ref. and actual results stats:\n' + dfDiffStats.to_string())

        # e. Save dfDiffStats, to later tweak the thresholds below.
        dfNewDiffStats = dfDiffStats.reset_index()
        dfNewDiffStats.insert(0, 'timestamp', pd.Timestamp.now())
        dfNewDiffStats.insert(1, 'table', table)
        dfNewDiffStats.set_index('timestamp', inplace=True)

        fpnDiffStatHistory = uivu.pTmpDir / f'dont-remove-valtst-optanlys-filsor-diff-stats-history.xlsx'
        dfDiffStatHistory = pd.read_excel(fpnDiffStatHistory, index_col=0) if fpnDiffStatHistory.is_file() \
                            else pd.DataFrame()
        dfDiffStatHistory = pd.concat([dfDiffStatHistory, dfNewDiffStats])
        dfDiffStatHistory.to_excel(fpnDiffStatHistory, index=True)

        # f. Expect less that N% mean difference (after watching stat. history over any runs)
        KPctRelDiffThresh = 25
        meanCols = [col for col in dfDiffStats if col.endswith('mean')]
        minLogRelDiff = dfDiffStats[meanCols].min().min()
        maxPctRelDiff = 100 / 10 ** minLogRelDiff
        logger.info(f"  - minimum relative diff. for all samples, results, stats:"
                    f" 10**-{minLogRelDiff:.3f} <=> {maxPctRelDiff:.2f}%")
        assert maxPctRelDiff < KPctRelDiffThresh

        logger.info(f'... done with auto filter & sort {source} ({table}).')

    @classmethod
    def compareExcelReports(cls, ddfRefReport, ddfActReport, mode='api'):
        """Prerequisite: Reference report generated with either MCDS 7.4 or 6.2"""

        logger.info(f'Comparing reference to actual opt-analysis workbook reports ({mode}) ...')

        # Compare "AFS-Steps"
        dfRef = ddfRefReport['AFS-Steps']
        dfAct = ddfActReport['AFS-Steps']
        cls.compareAFSStepsTables(dfRef, dfAct, mode=mode)

        # Compare "AFSM-<filter & sort scheme>" sheets
        for sheetName in ddfRefReport:
            if not sheetName.startswith('AFSM-'):
                continue
            dfRef = ddfRefReport[sheetName].reset_index()  # Restore NumEchant column (loaded as index)
            dfAct = ddfActReport[sheetName].reset_index()  # Idem
            cls.compareAFSMTables(dfRef, dfAct, mode=mode, table=sheetName)

        # Compare "Synthesis" sheet
        dfRef = ddfRefReport['Synthesis']
        dfAct = ddfActReport['Synthesis']
        cls.compareUnoptResultTables(dfRef, dfAct, mode=mode, source='report', table='Synthesis')
        cls.compareOptResultTables(dfRef, dfAct, mode=mode, source='report', table='Synthesis')

        # Compare "Details" sheet :
        dfRef = ddfRefReport['Details']
        dfAct = ddfActReport['Details']
        cls.compareUnoptResultTables(dfRef, dfAct, mode=mode, source='report', table='Details')
        cls.compareOptResultTables(dfRef, dfAct, mode=mode, source='report', table='Details')

        # Compare "Samples" (if there), and "Analyses", "Analyser" and "Computing platform" sheets
        # (+ some preliminary tweaks to convert workbook to results specs "format")
        ddfRefSpecs = {cls.KRep2SpecNames[n]: sdf.copy() for n, sdf in ddfRefReport.items() if n in cls.KRep2SpecNames}
        ddfRefSpecs['analyser'] = ddfRefSpecs['analyser']['Value']
        ddfRefSpecs['runtime'] = ddfRefSpecs['runtime']['Version']
        ddfActSpecs = {cls.KRep2SpecNames[n]: sdf.copy() for n, sdf in ddfActReport.items() if n in cls.KRep2SpecNames}
        ddfActSpecs['analyser'] = ddfActSpecs['analyser']['Value']
        ddfActSpecs['runtime'] = ddfActSpecs['runtime']['Version']
        cls.compareResultSpecs(ddfRefSpecs, ddfActSpecs, source='report')

        logger.info(f'Done comparing reference to actual opt-analysis workbook reports ({mode}).')

    @pytest.fixture()
    def htmlRefReportLines_fxt(self):

        with open(uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-OptRapport.ExAicMQua-r900m6q3d15.html') as file:
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

    @pytest.fixture()
    def filSorSchemes_fxt(self):

        # Filter and sort sub-reports : schemes to apply
        whichFinalQua = RS.CLCmbQuaBal3  # The optimised criteria
        ascFinalQua = False

        whichBestQua = [RS.CLGrpOrdClTrChi2KSDCv, RS.CLGrpOrdClTrDCv, whichFinalQua,
                       RS.CLGrpOrdClTrQuaChi2, RS.CLGrpOrdClTrQuaKS, RS.CLGrpOrdClTrQuaDCv]

        dupSubset = [RS.CLNObs, RS.CLEffort, RS.CLDeltaAic, RS.CLChi2, RS.CLKS, RS.CLCvMUw, RS.CLCvMCw, RS.CLDCv,
                     RS.CLPDetec, RS.CLPDetecMin, RS.CLPDetecMax, RS.CLDensity, RS.CLDensityMin, RS.CLDensityMax]
        dDupRounds = {RS.CLDeltaAic: 1, RS.CLChi2: 2, RS.CLKS: 2, RS.CLCvMUw: 2, RS.CLCvMCw: 2, RS.CLDCv: 2,
                      RS.CLPDetec: 3, RS.CLPDetecMin: 3, RS.CLPDetecMax: 3, RS.CLDensity: 2, RS.CLDensityMin: 2, RS.CLDensityMax: 2}

        filSorSchemes = [dict(method=RS.filterSortOnExecCode,
                              deduplicate=dict(dupSubset=dupSubset, dDupRounds=dDupRounds),
                              filterSort=dict(whichFinalQua=whichFinalQua, ascFinalQua=ascFinalQua),
                              preselCols=[RS.CLCmbQuaBal1, RS.CLCmbQuaBal2, RS.CLCmbQuaBal3],
                              preselAscs=False, preselThrhs=0.2, preselNum=4),
                         dict(method=RS.filterSortOnExCAicMulQua,
                              deduplicate=dict(dupSubset=dupSubset, dDupRounds=dDupRounds),
                              filterSort=dict(sightRate=87.5, nBestAIC=5, nBestQua=4, whichBestQua=whichBestQua,
                                              nFinalRes=18, whichFinalQua=whichFinalQua, ascFinalQua=ascFinalQua),
                              preselCols=[RS.CLCmbQuaBal1, RS.CLCmbQuaBal2, RS.CLCmbQuaBal3],
                              preselAscs=False, preselThrhs=0.2, preselNum=5),
                         dict(method=RS.filterSortOnExCAicMulQua,
                              deduplicate=dict(dupSubset=dupSubset, dDupRounds=dDupRounds),
                              filterSort=dict(sightRate=90, nBestAIC=4, nBestQua=3, whichBestQua=whichBestQua,
                                              nFinalRes=15, whichFinalQua=whichFinalQua, ascFinalQua=ascFinalQua),
                              preselCols=[RS.CLCmbQuaBal1, RS.CLCmbQuaBal2, RS.CLCmbQuaBal3],
                              preselAscs=False, preselThrhs=0.2, preselNum=4),
                         dict(method=RS.filterSortOnExCAicMulQua,
                              deduplicate=dict(dupSubset=dupSubset, dDupRounds=dDupRounds),
                              filterSort=dict(sightRate=92.5, nBestAIC=3, nBestQua=2, whichBestQua=whichBestQua,
                                              nFinalRes=12, whichFinalQua=whichFinalQua, ascFinalQua=ascFinalQua),
                              preselCols=[RS.CLCmbQuaBal1, RS.CLCmbQuaBal2, RS.CLCmbQuaBal3],
                              preselAscs=False, preselThrhs=0.2, preselNum=3)]

        # The selected one for HTML report
        htmlFilSorScheme = next(sch for sch in filSorSchemes if sch['method'] is RS.filterSortOnExCAicMulQua
                                                                and sch['filterSort']['sightRate'] == 90)

        return filSorSchemes, htmlFilSorScheme, whichFinalQua

    # 7a. Generate HTML and Excel analyses reports through pyaudisam API
    def testReports(self, optAnalyser_fxt, filSorSchemes_fxt, excelRefReport_fxt, htmlRefReportLines_fxt):

        build = True  # Debug only: Set to False to avoid rebuilding the reports, and only check them
        cleanup = True  # Debug only: Set to False to prevent cleaning at the end

        # Pre-requisites : uncleaned analyser work dir (we need the results file and analysis folders).
        optanlr, _, _ = optAnalyser_fxt
        filSorSchemes, htmlFilSorScheme, whichFinalQua = filSorSchemes_fxt
        KExptdMaxAnlysFolders = 120 + 48
        KExptdMinAnlysFolders = KExptdMaxAnlysFolders - 20  # Note: No real theoretical min, less may occur !

        if build:
            logger.info('Checking opt-analyser results presence (build=True => you must once have run testRunCli !)')
            anlysrResFilePath = optanlr.workDir / 'valtests-optanalyses-results.xlsx'
            assert optanlr.workDir.is_dir() and anlysrResFilePath.is_file()

            anlysFolders = [path for path in optanlr.workDir.iterdir() if path.is_dir()]
            assert KExptdMinAnlysFolders <= len(anlysFolders) <= KExptdMaxAnlysFolders

            dfActRes = pd.read_excel(anlysrResFilePath, header=[0, 1, 2], skiprows=[3], index_col=0)
            uivu.checkAnalysisFolders(dfActRes[RS.CLRunFolder],
                                      expectedCount=(KExptdMinAnlysFolders, KExptdMaxAnlysFolders),
                                      anlysKind='opt-analysis')
            logger.info('Done checking opt-analyser results presence: OK.')

            # a. Load results
            rsAct = self.loadResults(optanlr, anlysrResFilePath)
            logger.info(f'Actual results: n={len(rsAct)} =>\n'
                        + rsAct.dfTransData(columns=KResLogCols, lang='en').to_string(min_rows=99, max_rows=99))

            # b. Generate Excel and HTML reports
            # b.i. Super-synthesis sub-report : Selected analysis results columns for the 3 textual columns of the table
            sampleCols = [
                ('header (head)', 'NumEchant', 'Value'),
                ('header (sample)', 'Espèce', 'Value'),
                ('header (sample)', 'Passage', 'Value'),
                ('header (sample)', 'Adulte', 'Value'),
                ('header (sample)', 'Durée', 'Value'),
                RS.CLNTotObs, RS.CLMinObsDist, RS.CLMaxObsDist]

            paramCols = [
                ('header (head)', 'NumAnlys', 'Value'),
                RS.CLParEstKeyFn, RS.CLParEstAdjSer,
                RS.CLParTruncLeft, RS.CLParTruncRight, RS.CLParModFitDistCuts]

            resultCols = [
                RS.CLRunStatus,
                RS.CLNObs, RS.CLEffort, RS.CLSightRate, RS.CLNAdjPars,
                RS.CLAic, RS.CLChi2, RS.CLKS, RS.CLDCv,
                RS.CLCmbQuaBal1, RS.CLCmbQuaBal2, RS.CLCmbQuaBal3,

                RS.CLEswEdr, RS.CLPDetec,
                RS.CLDensity, RS.CLDensityMin, RS.CLDensityMax,
                RS.CLNumber, RS.CLNumberMin, RS.CLNumberMax]

            # b.ii. Synthesis and Filter & sort sub-reports : Selected analysis results columns for the table
            synthCols = [
                ('header (head)', 'NumEchant', 'Value'),
                ('header (sample)', 'Espèce', 'Value'),
                ('header (sample)', 'Passage', 'Value'),
                ('header (sample)', 'Adulte', 'Value'),
                ('header (sample)', 'Durée', 'Value'),

                ('header (head)', 'NumAnlys', 'Value'),
                RS.CLParEstKeyFn, RS.CLParEstAdjSer,
                # RS.CLParEstSelCrit, RS.CLParEstCVInt,
                RS.CLParTruncLeft, RS.CLParTruncRight, RS.CLParModFitDistCuts,

                RS.CLOptimTruncFlag,  # Mostly useful for ref/actual comparison (for sorting out opt. and unopt. stuff)

                RS.CLNTotObs, RS.CLNObs, RS.CLNTotPars, RS.CLEffort,
                RS.CLDeltaAic, RS.CLChi2, RS.CLKS, RS.CLCvMUw, RS.CLCvMCw, RS.CLDCv,
                RS.CLSightRate,
                RS.CLCmbQuaBal1, RS.CLCmbQuaBal2, RS.CLCmbQuaBal3,
                RS.CLCmbQuaChi2, RS.CLCmbQuaKS, RS.CLCmbQuaDCv,

                RS.CLPDetec, RS.CLPDetecMin, RS.CLPDetecMax,
                RS.CLDensity, RS.CLDensityMin, RS.CLDensityMax,
                RS.CLNumber, RS.CLNumberMin, RS.CLNumberMax,

                RS.CLGrpOrdSmTrAic,
                RS.CLGrpOrdClTrChi2KSDCv,  # RS.CLGrpOrdClTrChi2,
                RS.CLGrpOrdClTrDCv,
                RS.CLGrpOrdClTrQuaBal1, RS.CLGrpOrdClTrQuaBal2, RS.CLGrpOrdClTrQuaBal3, RS.CLGrpOrdClTrQuaChi2,
                RS.CLGrpOrdClTrQuaKS, RS.CLGrpOrdClTrQuaDCv,
                RS.CLGblOrdChi2KSDCv, RS.CLGblOrdQuaBal1, RS.CLGblOrdQuaBal2, RS.CLGblOrdQuaBal3,
                RS.CLGblOrdQuaChi2, RS.CLGblOrdQuaKS, RS.CLGblOrdQuaDCv,
                RS.CLGblOrdDAicChi2KSDCv]

            # b.iii. Filter & sort sub-reports : schemes to apply
            # See filSorSchemes_fxt fixture

            # b.iv. Sorting columns for all the sub-reports
            # Super-synthesis, synthesis and detail tables, HTML or Excel : sort parameters.
            sortCols = [('header (head)', 'NumEchant', 'Value'), whichFinalQua]
            sortAscend = [True, False]

            # b.v. Report object
            report = ads.MCDSResultsFilterSortReport(
                             resultsSet=rsAct,
                             title="PyAuDiSam Validation : Analyses with optimised truncations",
                             subTitle="Auto-selection of best results",
                             description='Automated filtering et sorting : method "{fsId}" ; after '
                                         'easy and parallel run through MCDSTruncationOptAnalyser',
                             anlysSubTitle='Analyses details',
                             lang='en', keywords='pyaudisam, validation, optimisation',
                             superSynthPlotsHeight=280, plotImgSize=(512, 280),
                             sampleCols=sampleCols, paramCols=paramCols,
                             resultCols=resultCols, synthCols=synthCols,
                             sortCols=sortCols, sortAscend=sortAscend,
                             filSorSchemes=filSorSchemes,
                             tgtFolder=optanlr.workDir,
                             tgtPrefix='valtests-optanalyses-report')

            # b.vi. Excel report
            xlsxRep = report.toExcel()
            logger.info('Excel report: ' + pl.Path(xlsxRep).resolve().as_posix())

            # b.vii. HTML report
            logger.info(', '.join(rsAct.filSorIdMgr.dFilSorSchemes.keys()) + '\n=> ' + rsAct.filSorSchemeId(htmlFilSorScheme))
            htmlRep = report.toHtml(htmlFilSorScheme, rebuild=False)
            logger.info('HTML report: ' + pl.Path(htmlRep).resolve().as_posix())

        else:
            xlsxRep = optanlr.workDir / 'valtests-optanalyses-report.xlsx'

            schemeId = ads.FilterSortSchemeIdManager().schemeId(htmlFilSorScheme)
            htmlRep = optanlr.workDir / f'valtests-optanalyses-report.{schemeId}.html'

            logger.info('Checking opt-analyser reports presence (build=False => you must once have run with True !)')
            assert optanlr.workDir.is_dir() and xlsxRep.is_file() and htmlRep.is_file()
            anlysFolders = [path for path in optanlr.workDir.iterdir() if path.is_dir()]
            assert KExptdMinAnlysFolders <= len(anlysFolders) <= KExptdMaxAnlysFolders
            logger.info('Done checking opt-analyser reports presence: OK.')

        # c. Load generated Excel report and compare it to reference one
        ddfRefRep = excelRefReport_fxt

        ddfActRep = pd.read_excel(xlsxRep, sheet_name=None, index_col=0)
        self.compareExcelReports(ddfRefRep, ddfActRep, mode='api')

        # c. Load generated HTML report and compare it to reference one
        htmlRefRepLines = htmlRefReportLines_fxt

        with open(htmlRep) as file:
            htmlActRepLines = file.readlines()
        # TODO self.compareHtmlReports(htmlRefRepLines, htmlActRepLines)

        # e. Cleanup generated report (well ... partially at least) for clearing next function's ground
        if cleanup:
            pl.Path(xlsxRep).unlink()
            pl.Path(htmlRep).unlink()

        # f. Done.
        logger.info(f'PASS testReports: MCDSResultsFilterSortReport ctor, toExcel, toHtml')

    # 7b. Generate HTML and Excel analyses reports through pyaudisam command line
    def testReportsCli(self, excelRefReport_fxt, filSorSchemes_fxt, htmlRefReportLines_fxt):

        build = True  # Debug only: Set to False to avoid rebuilding the report (only compare)

        # a. Report "through the commande line"
        if build:
            argv = f'-p {uivu.pTestDir.as_posix()}/valtests-ds-params.py -w {uivu.pWorkDir.as_posix()}' \
                   ' -n --optreports excel,html:mqua-r90 -u'.split()
            rc = ads.main(argv, standaloneLogConfig=False)
            logger.info(f'CLI run: rc={rc}')

        # b. Load generated Excel report and compare it to the reference one
        ddfActRep = pd.read_excel(uivu.pWorkDir / 'valtests-optanalyses-report.xlsx', sheet_name=None, index_col=0)

        ddfRefRep = excelRefReport_fxt
        self.compareExcelReports(ddfRefRep, ddfActRep, mode='cli')

        # c. Load generated HTML report and compare it to reference one
        _, htmlFilSorScheme, _ = filSorSchemes_fxt
        schemeId = ads.FilterSortSchemeIdManager().schemeId(htmlFilSorScheme)
        htmlRep = uivu.pWorkDir / f'valtests-optanalyses-report.{schemeId}.html'
        with open(htmlRep) as file:
            htmlActRepLines = file.readlines()

        htmlRefRepLines = htmlRefReportLines_fxt
        # TODO self.compareHtmlReports(htmlRefRepLines, htmlActRepLines)

        # d. No cleanup: let the final test class cleaner operate: _inifinalizeClass()

        # e. Done.
        logger.info(f'PASS testReports: main, MCDSResultsFullReport ctor, toExcel, toHtml (command line mode)')
