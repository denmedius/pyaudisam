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

# Automated validation tests for "analyser" submodule: MCDSPreAnalyser class,
# and "reports" submodule: MCDSResultsPreReport class and bases (adapted from valtest.ipynb notebook)

# To run : simply run "pytest" and see ./tmp/val.pnr.{datetime}.log for detailed test log

import time
import pathlib as pl
import shutil

import numpy as np
import pandas as pd

import pytest

import pyaudisam as ads

import unintval_utils as uivu


# Setup local logger.
logger = uivu.setupLogger('val.pnr', level=ads.DEBUG)

# Some constants
RS = ads.MCDSPreAnalysisResultsSet
KResLogCols = [
    ('header (head)', 'NumEchant', 'Value'),
    ('header (head)', 'NumAnlys', 'Value'),
    ('header (tail)', 'AbrevAnlys', 'Value'),
    RS.CLNTotObs,
    RS.CLMinObsDist, RS.CLMaxObsDist,
    RS.CLNObs,
    RS.CLRunStatus,
    RS.CLCmbQuaBal3,
    RS.CLDensity, RS.CLDensityMin, RS.CLDensityMax,
]


@pytest.mark.valtests
@pytest.mark.parametrize("sampleSpecMode", ['implicit', 'explicit'])
class TestMcdsPreAnalyser:

    # Set to False to skip final cleanup (useful for debugging)
    KFinalCleanup = True

    # Class and test function initializers / finalizers ###########################
    @pytest.fixture(autouse=True, scope='class')
    def _inifinalizeClass(self):

        KWhat2Test = 'pre-analyser'

        uivu.logBegin(what=KWhat2Test)

        # Set up a clear ground before starting
        uivu.setupWorkDir('val-panlr', cleanup=self.KFinalCleanup)

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

    # II. Run and report pre-analyses
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

        logger.info(f'Invidual. sightings: n={len(dfObsIndiv)} =>\n' + dfTransects.to_string(min_rows=30, max_rows=30))

        logger.info(f'Done preparing individual. sightings.\n')

        return dfObsIndiv, dfTransects

    @staticmethod
    def sampleSpecMode(implicit):
        return 'im' if implicit else 'ex'

    @pytest.fixture()
    def preAnalyser_fxt(self, sampleSpecMode, inputDataSet_fxt):

        logger.info(f'Preparing pre-analyser ...')

        # ## 0. Data description
        transectPlaceCols = ['Point']
        passIdCol = 'Passage'
        effortCol = 'Effort'

        sampleDecCols = [effortCol, 'Distance']

        sampleNumCol = 'NumEchant'
        sampleSelCols = ['Espèce', passIdCol, 'Adulte', 'Durée']

        sampleAbbrevCol = 'AbrevEchant'

        speciesAbbrevCol = 'AbrevEsp'

        dSurveyArea = dict(Zone='ACDC', Surface='2400')

        # ## 1. Individuals data set
        # ## 2. Actual transects
        # See parameters

        # ## 4A. Really run pre-analyses
        # ### a. MCDSPreAnalyser object
        areSpecsImplicit = sampleSpecMode == 'implicit'
        dfObsIndiv, dfTransects = inputDataSet_fxt
        preAnlysr = \
            ads.MCDSPreAnalyser(dfObsIndiv, dfTransects=dfTransects, dSurveyArea=dSurveyArea,
                                transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                                sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols,
                                sampleSpecCustCols=[speciesAbbrevCol],
                                abbrevCol=sampleAbbrevCol, abbrevBuilder=uivu.sampleAbbrev, sampleIndCol=sampleNumCol,
                                distanceUnit='Meter', areaUnit='Hectare',
                                surveyType='Point', distanceType='Radial', clustering=False,
                                resultsHeadCols=dict(before=[sampleNumCol], sample=sampleSelCols,
                                                     after=([] if areSpecsImplicit else [speciesAbbrevCol])
                                                           + [sampleAbbrevCol]),
                                workDir=uivu.pWorkDir, logProgressEvery=5)

        assert preAnlysr.specs == dict(Zone='ACDC',
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
                                       defDiscrDistCuts=None)

        logger.info(f'Pre-analyser specs:\n{preAnlysr.specs}')

        logger.info(f'Done preparing pre-analyser.\n')

        logger.info(f'Preparing {sampleSpecMode} sample specs ...')

        # a. Implicit variants
        varEspeces = ['Sylvia atricapilla', 'Turdus merula', 'Luscinia megarhynchos']  # 1 species variant per species
        varPassages = ['b', 'a+b']  # Pass b or a+b => 2 variants
        varAdultes = ['m']  # Males only => 1 variant
        varDurees = ['5mn', '10mn']  # first 5mn, and then all 10mn => 2 variants

        dImplSampleSpecs = {'Espèce': varEspeces, 'Passage': varPassages, 'Adulte': varAdultes, 'Durée': varDurees}
        implSampleSpecs = dict(_impl=dImplSampleSpecs)

        logger.info(f'Implicit sample specs: {implSampleSpecs}')

        # b. Explicit variants
        dfExplSampleSpecs = None
        if not areSpecsImplicit:
            dfExplSampleSpecs = ads.Analyser.explicitVariantSpecs(implSampleSpecs)
            # Just the same, but less generic.
            # dfExplSampleSpecs = ads.Analyser.explicitPartialVariantSpecs(dImplSampleSpecs)

            # Added neutral pass-through column (from sample specs to results)
            speciesAbbrevCol = 'AbrevEsp'
            dfExplSampleSpecs[speciesAbbrevCol] = \
                dfExplSampleSpecs['Espèce'].apply(lambda s: ''.join(m[:4] for m in s.split()))

            logger.info(f'Explicit sample specs:\n{dfExplSampleSpecs.to_string()}')

        # c. Check pre-analyses specs
        dfExplSampleSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols, verdict, reasons = \
            preAnlysr.explicitParamSpecs(dfExplParamSpecs=dfExplSampleSpecs if not areSpecsImplicit else None,
                                         implParamSpecs=implSampleSpecs if areSpecsImplicit else None,
                                         dropDupes=True, check=True)

        logger.info('Sample spec. explicitations:')
        logger.info(f'* verdict: {verdict}')
        logger.info(f'* reasons: {reasons}')
        logger.info(f'* userParamSpecCols: n={len(userParamSpecCols)} => {userParamSpecCols}')
        logger.info(f'* intParamSpecCols: n={len(intParamSpecCols)} => {intParamSpecCols}')
        logger.info(f'* unmUserParamSpecCols: n={len(unmUserParamSpecCols)} => {unmUserParamSpecCols}')

        logger.info(f'Explicitated sample specs: n={len(dfExplSampleSpecs)} =>\n'
                    + dfExplSampleSpecs.to_string(min_rows=30, max_rows=30))

        assert len(dfExplSampleSpecs) == 12
        assert userParamSpecCols == []  # No analysis params here (auto. generated by PreAnalyser)
        assert intParamSpecCols == []  # Idem
        assert unmUserParamSpecCols == []
        assert verdict
        assert not reasons

        # Done.
        logger.info(f'Done preparing {sampleSpecMode} sample specs.\n')

        return (preAnlysr, implSampleSpecs) if areSpecsImplicit else (preAnlysr, dfExplSampleSpecs)

    @pytest.fixture()
    def expectedSampleStats_fxt(self):

        return pd.DataFrame(columns=('NumEchant', 'Distance Min', 'Distance Max', 'NTot Obs'),
                            data=[(0, 13.7, 488.2, 137),
                                  (1, 8.4, 488.2, 207),
                                  (2, 10.8, 488.2, 261),
                                  (3, 1.2, 511.4, 388),
                                  (4, 2.9, 714.1, 131),
                                  (5, 2.9, 786.0, 220),
                                  (6, 2.9, 714.1, 231),
                                  (7, 2.9, 786.0, 400),
                                  (8, 32.7, 1005.4, 57),
                                  (9, 32.7, 1005.4, 84),
                                  (10, 14.4, 1005.4, 107),
                                  (11, 14.4, 1005.4, 156)]).set_index('NumEchant')

    def testComputeSampleStats(self, sampleSpecMode, preAnalyser_fxt, expectedSampleStats_fxt):

        if sampleSpecMode != 'implicit':
            msg = 'testComputeSampleStats(explicit): skipped, as not relevant'
            logger.info(msg)
            pytest.skip(msg)  # Raises an exception => function execution stops here.

        preAnlysr, sampleSpecs = preAnalyser_fxt
        dfSampleStats = preAnlysr.computeSampleStats(implSampleSpecs=sampleSpecs, sampleDistCol='Distance')
        logger.info(f'Sample stats: n={len(dfSampleStats)} =>\n{dfSampleStats.to_string(min_rows=30, max_rows=30)}')

        dfSampleStats['Distance Min'] = dfSampleStats['Distance Min'].round(1)
        dfSampleStats['Distance Max'] = dfSampleStats['Distance Max'].round(1)
        dfSampleStats.set_index('NumEchant', inplace=True)
        dfSampleStats = dfSampleStats[['Distance Min', 'Distance Max', 'NTot Obs']]

        dfExptdStats = expectedSampleStats_fxt
        assert dfSampleStats.compare(dfExptdStats).empty

        logger.info(f'PASS testComputeSampleStats: computeSampleStats')

    # Only minimalistic checks done here, as already more deeply tested in test_unint_engine.py
    @staticmethod
    def checkExportedDsInputData(exportDir, sampleSpecs, sampleSpecMode):

        areSpecsImplicit = sampleSpecMode == 'implicit'

        # ii. Check list of generated files
        expdGenFileNames = []
        if areSpecsImplicit:
            for esp in sampleSpecs['_impl']['Espèce']:
                for pas in sampleSpecs['_impl']['Passage']:
                    for ad in sampleSpecs['_impl']['Adulte']:
                        for dur in sampleSpecs['_impl']['Durée']:
                            sampAbbrv = uivu.sampleAbbrev(pd.Series({'Espèce': esp, 'Passage': pas,
                                                                     'Adulte': ad, 'Durée': dur}))
                            expdGenFileNames.append(f'{sampAbbrv}-dist.txt')
        else:
            for _, sSampSpec in sampleSpecs.iterrows():
                sampAbbrv = uivu.sampleAbbrev(sSampSpec[['Espèce', 'Passage', 'Adulte', 'Durée']])
                expdGenFileNames.append(f'{sampAbbrv}-dist.txt')
        assert all(fpn.name in expdGenFileNames for fpn in exportDir.glob('*-dist.txt'))

        # iii. Check first generated file
        sampAbbrv = uivu.sampleAbbrev(pd.Series({'Espèce': 'Luscinia megarynchos', 'Passage': 'a+b',
                                                 'Adulte': 'm', 'Durée': '10mn'}))
        fpnSamp = exportDir / f'{sampAbbrv}-dist.txt'
        dfSampDist = pd.read_csv(fpnSamp, sep='\t', decimal=',')
        logger.info(f'{fpnSamp.as_posix()}:\n{dfSampDist.to_string(min_rows=30, max_rows=30)}')

        assert len(dfSampDist) == 182
        assert dfSampDist.columns.tolist() == ['Region*Label', 'Region*Area', 'Point transect*Label',
                                               'Point transect*Survey effort', 'Observation*Radial distance']
        assert dfSampDist['Point transect*Label'].nunique() == 96
        assert all(dfSampDist['Region*Label'].unique() == ['ACDC'])
        assert all(dfSampDist['Region*Area'].unique() == [2400])
        assert dfSampDist['Point transect*Survey effort'].sum() == 362
        assert dfSampDist['Observation*Radial distance'].isnull().sum() == 26

        # iv. Cleanup all generated files
        for fpnSamp in exportDir.glob('*-dist.txt'):
            fpnSamp.unlink()

    # ### c. Generate input files for manual analyses with Distance GUI (not needed for pre-analyses)
    #        through pyaudisam API
    def testExportDsInputData(self, sampleSpecMode, preAnalyser_fxt):

        # i. Export distance files
        logger.info(f'Exporting Distance files ({sampleSpecMode} sample specs) ...')

        areSpecsImplicit = sampleSpecMode == 'implicit'
        preAnlysr, sampleSpecs = preAnalyser_fxt
        preAnlysr.exportDSInputData(implSampleSpecs=sampleSpecs if areSpecsImplicit else None,
                                    dfExplSampleSpecs=sampleSpecs if not areSpecsImplicit else None,
                                    format='Distance')

        # ii. Check exported files
        self.checkExportedDsInputData(preAnlysr.workDir, sampleSpecs, sampleSpecMode)

        logger.info(f'PASS testExportDsInputData: exportDsInputData({sampleSpecMode} sample specs)')

    # ### c. Generate input files for manual analyses with Distance GUI (not needed for pre-analyses),
    #        through pyaudisam command line interface (implicit mode only, see valtests-ds-params.py)
    def testExportDsInputDataCli(self, sampleSpecMode, preAnalyser_fxt):

        if sampleSpecMode != 'implicit':
            msg = 'testExportDsInputDataCli(explicit): skipped, as not relevant'
            logger.info(msg)
            pytest.skip(msg)  # Raises an exception => function execution stops here.

        # i. Export distance files
        logger.info(f'Exporting Distance files (command line mode) ...')

        testPath = pl.Path(__file__).parent
        preAnlysr, sampleSpecs = preAnalyser_fxt
        workPath = preAnlysr.workDir

        # a. Export files "through the commande line"
        argv = f'-p {testPath.as_posix()}/valtests-ds-params.py -w {workPath.as_posix()} -n --distexport -u'.split()
        rc = ads.main(argv, standaloneLogConfig=False)
        logger.info(f'CLI run: rc={rc}')

        # ii. Check exported files
        self.checkExportedDsInputData(workPath, sampleSpecs, sampleSpecMode)

        logger.info(f'PASS testExportDsInputDataCli: exportDsInputData(command line mode)')

    @staticmethod
    def loadResults(preAnlysr, filePath, postComputed=False):

        logger.info(f'Loading pre-results from {filePath.as_posix()} ...')

        rsRes = preAnlysr.setupResults()
        rsRes.fromFile(filePath, postComputed=postComputed)

        assert isinstance(rsRes, ads.MCDSPreAnalysisResultsSet)

        return rsRes

    @pytest.fixture()
    def refResults_fxt(self, sampleSpecMode, preAnalyser_fxt):

        logger.info(f'Preparing reference pre-results ({sampleSpecMode} mode) ...')

        # Prevent re-postComputation as this ref. file is old, with now missing computed cols
        preAnlysr, _ = preAnalyser_fxt
        rsRef = self.loadResults(preAnlysr, uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-PreResultats.ods',
                                 postComputed=True)

        logger.info(f'Reference results: n={len(rsRef)} =>\n'
                    + rsRef.dfTransData(columns=KResLogCols, lang='en').to_string(min_rows=99, max_rows=99))

        return rsRef

    @classmethod
    def compareResultSpecs(cls, dsdfRefSpecs, dsdfActSpecs, source='results'):
        """dsdfXxx : dict(DataFrame | Series)
        => expecting Series for 'analyser' and 'runtime' keys, DataFrames otherwise"""

        # a. Samples (DataFrame)
        #    Note: Remove neutral pass-through column (from sample specs to results)
        #          from actual sample specs if here and not present in ref.
        logger.info(f'* {source} specs: samples ...')

        dfRefSmpSpecs = dsdfRefSpecs['samples']
        logger.info(f'  - ref. sample specs: n={len(dfRefSmpSpecs)} =>\n' + dfRefSmpSpecs.to_string())
        dfActSmpSpecs = dsdfActSpecs['samples']
        if 'AbrevEsp' not in dfRefSmpSpecs:
            dfActSmpSpecs.drop(columns=['AbrevEsp'], inplace=True, errors='ignore')
        logger.info(f'  - actual sample specs: n={len(dfActSmpSpecs)} =>\n' + dfActSmpSpecs.to_string())

        dfComp = dfRefSmpSpecs.compare(dfActSmpSpecs)
        logger.info(f'  - sample specs comparison: n={len(dfComp)} =>\n' + dfComp.to_string())
        assert dfComp.empty

        # b. Specs: Models (DataFrame)
        logger.info(f'* {source} specs: models ...')

        dfRefModSpecs = dsdfRefSpecs['models']
        logger.info(f'Ref. model specs: n={len(dfRefModSpecs)} =>\n' + dfRefModSpecs.to_string())
        dfActModSpecs = dsdfActSpecs['models']
        logger.info(f'Actual model specs: n={len(dfActModSpecs)} =>\n' + dfActModSpecs.to_string())
        assert dfRefModSpecs.compare(dfActModSpecs).empty

        # c. Analyser (Series)
        logger.info(f'* {source} specs: analyser ...')

        sRefAnrSpecs = dsdfRefSpecs['analyser']
        logger.info(f'  - ref. analyser specs: n={len(sRefAnrSpecs)} =>\n' + sRefAnrSpecs.to_string())
        sActAnrSpecs = dsdfActSpecs['analyser']
        logger.info(f'  - actual analyser specs: n={len(sActAnrSpecs)} =>\n' + sActAnrSpecs.to_string())
        dfComp = sRefAnrSpecs.compare(sActAnrSpecs)

        logger.info(f'  - analyser specs comparison: n={len(dfComp)} =>\n' + dfComp.to_string())
        assert dfComp.empty

        # d. Run-time (Series): whatever ref, expect a specific up-to-date list of item names, but nothing more
        # (values may vary, 'cause they are mostly software versions: it's OK)
        logger.info(f'* {source} specs: run platform ...')

        sRefRunSpecs = dsdfRefSpecs['runtime']
        logger.info(f'  - ref. runtime specs: n={len(sRefRunSpecs)} =>\n' + sRefRunSpecs.to_string())
        sActRunSpecs = dsdfActSpecs['runtime']
        logger.info(f'  - actual runtime specs: n={len(sActRunSpecs)} =>\n' + sActRunSpecs.to_string())
        assert set(sActRunSpecs.index) \
               == {'os', 'processor', 'python', 'numpy', 'pandas', 'zoopt', 'matplotlib',
                   'jinja2', 'pyaudisam', 'MCDS engine', 'MCDS engine version'}

        logger.info(f'  ... done with pre-analyses {source} specs.')

    @classmethod
    def compareResults(cls, rsRef, rsAct):
        """Prerequisite: Reference results generated with either MCDS 7.4 or 6.2"""

        logger.info('Comparing reference to actual pre-results ...')

        # 1. Actual pre-analysis results: "all-results" sheet
        # 1a. Comparison index = analysis "Id": sample Id columns and analysis indexes.
        indexPreCols = [col for col in rsAct.miCustomCols.to_list() if '(sample)' in col[0]] \
                       + [('parameters', 'estimator key function', 'Value'),
                          ('parameters', 'estimator adjustment series', 'Value')]

        # 1b. Comparison columns: ignore ...
        #   - sample Id columns and analysis indexes (used as comparison index = analysis "Id")
        #   - 'run output' chapter results (start time, elapsed time, run folder ... always different)
        #   - text columns (not supported by ResultsSet.compare).
        subsetPreCols = [col for col in rsAct.dfData.columns.to_list()
                         if col in rsRef.columns
                         and col not in indexPreCols + [col for col in rsAct.miCustomCols.to_list()
                                                        if '(sample)' not in col[0]]
                         + [('parameters', 'estimator selection criterion', 'Value'),
                            ('parameters', 'CV interval', 'Value'),
                            ('run output', 'start time', 'Value'),
                            ('run output', 'elapsed time', 'Value'),
                            ('run output', 'run folder', 'Value'),
                            ('detection probability', 'key function type', 'Value'),
                            ('detection probability', 'adjustment series type', 'Value'),
                            ('detection probability', 'Delta AIC', 'Value'),
                            ('density/abundance', 'density of animals', 'Delta Cv')]]

        # 1c. Compare
        dfDiff = rsRef.compare(rsAct, indexCols=indexPreCols, subsetCols=subsetPreCols,
                               noneIsNan=True, dropCloser=13, dropNans=True)

        logger.info(f'Diff. to reference (relative): n={len(dfDiff)} =>\n'
                    + dfDiff.to_string(min_rows=30, max_rows=30))
        if not dfDiff.empty:
            dfDiff.to_excel(uivu.pWorkDir / 'res-comp-13.xlsx')

        assert dfDiff.empty, 'Oh oh ... some unexpected differences !'

        # 1d. To be perfectly honest ... there may be some 10**-14/-16 glitches (due to worksheet I/O ?) ... or not.
        dfComp = rsRef.compare(rsAct, indexCols=indexPreCols, subsetCols=subsetPreCols,
                               noneIsNan=True, dropNans=True)
        dfComp = dfComp[(dfComp != np.inf).all(axis='columns')]

        logger.info(f'Diff. to reference (absolute): n={len(dfComp)} =>\n'
                    + dfComp.to_string(min_rows=30, max_rows=30))

        # 2. Specs
        cls.compareResultSpecs(rsRef.specs, rsAct.specs, source='results')

    # ### d. Run pre-analyses through pyaudisam API
    # Performance figures:
    # Ruindows 10 laptop with PCI-e SSD, "optimal performances" power scheme, Python 3.8 :
    # * 4-HT-core i5-8350U:
    #   * 2021 (precise date ?): 50s to ~1mn10s elapsed for 12 samples, 6-12 threads (N=?)
    # * 6-core i7-10750H (HT off):
    #   * 2022-01-17, 2023-11-02: 39-40s elapsed for 12 samples, 6-12 threads (N=5)
    # Ruindows 11 laptop with PCI-e SSD, "high performances" power scheme, Python 3.8 :
    # * 6-HT-core i7-10850H (HT on):
    #   * 2024-03-02: 40s elapsed for 12 samples, 6 threads (N=1)
    #   * 2024-03-02: 39s elapsed for 12 samples, 12 threads (N=1)
    def testRun(self, sampleSpecMode, preAnalyser_fxt, refResults_fxt):

        postCleanup = True  # Debug only: Set to False to prevent cleaning at the end

        # i. Cleanup test folder (Note: avoid any Ruindows shell or explorer inside this folder !)
        preAnlysr, sampleSpecs = preAnalyser_fxt
        if preAnlysr.workDir.exists():
            shutil.rmtree(preAnlysr.workDir)

        # ii. Run and measure performance
        threads = 6
        logger.info(f'Running pre-analyses: {sampleSpecMode} sample specs, {threads} parallel threads ...')

        # Model fall-down strategy
        # Note: For real bird study analyses, you'll probably avoid NEXPON key function
        #       (a model with no shoulder: g'(0) << 1).
        #       And also HERMITE adjustment series (overkill fitting).
        modelStrategy = [dict(keyFn=kf, adjSr=js, estCrit='AIC', cvInt=95)
                         for js in ['COSINE', 'POLY', 'HERMITE']
                         for kf in ['HNORMAL', 'HAZARD', 'UNIFORM', 'NEXPON']]

        # BE CAREFUL: time.process_time() uses relative time for comparison only of codes among the same environment
        # NOT A REAL TIME reference
        areSpecsImplicit = sampleSpecMode == 'implicit'

        start = time.perf_counter()
        preAnlysr, sampleSpecs = preAnalyser_fxt
        rsAct = preAnlysr.run(implSampleSpecs=sampleSpecs if areSpecsImplicit else None,
                              dfExplSampleSpecs=sampleSpecs if not areSpecsImplicit else None,
                              dModelStrategy=modelStrategy, threads=threads)
        end = time.perf_counter()

        logger.info(f'Elapsed time={end - start:.2f}s')
        logger.info(f'Actual results: n={len(rsAct)} =>\n'
                    + rsAct.dfTransData(columns=KResLogCols, lang='en').to_string(min_rows=99, max_rows=99))

        # Export results
        rsAct.toOpenDoc(preAnlysr.workDir / f'valtests-preanalyses-results-{sampleSpecMode}api.ods')
        # rsAct.toExcel(preAnlysr.workDir / f'valtests-preanalyses-results-{sampleSpecMode}api-fr.xlsx', lang='fr')

        # ### e. Check results: Compare to reference
        # i. Check presence of neutral and pass-through column in explicit spec. mode
        #    (it should have effectively passed through :-)
        speciesAbbrevCol = 'AbrevEsp'
        logger.debug('dfTransData(en).columns: ' + str(rsAct.dfTransData('en').columns))
        assert areSpecsImplicit or speciesAbbrevCol in rsAct.dfTransData('en').columns

        # ii. Compare to reference results
        rsRef = refResults_fxt
        self.compareResults(rsRef, rsAct)

        # f. Minimal check of pre-analysis folders
        logger.info('Checking pre-analysis folders (minimal) ...')
        uivu.checkAnalysisFolders(rsAct.dfTransData('en').RunFolder, expectedCount=12, anlysKind='pre-analysis')

        # g. Cleanup analyser (analysis folders, not workbook results files)
        if postCleanup:
            preAnlysr.cleanup()
        else:
            logger.warning('NOT cleaning up the pre-analyser: this is not the normal testing scheme !')

        # h. Done.
        logger.info(f'PASS testRun: run({sampleSpecMode} sample specs), cleanup')

    # Run pre-analyses through pyaudisam command line interface
    # (implicit mode only, see valtests-ds-params.py)
    def testRunCli(self, sampleSpecMode, preAnalyser_fxt, refResults_fxt):

        if sampleSpecMode != 'implicit':
            msg = 'testRunCli(explicit): skipped, as not relevant'
            logger.info(msg)
            pytest.skip(msg)  # Raises an exception => function execution stops here.

        # a. Cleanup test folder (Note: avoid any Ruindows shell or explorer inside this folder !)
        preAnlysr, _ = preAnalyser_fxt
        shutil.rmtree(preAnlysr.workDir)

        # b. Run "through the commande line"
        argv = f'-p {uivu.pTestDir.as_posix()}/valtests-ds-params.py -w {preAnlysr.workDir.as_posix()}' \
               ' -n --preanalyses -u'.split()
        rc = ads.main(argv, standaloneLogConfig=False)
        logger.info(f'CLI run: rc={rc}')

        # c. Load pre-results
        rsAct = self.loadResults(preAnlysr, preAnlysr.workDir / 'valtests-preanalyses-results.xlsx')
        logger.info(f'Actual results: n={len(rsAct)} =>\n'
                    + rsAct.dfTransData(columns=KResLogCols, lang='en').to_string())

        # d. Compare to reference.
        rsRef = refResults_fxt
        self.compareResults(rsRef, rsAct)

        # e. Minimal check of pre-analysis folders
        uivu.checkAnalysisFolders(rsAct.dfTransData('en').RunFolder, expectedCount=12, anlysKind='pre-analysis')

        # f. Don't clean up work folder / analysis folders : needed for report generations below

        # g. Done.
        logger.info(f'PASS testRunCli: main, run (command line mode)')

    @staticmethod
    def loadWorkbookReport(filePath):

        logger.info(f'Loading workbook pre-report from {filePath.as_posix()} ...')

        return pd.read_excel(filePath, sheet_name=None, index_col=0)

    @pytest.fixture()
    def workbookRefReport_fxt(self):

        return self.loadWorkbookReport(uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-PreRapport.ods')

    KRep2SpecNames = {'Samples': 'samples', 'Models': 'models', 'Analyser': 'analyser',
                      'Computing platform': 'runtime'}

    @classmethod
    def compareReports(cls, ddfRefReport, ddfActReport, mode='api', kind='workbook'):
        """Prerequisite: Reference report generated with either MCDS 7.4 or 6.2"""

        assert kind in {'workbook', 'html'}

        logger.info(f'Comparing reference to actual pre-analysis {kind} report ({mode}) ...')

        logger.info(f'* {kind} report tables ...')
        # logger.info('  - expected tables: ' + ', '.join(expectTables or [None]))
        logger.info('  - ref. report tables: ' + ', '.join(ddfRefReport.keys()))
        logger.info('  - actual report tables: ' + ', '.join(ddfActReport.keys()))

        assert set(ddfRefReport.keys()) == set(ddfActReport.keys()), 'Missing ref. tables in actual pre-report'

        if kind == 'html':

            # Compare "Title" table when html kind
            logger.info(f'* {kind} pre-report title ...')
            logger.info(f'  - ref. : n={len(ddfRefReport["Title"])} =>\n' + ddfRefReport['Title'].to_string())
            logger.info(f'  - actual : n={len(ddfActReport["Title"])} =>\n' + ddfActReport['Title'].to_string())
            assert ddfRefReport['Title'].compare(ddfActReport['Title']).empty

            # Compare "SuperSynthesis" table when html kind
            logger.info(f'* {kind} super-synthesis table ...')
            dfRef = ddfRefReport['SuperSynthesis'].reset_index()  # Restore NumEchant column (loaded as index)
            dfAct = ddfActReport['SuperSynthesis'].reset_index()  # Idem

            # a. Compare all the string columns and a few "no precision issue" more.
            idCols = ['NumEchant', 'Espèce', 'Passage', 'Adulte', 'Durée']
            simpleCompCols = idCols
            simpleCompCols += ['NTot Obs', 'Mod Key Fn', 'Mod Adj Ser']
            assert dfRef[simpleCompCols].set_index('NumEchant') \
                     .compare(dfAct[simpleCompCols].set_index('NumEchant')).empty

            # b. Compare other (all numerical) columns with a small margin (1e-14 relative diff)
            otherCompCols = [col for col in dfRef if col not in simpleCompCols]
            otherCompColsStr = '\n    . '.join(str(t) for t in otherCompCols)
            logger.info(f'  - data columns for comparison:\n    . ' + otherCompColsStr)
            dfDiff = ads.DataSet.compareDataFrames(dfLeft=dfRef, dfRight=dfAct,
                                                   subsetCols=otherCompCols, indexCols=idCols,
                                                   noneIsNan=True, dropCloserCols=True,
                                                   dropCloser=14, dropNans=True)
            logger.info(f'  - diff. to reference (relative): n={len(dfDiff)} =>\n' + dfDiff.to_string())
            dfDiff.reset_index().to_excel(uivu.pWorkDir / f'super-synth-{kind}-{mode}-comp-14.xlsx')
            assert dfDiff.empty

        else:

            # Compare "Synthesis" sheet when workbook kind
            dfRef = ddfRefReport['Synthesis'].drop(columns=['RunFolder']).set_index('NumEchant')
            dfAct = ddfActReport['Synthesis'].drop(columns=['RunFolder']).set_index('NumEchant')
            assert dfRef.compare(dfAct).empty

            # Compare "Details" sheet when workbook kind: not that simple ...
            # * 11 more "No Doc" columns with MCDS 7.4 (the ref) compared to MCDS 6.2,
            # * very small differences in "Qua" indicators between MCDS 7.4 compared to MCDS 6.2
            dfRef = ddfRefReport['Details'].drop(columns=['StartTime', 'ElapsedTime', 'RunFolder'])
            dfAct = ddfActReport['Details'].drop(columns=['StartTime', 'ElapsedTime', 'RunFolder'])
            # a. Compare all the string columns and a few "no precision issue" more.
            idCols = ['NumEchant', 'Espèce', 'Passage', 'Adulte', 'Durée', 'AbrevEchant']
            simpleCompCols = idCols
            simpleCompCols += ['NTot Obs', 'Mod Key Fn', 'Mod Adj Ser', 'Mod Chc Crit',
                               'Conf Interv', 'Key Fn', 'Adj Ser']
            assert dfRef[simpleCompCols].set_index('NumEchant') \
                      .compare(dfAct[simpleCompCols].set_index('NumEchant')).empty

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
            if not dfDiff.empty:
                dfDiff.reset_index().to_excel(uivu.pWorkDir / 'rep-comp-14.xlsx')

        # Compare "Samples", "Models", "Analyser" and "Computing platform" tables
        # (+ some preliminary tweaks to convert workbook to results specs "format")
        ddfRefSpecs = {cls.KRep2SpecNames[n]: sdf.copy() for n, sdf in ddfRefReport.items() if n in cls.KRep2SpecNames}
        ddfRefSpecs['analyser'] = ddfRefSpecs['analyser']['Value']
        ddfRefSpecs['runtime'] = ddfRefSpecs['runtime']['Version']
        ddfActSpecs = {cls.KRep2SpecNames[n]: sdf.copy() for n, sdf in ddfActReport.items() if n in cls.KRep2SpecNames}
        ddfActSpecs['analyser'] = ddfActSpecs['analyser']['Value']
        ddfActSpecs['runtime'] = ddfActSpecs['runtime']['Version']
        cls.compareResultSpecs(ddfRefSpecs, ddfActSpecs, source='report')

        logger.info(f'Done comparing reference to actual pre-analysis {kind} report ({mode}).')

    @pytest.fixture()
    def htmlRefReport_fxt(self):

        fpnRep = uivu.pRefOutDir / f'ACDC2019-Naturalist-extrait-PreRapport.html'

        return self.loadHtmlReport(fpnRep)

    @staticmethod
    def loadHtmlReport(filePath):
        """Produce a dict of DataFrames with exact same layout as loadWorkbookReport"""

        assert filePath.is_file(), f'Expected HTML report file not found {filePath.as_posix()}'

        logger.info(f'Loading Html report tables from {filePath.as_posix()} ...')

        ldfTables = pd.read_html(filePath)

        nResults = (len(ldfTables) - 1 - 5) // 3

        # Get the title table
        ddfRep = {'Title': ldfTables[0]}

        # Build the super-synthesis table
        # from the sub-tables (columns 1, 2, and 3) of the HTML super-synthesis table (1 row per analysis)
        ddfRep['SuperSynthesis'] = pd.DataFrame([pd.concat([ldfTables[subTblInd]
                                                            for subTblInd in range(resInd, resInd + 3)])
                                                   .set_index(0).loc[:, 1]
                                                 for resInd in range(2, 2 + 3 * nResults, 3)])
        ddfRep['SuperSynthesis'].set_index('NumEchant', inplace=True)
        # Fix some float columns, strangely loaded as string
        for col in ['Min Dist', 'Max Dist']:
            ddfRep['SuperSynthesis'][col] = ddfRep['SuperSynthesis'][col].astype(float)

        # Get and format the analyses, analyser and runtime tables
        sampTableInd = 2 + 3 * nResults
        ddfRep['Samples'] = ldfTables[sampTableInd]
        ddfRep['Samples'].set_index(ddfRep['Samples'].columns[0], inplace=True)
        ddfRep['Samples'].index.name = None

        ddfRep['Models'] = ldfTables[sampTableInd + 1]
        ddfRep['Models'].set_index(ddfRep['Models'].columns[0], inplace=True)
        ddfRep['Models'].index.name = None

        ddfRep['Analyser'] = ldfTables[sampTableInd + 2]
        ddfRep['Analyser'].set_index(ddfRep['Analyser'].columns[0], inplace=True)
        ddfRep['Analyser'].index.name = None

        ddfRep['Computing platform'] = ldfTables[sampTableInd + 3]
        ddfRep['Computing platform'].set_index(ddfRep['Computing platform'].columns[0], inplace=True)
        ddfRep['Computing platform'].index.name = None

        return ddfRep

    @staticmethod
    def loadHtmlReportLines(filePath):

        logger.info(f'Loading HTML pre-report lines from {filePath.as_posix()} ...')

        return uivu.loadPrettyHtmlLines(filePath)

    @pytest.fixture()
    def htmlRefReportLines_fxt(self):

        return self.loadHtmlReportLines(uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-PreRapport.html')

    @staticmethod
    def compareHtmlReports(refReportLines, actReportLines):
        """Prerequisite: Reference report generated with either MCDS 7.4 or 6.2"""

        DEBUG = False

        logger.info('Preprocessing HTML pre-reports for comparison ...')

        if DEBUG:
            with open(uivu.pTmpDir / 'prereport-ref-before.html', 'w') as file:
                file.write('\n'.join(refReportLines))
            with open(uivu.pTmpDir / 'prereport-act-before.html', 'w') as file:
                file.write('\n'.join(actReportLines))

        # Pre-process actual report lines
        remRefLines = remActLines = 0

        # * list unique analysis folders (keeping the original order) in both reports
        KREAnlysDir = r'="./([a-zA-Z0-9-_]+)/'
        refAnlysDirs = uivu.listUniqueStrings(KREAnlysDir, refReportLines)
        actAnlysDirs = uivu.listUniqueStrings(KREAnlysDir, actReportLines)
        logger.info(f'* found {len(refAnlysDirs)}/{len(actAnlysDirs)} ref./act. pre-analysis folder')
        assert len(refAnlysDirs) == len(actAnlysDirs)

        # * replace each analysis folder in the actual report by the corresponding ref. report one
        repAnlysDirLines = uivu.replaceStrings(froms=actAnlysDirs, tos=refAnlysDirs, lines=actReportLines)
        logger.info(f'* replaced analysis folder by ref. one in {repAnlysDirLines} act. lines')

        # * remove specific lines in both reports:
        #   - header meta "DateTime"
        KREDateTime = r'[0-9]{2}/[0-9]{2}/[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}'
        KREMetaDateTime = rf'<meta content="{KREDateTime}" name="datetime"/>'
        remRefLines += uivu.removeLines(KREMetaDateTime, refReportLines)
        remActLines += uivu.removeLines(KREMetaDateTime, actReportLines)

        #   - footer "Generated on <date+time>"  => not needed, 'cause in final ignored blocks (see below)
        # KREGenDateTime = rf'Generated on {KREDateTime}'
        # remRefLines += uivu.removeLines(KREGenDateTime, refReportLines)
        # remActLines += uivu.removeLines(KREGenDateTime, actReportLines)

        logger.info(f'* removed {remRefLines} ref. and {remActLines} act. lines')

        # Build the list of unified diff blocks
        blocks = uivu.unifiedDiff(refReportLines, actReportLines, logger=logger, subject='HTML pre-reports')

        # Filter diff blocks to check (ignore some that are expected to change without any issue:
        # * header meta "datetime" generation date,
        # * computation platform table, with component versions,
        # * generation date, credits to components with versions, sources)
        blocks_to_check = []
        for block in blocks:
            if block.startLines.expected >= 3955 - remRefLines:  # <h3>Computing platform</h3><table ...<tbody>
                logger.info(f'Ignoring block @ -{block.startLines.expected} +{block.startLines.real} @')
                continue
            blocks_to_check.append(block)

        # Check filtered blocks : should not be any left !
        assert len(blocks_to_check) == 0

        logger.info('HTML pre-reports comparison: success !')

    # ## 7. Generate HTML and Excel pre-analyses reports through pyaudisam API
    def testReports(self, sampleSpecMode, preAnalyser_fxt,
                    workbookRefReport_fxt, htmlRefReport_fxt, htmlRefReportLines_fxt):

        if sampleSpecMode != 'implicit':
            msg = 'testReports(explicit): skipped, as not relevant'
            logger.info(msg)
            pytest.skip(msg)  # Raises an exception => function execution stops here.

        build = True  # Debug only: Set to False to avoid rebuilding the reports, and only check them
        postCleanup = True  # Debug only: Set to False to prevent cleaning at the end

        # Pre-requisites : uncleaned pre-analyser work dir (we need the results file and analysis folders).
        preAnlysr, _ = preAnalyser_fxt
        if build:
            logger.info('Checking analyser results presence ...')
            preAnlysrResFilePath = preAnlysr.workDir / 'valtests-preanalyses-results.xlsx'
            assert preAnlysr.workDir.exists() and preAnlysrResFilePath.is_file()
            anlysFolders = [path for path in preAnlysr.workDir.iterdir() if path.is_dir()]
            assert len(anlysFolders) == 12
            logger.info('Done checking analyser results presence: OK.')

            # a. Load pre-results
            # (the last generated one, through implicit or explicit sample specs:
            #  never mind, they are the same as checked above)
            rsAct = self.loadResults(preAnlysr, preAnlysrResFilePath)
            logger.info(f'Actual results: n={len(rsAct)} =>\n' + rsAct.dfData.to_string(min_rows=30, max_rows=30))

            # # b. Generate Excel and HTML reports
            # b.i. Super-synthesis sub-report : Selected analysis results columns for the 3 textual columns of the table
            sampleRepCols = [
                ('header (head)', 'NumEchant', 'Value'),
                ('header (sample)', 'Espèce', 'Value'),
                ('header (sample)', 'Passage', 'Value'),
                ('header (sample)', 'Adulte', 'Value'),
                ('header (sample)', 'Durée', 'Value'),
                RS.CLNTotObs, RS.CLMinObsDist, RS.CLMaxObsDist,
            ]

            paramRepCols = [
                RS.CLParEstKeyFn, RS.CLParEstAdjSer,
                # RS.CLParEstSelCrit, RS.CLParEstCVInt
            ]

            resultRepCols = [
                RS.CLRunStatus,
                RS.CLNObs, RS.CLEffort,
                RS.CLAic, RS.CLChi2, RS.CLKS, RS.CLDCv,

                RS.CLCmbQuaBal1, RS.CLCmbQuaBal2, RS.CLCmbQuaBal3,

                RS.CLPDetec,
                RS.CLEswEdr,
                RS.CLDensity, RS.CLDensityMin, RS.CLDensityMax,
                RS.CLNumber, RS.CLNumberMin, RS.CLNumberMax,
            ]

            # b.ii. Synthesis sub-report : Selected analysis results columns for the
            synthRepCols = [
                ('header (head)', 'NumEchant', 'Value'),
                ('header (sample)', 'Espèce', 'Value'),
                ('header (sample)', 'Passage', 'Value'),
                ('header (sample)', 'Adulte', 'Value'),
                ('header (sample)', 'Durée', 'Value'),
                RS.CLParEstKeyFn,
                RS.CLParEstAdjSer,

                RS.CLNTotObs, RS.CLNObs, RS.CLNTotPars, RS.CLEffort, RS.CLDeltaAic,
                RS.CLChi2, RS.CLKS, RS.CLCvMUw, RS.CLCvMCw, RS.CLDCv,

                RS.CLSightRate,
                RS.CLCmbQuaBal1, RS.CLCmbQuaBal2, RS.CLCmbQuaBal3,
                RS.CLCmbQuaChi2, RS.CLCmbQuaKS, RS.CLCmbQuaDCv,

                RS.CLPDetec, RS.CLPDetecMin, RS.CLPDetecMax,
                RS.CLDensity, RS.CLDensityMin, RS.CLDensityMax,
                RS.CLNumber, RS.CLNumberMin, RS.CLNumberMax,
            ]

            # b.iii. Sorting columns for all the sub-reports
            sortRepCols = [('header (head)', 'NumEchant', 'Value')]
            sortRepAscend = True

            # b.iv. Report object
            report = ads.MCDSResultsPreReport(resultsSet=rsAct,
                                              title='PyAuDiSam Validation: Pre-analyses',
                                              subTitle='Pre-analysis results report',
                                              anlysSubTitle='Pre-analysis results details',
                                              description='Easy and parallel run through MCDSPreAnalyser',
                                              keywords='pyaudisam, validation, pre-analysis',
                                              lang='en', superSynthPlotsHeight=288,
                                              # plotImgSize=(640, 400), plotLineWidth=1, plotDotWidth=4,
                                              # plotFontSizes=dict(title=11, axes=10, ticks=9, legend=10),
                                              sampleCols=sampleRepCols, paramCols=paramRepCols,
                                              resultCols=resultRepCols, synthCols=synthRepCols,
                                              sortCols=sortRepCols, sortAscend=sortRepAscend,
                                              tgtFolder=preAnlysr.workDir,
                                              tgtPrefix='valtests-preanalyses-report-api')

            # b.iv. Excel report
            xlsxRep = pl.Path(report.toExcel())
            logger.info('Excel pre-report: ' + xlsxRep.resolve().as_posix())

            # b.v. HTML report
            htmlRep = pl.Path(report.toHtml())
            logger.info('HTML pre-report: ' + htmlRep.resolve().as_posix())

        else:
            logger.warning('NOT building the reports: this is not the normal testing scheme !')
            xlsxRep = preAnlysr.workDir / 'valtests-preanalyses-report.xlsx'
            htmlRep = preAnlysr.workDir / 'valtests-preanalyses-report.html'

        # c. Load generated Excel report and compare it to reference one
        ddfRefRep = workbookRefReport_fxt
        ddfActRep = self.loadWorkbookReport(xlsxRep)
        self.compareReports(ddfRefRep, ddfActRep, mode='api', kind='workbook')

        # d. Load generated HTML report and compare it to reference one
        #    (results only = only tables)
        ddfRefHtmlRep = htmlRefReport_fxt
        ddfActHtmlRep = self.loadHtmlReport(htmlRep)
        self.compareReports(ddfRefHtmlRep, ddfActHtmlRep, mode='api', kind='html')

        # e. Load generated HTML report and compare it to reference one
        #    (results + layout)
        # c. Load generated HTML report and compare it to reference one
        htmlRefRepLines = htmlRefReportLines_fxt
        htmlActRepLines = self.loadHtmlReportLines(htmlRep)
        self.compareHtmlReports(htmlRefRepLines, htmlActRepLines)

        # e. Cleanup generated report (well ... partially at least)
        #    for clearing next function's ground
        if postCleanup:
            xlsxRep.unlink()
            htmlRep.unlink()
        else:
            logger.warning('NOT cleaning up reports: this is not the normal testing scheme !')

        # f. Done.
        logger.info(f'PASS testReports: MCDSResultsReport ctor, toExcel, toHtml')

    # ## 7. Generate HTML and Excel pre-analyses reports through pyaudisam command line
    def testReportsCli(self, sampleSpecMode, workbookRefReport_fxt, htmlRefReportLines_fxt):

        if sampleSpecMode != 'implicit':
            msg = 'testReportsCli(explicit): skipped, as not relevant'
            logger.info(msg)
            pytest.skip(msg)  # Raises an exception => function execution stops here.

        build = True  # Debug only: Set to False to avoid rebuilding the report

        # a. Report "through the commande line"
        workPath = uivu.pWorkDir
        if build:
            argv = f'-p {uivu.pTestDir.as_posix()}/valtests-ds-params.py -w {workPath.as_posix()}' \
                   ' -n --prereports excel,html -u'.split()
            rc = ads.main(argv, standaloneLogConfig=False)
            logger.info(f'CLI run: rc={rc}')
        else:
            logger.warning('NOT building the reports: this is not the normal testing scheme !')

        # b. Load generated Excel report and compare it to reference one
        ddfRefRep = workbookRefReport_fxt
        ddfActRep = self.loadWorkbookReport(workPath / 'valtests-preanalyses-report.xlsx')
        self.compareReports(ddfRefRep, ddfActRep)

        # c. Load generated HTML report and compare it to reference one
        htmlRefRepLines = htmlRefReportLines_fxt
        htmlActRepLines = self.loadHtmlReportLines(workPath / 'valtests-preanalyses-report.html')
        self.compareHtmlReports(htmlRefRepLines, htmlActRepLines)

        # d. No cleanup: let the final cleaning code operate in _inifinalizeClass()

        # e. Done.
        logger.info(f'PASS testReports: main, MCDSResultsReport ctor, toExcel, toHtml (command line mode)')
