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
logger = uivu.setupLogger('val.pnr', level=ads.DEBUG, otherLoggers={'ads.eng': ads.INFO2})


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
        uivu.setupWorkDir('val-mpanlr', cleanup=self.KFinalCleanup)

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

        assert preAnlysr.specs == {
            'Zone': 'ACDC',
            'Surface': '2400',
            'distanceUnit': 'Meter',
            'areaUnit': 'Hectare',
            'runMethod': 'subprocess.run',
            'runTimeOut': 300,
            'surveyType': 'Point',
            'distanceType': 'Radial',
            'clustering': False,
            'defEstimKeyFn': 'UNIFORM',
            'defEstimAdjustFn': 'POLY',
            'defEstimCriterion': 'AIC',
            'defCVInterval': 95,
            'defMinDist': None,
            'defMaxDist': None,
            'defFitDistCuts': None,
            'defDiscrDistCuts': None
        }

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

        rsPreRes = preAnlysr.setupResults()
        rsPreRes.fromFile(filePath, postComputed=postComputed)

        return rsPreRes

    @pytest.fixture()
    def refResults_fxt(self, sampleSpecMode, preAnalyser_fxt):

        logger.info(f'Preparing reference pre-results ({sampleSpecMode} mode) ...')

        # Prevent re-postComputation as this ref. file is old, with now missing computed cols
        preAnlysr, _ = preAnalyser_fxt
        rsRef = self.loadResults(preAnlysr, uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-PreResultats.ods',
                                 postComputed=True)

        logger.info(f'Reference results: n={len(rsRef)} =>\n' + rsRef.dfData.to_string(min_rows=30, max_rows=30))

        return rsRef

    @staticmethod
    def compareResults(rsRef, rsAct):

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

        assert dfDiff.empty, 'Oh oh ... some unexpected differences !'

        # 1d. To be perfectly honest ... there may be some 10**-14/-16 glitches (due to worksheet I/O ?) ... or not.
        dfComp = rsRef.compare(rsAct, indexCols=indexPreCols, subsetCols=subsetPreCols,
                               noneIsNan=True, dropNans=True)
        dfComp = dfComp[(dfComp != np.inf).all(axis='columns')]

        logger.info(f'Diff. to reference (absolute): n={len(dfComp)} =>\n'
                    + dfComp.to_string(min_rows=30, max_rows=30))

        # 2. Specs: Samples
        dfRefSampSpecs = rsRef.specs['samples']
        logger.info(f'Ref sample specs: n={len(dfRefSampSpecs)} =>\n'
                    + dfRefSampSpecs.to_string(min_rows=30, max_rows=30))
        # Remove neutral pass-through column (from sample specs to results)
        # from actual sample specs if there (not present in ref.)
        dfActSampSpecs = rsAct.specs['samples'].drop(columns=['AbrevEsp'], errors='ignore')
        logger.info(f'Actual sample specs: n={len(dfActSampSpecs)} =>\n'
                    + dfActSampSpecs.to_string(min_rows=30, max_rows=30))
        assert dfRefSampSpecs.compare(dfActSampSpecs).empty

        # 3. Specs: Models
        dfRefModSpecs = rsRef.specs['models']
        logger.info(f'Ref. model specs: n={len(dfRefModSpecs)} =>\n'
                    + dfRefModSpecs.to_string(min_rows=30, max_rows=30))
        dfActModSpecs = rsAct.specs['models']
        logger.info(f'Actual model specs: n={len(dfActModSpecs)} =>\n'
                    + dfActModSpecs.to_string(min_rows=30, max_rows=30))
        assert dfRefModSpecs.compare(dfActModSpecs).empty

        # 4. Specs: Analyser
        dfRefAnrSpecs = rsRef.specs['analyser']
        logger.info(f'Ref. analyser specs: n={len(dfRefAnrSpecs)} =>\n'
                    + dfRefAnrSpecs.to_string(min_rows=30, max_rows=30))
        dfActAnrSpecs = rsAct.specs['analyser']
        logger.info(f'Actual analyser specs: n={len(dfActAnrSpecs)} =>\n'
                    + dfActAnrSpecs.to_string(min_rows=30, max_rows=30))
        assert dfRefAnrSpecs.compare(dfActAnrSpecs).empty

        # 5. Specs: Run-time : whatever ref, expect a specific list of item names, no more (values may vary, it's OK)
        sRefRunSpecs = rsRef.specs['runtime']
        logger.info(f'Ref. runtime specs: n={len(sRefRunSpecs)} =>\n' + sRefRunSpecs.to_string())
        sActRunSpecs = rsAct.specs['runtime']
        logger.info(f'Actual runtime specs: n={len(sActRunSpecs)} =>\n' + sActRunSpecs.to_string())
        assert set(sRefRunSpecs.index) \
               == {'os', 'processor', 'python', 'numpy', 'pandas', 'zoopt', 'matplotlib',
                   'jinja2', 'pyaudisam', 'MCDS engine', 'MCDS engine version'}

        logger.info('Done comparing reference to actual pre-results.')

    # ### d. Run pre-analyses through pyaudisam API
    def testRun(self, sampleSpecMode, preAnalyser_fxt, refResults_fxt):

        # i. Cleanup test folder (Note: avoid any Ruindows shell or explorer inside this folder !)
        preAnlysr, sampleSpecs = preAnalyser_fxt
        if preAnlysr.workDir.exists():
            shutil.rmtree(preAnlysr.workDir)

        # ii. Run and measure performance
        # Ruindows 10 laptop with PCI-e SSD, "optimal performances" power scheme, Python 3.8 :
        # * 4-HT-core i5-8350U:
        #   * 2021 (precise date ?): 50s to ~1mn10s elapsed for 12 samples, 6-12 threads (N=?)
        # * 6-core i7-10750H (HT off):
        #   * 2022-01-17, 2023-11-02: 39-40s elapsed for 12 samples, 6-12 threads (N=5)
        # Ruindows 11 laptop with PCI-e SSD, "high performances" power scheme, Python 3.8 :
        # * 6-HT-core i7-10850H (HT on):
        #   * 2024-03-02: 40s elapsed for 12 samples, 6 threads (N=1)
        #   * 2024-03-02: 39s elapsed for 12 samples, 12 threads (N=1)
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
        dfEnRes = rsAct.dfTransData('en')
        for anlysFolderPath in dfEnRes.RunFolder:
            assert {fpn.name for fpn in pl.Path(anlysFolderPath).iterdir()} \
                   == {'cmd.txt', 'data.txt', 'log.txt', 'output.txt', 'plots.txt', 'stats.txt'}
        logger.info('... done checking pre-analysis folders (minimal).')

        # g. Cleanup pre-analyser (analysis folders)
        preAnlysr.cleanup()

        # h. Done.
        logger.info(f'PASS testRun: run({sampleSpecMode} sample specs), cleanup')

    # Run pre-analyses through pyaudisam command line interface
    # (implicit mode only, see valtests-ds-params.py)
    def testRunCli(self, sampleSpecMode, preAnalyser_fxt, refResults_fxt):

        if sampleSpecMode != 'implicit':
            msg = 'testRunCli(explicit): skipped, as not relevant'
            logger.info(msg)
            pytest.skip(msg)  # Raises an exception => function execution stops here.

        testPath = pl.Path(__file__).parent
        preAnlysr, _ = preAnalyser_fxt
        workPath = preAnlysr.workDir

        # a. Cleanup test folder (Note: avoid any Ruindows shell or explorer inside this folder !)
        shutil.rmtree(workPath)

        # b. Run "through the commande line"
        argv = f'-p {testPath.as_posix()}/valtests-ds-params.py -w {workPath.as_posix()} -n --preanalyses -u'.split()
        rc = ads.main(argv, standaloneLogConfig=False)
        logger.info(f'CLI run: rc={rc}')

        # c. Load pre-results
        rsAct = self.loadResults(preAnlysr, workPath / 'valtests-preanalyses-results.xlsx')
        logger.info(f'Actual results: n={len(rsAct)} =>\n' + rsAct.dfData.to_string(min_rows=30, max_rows=30))

        # d. Compare to reference.
        rsRef = refResults_fxt
        self.compareResults(rsRef, rsAct)

        # e. Minimal check of pre-analysis folders
        logger.info('Checking pre-analysis folders (minimal) ...')
        dfEnRes = rsAct.dfTransData('en')
        for anlysFolderPath in dfEnRes.RunFolder:
            assert {fpn.name for fpn in pl.Path(anlysFolderPath).iterdir()} \
                   == {'cmd.txt', 'data.txt', 'log.txt', 'output.txt', 'plots.txt', 'stats.txt'}
        logger.info('... done checking pre-analysis folders (minimal).')

        # f. Don't clean up work folder / analysis folders : needed for report generations below

        # g. Done.
        logger.info(f'PASS testRunCli: main, run (command line mode)')

    @pytest.fixture()
    def excelRefReport_fxt(self):

        return pd.read_excel(uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-PreRapport.xlsx',
                             sheet_name=None, index_col=0)

    @staticmethod
    def compareExcelReports(ddfRefReport, ddfActReport):

        # Compare "Synthesis" sheet
        dfRef = ddfRefReport['Synthesis'].drop(columns=['RunFolder']).set_index('NumEchant')
        dfAct = ddfActReport['Synthesis'].drop(columns=['RunFolder']).set_index('NumEchant')
        assert dfRef.compare(dfAct).empty

        # Compare "Details" sheet : not that simple ...
        # * 11 more "No Doc" columns with MCDS 7.4 (the ref) compared to MCDS 6.2,
        # * very small differences in "Qua" indicators between MCDS 7.4 compared to MCDS 6.2
        dfRef = ddfRefReport['Details'].drop(columns=['StartTime', 'ElapsedTime', 'RunFolder'])
        dfAct = ddfActReport['Details'].drop(columns=['StartTime', 'ElapsedTime', 'RunFolder'])
        # a. Compare all the string columns and a few "no precision issue" more.
        idCols = ['NumEchant', 'Espèce', 'Passage', 'Adulte', 'Durée', 'AbrevEchant']
        simpleCompCols = idCols
        simpleCompCols += ['NTot Obs', 'Mod Key Fn', 'Mod Adj Ser', 'Mod Chc Crit', 'Conf Interv', 'Key Fn', 'Adj Ser']
        assert dfRef[simpleCompCols].set_index('NumEchant').compare(dfAct[simpleCompCols].set_index('NumEchant')).empty
        # b. Compare other (all numerical) columns with a small margin (1e-14 relative diff)
        otherCompCols = [col for col in dfRef if col not in simpleCompCols]
        if len(dfAct.columns) == len(dfRef.columns):  # MCDS 7.4 ?
            assert ads.DataSet.compareDataFrames(dfLeft=dfRef, dfRight=dfAct,
                                                 subsetCols=otherCompCols, indexCols=idCols,
                                                 noneIsNan=True, dropCloserCols=True,
                                                 dropCloser=14, dropNans=True).empty
        else:  # Last chance: MCDS 6.2 ?
            spe74CompCols = [col for col in otherCompCols if col.startswith('SansDoc #')]  # 7.4-specifics
            assert len(spe74CompCols) == 11
            otherCompCols = [col for col in otherCompCols if col not in spe74CompCols]  # Remove 7.4-specifics
            assert ads.DataSet.compareDataFrames(dfLeft=dfRef, dfRight=dfAct,
                                                 subsetCols=otherCompCols, indexCols=idCols,
                                                 noneIsNan=True, dropCloserCols=True,
                                                 dropCloser=15, dropNans=True).empty

        # Compare "Samples" sheet
        dfRef = ddfRefReport['Samples'].set_index('NumEchant', drop=True)
        dfAct = ddfActReport['Samples'].set_index('NumEchant', drop=True)
        assert dfRef.compare(dfAct).empty

        # Compare "Models" sheet
        dfRef = ddfRefReport['Models']
        dfAct = ddfActReport['Models']
        assert dfRef.compare(dfAct).empty

        # Compare "Analyser" sheet
        dfRef = ddfRefReport['Analyser']
        dfAct = ddfActReport['Analyser']
        assert dfRef.compare(dfAct).empty

    @pytest.fixture()
    def htmlRefReportLines_fxt(self):

        with open(uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-PreRapport.html') as file:
            repLines = file.readlines()

        return repLines

    @staticmethod
    def compareHtmlReports(refReportLines, actReportLines):

        logger.info('Preprocessing HTML pre-reports for comparison ...')

        # Pre-process actual report lines
        remRefLines = remActLines = 0

        # * list unique analysis folders (keeping the original order) in both reports
        KREAnlysDir = r'="./([a-zA-Z0-9-_]+)/'
        refAnlysDirs = uivu.listUniqueStrings(KREAnlysDir, refReportLines)
        actAnlysDirs = uivu.listUniqueStrings(KREAnlysDir, actReportLines)
        assert len(refAnlysDirs) == len(actAnlysDirs)

        logger.info(f'* found {len(actAnlysDirs)} pre-analysis folders')

        # * replace each analysis folder in the actual report by the corresponding ref. report one
        uivu.replaceStrings(actAnlysDirs, refAnlysDirs, actReportLines)

        # * remove specific lines in both reports:
        #   - header meta "DateTime"
        KREDateTime = r'[0-9]{2}/[0-9]{2}/[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}'
        KREMetaDateTime = rf'<meta name="datetime" content="{KREDateTime}"/>'
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
            if block.startLines.expected >= 552 - remRefLines:  # <h3>Computing platform</h3><table ...<tbody>
                logger.info(f'Ignoring block @ -{block.startLines.expected} +{block.startLines.real} @')
                continue
            blocks_to_check.append(block)

        # Check filtered blocks : should not be any left !
        assert len(blocks_to_check) == 0

        logger.info('HTML pre-reports comparison: success !')

    # ## 7. Generate HTML and Excel pre-analyses reports through pyaudisam API
    def testReports(self, sampleSpecMode, preAnalyser_fxt, excelRefReport_fxt, htmlRefReportLines_fxt):

        if sampleSpecMode != 'implicit':
            msg = 'testReports(explicit): skipped, as not relevant'
            logger.info(msg)
            pytest.skip(msg)  # Raises an exception => function execution stops here.

        build = True  # Debug only: Set to False to avoid rebuilding the report, and only check it
        cleanup = True  # Debug only: Set to False to prevent cleaning at the end

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
            R = rsAct.__class__
            # b.i. Super-synthesis sub-report : Selected analysis results columns for the 3 textual columns of the table
            sampleRepCols = [
                ('header (head)', 'NumEchant', 'Value'),
                ('header (sample)', 'Espèce', 'Value'),
                ('header (sample)', 'Passage', 'Value'),
                ('header (sample)', 'Adulte', 'Value'),
                ('header (sample)', 'Durée', 'Value'),
                R.CLNTotObs, R.CLMinObsDist, R.CLMaxObsDist
            ]

            paramRepCols = [
                R.CLParEstKeyFn, R.CLParEstAdjSer
                # R.CLParEstSelCrit, R.CLParEstCVInt
            ]

            resultRepCols = [
                R.CLRunStatus,
                R.CLNObs, R.CLEffort,
                R.CLAic, R.CLChi2, R.CLKS, R.CLDCv,

                R.CLCmbQuaBal1, R.CLCmbQuaBal2, R.CLCmbQuaBal3,

                R.CLPDetec,
                R.CLEswEdr,
                R.CLDensity, R.CLDensityMin, R.CLDensityMax,
                R.CLNumber, R.CLNumberMin, R.CLNumberMax
            ]

            # b.ii. Synthesis sub-report : Selected analysis results columns for the
            synthRepCols = [
                ('header (head)', 'NumEchant', 'Value'),
                ('header (sample)', 'Espèce', 'Value'),
                ('header (sample)', 'Passage', 'Value'),
                ('header (sample)', 'Adulte', 'Value'),
                ('header (sample)', 'Durée', 'Value'),
                R.CLParEstKeyFn,
                R.CLParEstAdjSer,
                # R.CLParEstSelCrit,
                # R.CLParEstCVInt,
                # R.CLParTruncLeft,
                # R.CLParTruncRight,
                # R.CLParModFitDistCuts,

                R.CLNTotObs, R.CLNObs, R.CLNTotPars, R.CLEffort, R.CLDeltaAic,
                R.CLChi2, R.CLKS, R.CLCvMUw, R.CLCvMCw, R.CLDCv,

                R.CLSightRate,
                R.CLCmbQuaBal1, R.CLCmbQuaBal2, R.CLCmbQuaBal3,
                R.CLCmbQuaChi2, R.CLCmbQuaKS, R.CLCmbQuaDCv,

                R.CLPDetec, R.CLPDetecMin, R.CLPDetecMax,
                R.CLDensity, R.CLDensityMin, R.CLDensityMax,
                R.CLNumber, R.CLNumberMin, R.CLNumberMax
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
            xlsxRep = report.toExcel()
            logger.info('Excel pre-report: ' + pl.Path(xlsxRep).resolve().as_posix())

            # b.v. HTML report
            htmlRep = report.toHtml()
            logger.info('HTML pre-report: ' + pl.Path(htmlRep).resolve().as_posix())

        else:
            xlsxRep = preAnlysr.workDir / 'valtests-preanalyses-report.xlsx'
            htmlRep = preAnlysr.workDir / 'valtests-preanalyses-report.html'

        # c. Load generated Excel report and compare it to reference one
        ddfRefRep = excelRefReport_fxt

        ddfActRep = pd.read_excel(xlsxRep, sheet_name=None, index_col=0)

        self.compareExcelReports(ddfRefRep, ddfActRep)

        # c. Load generated HTML report and compare it to reference one
        htmlRefRepLines = htmlRefReportLines_fxt

        with open(htmlRep) as file:
            htmlActRepLines = file.readlines()

        self.compareHtmlReports(htmlRefRepLines, htmlActRepLines)

        # e. Cleanup generated report (well ... partially at least)
        #    for clearing next function's ground
        if cleanup:
            pl.Path(xlsxRep).unlink()
            pl.Path(htmlRep).unlink()

        # f. Done.
        logger.info(f'PASS testReports: MCDSResultsReport ctor, toExcel, toHtml')

    # ## 7. Generate HTML and Excel pre-analyses reports through pyaudisam command line
    def testReportsCli(self, sampleSpecMode, preAnalyser_fxt, excelRefReport_fxt, htmlRefReportLines_fxt):

        if sampleSpecMode != 'implicit':
            msg = 'testReportsCli(explicit): skipped, as not relevant'
            logger.info(msg)
            pytest.skip(msg)  # Raises an exception => function execution stops here.

        build = True  # Debug only: Set to False to avoid rebuilding the report

        testPath = pl.Path(__file__).parent
        preAnlysr, _ = preAnalyser_fxt
        workPath = preAnlysr.workDir

        # a. Report "through the commande line"
        if build:
            argv = f'-p {testPath.as_posix()}/valtests-ds-params.py -w {workPath.as_posix()}' \
                   ' -n --prereports excel,html -u'.split()
            rc = ads.main(argv, standaloneLogConfig=False)
            logger.info(f'CLI run: rc={rc}')

        # b. Load generated Excel report and compare it to reference one
        ddfActRep = pd.read_excel(workPath / 'valtests-preanalyses-report.xlsx',
                                  sheet_name=None, index_col=0)

        ddfRefRep = excelRefReport_fxt
        self.compareExcelReports(ddfRefRep, ddfActRep)

        # c. Load generated HTML report and compare it to reference one
        with open(workPath / 'valtests-preanalyses-report.html') as file:
            htmlActRepLines = file.readlines()

        htmlRefRepLines = htmlRefReportLines_fxt
        self.compareHtmlReports(htmlRefRepLines, htmlActRepLines)

        # d. No cleanup: let the final cleaning code operate in _inifinalizeClass()

        # e. Done.
        logger.info(f'PASS testReports: main, MCDSResultsReport ctor, toExcel, toHtml (command line mode)')
