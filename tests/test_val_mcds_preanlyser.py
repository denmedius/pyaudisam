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

# Automated validation tests for "analyser" submodule, MCDSPreAnalyser class part
# (adapted from valtest.ipynb notebook)

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

    # The folder where all files are generated
    KPreAnlalyserWorkDir = uivu.pTmpDir / 'mcds-preanlr'

    # Class and test function initialisers / finalisers ###########################
    @pytest.fixture(autouse=True, scope='class')
    def _inifinalise_class(self):

        what2Test = 'pre-analyser'

        uivu.logBegin(what=what2Test)

        # Make sure the ground is clear at start
        logger.info('Removing work folder ' + self.KPreAnlalyserWorkDir.as_posix())
        if self.KPreAnlalyserWorkDir.is_dir():
            shutil.rmtree(self.KPreAnlalyserWorkDir)

        # The code before yield is run before the first test function in this class
        yield
        # The code after yield is run after the last test function in this class

        # Let the ground clear after passing there
        logger.info('Removing work folder ' + self.KPreAnlalyserWorkDir.as_posix())
        if self.KPreAnlalyserWorkDir.is_dir():
            shutil.rmtree(self.KPreAnlalyserWorkDir)

        uivu.logEnd(what=what2Test)

    @pytest.fixture(autouse=True, scope='function')
    def _inifinalise_function(self):

        # The code before yield is run before every test function
        yield
        # The code after yield is run after every test function

    # Test functions #############################################################

    # II. Run and report pre-analyses
    # Thanks to MCDSPreAnalyser and MCDSPreReport.
    # Short code, fast (parallel) run.
    # Note: 2 modes here, with explicit or implicit sample specification (manual switch).
    # Note: The exact same results (implicit mode) and reports can be produced through the command line :
    # $ cd ..
    # $ python -m pyaudisam -p tests/valtests-ds-params.py -w tests/tmp/mcds-preanlr -n
    #   --preanalyses --prereports excel,html -u
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

        dfObsIndiv, dfTransects = inputDataSet_fxt

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
        # Note: The exact same results can be produced through the command line :
        # $ cd ..
        # $ python -m pyaudisam -p tests/valtests-ds-params.py -w tests/tmp/mcds-preanlr -n --preanalyses -u

        # ### a. MCDSPreAnalyser object
        areSpecsImplicit = sampleSpecMode == 'implicit'
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
                                workDir=self.KPreAnlalyserWorkDir, logProgressEvery=5)

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

        logger.info(f'Pre-analyser specs:\n' + str(preAnlysr.specs))

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
        dfExptdStats = expectedSampleStats_fxt

        dfSampleStats = preAnlysr.computeSampleStats(implSampleSpecs=sampleSpecs, sampleDistCol='Distance')
        logger.info(f'Sample stats: n={len(dfSampleStats)} =>\n{dfSampleStats.to_string(min_rows=30, max_rows=30)}')

        dfSampleStats['Distance Min'] = dfSampleStats['Distance Min'].round(1)
        dfSampleStats['Distance Max'] = dfSampleStats['Distance Max'].round(1)
        dfSampleStats.set_index('NumEchant', inplace=True)
        dfSampleStats = dfSampleStats[['Distance Min', 'Distance Max', 'NTot Obs']]

        assert dfSampleStats.compare(dfExptdStats).empty

        logger.info(f'PASS testComputeSampleStats: computeSampleStats')

    # Only minimalistic checks done here, as already more deeply tested in test_unint_engine.py
    def checkExportedDsInputData(self, exportDir, sampleSpecs, sampleSpecMode):

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

        preAnlysr, sampleSpecs = preAnalyser_fxt

        # i. Export distance files
        logger.info(f'Exporting Distance files ({sampleSpecMode} sample specs) ...')

        areSpecsImplicit = sampleSpecMode == 'implicit'
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

        preAnlysr, sampleSpecs = preAnalyser_fxt

        # i. Export distance files
        logger.info(f'Exporting Distance files (command line mode) ...')

        testPath = pl.Path(__file__).parent
        workPath = preAnlysr.workDir

        # a. Export files "through the commande line"
        argv = f'-p {testPath.as_posix()}/valtests-ds-params.py -w {workPath.as_posix()} -n --distexport -u'.split()
        rc = ads.main(argv, standaloneLogConfig=False)
        logger.info(f'CLI run: rc={rc}')

        # ii. Check exported files
        self.checkExportedDsInputData(workPath, sampleSpecs, sampleSpecMode)

        logger.info(f'PASS testExportDsInputDataCli: exportDsInputData(command line mode)')

    @staticmethod
    def loadPreResults(preAnlysr, filePath, postComputed=False):

        logger.info(f'Loading pre-results from file ...')

        rsPreRes = preAnlysr.setupResults()
        rsPreRes.fromFile(filePath, postComputed=postComputed)

        return rsPreRes

    @pytest.fixture()
    def refPreResults_fxt(self, sampleSpecMode, preAnalyser_fxt):

        preAnlysr, _ = preAnalyser_fxt

        logger.info(f'Preparing reference pre-results ({sampleSpecMode} mode) ...')

        # Prevent re-postComputation as this ref. file is old, with now missing computed cols
        rsPreRef = self.loadPreResults(preAnlysr, uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-PreResultats.ods',
                                       postComputed=True)

        logger.info(f'Reference results: n={len(rsPreRef)} =>\n' + rsPreRef.dfData.to_string(min_rows=30, max_rows=30))

        return rsPreRef

    @staticmethod
    def comparePreResults(rsRef, rsAct):

        # * index = analysis "Id": sample Id columns and analysis indexes.
        indexPreCols = [col for col in rsAct.miCustomCols.to_list() if '(sample)' in col[0]] \
                       + [('parameters', 'estimator key function', 'Value'),
                          ('parameters', 'estimator adjustment series', 'Value')]
        # * ignore:
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

        dfDiff = rsRef.compare(rsAct, indexCols=indexPreCols, subsetCols=subsetPreCols,
                               noneIsNan=True, dropCloser=13, dropNans=True)

        logger.info(f'Diff. to reference (relative): n={len(dfDiff)} =>\n'
                    + dfDiff.to_string(min_rows=30, max_rows=30))

        assert dfDiff.empty, 'Oh oh ... some unexpected differences !'

        # iv. To be perfectly honest ... there may be some 10**-14/-16 glitches (due to worksheet I/O ?) ... or not.
        dfComp = rsRef.compare(rsAct, indexCols=indexPreCols, subsetCols=subsetPreCols,
                               noneIsNan=True, dropNans=True)
        dfComp = dfComp[(dfComp != np.inf).all(axis='columns')]

        logger.info(f'Diff. to reference (absolute): n={len(dfComp)} =>\n'
                    + dfComp.to_string(min_rows=30, max_rows=30))

    # ### d. Run pre-analyses through pyaudisam API
    def testRun(self, sampleSpecMode, preAnalyser_fxt):

        preAnlysr, sampleSpecs = preAnalyser_fxt

        # i. Cleanup test folder (Note: avoid any Ruindows shell or explorer inside this folder !)
        if self.KPreAnlalyserWorkDir.is_dir():
            shutil.rmtree(self.KPreAnlalyserWorkDir)

        # ii. Run and measure performance
        # Ruindows 10 laptop with PCI-e SSD, "optimal performances" power scheme, Python 3.8 :
        # * 4-HT-core i5-8350U:
        #   * 2021 (precise date ?): 50s to ~1mn10s elapsed for 12 samples, 6-12 threads (N=?)
        # * 6-core i7-10750H (HT off):
        #   * 2022-01-17, 2023-11-02: 39-40s elapsed for 12 samples, 6-12 threads (N=5)
        # Ruindows 11 laptop with PCI-e SSD, "high performances" power scheme, Python 3.8 :
        # * 6-core i7-10750H (HT on):
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
        rsPreAct = preAnlysr.run(implSampleSpecs=sampleSpecs if areSpecsImplicit else None,
                                 dfExplSampleSpecs=sampleSpecs if not areSpecsImplicit else None,
                                 dModelStrategy=modelStrategy, threads=threads)
        end = time.perf_counter()

        logger.info(f'Elapsed time={end - start:.2f}s')

        # preResults.toExcel(preAnlysr.workDir / f'valtests-preanalyses-results-{specMode}api.xlsx')
        # preResults.toExcel(preAnlysr.workDir / f'valtests-preanalyses-results-{specMode}api-fr.xlsx', lang='fr')

        # ### e. Check results: Compare to reference
        # (reference generated with same kind of "long" code like in III above, but on another data set)
        # i. Check presence of neutral and pass-through column in explicit spec. mode
        #    (it should have effectively passed through :-)
        speciesAbbrevCol = 'AbrevEsp'
        logger.debug('dfTransData(en).columns: ' + str(rsPreAct.dfTransData('en').columns))
        assert areSpecsImplicit or speciesAbbrevCol in rsPreAct.dfTransData('en').columns

        # ii. Load reference (prevent re-postComputation as this ref. file is old, with now missing computed cols)
        rsPreRef = self.loadPreResults(preAnlysr, uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-PreResultats.ods',
                                       postComputed=True)
        logger.info(f'Reference results: n={len(rsPreRef)} =>\n' + rsPreRef.dfData.to_string(min_rows=30, max_rows=30))

        # iii Compare:
        self.comparePreResults(rsPreRef, rsPreAct)

        # f. Minimal check of pre-analysis folders
        dfEnRes = rsPreAct.dfTransData('en')
        for anlysFolderPath in dfEnRes.RunFolder:
            assert {fpn.name for fpn in pl.Path(anlysFolderPath).iterdir()} \
                   == {'cmd.txt', 'data.txt', 'log.txt', 'output.txt', 'plots.txt', 'stats.txt'}

        # g. Cleanup pre-analyser (analysis folders)
        preAnlysr.cleanup()

        # h. Done.
        logger.info(f'PASS testRun: run({sampleSpecMode} sample specs), cleanup')

    # Run pre-analyses through pyaudisam command line interface
    # (implicit mode only, see valtests-ds-params.py)
    def testRunCli(self, sampleSpecMode, preAnalyser_fxt, refPreResults_fxt):

        if sampleSpecMode != 'implicit':
            msg = 'testRunCli(explicit): skipped, as not relevant'
            logger.info(msg)
            pytest.skip(msg)  # Raises an exception => function execution stops here.

        preAnlysr, _ = preAnalyser_fxt
        rsPreRef = refPreResults_fxt

        testPath = pl.Path(__file__).parent
        workPath = preAnlysr.workDir

        # a. Cleanup test folder (Note: avoid any Ruindows shell or explorer inside this folder !)
        shutil.rmtree(workPath)

        # b. Run "through the commande line"
        argv = f'-p {testPath.as_posix()}/valtests-ds-params.py -w {workPath.as_posix()} -n --preanalyses -u'.split()
        rc = ads.main(argv, standaloneLogConfig=False)
        logger.info(f'CLI run: rc={rc}')

        # c. Load pre-results
        rsPreAct = self.loadPreResults(preAnlysr, workPath / 'valtests-preanalyses-results.xlsx')
        logger.info(f'Actual results: n={len(rsPreAct)} =>\n' + rsPreAct.dfData.to_string(min_rows=30, max_rows=30))

        # d. Compare to reference.
        self.comparePreResults(rsPreRef, rsPreAct)

        # e. Don't clean up work folder / analysis folders : needed for report generations below

        # f. Done.
        logger.info(f'PASS testRunCli: main, run (command line mode)')

    @pytest.fixture()
    def excelRefPreReport_fxt(self):

        return pd.read_excel(uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-PreRapport.xlsx',
                             sheet_name=None, index_col=0)

    @staticmethod
    def compareExcelReports(ddfRefReport, ddfActReport):

        # Compare "Synthesis" sheet
        dfRef = ddfRefReport['Synthesis'].drop(columns=['RunFolder'])
        dfAct = ddfActReport['Synthesis'].drop(columns=['RunFolder'])
        assert dfRef.set_index('NumEchant').compare(dfAct.set_index('NumEchant')).empty

        # Compare "Details" sheet
        dfRef = ddfRefReport['Details'].drop(columns=['StartTime', 'ElapsedTime', 'RunFolder'])
        dfAct = ddfActReport['Details'].drop(columns=['StartTime', 'ElapsedTime', 'RunFolder'])
        assert dfRef.set_index('NumEchant').compare(dfAct.set_index('NumEchant')).empty

        # Compare "Samples" sheet
        dfRef = ddfRefReport['Samples']
        dfAct = ddfActReport['Samples']
        assert dfRef.set_index('NumEchant').compare(dfAct.set_index('NumEchant')).empty

        # Compare "Models" sheet
        dfRef = ddfRefReport['Models']
        dfAct = ddfActReport['Models']
        assert dfRef.compare(dfAct).empty

        # Compare "Analyser" sheet
        dfRef = ddfRefReport['Analyser']
        dfAct = ddfActReport['Analyser']
        assert dfRef.compare(dfAct).empty

    @pytest.fixture()
    def htmlRefPreReportLines_fxt(self):

        with open(uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-PreRapport.html') as file:
            repLines = file.readlines()

        return repLines

    @staticmethod
    def compareHtmlReports(refReportLines, actReportLines, cliMode=False):

        blocks = uivu.unifiedDiff(refReportLines, actReportLines, logger=logger)

        assert len(blocks) == 15 + (0 if cliMode else 1)
        assert blocks[0].startLines.expected == 26
        assert blocks[-1].startLines.expected == (600 if cliMode else 610)

        for block in blocks:

            logger.info(f'Block @ -{block.startLines.expected} +{block.startLines.real} @')

            if block.startLines.expected == 26:
                assert block.startLines.real == 26, 'Wrong start line nums on first diff block'
                assert len(block.expectedLines) == 1, 'Wrong num. of expected lines on first diff block'
                assert len(block.realLines) == 1, 'Wrong num. of real lines on first diff block'
                assert block.expectedLines[0].strip().startswith('<meta name="datetime" content="'), \
                    'Wrong first diff block expected line'
                assert block.realLines[0].strip().startswith('<meta name="datetime" content="'), \
                    'Wrong first diff block real line'
                continue

            if block.startLines.expected == 114:
                assert block.startLines.real == 114, 'Wrong start line nums on 2nd diff block'
                assert len(block.expectedLines) == 1, 'Wrong num. of expected lines on 2nd diff block'
                assert len(block.realLines) == 1, 'Wrong num. of real lines on 2nd diff block'
                continue

            if block.startLines.expected == 600:
                assert block.startLines.real == 600, 'Wrong start line nums on last diff block'
                assert len(block.expectedLines) == 1, 'Wrong num. of expected lines on last diff block'
                assert len(block.realLines) == 1, 'Wrong num. of real lines on last diff block'
                assert block.expectedLines[0].strip().startswith('Generated on'), \
                    'Wrong last diff block expected line'
                assert block.realLines[0].strip().startswith('Generated on'), \
                    'Wrong last diff block real line'
                continue

            if block.startLines.expected == 217:
                assert block.startLines.real == 217, 'Wrong start line nums on last diff block'
                assert len(block.expectedLines) == 3, 'Wrong num. of expected lines on diff block'
                assert len(block.realLines) == 3, 'Wrong num. of real lines on diff block'
                for line in block.expectedLines:
                    line = line.strip()
                    assert line.startswith('<td><img src="./'), f"Wrong diff block expected line: '{line}'"
                for line in block.realLines:
                    line = line.strip()
                    assert line.startswith('<td><img src="./'), f"Wrong diff block real line: '{line}'"
                continue

            if block.startLines.expected == 610:
                assert not cliMode  # This block only when cliMode (sources include valtests-ds-params.py)
                assert block.startLines.real == 609, 'Wrong start line nums on sources diff block'
                assert len(block.expectedLines) == 3, 'Wrong num. of expected lines on sources diff block'
                assert len(block.realLines) == 0, 'Wrong num. of real lines on sources diff block'
                assert any('valtests-ds-params.py' in line for line in block.expectedLines), \
                    'Wrong diff block: expecting valtests-ds-params.py in expected sources'
                continue

            # All other blocks ...
            assert block.startLines.expected == block.startLines.real, 'Wrong start line nums on last diff block'
            assert len(block.expectedLines) == 6, 'Wrong num. of expected lines on diff block'
            assert len(block.realLines) == 6, 'Wrong num. of real lines on diff block'
            for line in block.expectedLines:
                line = line.strip()
                assert line.startswith('<tr>') or line.startswith('</tr>') \
                       or line.startswith('<th><a href="./') or line.startswith('<td><img src="./'), \
                    f"Wrong diff block expected line: '{line}'"
            for line in block.realLines:
                line = line.strip()
                assert line.startswith('<tr>') or line.startswith('</tr>') \
                       or line.startswith('<th><a href="./') or line.startswith('<td><img src="./'), \
                    f"Wrong diff block real line: '{line}'"

    # ## 7. Generate HTML and Excel pre-analyses reports through pyaudisam API
    def testReports(self, sampleSpecMode, preAnalyser_fxt, excelRefPreReport_fxt, htmlRefPreReportLines_fxt):

        if sampleSpecMode != 'implicit':
            msg = 'testReports(explicit): skipped, as not relevant'
            logger.info(msg)
            pytest.skip(msg)  # Raises an exception => function execution stops here.

        build = True  # Debug only: Set to False to avoid rebuilding the report, and only check it
        cleanup = True  # Debug only: Set to False to prevent cleaning at the end

        preAnlysr, sampleSpecs = preAnalyser_fxt

        # Pre-requisites : uncleaned pre-analyser work dir (we need the results file and analysis folders).
        if build:
            logger.info('Checking analyser results presence ...')
            preAnlysrResFilePath = preAnlysr.workDir / 'valtests-preanalyses-results.xlsx'
            assert self.KPreAnlalyserWorkDir.is_dir() and preAnlysrResFilePath.is_file()
            anlysFolders = [path for path in self.KPreAnlalyserWorkDir.iterdir() if path.is_dir()]
            assert len(anlysFolders) == 12
            logger.info('Done checking analyser results presence: OK.')

            # a. Load pre-results
            # (the last generated one, through implicit or explicit sample specs:
            #  never mind, they are the same as checked above)
            rsPreAct = self.loadPreResults(preAnlysr, preAnlysrResFilePath)
            logger.info(f'Actual results: n={len(rsPreAct)} =>\n' + rsPreAct.dfData.to_string(min_rows=30, max_rows=30))

            # # b. Generate Excel and HTML reports
            R = rsPreAct.__class__
            # b.i. Super-synthesis sub-report : Selected analysis results columns for the 3 textual columns of the table
            samplePreRepCols = [
                ('header (head)', 'NumEchant', 'Value'),
                ('header (sample)', 'Espèce', 'Value'),
                ('header (sample)', 'Passage', 'Value'),
                ('header (sample)', 'Adulte', 'Value'),
                ('header (sample)', 'Durée', 'Value'),
                R.CLNTotObs, R.CLMinObsDist, R.CLMaxObsDist
            ]

            paramPreRepCols = [
                R.CLParEstKeyFn, R.CLParEstAdjSer
                # R.CLParEstSelCrit, R.CLParEstCVInt
            ]

            resultPreRepCols = [
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
            synthPreRepCols = [
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
            sortPreRepCols = [('header (head)', 'NumEchant', 'Value')]
            sortPreRepAscend = True

            # b.iv. Report object
            preReport = ads.MCDSResultsPreReport(resultsSet=rsPreAct,
                                                 title='PyAuDiSam Validation: Pre-analyses',
                                                 subTitle='Pre-analysis results report',
                                                 anlysSubTitle='Pre-analysis results details',
                                                 description='Easy and parallel run through MCDSPreAnalyser',
                                                 keywords='pyaudisam, validation, pre-analysis',
                                                 lang='en', superSynthPlotsHeight=288,
                                                 # plotImgSize=(640, 400), plotLineWidth=1, plotDotWidth=4,
                                                 # plotFontSizes=dict(title=11, axes=10, ticks=9, legend=10),
                                                 sampleCols=samplePreRepCols, paramCols=paramPreRepCols,
                                                 resultCols=resultPreRepCols, synthCols=synthPreRepCols,
                                                 sortCols=sortPreRepCols, sortAscend=sortPreRepAscend,
                                                 tgtFolder=preAnlysr.workDir,
                                                 tgtPrefix='valtests-preanalyses-report-api')

            # b.iv. Excel report
            xlsxPreRep = preReport.toExcel()
            logger.info('Excel pre-report: ' + pl.Path(xlsxPreRep).resolve().as_posix())

            # b.v. HTML report
            htmlPreRep = preReport.toHtml()
            logger.info('HTML pre-report: ' + pl.Path(htmlPreRep).resolve().as_posix())

        # c. Load generated Excel report and compare it to reference one
        ddfRefPreRep = excelRefPreReport_fxt

        if not build:
            xlsxPreRep = preAnlysr.workDir / 'valtests-preanalyses-report.xlsx'
        ddfActPreRep = pd.read_excel(xlsxPreRep, sheet_name=None, index_col=0)

        self.compareExcelReports(ddfRefPreRep, ddfActPreRep)

        # c. Load generated HTML report and compare it to reference one
        htmlRefPreRepLines = htmlRefPreReportLines_fxt

        if not build:
            htmlPreRep = preAnlysr.workDir / 'valtests-preanalyses-report.html'
        with open(htmlPreRep) as file:
            htmlActPreRepLines = file.readlines()

        self.compareHtmlReports(htmlRefPreRepLines, htmlActPreRepLines, cliMode=False)

        # e. Cleanup generated report (well ... partially at least)
        #    for clearing next function's ground
        if cleanup:
            pl.Path(xlsxPreRep).unlink()
            pl.Path(htmlPreRep).unlink()

        # f. Done.
        logger.info(f'PASS testReports: MCDSResultsPreReport ctor, toExcel, toHtml')

    # ## 7. Generate HTML and Excel pre-analyses reports through pyaudisam command line
    def testReportsCli(self, sampleSpecMode, preAnalyser_fxt, excelRefPreReport_fxt, htmlRefPreReportLines_fxt):

        if sampleSpecMode != 'implicit':
            msg = 'testReportsCli(explicit): skipped, as not relevant'
            logger.info(msg)
            pytest.skip(msg)  # Raises an exception => function execution stops here.

        build = True  # Debug only: Set to False to avoid rebuilding the report

        preAnlysr, sampleSpecs = preAnalyser_fxt

        testPath = pl.Path(__file__).parent
        workPath = preAnlysr.workDir

        # a. Report "through the commande line"
        if build:
            argv = f'-p {testPath.as_posix()}/valtests-ds-params.py -w {workPath.as_posix()}' \
                   ' -n --prereports excel,html -u'.split()
            rc = ads.main(argv, standaloneLogConfig=False)
            logger.info(f'CLI run: rc={rc}')

        # b. Load generated Excel report and compare it to reference one
        ddfActPreRep = pd.read_excel(workPath / 'valtests-preanalyses-report.xlsx',
                                     sheet_name=None, index_col=0)

        ddfRefPreRep = excelRefPreReport_fxt
        self.compareExcelReports(ddfRefPreRep, ddfActPreRep)

        # c. Load generated HTML report and compare it to reference one
        with open(workPath / 'valtests-preanalyses-report.html') as file:
            htmlActPreRepLines = file.readlines()

        htmlRefPreRepLines = htmlRefPreReportLines_fxt
        self.compareHtmlReports(htmlRefPreRepLines, htmlActPreRepLines, cliMode=True)

        # d. No cleanup: let the final test class cleaner operate: _inifinalise_class()

        # e. Done.
        logger.info(f'PASS testReports: main, MCDSResultsPreReport ctor, toExcel, toHtml (command line mode)')
