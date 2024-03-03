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

import numpy as np
import pandas as pd

import pytest

import pyaudisam as ads

import unintval_utils as uivu


# Setup local logger.
logger = uivu.setupLogger('val.pnr', level=ads.DEBUG, otherLoggers={'ads.eng': ads.INFO2})


@pytest.mark.parametrize("implicitMode", [True, False])
class TestMcdsPreAnalyser:

    # Class and test function initialisers / finalisers ###########################
    @pytest.fixture(autouse=True, scope='class')
    def _inifinalise_class(self):

        what2Test = 'pre-analyser'

        uivu.logBegin(what=what2Test)

        # The code before yield is run before the first test function in this class
        yield
        # The code after yield is run after the last test function in this class

        uivu.logEnd(what=what2Test)

    @pytest.fixture(autouse=True, scope='function')
    def _inifinalise_function(self):

        # The code before yield is run before every test function
        yield
        # The code after yield is run after every test function

    # Test functions #############################################################

    # Short string for sample "identification"
    @staticmethod
    def sampleAbbrev(sSample):
        abrvSpe = ''.join(word[:4].title() for word in sSample['Espèce'].split(' ')[:2])
        sampAbbrev = '{}-{}-{}-{}'.format(abrvSpe, sSample.Passage.replace('+', ''),
                                          sSample.Adulte.replace('+', ''), sSample['Durée'])
        return sampAbbrev

    # Short string for analysis "identification"
    @classmethod
    def analysisAbbrev(cls, sAnlys):
        # Sample abbreviation
        abbrevs = [cls.sampleAbbrev(sAnlys)]

        # Model + Parameters abbreviation
        abbrevs += [sAnlys['FonctionClé'][:3].lower(), sAnlys['SérieAjust'][:3].lower()]
        dTroncAbrv = {'l': 'TrGche' if 'TrGche' in sAnlys.index else 'TroncGche',
                      'r': 'TrDrte' if 'TrDrte' in sAnlys.index else 'TroncDrte',
                      'm': 'NbTrches' if 'NbTrches' in sAnlys.index else 'NbTrModel'
                           if 'NbTrModel' in sAnlys.index else 'NbTrchMod', 'd': 'NbTrDiscr'}
        for abrv, name in dTroncAbrv.items():
            if name in sAnlys.index and not pd.isnull(sAnlys[name]):
                trcAbrv = sAnlys[name][0].lower() if isinstance(sAnlys[name], str) \
                                                  else int(sAnlys[name])
                abbrevs.append('{}{}'.format(abrv, trcAbrv))

        return '-'.join(abbrevs)

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

        logger.info(f'Preparing invidual. sightings ...')

        # ## 1. Individuals data set
        dfObsIndiv = ads.DataSet(uivu.pRefInDir / 'ACDC2019-Naturalist-ExtraitObsIndiv.ods',
                                 sheet='DonnéesIndiv').dfData

        logger.info(f'Invidual. sightings: n={len(dfObsIndiv)} =>\n'
                    + dfObsIndiv.to_string(min_rows=30, max_rows=30))
        indObsDesc = {col: dfObsIndiv[col].unique()
                      for col in ['Observateur', 'Point', 'Passage', 'Adulte', 'Durée', 'Espèce']}
        logger.info(f'... {indObsDesc}')

        # ## 2. Actual transects
        # (can't deduce them from data, some points are missing because of data selection)
        dfTransects = ads.DataSet(uivu.pRefInDir / 'ACDC2019-Naturalist-ExtraitObsIndiv.ods',
                                  sheet='Inventaires').dfData

        logger.info(f'Invidual. sightings: n={len(dfObsIndiv)} =>\n' + dfTransects.to_string(min_rows=30, max_rows=30))

        logger.info(f'Done preparing invidual. sightings.\n')

        return dfObsIndiv, dfTransects

    @pytest.fixture()
    def preAnalyser_fxt(self, implicitMode, inputDataSet_fxt):

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
        workDir = uivu.pTmpDir / 'mcds-preanlr'
        preAnlysr = \
            ads.MCDSPreAnalyser(dfObsIndiv, dfTransects=dfTransects, dSurveyArea=dSurveyArea,
                                transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                                sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols,
                                sampleSpecCustCols=[speciesAbbrevCol],
                                abbrevCol=sampleAbbrevCol, abbrevBuilder=self.sampleAbbrev, sampleIndCol=sampleNumCol,
                                distanceUnit='Meter', areaUnit='Hectare',
                                surveyType='Point', distanceType='Radial', clustering=False,
                                resultsHeadCols=dict(before=[sampleNumCol], sample=sampleSelCols,
                                                     after=([] if implicitMode else [speciesAbbrevCol])
                                                           + [sampleAbbrevCol]),
                                workDir=workDir, logProgressEvery=5)

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

        specMode = f"{'im' if implicitMode else 'ex'}plicit"
        logger.info(f"Preparing {specMode} sample specs ...")

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
        if not implicitMode:
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
            preAnlysr.explicitParamSpecs(dfExplParamSpecs=dfExplSampleSpecs if not implicitMode else None,
                                         implParamSpecs=implSampleSpecs if implicitMode else None,
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
        logger.info(f'Done preparing {specMode} sample specs.\n')

        return (preAnlysr, implSampleSpecs) if implicitMode else (preAnlysr, dfExplSampleSpecs)

    # ### c. Generate input files for manual analyses with Distance GUI (not needed for pre-analyses)
    def testExportDsInputData(self, preAnalyser_fxt, implicitMode):

        preAnlysr, sampleSpecs = preAnalyser_fxt

        # ### c. Generate input files for manual analyses with Distance GUI (not needed for pre-analyses)
        # * Only minimalistic checks done here, as already more deeply tested in test_unint_engine.py
        # * The exact same results can be produced through the command line for the implicit mode:
        # $ cd ..
        # $ python -m pyaudisam -p tests/valtests-ds-params.py -w tests/tmp/mcds-preanlr -n --distexport -u
        # i. Export distance files
        specMode = f"{'im' if implicitMode else 'ex'}plicit"
        logger.info(f"Exporting Distance files ({specMode} sample specs) ...")

        preAnlysr.exportDSInputData(implSampleSpecs=sampleSpecs if implicitMode else None,
                                    dfExplSampleSpecs=sampleSpecs if not implicitMode else None,
                                    format='Distance')

        # ii. Check list of generated files
        expdGenFileNames = []
        if implicitMode:
            for esp in sampleSpecs['_impl']['Espèce']:
                for pas in sampleSpecs['_impl']['Passage']:
                    for ad in sampleSpecs['_impl']['Adulte']:
                        for dur in sampleSpecs['_impl']['Durée']:
                            sampAbbrv = self.sampleAbbrev(pd.Series({'Espèce': esp, 'Passage': pas,
                                                                     'Adulte': ad, 'Durée': dur}))
                            expdGenFileNames.append(f'{sampAbbrv}-dist.txt')
        else:
            for _, sSampSpec in sampleSpecs.iterrows():
                sampAbbrv = self.sampleAbbrev(sSampSpec[['Espèce', 'Passage', 'Adulte', 'Durée']])
                expdGenFileNames.append(f'{sampAbbrv}-dist.txt')
        assert all(fpn.name in expdGenFileNames for fpn in preAnlysr.workDir.glob('*-dist.txt'))

        # iii. Check first generated file
        sampAbbrv = self.sampleAbbrev(pd.Series({'Espèce': 'Luscinia megarynchos', 'Passage': 'a+b',
                                                 'Adulte': 'm', 'Durée': '10mn'}))
        fpnSamp = preAnlysr.workDir / f'{sampAbbrv}-dist.txt'
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

        # iv. Cleanup generated files
        for fpnSamp in preAnlysr.workDir.glob('*-dist.txt'):
            fpnSamp.unlink()

        logger.info(f"PASS testMcdsPreAnalyser: exportDsInputData({specMode} sample specs)")

    # ### d. Run pre-analyses
    def testRun(self, preAnalyser_fxt, implicitMode):

        preAnlysr, sampleSpecs = preAnalyser_fxt

        # i. Run and measure performance
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
        specMode = f"{'im' if implicitMode else 'ex'}plicit"
        logger.info(f"Running pre-analyses: {specMode} sample specs, {threads} parallel threads ...")

        # Model fall-down strategy
        # Note: For real bird study analyses, you'll probably avoid NEXPON key function
        #       (a model with no shoulder: g'(0) << 1).
        #       And also HERMITE adjustment series (overkill fitting).
        modelStrategy = [dict(keyFn=kf, adjSr=js, estCrit='AIC', cvInt=95)
                         for js in ['COSINE', 'POLY', 'HERMITE']
                         for kf in ['HNORMAL', 'HAZARD', 'UNIFORM', 'NEXPON']]

        # BE CAREFUL: time.process_time() uses relative time for comparison only of codes among the same environment
        # NOT A REAL TIME reference
        start = time.perf_counter()
        preResults = preAnlysr.run(implSampleSpecs=sampleSpecs if implicitMode else None,
                                   dfExplSampleSpecs=sampleSpecs if not implicitMode else None,
                                   dModelStrategy=modelStrategy, threads=threads)
        end = time.perf_counter()

        logger.info(f'Elapsed time={end - start:.2f}s')

        # preResFileName = workDir / 'valtests-preanalyses-results.xlsx'
        # preResults.toExcel(preResFileName)

        # preResults.toExcel(workDir / 'valtests-preanalyses-results-fr.xlsx', lang='fr')

        # ### e. Check results: Compare to reference
        # (reference generated with same kind of "long" code like in III above, but on another data set)
        # i. Check presence of neutral and pass-through column in explicit spec. mode
        #    (it should have effectively passed through :-)
        speciesAbbrevCol = 'AbrevEsp'
        logger.debug('dfTransData(en).columns: ' + str(preResults.dfTransData('en').columns))
        assert implicitMode or speciesAbbrevCol in preResults.dfTransData('en').columns

        # ii. Load reference (prevent re-postComputation as this ref. file is old, with now missing computed cols)
        rsRef = preResults.copy(withData=False)
        rsRef.fromOpenDoc(uivu.pRefOutDir / 'ACDC2019-Naturalist-ExtraitPreResultats.ods', postComputed=True)

        logger.info(f'Reference results: n={len(rsRef)} =>\n' + rsRef.dfData.to_string(min_rows=30, max_rows=30))

        # iii Compare (ignore sample and analysis indexes, no use here).
        indexPreCols = [col for col in preResults.miCustomCols.to_list() if '(sample)' in col[0]] \
                       + [('parameters', 'estimator key function', 'Value'),
                          ('parameters', 'estimator adjustment series', 'Value')]

        subsetPreCols = [col for col in preResults.dfData.columns.to_list()
                         if col in rsRef.columns
                         and col not in indexPreCols + [col for col in preResults.miCustomCols.to_list()
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

        dfDiff = rsRef.compare(preResults, indexCols=indexPreCols, subsetCols=subsetPreCols,
                               dropCloser=13, dropNans=True)

        logger.info(f'Diff. to reference (relative): n={len(dfDiff)} =>\n'
                    + dfDiff.to_string(min_rows=30, max_rows=30))

        assert dfDiff.empty, 'Oh oh ... some unexpected differences !'

        # iv. To be perfectly honest ... there may be some 10**-14/-16 glitches (due to worksheet I/O ?) ... or not.
        dfComp = rsRef.compare(preResults, indexCols=indexPreCols, subsetCols=subsetPreCols, dropNans=True)
        dfComp = dfComp[(dfComp != np.inf).all(axis='columns')]

        logger.info(f'Diff. to reference (absolute): n={len(dfComp)} =>\n'
                    + dfComp.to_string(min_rows=30, max_rows=30))

        # f. Minimal check of pre-analysis folders
        dfEnRes = preResults.dfTransData('en')
        for anlysFolderPath in dfEnRes.RunFolder:
            assert {fpn.name for fpn in pl.Path(anlysFolderPath).iterdir()} \
                   == {'cmd.txt', 'data.txt', 'log.txt', 'output.txt', 'plots.txt', 'stats.txt'}

        # g. Cleanup pre-analyser (analysis folders)
        preAnlysr.cleanup()

        # h. Done.
        logger.info(f"PASS testMcdsPreAnalyser: run({specMode} sample specs)")
