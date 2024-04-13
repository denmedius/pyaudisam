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

# Automated validation tests for "analyser" submodule, MCDSAnalyser class part
# (adapted from valtest.ipynb notebook)

# To run : simply run "pytest" and see ./tmp/val.anr.{datetime}.log for detailed test log

import time
import pathlib as pl
import shutil
import re

import numpy as np
import pandas as pd

import pytest

import pyaudisam as ads

import unintval_utils as uivu


# Setup local logger.
logger = uivu.setupLogger('val.anr', level=ads.DEBUG, otherLoggers={'ads.eng': ads.INFO2})


class TestMcdsAnalyser:

    # Set to False to skip final cleanup (useful for debugging)
    KFinalCleanup = True

    # Class and test function initializers / finalizers ###########################
    @pytest.fixture(autouse=True, scope='class')
    def _inifinalizeClass(self):

        KWhat2Test = 'analyser'

        uivu.logBegin(what=KWhat2Test)

        # Setup a clear ground before starting
        uivu.setupWorkDir('val-anlr', cleanup=self.KFinalCleanup)

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

    # II. Run and report analyses
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
    def analyser_fxt(self, inputDataSet_fxt):

        logger.info(f'Preparing analysis specs ...')

        # ## 0. Data description
        transectPlaceCols = ['Point']
        passIdCol = 'Passage'
        effortCol = 'Effort'

        sampleDecCols = [effortCol, 'Distance']

        sampleNumCol = 'NumEchant'
        sampleSelCols = ['Espèce', passIdCol, 'Adulte', 'Durée']

        varIndCol = 'NumAnlys'
        anlysAbbrevCol = 'AbrevAnlys'

        dSurveyArea = dict(Zone='ACDC', Surface='2400')

        # ## 1. Individuals data set
        # ## 2. Actual transects
        # See inputDataSet_fxt

        # ## 3. Analysis specs
        dfAnlysSpecs = \
            ads.Analyser.explicitVariantSpecs(uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-SpecsAnalyses.xlsx',
                                              keep=['Echant1_impl', 'Echant2_impl', 'Modl_impl',
                                                    'Params1_expl', 'Params2_expl'],
                                              varIndCol=varIndCol,
                                              # convertCols={ 'Durée': int }, # float 'cause of Excel
                                              computedCols={anlysAbbrevCol: uivu.analysisAbbrev})

        logger.info(f'Analysis specs:\n{dfAnlysSpecs.to_string()}')

        # Recall analysis set without truncation params
        _dfAnlysSpecsNoOpt = dfAnlysSpecs[['Espèce', 'Passage', 'Adulte', 'Durée', 'FonctionClé', 'SérieAjust']] \
                                .drop_duplicates().reset_index(drop=True)
        logger.info(f'Analysis specs without opt. truncation:\n{_dfAnlysSpecsNoOpt.to_string()}')

        logger.info(f'Done preparing analysis specs.\n')

        logger.info(f'Preparing analyser ...')

        # ## 4A. Really run analyses
        # ### a. MCDSAnalyser object
        dfObsIndiv, dfTransects = inputDataSet_fxt
        anlysr = ads.MCDSAnalyser(dfObsIndiv, dfTransects=dfTransects, dSurveyArea=dSurveyArea,
                                  transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                                  sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols,
                                  abbrevCol=anlysAbbrevCol, anlysIndCol=varIndCol, sampleIndCol=sampleNumCol,
                                  distanceUnit='Meter', areaUnit='Hectare',
                                  surveyType='Point', distanceType='Radial', clustering=False,
                                  resultsHeadCols=dict(before=[varIndCol, sampleNumCol], sample=sampleSelCols,
                                                       after=[anlysAbbrevCol]),
                                  workDir=uivu.pWorkDir, logProgressEvery=5,
                                  defEstimCriterion='AIC', defCVInterval=95)

        logger.info(f'analyser specs:\n{anlysr.specs}')

        assert anlysr.specs == {
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

        logger.info(f'Done preparing analyser.\n')

        logger.info(f'Checking analysis specs ...')

        dfAnlysSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols, verdict, reasons = \
            anlysr.explicitParamSpecs(dfExplParamSpecs=dfAnlysSpecs, dropDupes=True, check=True)

        logger.info('Analysis spec. explicitations:')
        logger.info(f'* verdict: {verdict}')
        logger.info(f'* reasons: {reasons}')
        logger.info(f'* userParamSpecCols: n={len(userParamSpecCols)} => {userParamSpecCols}')
        logger.info(f'* intParamSpecCols: n={len(intParamSpecCols)} => {intParamSpecCols}')
        logger.info(f'* unmUserParamSpecCols: n={len(unmUserParamSpecCols)} => {unmUserParamSpecCols}')

        logger.info(f'Explicitated analysis specs: n={len(dfAnlysSpecs)} =>\n'
                    + dfAnlysSpecs.to_string(min_rows=30, max_rows=30))

        assert len(dfAnlysSpecs) == 48
        assert userParamSpecCols == ['FonctionClé', 'SérieAjust', 'TrGche', 'TrDrte', 'NbTrchMod']
        assert intParamSpecCols == ['EstimKeyFn', 'EstimAdjustFn', 'MinDist', 'MaxDist', 'FitDistCuts']
        assert unmUserParamSpecCols == []
        assert verdict
        assert not reasons

        # Done.
        logger.info(f'Done checking analysis specs.\n')

        return anlysr, dfAnlysSpecs

    @staticmethod
    def loadResults(anlysr, filePath, postComputed=False):

        logger.info(f'Loading results from {filePath.as_posix()} ...')

        rsRes = anlysr.setupResults()
        rsRes.fromFile(filePath, postComputed=postComputed)

        return rsRes

    @pytest.fixture()
    def refResults_fxt(self, analyser_fxt):

        logger.info(f'Preparing reference results ...')

        # Prevent re-postComputation as this ref. file is old, with now missing computed cols
        anlysr, _ = analyser_fxt
        rsRef = self.loadResults(anlysr, uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-Resultats.ods',
                                 postComputed=True)

        logger.info(f'Reference results: n={len(rsRef)} =>\n' + rsRef.dfData.to_string(min_rows=30, max_rows=30))

        return rsRef

    @staticmethod
    def compareResults(rsRef, rsAct):

        # * index = analysis "Id": sample Id columns and analysis indexes.
        indexCols = [col for col in rsAct.miCustomCols.to_list() if '(sample)' in col[0]] \
                    + [('parameters', 'estimator key function', 'Value'),
                       ('parameters', 'estimator adjustment series', 'Value'),
                       ('parameters', 'left truncation distance', 'Value'),
                       ('parameters', 'right truncation distance', 'Value'),
                       ('parameters', 'model fitting distance cut points', 'Value')]
        # * ignore:
        #   - sample Id columns and analysis indexes (used as comparison index = analysis "Id"),
        #   - 'run output' chapter results (start time, elapsed time, run folder ... always different),
        #   - text columns (not supported by ResultsSet.compare),
        #   - a few computed columns.
        subsetCols = [col for col in rsAct.dfData.columns.to_list()
                      if col in rsRef.columns
                      and col not in (indexCols + [col for col in rsAct.miCustomCols.to_list()
                                                   if '(sample)' not in col[0]]
                                      + [('parameters', 'estimator selection criterion', 'Value'),
                                         ('parameters', 'CV interval', 'Value'),
                                         ('run output', 'start time', 'Value'),
                                         ('run output', 'elapsed time', 'Value'),
                                         ('run output', 'run folder', 'Value'),
                                         ('detection probability', 'key function type', 'Value'),
                                         ('detection probability', 'adjustment series type', 'Value'),
                                         ('detection probability', 'Delta AIC', 'Value'),
                                         ('density/abundance', 'density of animals', 'Delta Cv')])]

        dfDiff = rsRef.compare(rsAct, indexCols=indexCols, subsetCols=subsetCols,
                               noneIsNan=True, dropCloser=12, dropNans=True)

        logger.info(f'Diff. to reference (relative): n={len(dfDiff)} =>\n'
                    + dfDiff.to_string(min_rows=30, max_rows=30))

        assert dfDiff.empty, 'Oh oh ... some unexpected differences !'

        # iv. To be perfectly honest ... there may be some 10**-14/-16 glitches (due to worksheet I/O ?) ... or not.
        dfComp = rsRef.compare(rsAct, indexCols=indexCols, subsetCols=subsetCols,
                               noneIsNan=True, dropNans=True)
        dfComp = dfComp[(dfComp != np.inf).all(axis='columns')]

        logger.info(f'Diff. to reference (absolute): n={len(dfComp)} =>\n'
                    + dfComp.to_string(min_rows=30, max_rows=30))

    # ### d. Run analyses through pyaudisam API
    def testRun(self, analyser_fxt):

        # i. Cleanup test folder (Note: avoid any Ruindows shell or explorer inside this folder !)
        anlysr, dfAnlysSpecs = analyser_fxt
        if anlysr.workDir.exists():
            shutil.rmtree(anlysr.workDir)

        # ii. Run and measure performance
        # Figures on a Ruindows 10 laptop with PCI-e SSD, "optimal performances" power scheme:
        # * 6-HT-core i7-8850H (python 3.7?):
        #   * 2019 or 2020 before 06: min=5, max=11s elapsed for 64 analyses, 6 threads ?
        # * 4-HT-core i5-8350U (python 3.8):
        #   * 2021-01: min=5.3, max=5.7s elapsed for 48 analyses, 6 threads ?
        #   * 2021-10-02: min=4.2s, max=5.7s (n=3) elapsed for 48 analyses, 6 threads ?
        # * 6-core i7-10750H, HT disabled (python 3.8):
        #   * 2022-01-01: mean=3.4s (n=4) elapsed for 48 analyses, 6 threads
        # * 6-HT-core i7-10750H, HT on (python 3.8):
        #   * 2023-11-02: mean=3.1s (n=2) elapsed for 48 analyses, 6 threads
        threads = 6
        logger.info(f'Running analyses: {threads} parallel threads ...')

        # BE CAREFUL: time.process_time() uses relative time for comparison only of codes among the same environment
        # NOT A REAL TIME reference
        start = time.perf_counter()
        rsAct = anlysr.run(dfAnlysSpecs, threads=6)
        end = time.perf_counter()

        logger.info(f'Elapsed time={end - start:.2f}s')

        # results.toExcel(anlysr.workDir / 'valtests-analyses-results-api.xlsx')
        # results.toExcel(anlysr.workDir / 'valtests-analyses-results-api-fr.xlsx', lang='fr')

        # ### e. Check results: Compare to reference
        # (reference generated with same kind of "long" code like in III above, but on another data set)
        # i. Load reference (prevent re-postComputation as this ref. file is old, with now missing computed cols)
        rsRef = self.loadResults(anlysr, uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-Resultats.ods',
                                 postComputed=True)
        logger.info(f'Reference results: n={len(rsRef)} =>\n' + rsRef.dfData.to_string(min_rows=30, max_rows=30))

        # ii. Compare
        self.compareResults(rsRef, rsAct)

        # f. Minimal check of analysis folders
        logger.info('Checking analysis folders (minimal) ...')
        dfEnRes = rsAct.dfTransData('en')
        for anlysFolderPath in dfEnRes.RunFolder:
            assert {fpn.name for fpn in pl.Path(anlysFolderPath).iterdir()} \
                   == {'cmd.txt', 'data.txt', 'log.txt', 'output.txt', 'plots.txt', 'stats.txt'}
        logger.info('... done checking analysis folders (minimal).')

        # g. Cleanup analyser (analysis folders)
        anlysr.cleanup()

        # h. Done.
        logger.info(f'PASS testRun: run, cleanup')

    # Run analyses through pyaudisam command line interface
    # (implicit mode only, see valtests-ds-params.py)
    def testRunCli(self, analyser_fxt, refResults_fxt):

        testPath = pl.Path(__file__).parent
        anlysr, _ = analyser_fxt

        # a. Cleanup test folder (Note: avoid any Ruindows shell or explorer inside this folder !)
        if anlysr.workDir.exists():
            shutil.rmtree(anlysr.workDir)

        # b. Run "through the commande line"
        argv = f'-p {testPath.as_posix()}/valtests-ds-params.py -w {anlysr.workDir.as_posix()} -n --analyses -u'.split()
        rc = ads.main(argv, standaloneLogConfig=False)
        logger.info(f'CLI run: rc={rc}')

        # c. Load results
        rsAct = self.loadResults(anlysr, anlysr.workDir / 'valtests-analyses-results.xlsx')
        logger.info(f'Actual results: n={len(rsAct)} =>\n' + rsAct.dfData.to_string(min_rows=30, max_rows=30))

        # d. Compare to reference.
        rsRef = refResults_fxt
        self.compareResults(rsRef, rsAct)

        # e. Minimal check of analysis folders
        logger.info('Checking analysis folders (minimal) ...')
        dfEnRes = rsAct.dfTransData('en')
        for anlysFolderPath in dfEnRes.RunFolder:
            assert {fpn.name for fpn in pl.Path(anlysFolderPath).iterdir()} \
                   == {'cmd.txt', 'data.txt', 'log.txt', 'output.txt', 'plots.txt', 'stats.txt'}
        logger.info('... done checking analysis folders (minimal).')

        # f. Don't clean up work folder / analysis folders : needed for report generations below

        # g. Done.
        logger.info(f'PASS testRunCli: main, run (command line mode)')

    @pytest.fixture()
    def excelRefReport_fxt(self):

        return pd.read_excel(uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-Rapport.xlsx',
                             sheet_name=None, index_col=0)

    @staticmethod
    def compareExcelReports(ddfRefReport, ddfActReport):

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

    @pytest.fixture()
    def htmlRefReportLines_fxt(self):

        with open(uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-Rapport.html') as file:
            repLines = file.readlines()

        return repLines

    @staticmethod
    def compareHtmlReports(refReportLines, actReportLines, cliMode=False):

        logger.info('Preprocessing HTML reports for comparison ...')

        # Pre-process actual report lines
        remRefLines = remActLines = 0

        # * list unique cell ids (keeping the original order) in both reports
        KRETableId = r'#T_([0-9a-f]{5})_'
        refTableIds = uivu.listUniqueStrings(KRETableId, refReportLines)
        actTableIds = uivu.listUniqueStrings(KRETableId, actReportLines)
        assert len(refTableIds) == 2
        assert len(actTableIds) == len(refTableIds)

        refDetTableId = refTableIds[1]  # "Details" table Id

        logger.info(f'* found table Ids: ref={refTableIds}, act={actTableIds}')

        # * replace each cell id in the actual report by the corresponding ref. report one
        uivu.replaceStrings(actTableIds, refTableIds, actReportLines)

        # * list unique analysis folders (keeping the original order) in both reports
        KREAnlysDir = r'="./([a-zA-Z0-9-_]+)/'
        refAnlysDirs = uivu.listUniqueStrings(KREAnlysDir, refReportLines)
        actAnlysDirs = uivu.listUniqueStrings(KREAnlysDir, actReportLines)
        assert len(refAnlysDirs) == len(actAnlysDirs)

        # * replace each analysis folder in the actual report by the corresponding ref. report one
        uivu.replaceStrings(froms=actAnlysDirs, tos=refAnlysDirs, lines=actReportLines)

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
        KREDetailsMcds74LastCol = rf'<td id="T_{refDetTableId}_row.+_col122"'  # No such column prior with MCDS 6.2
        if not uivu.selectLines(KREDetailsMcds74LastCol, actReportLines):
            KREDetailsUndocCol = rf'<td id="T_{refDetTableId}_row.+_col(66|67|68|69|70|71|72|73|74|75|76)"'
            remRefLines += uivu.removeLines(KREDetailsUndocCol, refReportLines)

        #   - header and style lines in ref "Details" table (simpler report comparison, at the price of not checking
        #     cell styling ... but this is not really useful / much examined after all,
        #     and column headers, which is a bit more annoying ...)
        KREDetailsLine2Drop = rf'^#T_{refDetTableId}_row0_col0,'
        remRefLines += uivu.removeLines(KREDetailsLine2Drop, refReportLines)
        remActLines += uivu.removeLines(KREDetailsLine2Drop, actReportLines)

        KREDetailsLine2Drop = rf'^ +}}#T_{refDetTableId}_row'
        remRefLines += uivu.removeLines(KREDetailsLine2Drop, refReportLines)
        remActLines += uivu.removeLines(KREDetailsLine2Drop, actReportLines)

        KREDetailsLine2Drop = rf'<table id="T_{refDetTableId}_" ><thead>'
        remRefLines += uivu.removeLines(KREDetailsLine2Drop, refReportLines)
        remActLines += uivu.removeLines(KREDetailsLine2Drop, actReportLines)

        logger.info(f'* removed {remRefLines} ref. and {remActLines} act. lines')

        # * remove cell id/class in "Details" table (simpler report comparison, at the price of not checking
        #   cell styling ... but this is not really useful / much examined after all)
        KDetailsConstCellIdClass = '<td>'
        KREDetailsCellIdClass = rf'<td id="T_{refDetTableId}_row.+_col.+" class=".+" >'
        repRefLines = uivu.replaceRegExps(re2Search=KREDetailsCellIdClass, repl=KDetailsConstCellIdClass,
                                          lines=refReportLines)
        repActLines = uivu.replaceRegExps(re2Search=KREDetailsCellIdClass, repl=KDetailsConstCellIdClass,
                                          lines=actReportLines)

        KDetailsConstCellIdClass = '<th>'
        KREDetailsCellIdClass = rf'<th id="T_{refDetTableId}_level.+_row.+" class=".+" >'
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

    # ## 7. Generate HTML and Excel analyses reports through pyaudisam API
    def testReports(self, analyser_fxt, excelRefReport_fxt, htmlRefReportLines_fxt):

        build = True  # Debug only: Set to False to avoid rebuilding the report, and only check it
        cleanup = True  # Debug only: Set to False to prevent cleaning at the end

        # Pre-requisites : uncleaned analyser work dir (we need the results file and analysis folders).
        anlysr, _ = analyser_fxt
        if build:
            logger.info('Checking analyser results presence ...')
            anlysrResFilePath = anlysr.workDir / 'valtests-analyses-results.xlsx'
            assert anlysr.workDir.is_dir() and anlysrResFilePath.is_file()
            anlysFolders = [path for path in anlysr.workDir.iterdir() if path.is_dir()]
            assert len(anlysFolders) == 48
            logger.info('Done checking analyser results presence: OK.')

            # a. Load results
            # (the last generated one, through implicit or explicit sample specs:
            #  never mind, they are the same as checked above)
            rsAct = self.loadResults(anlysr, anlysrResFilePath)
            logger.info(f'Actual results: n={len(rsAct)} =>\n' + rsAct.dfData.to_string(min_rows=30, max_rows=30))

            # b. Generate Excel and HTML reports
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
                R.CLParEstKeyFn, R.CLParEstAdjSer,
                # R.CLParEstSelCrit, R.CLParEstCVInt,
                R.CLParTruncLeft, R.CLParTruncRight, R.CLParModFitDistCuts
            ]

            resultRepCols = [
                ('header (head)', 'NumAnlys', 'Value'),
                R.CLRunStatus,
                R.CLNObs, R.CLEffort, R.CLSightRate, R.CLNAdjPars,
                R.CLAic, R.CLChi2, R.CLKS, R.CLDCv,
                R.CLCmbQuaBal3, R.CLCmbQuaBal2, R.CLCmbQuaBal1,
                R.CLDensity, R.CLDensityMin, R.CLDensityMax,
                R.CLNumber, R.CLNumberMin, R.CLNumberMax,
                R.CLEswEdr, R.CLPDetec
            ]

            # b.ii. Synthesis sub-report : Selected analysis results columns for the table
            synthRepCols = [
                ('header (head)', 'NumEchant', 'Value'),
                ('header (sample)', 'Espèce', 'Value'),
                ('header (sample)', 'Passage', 'Value'),
                ('header (sample)', 'Adulte', 'Value'),
                ('header (sample)', 'Durée', 'Value'),
                ('header (head)', 'NumAnlys', 'Value'),

                R.CLParEstKeyFn, R.CLParEstAdjSer,
                # R.CLParEstSelCrit, R.CLParEstCVInt,
                R.CLParTruncLeft, R.CLParTruncRight, R.CLParModFitDistCuts,

                R.CLNTotObs, R.CLNObs, R.CLNTotPars, R.CLEffort, R.CLDeltaAic,
                R.CLChi2, R.CLKS, R.CLCvMUw, R.CLCvMCw, R.CLDCv,
                R.CLPDetec, R.CLPDetecMin, R.CLPDetecMax, R.CLDensity, R.CLDensityMin, R.CLDensityMax,

                R.CLSightRate,
                R.CLCmbQuaBal1, R.CLCmbQuaBal2, R.CLCmbQuaBal3,
                R.CLCmbQuaChi2, R.CLCmbQuaKS, R.CLCmbQuaDCv,

                R.CLGrpOrdSmTrAic,
                R.CLGrpOrdClTrChi2KSDCv,  # R.CLGrpOrdClTrChi2,
                R.CLGrpOrdClTrDCv,
                R.CLGrpOrdClTrQuaBal1, R.CLGrpOrdClTrQuaBal2, R.CLGrpOrdClTrQuaBal3, R.CLGrpOrdClTrQuaChi2,
                R.CLGrpOrdClTrQuaKS, R.CLGrpOrdClTrQuaDCv,
                R.CLGblOrdChi2KSDCv, R.CLGblOrdQuaBal1, R.CLGblOrdQuaBal2, R.CLGblOrdQuaBal3,
                R.CLGblOrdQuaChi2, R.CLGblOrdQuaKS, R.CLGblOrdQuaDCv,
                R.CLGblOrdDAicChi2KSDCv,
                R.CLRunFolder,
            ]

            # b.iii. Sorting columns for all the sub-reports
            sortRepCols = \
                [('header (head)', 'NumEchant', 'Value')] \
                + [R.CLParTruncLeft, R.CLParTruncRight,
                   R.CLDeltaAic,
                   R.CLCmbQuaBal3]

            sortRepAscend = [True] * (len(sortRepCols) - 1) + [False]

            # b.iv. Report object
            report = ads.MCDSResultsFullReport(resultsSet=rsAct,
                                               sampleCols=sampleRepCols, paramCols=paramRepCols,
                                               resultCols=resultRepCols, synthCols=synthRepCols,
                                               sortCols=sortRepCols, sortAscend=sortRepAscend,
                                               title='PyAuDiSam Validation: Analyses',
                                               subTitle='Global analysis full report',
                                               anlysSubTitle='Detailed report',
                                               description='Easy and parallel run through MCDSAnalyser',
                                               keywords='pyaudisam, validation, analysis',
                                               lang='en', superSynthPlotsHeight=288,
                                               # plotImgSize=(640, 400), plotLineWidth=1, plotDotWidth=4,
                                               # plotFontSizes=dict(title=11, axes=10, ticks=9, legend=10),
                                               tgtFolder=anlysr.workDir, tgtPrefix='valtests-analyses-report')

            # b.iv. Excel report
            xlsxRep = report.toExcel()
            logger.info('Excel report: ' + pl.Path(xlsxRep).resolve().as_posix())

            # b.v. HTML report
            htmlRep = report.toHtml()
            logger.info('HTML report: ' + pl.Path(htmlRep).resolve().as_posix())

        else:
            xlsxRep = anlysr.workDir / 'valtests-analyses-report.xlsx'
            htmlRep = anlysr.workDir / 'valtests-analyses-report.html'

        # c. Load generated Excel report and compare it to reference one
        ddfRefRep = excelRefReport_fxt

        ddfActRep = pd.read_excel(xlsxRep, sheet_name=None, index_col=0)

        self.compareExcelReports(ddfRefRep, ddfActRep)

        # c. Load generated HTML report and compare it to reference one
        htmlRefRepLines = htmlRefReportLines_fxt

        with open(htmlRep) as file:
            htmlActRepLines = file.readlines()

        self.compareHtmlReports(htmlRefRepLines, htmlActRepLines, cliMode=False)

        # e. Cleanup generated report (well ... partially at least)
        #    for clearing next function's ground
        if cleanup:
            pl.Path(xlsxRep).unlink()
            pl.Path(htmlRep).unlink()

        # f. Done.
        logger.info(f'PASS testReports: MCDSResultsFullReport ctor, toExcel, toHtml')

    # ## 7. Generate HTML and Excel analyses reports through pyaudisam command line
    def testReportsCli(self, analyser_fxt, excelRefReport_fxt, htmlRefReportLines_fxt):

        build = True  # Debug only: Set to False to avoid rebuilding the report

        testPath = pl.Path(__file__).parent
        anlysr, _ = analyser_fxt
        workPath = anlysr.workDir

        # a. Report "through the commande line"
        if build:
            argv = f'-p {testPath.as_posix()}/valtests-ds-params.py -w {workPath.as_posix()}' \
                   ' -n --reports excel:full,html:full -u'.split()
            rc = ads.main(argv, standaloneLogConfig=False)
            logger.info(f'CLI run: rc={rc}')

        # b. Load generated Excel report and compare it to reference one
        ddfActRep = pd.read_excel(workPath / 'valtests-analyses-report.xlsx',
                                  sheet_name=None, index_col=0)

        ddfRefRep = excelRefReport_fxt
        self.compareExcelReports(ddfRefRep, ddfActRep)

        # c. Load generated HTML report and compare it to reference one
        with open(workPath / 'valtests-analyses-report.html') as file:
            htmlActRepLines = file.readlines()

        htmlRefRepLines = htmlRefReportLines_fxt
        self.compareHtmlReports(htmlRefRepLines, htmlActRepLines, cliMode=True)

        # d. No cleanup: let the final test class cleaner operate: _inifinalizeClass()

        # e. Done.
        logger.info(f'PASS testReports: main, MCDSResultsFullReport ctor, toExcel, toHtml (command line mode)')
