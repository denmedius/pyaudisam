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

# Automated unit and integration tests for "analyser" submodule, results set part

# To run : simply run "pytest" or "python <this file>" in current folder
#          and check standard output ; and ./tmp/unt-ars.{datetime}.log for details

# WARNING : Work in progress ... not working yet ... but soon ... patience !

import sys

import pandas as pd

import pyaudisam as ads

import unintval_utils as uivu


# Setup local logger.
logger = uivu.setupLogger('unt.ars', level=ads.DEBUG,
                          otherLoggers={'ads.eng': ads.INFO2, 'ads.dat': ads.INFO,
                                        'ads.anr': ads.INFO5})

what2Test = 'analysis results set'


###############################################################################
#                         Actions to be done before any test                  #
###############################################################################
def testBegin():
    uivu.logBegin(what=what2Test)


###############################################################################
#                                Test Cases                                   #
###############################################################################

# AnalysisResultsSet class with specialised postComputeColumns
# (with an extra. post-computed column: Delta AIC)
class SpecialAnalysisResultsSet(ads.analyser.AnalysisResultsSet):

    def __init__(self, miCustomCols=None, dfCustomColTrans=None,
                 dComputedCols=None, dfComputedColTrans=None):
        super().__init__(ads.MCDSAnalysis, miCustomCols, dfCustomColTrans, dComputedCols, dfComputedColTrans)

    # Post-computations.
    def postComputeColumns(self):
        # Compute Delta AIC (AIC - min(group)) per { species, sample, precision, duration } group.
        # a. Minimum AIC per group
        aicColInd = ('detection probability', 'AIC value', 'Value')
        aicGroupColInds = [('sample', 'species', 'Value'), ('sample', 'periods', 'Value'),
                           ('sample', 'duration', 'Value'), ('variant', 'precision', 'Value')]
        df2Join = self._dfData.groupby(aicGroupColInds)[[aicColInd]].min()

        # b. Rename computed columns to target
        deltaAicColInd = ('detection probability', 'Delta AIC', 'Value')
        df2Join.columns = pd.MultiIndex.from_tuples([deltaAicColInd])

        # c. Join the column to the target data-frame
        self._dfData = self._dfData.join(df2Join, on=aicGroupColInds)

        # d. Compute delta-AIC in-place
        self._dfData[deltaAicColInd] = self._dfData[aicColInd] - self._dfData[deltaAicColInd]


# 5. AnalysisResultsSet and ResultsSet classes (1/2, see unint_mcds_anlys_results_test.py for 2/2)
# TODO: Split this too long test function !
def testArsCtorGettersSettersToFromFiles():

    # a. AnalysisResultsSet results object construction
    miCustCols = pd.MultiIndex.from_tuples([('id', 'index', 'Value'),
                                            ('sample', 'species', 'Value'),
                                            ('sample', 'periods', 'Value'),
                                            ('sample', 'duration', 'Value'),
                                            ('variant', 'precision', 'Value')])
    dfCustColTrans = \
        pd.DataFrame(index=miCustCols,
                     data=dict(en=['index', 'species', 'periods', 'duration', 'precision'],
                               fr=['numéro', 'espèce', 'périodes', 'durée', 'précision']))
    dCompCols = {('detection probability', 'Delta AIC', 'Value'):
                 len(ads.MCDSEngine.statSampCols()) + len(ads.MCDSAnalysis.MIRunColumns) + 11}  # Right before AIC
    dfCompColTrans = \
        pd.DataFrame(index=dCompCols.keys(),
                     data=dict(en=['Delta AIC'], fr=['Delta AIC']))

    rs = SpecialAnalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                   dComputedCols=dCompCols, dfComputedColTrans=dfCompColTrans)

    # ### b. Some getters
    # empty
    assert rs.empty
    logger.info('empty OK')

    # len
    assert len(rs) == 0
    logger.info('len OK')

    # index
    assert len(rs.index) == 0
    assert rs.index.to_list() == []
    logger.info('index OK')
    logger.info('index OK')

    # columns
    assert len(rs.columns) == 0
    logger.info('columns OK')

    # ### c. Append result rows
    # append
    sHead = pd.Series(index=miCustCols, data=list(range(len(miCustCols))))
    miResCols = ads.MCDSEngine.statSampCols().append(ads.MCDSAnalysis.MIRunColumns).append(ads.MCDSEngine.statModCols())
    sResult = pd.Series(index=miResCols, data=list(range(len(miResCols))))  # Fictive data, never mind !
    rs.append(sResult, sCustomHead=sHead)
    sResult = pd.Series(index=miResCols, data=list(range(1, len(miResCols) + 1)))  # Fictive data, never mind !
    rs.append(sResult, sCustomHead=sHead)
    sResult = pd.Series(index=miResCols, data=list(range(2, len(miResCols) + 2)))  # Fictive data, never mind !
    rs.append(sResult, sCustomHead=sHead)

    # ### d. Some getters again
    # dfRawData (no post-computed columns)
    dfRaw = rs.dfRawData
    logger.info('dfRaw: ' + dfRaw.to_string())

    # columns (Beware: rs.columns does trigger computation of ... computed columns !)
    assert len(rs._dfData.columns) == len(dfRaw.columns) and len(dfRaw.columns) == 113
    rawCols = rs._dfData.columns.to_list()
    logger.info('raw columns: ' + str(rawCols))

    # columns
    assert len(rs.columns) == 114  # The proof here !
    postCols = rs.columns.to_list()
    logger.info('raw columns: ' + str(postCols))

    # Check added == compute column
    assert (set(rs.columns.to_list()) - set(dfRaw.columns.to_list())
            == {('detection probability', 'Delta AIC', 'Value')})

    # dfData (post-computations already done, never mind)
    dfPost = rs.dfData
    logger.info('dfPost: ' + dfPost.to_string())

    # index
    assert len(rs.index) == 3
    assert rs.index.to_list() == [0, 1, 2]
    logger.info('dfData OK')

    # ### e. Getters: dfSubData
    columns = [('id', 'index', 'Value'), ('sample', 'species', 'Value'),
               ('sample', 'periods', 'Value'), ('sample', 'duration', 'Value'),
               ('detection probability', 'Delta AIC', 'Value')]
    index = [0, 2]
    dfSub = rs.dfSubData(index=index, columns=columns)
    logger.info('dfSub: ' + dfSub.to_string())
    logger.info('dfSubData OK')
    assert len(dfSub) == 2
    assert dfSub.index.to_list() == index
    assert dfSub.columns.to_list() == columns

    # ### f. Getters: Translation
    # dfTransData
    dfTrans = rs.dfTransData('fr')
    logger.info('dfTrans: ' + dfTrans.to_string())
    assert len(dfPost.columns) == len(dfTrans.columns)

    dfTrSub = rs.dfTransData('en', index=index, columns=columns)
    assert len(dfTrSub) == 2
    assert dfTrSub.index.to_list() == index
    assert dfTrSub.columns.to_list() == ['index', 'species', 'periods', 'duration', 'Delta AIC']
    logger.info('dfTrSub: ' + dfTrSub.to_string())
    logger.info('dfTransData OK')

    # ### g. Specs management
    rs.updateSpecs(d=dict(a=1, b=2), df=pd.DataFrame([dict(a=3, b=4), dict(a=7, b=9, v=90)]), reset=True)
    rs.updateSpecs(l=[9, -9], s=pd.Series(dict(e=3, f=5), name='serie'))
    logger.info('rs.specs: ' + str(rs.specs))

    try:
        rs.updateSpecs(l=[8, -8, 0])
        assert False, "Error: Should have refused to overwrite already existing 'l'"
    except AssertionError:
        logger.info('Good: Refused to overwrite existing spec if not explicitly authorised to')
    logger.info('rs.specs: ' + str(rs.specs))
    assert rs.specs['l'] == [9, -9]

    rs.updateSpecs(**dict(l=[7, -7, 77]), overwrite=True)
    logger.info('Good: Accepted to overwrite existing spec if explicitly authorised to')
    logger.info('rs.specs: ' + str(rs.specs))
    assert rs.specs['l'] == [7, -7, 77]
    logger.info('updateSpecs OK')

    # ### h. Imports and exports
    # #### i. Exports (with specs)
    # (see imports tests below for exported content checks)
    rs.toExcel(uivu.pTmpDir / 'results-set-uni.xlsx', sheetName='utest')

    rs.toExcel(uivu.pTmpDir / 'results-set-uni.xls', sheetName='utest')

    rs.toOpenDoc(uivu.pTmpDir / 'results-set-uni.ods', sheetName='utest')

    rs.toPickle(uivu.pTmpDir / 'results-set-uni.pickle.xz')

    rs.toPickle(uivu.pTmpDir / 'results-set-uni.pickle')

    # ### h. Imports and exports
    # #### ii. Imports with explicit format (with specs)
    # A. XLSX Format
    rs1 = SpecialAnalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                    dComputedCols=dCompCols, dfComputedColTrans=dfCompColTrans)
    rs1.fromExcel(uivu.pTmpDir / 'results-set-uni.xlsx', sheetName='utest')
    logger.info('rs1.dfData: ' + rs1.dfData.to_string())

    # Data
    assert rs1.dfData.equals(rs.dfData)  # == fails on NaNs in same places ...

    # Specs
    logger.info('rs1.specs: ' + str(rs1.specs))
    assert isinstance(rs1.specs['d'], dict) and rs1.specs['d'] == rs.specs['d']
    assert (isinstance(rs1.specs['df'], pd.DataFrame)
            and rs1.specs['df'].equals(rs.specs['df']))  # == fails on NaNs in same places
    assert isinstance(rs1.specs['l'], list) and rs1.specs['l'] == rs.specs['l']
    assert (isinstance(rs1.specs['s'], pd.Series)
            and rs1.specs['s'].name == rs.specs['s'].name
            and rs1.specs['s'].equals(rs.specs['s']))  # == fails on NaNs in same places

    # B. XLS Format
    rs2 = SpecialAnalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                    dComputedCols=dCompCols, dfComputedColTrans=dfCompColTrans)
    rs2.fromExcel(uivu.pTmpDir / 'results-set-uni.xls', sheetName='utest')
    logger.info(rs2.dfData.to_string())

    # Data
    assert rs2.dfData.equals(rs.dfData)  # == fails on NaNs in same places ...

    # Specs
    logger.info('rs2.specs: ' + str(rs2.specs))
    assert isinstance(rs2.specs['d'], dict) and rs2.specs['d'] == rs.specs['d']
    assert (isinstance(rs2.specs['df'], pd.DataFrame)
            and rs2.specs['df'].equals(rs.specs['df']))  # == fails on NaNs in same places
    assert isinstance(rs2.specs['l'], list) and rs2.specs['l'] == rs.specs['l']
    assert (isinstance(rs2.specs['s'], pd.Series)
            and rs2.specs['s'].name == rs.specs['s'].name
            and rs2.specs['s'].equals(rs.specs['s']))  # == fails on NaNs in same places

    # C. Format ODS
    rs3 = SpecialAnalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                    dComputedCols=dCompCols, dfComputedColTrans=dfCompColTrans)
    rs3.fromOpenDoc(uivu.pTmpDir / 'results-set-uni.ods', sheetName='utest')
    logger.info('rs3.dfData: ' + rs3.dfData.to_string())

    # Data
    assert rs3.dfData.equals(rs.dfData)  # == fails on NaNs in same places ...

    # Specs
    logger.info('rs3.specs: ' + str(rs3.specs))
    assert isinstance(rs3.specs['d'], dict) and rs3.specs['d'] == rs.specs['d']
    assert isinstance(rs3.specs['df'], pd.DataFrame) and rs3.specs['df'].equals(
        rs.specs['df'])  # == fails on NaNs in same places
    assert isinstance(rs3.specs['l'], list) and rs3.specs['l'] == rs.specs['l']
    assert isinstance(rs3.specs['s'], pd.Series) and rs3.specs['s'].name == rs.specs['s'].name \
           and rs3.specs['s'].equals(rs.specs['s'])  # == fails on NaNs in same places

    # D. Format pickle comprimé
    rs4 = SpecialAnalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                    dComputedCols=dCompCols, dfComputedColTrans=dfCompColTrans)
    rs4.fromPickle(uivu.pTmpDir / 'results-set-uni.pickle.xz')
    logger.info('rs4.dfData: ' + rs4.dfData.to_string())

    # Data
    assert rs4.dfData.equals(rs.dfData)  # == fails on NaNs in same places ...

    # Specs
    logger.info('rs4.specs: ' + str(rs4.specs))
    assert isinstance(rs4.specs['d'], dict) and rs4.specs['d'] == rs.specs['d']
    assert isinstance(rs4.specs['df'], pd.DataFrame) and rs4.specs['df'].equals(
        rs.specs['df'])  # == fails on NaNs in same places
    assert isinstance(rs4.specs['l'], list) and rs4.specs['l'] == rs.specs['l']
    assert isinstance(rs4.specs['s'], pd.Series) and rs4.specs['s'].name == rs.specs['s'].name \
           and rs4.specs['s'].equals(rs.specs['s'])  # == fails on NaNs in same places

    # E. Format pickle non comprimé
    rs5 = SpecialAnalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                    dComputedCols=dCompCols, dfComputedColTrans=dfCompColTrans)
    rs5.fromPickle(uivu.pTmpDir / 'results-set-uni.pickle')
    logger.info('rs5.dfData: ' + rs5.dfData.to_string())

    # Data
    assert rs5.dfData.equals(rs.dfData)  # == fails on NaNs in same places ...

    # Specs
    logger.info('rs5.specs: ' + str(rs5.specs))
    assert isinstance(rs5.specs['d'], dict) and rs5.specs['d'] == rs.specs['d']
    assert isinstance(rs5.specs['df'], pd.DataFrame) and rs5.specs['df'].equals(
        rs.specs['df'])  # == fails on NaNs in same places
    assert isinstance(rs5.specs['l'], list) and rs5.specs['l'] == rs.specs['l']
    assert isinstance(rs5.specs['s'], pd.Series) and rs5.specs['s'].name == rs.specs['s'].name \
           and rs5.specs['s'].equals(rs.specs['s'])  # == fails on NaNs in same places

    # #### iii. Imports with auto-detected format (with specs)
    # A. XLSX Format
    rs1 = SpecialAnalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                    dComputedCols=dCompCols, dfComputedColTrans=dfCompColTrans)
    rs1.fromFile(uivu.pTmpDir / 'results-set-uni.xlsx', sheetName='utest')
    logger.info('rs1.dfData: ' + rs1.dfData.to_string())

    # Data
    assert rs1.dfData.equals(rs.dfData)  # == fails on NaNs in same places ...

    # Specs
    logger.info('rs1.specs: ' + str(rs1.specs))
    assert isinstance(rs1.specs['d'], dict) and rs1.specs['d'] == rs.specs['d']
    assert isinstance(rs1.specs['df'], pd.DataFrame) and rs1.specs['df'].equals(
        rs.specs['df'])  # == fails on NaNs in same places
    assert isinstance(rs1.specs['l'], list) and rs1.specs['l'] == rs.specs['l']
    assert isinstance(rs1.specs['s'], pd.Series) and rs1.specs['s'].name == rs.specs['s'].name \
           and rs1.specs['s'].equals(rs.specs['s'])  # == fails on NaNs in same places

    # B. XLS Format
    rs2 = SpecialAnalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                    dComputedCols=dCompCols, dfComputedColTrans=dfCompColTrans)
    rs2.fromFile(uivu.pTmpDir / 'results-set-uni.xls', sheetName='utest')
    logger.info('rs2.dfData: ' + rs2.dfData.to_string())

    # Data
    assert rs2.dfData.equals(rs.dfData)  # == fails on NaNs in same places ...

    # Specs
    logger.info('rs2.specs: ' + str(rs2.specs))
    assert isinstance(rs2.specs['d'], dict) and rs2.specs['d'] == rs.specs['d']
    assert isinstance(rs2.specs['df'], pd.DataFrame) and rs2.specs['df'].equals(
        rs.specs['df'])  # == fails on NaNs in same places
    assert isinstance(rs2.specs['l'], list) and rs2.specs['l'] == rs.specs['l']
    assert isinstance(rs2.specs['s'], pd.Series) and rs2.specs['s'].name == rs.specs['s'].name \
           and rs2.specs['s'].equals(rs.specs['s'])  # == fails on NaNs in same places

    # C. Format ODS
    rs3 = SpecialAnalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                    dComputedCols=dCompCols, dfComputedColTrans=dfCompColTrans)
    rs3.fromFile(uivu.pTmpDir / 'results-set-uni.ods', sheetName='utest')
    logger.info('rs3.dfData: ' + rs3.dfData.to_string())

    # Data
    assert rs3.dfData.equals(rs.dfData)  # == fails on NaNs in same places ...

    # Specs
    logger.info('rs3.specs: ' + str(rs3.specs))
    assert isinstance(rs3.specs['d'], dict) and rs3.specs['d'] == rs.specs['d']
    assert isinstance(rs3.specs['df'], pd.DataFrame) and rs3.specs['df'].equals(rs.specs['df'])  # == fails on NaNs in same places
    assert isinstance(rs3.specs['l'], list) and rs3.specs['l'] == rs.specs['l']
    assert isinstance(rs3.specs['s'], pd.Series) and rs3.specs['s'].name == rs.specs['s'].name \
           and rs3.specs['s'].equals(rs.specs['s'])  # == fails on NaNs in same places

    # D. Format pickle comprimé
    rs4 = SpecialAnalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                    dComputedCols=dCompCols, dfComputedColTrans=dfCompColTrans)
    rs4.fromFile(uivu.pTmpDir / 'results-set-uni.pickle.xz')
    logger.info('rs4.dfData: ' + rs4.dfData.to_string())

    # Data
    assert rs4.dfData.equals(rs.dfData)  # == fails on NaNs in same places ...

    # Specs
    logger.info('rs4.specs: ' + str(rs4.specs))
    assert isinstance(rs4.specs['d'], dict) and rs4.specs['d'] == rs.specs['d']
    assert isinstance(rs4.specs['df'], pd.DataFrame) and rs4.specs['df'].equals(
        rs.specs['df'])  # == fails on NaNs in same places
    assert isinstance(rs4.specs['l'], list) and rs4.specs['l'] == rs.specs['l']
    assert isinstance(rs4.specs['s'], pd.Series) and rs4.specs['s'].name == rs.specs['s'].name \
           and rs4.specs['s'].equals(rs.specs['s'])  # == fails on NaNs in same places

    # E. Format pickle non comprimé
    rs5 = SpecialAnalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                    dComputedCols=dCompCols, dfComputedColTrans=dfCompColTrans)
    rs5.fromFile(uivu.pTmpDir / 'results-set-uni.pickle')
    logger.info('rs5.dfData: ' + rs5.dfData.to_string())

    # Data
    assert rs5.dfData.equals(rs.dfData)  # == fails on NaNs in same places ...

    # Specs
    logger.info('rs5.specs: ' + str(rs5.specs))
    assert isinstance(rs5.specs['d'], dict) and rs5.specs['d'] == rs.specs['d']
    assert isinstance(rs5.specs['df'], pd.DataFrame) and rs5.specs['df'].equals(
        rs.specs['df'])  # == fails on NaNs in same places
    assert isinstance(rs5.specs['l'], list) and rs5.specs['l'] == rs.specs['l']
    assert isinstance(rs5.specs['s'], pd.Series) and rs5.specs['s'].name == rs.specs['s'].name \
           and rs5.specs['s'].equals(rs.specs['s'])  # == fails on NaNs in same places

    # #### iv. Imports with default values for missing columns
    # TODO
    # How ?
    # For each file format,
    # - read target file (written above) with pandas API (not ResultsSet one)
    # - remove some columns
    # - overwrite target file with pandas API
    # - load target file with ResultsSet API, specifying default values for the missing columns
    # - check that results is OK

    logger.info0('PASS testArsCtorGettersSettersToFromFiles: Constructor, empty, len, columns, index, append,'
                 ' dfRawData, dfData, dfTransData, updateSpecs, toExcel(xlsx, xls), toOpenDoc, toPickle(std, xz),'
                 ' fromFile(xlsx, xls, ods, pickle.xz, pickle)')


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

            # Tests for AnalysisResultsSet
            testArsCtorGettersSettersToFromFiles()

            # Done.
            testEnd()

            # Success !
            rc = 0

        except Exception as exc:
            logger.exception(f'Exception: {exc}')
            rc = 1

    uivu.logEnd(what=what2Test, rc=rc)

    sys.exit(rc)
