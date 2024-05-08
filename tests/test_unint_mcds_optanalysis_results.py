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

# Automated unit and integration tests for "optanalyser" submodule, results set part

# To run : simply run "pytest" and check standard output + ./tmp/unt-ors.{datetime}.log for details

import pandas as pd

import pytest

import pyaudisam as ads

import unintval_utils as uivu


# Mark module
pytestmark = pytest.mark.unintests

# Setup local logger.
logger = uivu.setupLogger('unt.ors', level=ads.DEBUG,
                          otherLoggers={'ads.eng': ads.INFO2, 'ads.dat': ads.INFO,
                                        'ads.anr': ads.INFO5, 'ads.onr': ads.DEBUG3})

# Set to False to skip final cleanup (useful for debugging)
KFinalCleanup = True

KWhat2Test = 'opt-analysis results set'


###############################################################################
#                         Actions to be done before any test                  #
###############################################################################
def testBegin():
    uivu.logBegin(what=KWhat2Test)
    # uivu.setupWorkDir('unt-mores')


###############################################################################
#                                Test Cases                                   #
###############################################################################
RS = ads.MCDSTruncOptanalysisResultsSet

# Results post-computation parameters
KLdTruncIntrvSpecs = [dict(col='left', minDist=5.0, maxLen=5.0),
                      dict(col='right', minDist=25.0, maxLen=25.0)]
KTruncIntrvEpsilon = 1e-6


# ## 14. MCDSTruncOptAnalysisResultsSet
# a. An MCDSTruncOptanalyser object for creating MCDSTruncOptanalysisResultsSet objects
def mcdsOptAnalyser():

    # Source / Results data
    transectPlaceCols = ['Point']
    passIdCol = 'Passage'
    effortCol = 'Effort'
    sampleDistCol = 'Distance'
    sampleDecCols = [effortCol, sampleDistCol]
    sampleNumCol = 'NumEchant'
    sampleSelCols = ['Espèce', passIdCol, 'Adulte', 'Durée']
    # sampleAbbrevCol = 'AbrevEchant'
    # optIndCol = 'IndOptim'
    # optAbbrevCol = 'AbrevOptim'
    dSurveyArea = dict(Zone='ACDC', Surface='2400')

    # General DS analysis parameters
    varIndCol = 'NumAnlys'
    anlysAbbrevCol = 'AbrevAnlys'
    anlysParamCols = ['FonctionClé', 'SérieAjust', 'TrGche', 'TrDrte', 'NbTrchMod']
    distanceUnit = 'Meter'
    areaUnit = 'Hectare'
    surveyType = 'Point'
    distanceType = 'Radial'
    clustering = False

    # Load individualised observations and actual transects
    indivObsFile = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-ObsIndiv.ods'
    dfObsIndiv = ads.DataSet(indivObsFile, sheet='DonnéesIndiv').dfData
    dfTransects = ads.DataSet(indivObsFile, sheet='Inventaires').dfData
    dict(indivObs=len(dfObsIndiv), transects=len(dfTransects))

    # What's better to create an MCDS(Opt)AnalysisResultsSet object than a MCDSTruncationOptanalyser instance ?
    return ads.MCDSTruncationOptanalyser(dfObsIndiv, dfTransects=dfTransects, dSurveyArea=dSurveyArea,
                                         transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                                         sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols,
                                         sampleDistCol=sampleDistCol,
                                         abbrevCol=anlysAbbrevCol, abbrevBuilder=uivu.analysisAbbrev,
                                         anlysIndCol=varIndCol, sampleIndCol=sampleNumCol,
                                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                                         surveyType=surveyType, distanceType=distanceType, clustering=clustering,
                                         resultsHeadCols=dict(before=[varIndCol, sampleNumCol], sample=sampleSelCols,
                                                              after=anlysParamCols + [anlysAbbrevCol]),
                                         ldTruncIntrvSpecs=KLdTruncIntrvSpecs, truncIntrvEpsilon=KTruncIntrvEpsilon)


@pytest.fixture()
def mcdsOptAnalyser_fxt():
    return mcdsOptAnalyser()


# ### b. _sampleDistTruncGroups
# (this one is specialized from MCDSAnalysisResultsSet's, so it's NOT the same)
def testMcdsOptArsCtorFromOpenDocGetData(mcdsOptAnalyser_fxt):
    
    optanlr = mcdsOptAnalyser_fxt
    
    # Load results to play with ...
    resFileName = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-UnitestOptResultats.ods'
    logger.info('Loading results from {} ...'.format(resFileName))
    results = optanlr.setupResults()
    results.fromOpenDoc(resFileName, postComputed=True)  # Prevent re-post-computation : not a problem here, but longer
    
    # Get results table and extract only needed columns
    # (Note: No post-computed column used here, so ... recomputation authorised, but not needed, and slower)
    CLSampNum = ('header (head)', 'NumEchant', 'Value')
    dfRes = results.getData(copy=True)[[CLSampNum, RS.CLOptimTruncFlag, RS.CLParTruncLeft, RS.CLParTruncRight]]
    dfRes.head()
    
    # Expected results per sample.
    KDSampGroupNums = \
        {
            0: dict(left={18: 0, 17: 1, 5: 0, 16: 1, 15: 2, 12: 0, 8: 1, 14: 0, 9: 1, 1: 2, 0: 1, 19: 2},
                    right={18: 0, 17: 0, 5: 1, 16: 1, 15: 1, 12: 0, 8: 0, 14: 1, 9: 1, 1: 1, 0: 2, 19: 1}),
            1: dict(left={2: 0, 22: 1, 24: 0, 21: 1, 20: 2, 26: 0, 23: 1, 27: 0, 28: 1, 30: 2},
                    right={2: 0, 22: 0, 24: 1, 21: 1, 20: 1, 26: 0, 23: 0, 27: 1, 28: 1, 30: 1}),
            2: dict(left={29: 0, 32: 0, 31: 0, 33: 0, 35: 0, 34: 0},
                    right={29: 0, 32: 1, 31: 2, 33: 0, 35: 1, 34: 2}),
            3: dict(left={37: 0, 36: 0, 40: 0, 39: 0, 43: 0, 44: 0, 41: 2, 47: 1},
                    right={37: 0, 36: 1, 40: 2, 39: 0, 43: 1, 44: 2, 41: 1, 47: 1}),
            4: dict(left={48: 0, 49: 1, 50: 0, 52: 1, 56: 0, 53: 1, 54: 0, 57: 1, 51: 0, 60: 0},
                    right={48: 0, 49: 0, 50: 1, 52: 1, 56: 0, 53: 0, 54: 1, 57: 1, 51: 2, 60: 1}),
            5: dict(left={62: 0, 69: 1, 63: 0, 65: 1, 67: 0, 68: 1, 66: 0, 64: 1},
                    right={62: 0, 69: 0, 63: 1, 65: 1, 67: 0, 68: 0, 66: 1, 64: 1})
        }
    
    dRefGroupNums = dict()  # For building _distTruncGroups reference :-)
    for lblSamp in dfRes[CLSampNum].unique():
    
        logger.debug(f'* {lblSamp}')
        dSampGroupNums = RS._sampleDistTruncGroups(dfRes[dfRes[CLSampNum] == lblSamp],
                                                   ldIntrvSpecs=KLdTruncIntrvSpecs, intrvEpsilon=KTruncIntrvEpsilon)
        assert all(sGroupNums.eq(pd.Series(KDSampGroupNums[lblSamp][colAlias])).all()
                   for colAlias, sGroupNums in dSampGroupNums.items())
    
        for colAlias, sGroupNums in dSampGroupNums.items():
            if colAlias not in dRefGroupNums:
                dRefGroupNums[colAlias] = sGroupNums
            else:
                dRefGroupNums[colAlias] = dRefGroupNums[colAlias].append(sGroupNums)

    logger.info0('PASS testMcdsOptArsCtorFromOpenDocGetData(1): Constructor, fromOpenDoc, getData')

    # Check and build results reference
    # lblSamp = 5
    #
    # df = dfRes[dfRes[CLSampNum] == lblSamp].copy()
    #
    # dGroupNums = RS._sampleDistTruncGroups(df, ldIntrvSpecs=KLdTruncIntrvSpecs, intrvEpsilon=KTruncIntrvEpsilon)
    # for colAlias, sGroupNums in dGroupNums.items():
    #    df[RS.DCLGroupTruncDist[colAlias]] = sGroupNums
    #    logger.info(colAlias, '=', sGroupNums.to_dict(), ',', sep='')
    # df

    # ### c. _distTruncGroups
    #
    # Well, no test needed actually, as not any specialized from MCDSAnalysisREsultsSet's ...
    dGroupNums = results._distTruncGroups()

    # Auto-check
    assert all(sGroupNums.eq(dRefGroupNums[colAlias]).all() for colAlias, sGroupNums in dGroupNums.items())

    logger.info0('PASS testMcdsOptArsCtorFromOpenDocGetData(2): _distTruncGroups')


# ### d. _filterSortKeySchemes
def testMcdsOptArsFilterSortKeySchemes(mcdsOptAnalyser_fxt):

    optanlr = mcdsOptAnalyser_fxt

    # Load results to play with ...
    resFileName = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-UnitestOptResultats.ods'
    logger.info('Loading results from {} ...'.format(resFileName))
    results = optanlr.setupResults(ldFilSorKeySchemes=None)  # Will use predefined filter-sort key generation schemes
    results.fromOpenDoc(resFileName, postComputed=True)  # Prevent re-post-computation : not a problem here, but longer

    assert results._filterSortKeySchemes() == ads.MCDSTruncOptanalysisResultsSet.AutoFilSorKeySchemes

    # Load results to play with ...
    resFileName = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-UnitestOptResultats.ods'
    logger.info('Loading results from {} ...'.format(resFileName))
    ldFilSorKeySchemes = \
        [dict(key=RS.CLGrpOrdClTrChi2KSDCv,  # Best Chi2 & KS & DCv inside groups of close truncation params
              sort=[RS.CLOptimTruncFlag, RS.CLGroupTruncLeft, RS.CLGroupTruncRight,
                    RS.CLChi2, RS.CLKS, RS.CLDCv, RS.CLNObs, RS.CLRunStatus],
              ascend=[True, True, False, False, True, False, True],
              group=[RS.CLOptimTruncFlag, RS.CLGroupTruncLeft, RS.CLGroupTruncRight]),
         dict(key=RS.CLGblOrdDAicChi2KSDCv,
              sort=[RS.CLParTruncLeft, RS.CLParTruncRight, RS.CLParModFitDistCuts,
                    RS.CLDeltaAic, RS.CLChi2, RS.CLKS, RS.CLDCv, RS.CLNObs, RS.CLRunStatus],
              ascend=[True, True, True, True, False, False, True, False, True], napos='first')]
    results = optanlr.setupResults(ldFilSorKeySchemes=ldFilSorKeySchemes)  # Will not use predefined ones.
    results.fromOpenDoc(resFileName, postComputed=True)  # Prevent re-post-computation : not a problem here, but longer

    assert results._filterSortKeySchemes() == ldFilSorKeySchemes

    logger.info0('PASS testMcdsOptArsFilterSortKeySchemes: Constructor, fromOpenDoc, _filterSortKeySchemes')


# ### e. _filterOnExecCode
def testMcdsOptArsFilterOnExecCode(mcdsOptAnalyser_fxt):

    optanlr = mcdsOptAnalyser_fxt

    # Load results to play with ...
    # (OK, it's MCDSAnalysisResultsSet's one, but it's not any specialized
    # in MCDS(TruncOpt)AnalysisResultsSet, so it's the same)
    resFileName = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-UnitestOptResultats.ods'
    logger.info('Loading results from {} ...'.format(resFileName))
    results = optanlr.setupResults()
    results.fromOpenDoc(resFileName, postComputed=True)  # Prevent re-post-computation : not a problem here, but longer

    # Get results table (Note: No post-computed column used here,
    # so ... re-computation authorised, but not needed, and slower)
    dfFilSorRes = results.getData(copy=True)
    # dfFilSorRes = dfFilSorRes[dfFilSorRes[('header (head)', 'NumEchant', 'Value')] == 5].copy()  # For debugging.

    # Filter params
    dupSubset = [RS.CLDensity, RS.CLDensityMin, RS.CLDensityMax]
    dDupRounds = {RS.CLDensity: 1, RS.CLDensityMin: 2, RS.CLDensityMax: 1}

    # Save index before filtering
    iBefore = dfFilSorRes.index
    len(dfFilSorRes)

    # Filter
    filSorSteps = ads.analyser._FilterSortSteps(filSorSchId='ExCodeTst', resultsSet=results, lang='fr')  # Steps logger
    RS._filterOnExecCode(dfFilSorRes, filSorSteps, results.sampleIndCol,
                         dupSubset=dupSubset, dDupRounds=dDupRounds)
    # Look at steps
    filSorSteps.toList()

    # List filtered-out results
    sFiltered = set(iBefore) - set(dfFilSorRes.index)
    logger.info(', '.join(str(i) for i in sFiltered))

    # Auto-check
    sExpected = {+0, 14, 17, 8, 9,  # sample 0: + => because of poor status elimination
                 +21, 22, 23, 27,  # sample 1: otherwise, because of non-first duplicate
                 31,  # sample 2 ... etc.
                 +41, +39,  # sample 3
                 +56, 49, 53, 52, 57,  # sample 4
                 66, 69, 68, 65, 64}  # sample 5
    logger.info(', '.join(str(i) for i in sExpected))
    assert sFiltered == sExpected, 'Oh, oh ... not what we expected'

    logger.info0('PASS testMcdsOptArsFilterOnExecCode: Constructor, fromOpenDoc, _filterOnExecCode')


# ### f. _filterOnAicMultiQua
def testMcdsOptArsFilterOnAicMultiQua(mcdsOptAnalyser_fxt):

    optanlr = mcdsOptAnalyser_fxt

    # Load results to play with ...
    # (OK, it's MCDSAnalysisResultsSet's one, but it's not any specialized
    # in MCDS(TruncOpt)AnalysisResultsSet, so it's the same)
    resFileName = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-UnitestOptResultats.ods'
    logger.info('Loading results from {} ...'.format(resFileName))
    results = optanlr.setupResults()
    results.fromFile(resFileName, postComputed=True)  # Prevent re-post-computation : we don't want it !

    # Get results table without re-post computation : we want post-computed columns as in source workbook !
    dfFilSorRes = results.getData(copy=True)
    # dfFilSorRes = dfFilSorRes[dfFilSorRes[('header (head)', 'NumEchant', 'Value')] == 5].copy()  # For debugging

    # Filter params
    minSightRate = 92.0
    nBestAicOrd = 2
    nBestMQuaOrd = 1
    whichBestMQuaOrd = [RS.CLGrpOrdClTrChi2KSDCv, RS.CLGrpOrdClTrQuaBal3, RS.CLGrpOrdClTrQuaChi2]
    nFinalQua = 3
    whichFinalQua = RS.CLCmbQuaBal3
    ascFinalQua = False

    # Save index before filtering
    iBefore = dfFilSorRes.index
    len(dfFilSorRes)

    # Filter
    filSorSteps = ads.analyser._FilterSortSteps(filSorSchId='ExAicMQuaTst', resultsSet=results, lang='fr')
    RS._filterOnAicMultiQua(dfFilSorRes, filSorSteps, results.sampleIndCol,
                            minSightRate=minSightRate, nBestAicOrd=nBestAicOrd,
                            nBestMQuaOrd=nBestMQuaOrd, whichBestMQuaOrd=whichBestMQuaOrd,
                            nFinalQua=nFinalQua, whichFinalQua=whichFinalQua, ascFinalQua=ascFinalQua)
    # Look at steps
    filSorSteps.toList()

    # List filtered-out results
    sFiltered = set(iBefore) - set(dfFilSorRes.index)
    logger.info(', '.join(str(i) for i in sFiltered))

    # Auto-check filtered-out results
    # (causes => +: lower AIC, 0+: not of best multi-qua. orders, -0: poor sight rate, other: no of N best finalQua)
    sExpected = {+5, +8, +9, 0 + 16, 0 + 17, 0 + 18, 14 - 0, 19 - 0, 12,  # sample 0
                 +23, +24, 0 + 20, 0 + 21, +26, 30 - 0, 22,  # sample 1
                 +31, 0 + 35, 33,  # sample 2
                 +43, 0 + 44, 41 - 0, 47 - 0, 39,  # sample 3
                 +49, 0 + 48, 50 - 0, 52 - 0, 54 - 0, 57 - 0, 56,  # sample 4
                 +64, 0 + 65, 0 + 66, 63 - 0, 62}  # sample 5
    logger.info(', '.join(str(i) for i in sExpected))
    assert sFiltered == sExpected, 'Oh, oh ... not what we expected'

    logger.info0('PASS testMcdsOptArsFilterOnAicMultiQua: Constructor, fromFile, _filterOnExecCode')


# ### g. Non regression (reload results and re-post-compute)
def testMcdsOptArsNonRegression(mcdsOptAnalyser_fxt):

    ads.logger('ads.anr', level=ads.DEBUG1)
    ads.logger('ads.onr', level=ads.DEBUG)

    optanlr = mcdsOptAnalyser_fxt

    # #### i. Load reference results
    # (generated once through valtests.ipynb/IV. Run truncation opt-analyses ...)
    refRes = optanlr.setupResults()
    resFileName = uivu.pRefOutDir / 'ACDC2019-Naturalist-extrait-OptResultats.ods'
    logger.info('Loading results from {} ...'.format(resFileName))
    refRes.fromFile(resFileName, postComputed=True)  # Prevent re-post-computation : this is our reference !
    optanlr.shutdown()

    # #### ii Trigger re-post-computation on a copy
    #
    # (post-computations are the first thing we want to check for non regression)
    # ads.logger('ads.dat', level=ads.INFO, reset=True)
    # ads.logger('ads.anr', level=ads.DEBUG4, reset=True)
    # ads.logger('ads.onr', level=ads.DEBUG4, reset=True)

    res = refRes.copy()
    res.setPostComputed(False)

    # Trigger now !
    logger.info(f'res.dfData: {len(res)} rows =>\n' + res.dfData.to_string())
    logger.info('refRes.columns: ' + str(refRes.columns.to_list()))

    # #### iii Compare re-post-computed filter & sort Group and Order columns to reference.
    indexCols = [('header (head)', 'NumAnlys', 'Value'),
                 ('run output', 'run folder', 'Value'),  # Needed for multiple opt. tries kept per (same Id) analysis
                 ('header (tail)', 'TrGche', 'Value'),
                 ('header (tail)', 'TrDrte', 'Value'),
                 ('header (tail)', 'NbTrchMod', 'Value'),
                 ('header (tail)', 'OptimTrunc', 'Value')]
    subsetCols = [col for col in refRes.columns if col[0] == ads.MCDSAnalysisResultsSet.CLCAutoFilSor]
    logger.info('Checking filter & sort Group and Order columns: ' + str(subsetCols))

    dfComp = refRes.dfData.set_index(indexCols).sort_index()[subsetCols] \
                   .compare(res.dfData.set_index(indexCols).sort_index()[subsetCols])
    logger.info('refRes.dfData.compare(ref.dfData): ' + dfComp.to_string())
    assert dfComp.empty

    # #### iv Compare all other re-post-computed columns to reference.
    indexCols = ['DossierExec']
    relDiffCols = ['Chi2 P', 'Delta AIC', 'Delta CoefVar Densité',
                   'Qual Equi 1', 'Qual Equi 2', 'Qual Equi 3', 'Qual Chi2+', 'Qual KS+', 'Qual DCv+']
    dfRelDiff = ads.DataSet.compareDataFrames(refRes.dfTransData('fr').sort_values(by=indexCols),
                                              res.dfTransData('fr').sort_values(by=indexCols),
                                              indexCols=indexCols, subsetCols=relDiffCols, dropCloser=8)
    assert dfRelDiff.empty

    dfRelDiff = ads.DataSet.compareDataFrames(refRes.dfTransData('fr').sort_values(by=indexCols),
                                              res.dfTransData('fr').sort_values(by=indexCols),
                                              indexCols=indexCols, subsetCols=relDiffCols, dropCloser=14)
    logger.info('compareDataFrames(refres, res, 14): ' + dfRelDiff.to_string())

    # #### v. Compare all other non-re-computed columns to reference (one never knows ;-).
    relDiffCols = ['TrGche', 'TrDrte', 'NbTrchMod', 'NTot Obs', 'Min Dist', 'Max Dist', 'NObs', 'NEchant', 'Effort',
                   'TxContact', 'CoefVar TxContact', 'Min TxContact', 'Max TxContact', 'DegLib TxContact',
                   'Taux Obs', 'NbTot Pars', 'AIC', 'AICc', 'BIC', 'LogProba', 'KS P', 'CvM Uw P', 'CvM Cw P',
                   'f/h(0)', 'CoefVar f/h(0)', 'Min f/h(0)', 'Max f/h(0)', 'DegLib f/h(0)',
                   'PDetec', 'CoefVar PDetec', 'Min PDetec', 'Max PDetec', 'DegLib PDetec',
                   'EDR/ESW', 'CoefVar EDR/ESW', 'Min EDR/ESW', 'Max EDR/ESW', 'DegLib EDR/ESW',
                   'DensClu', 'CoefVar DensClu', 'Min DensClu', 'Max DensClu', 'DegLib DensClu',
                   'Densité', 'Delta CoefVar Densité', 'CoefVar Densité',
                   'Min Densité', 'Max Densité', 'DegLib Densité',
                   'Nombre', 'CoefVar Nombre', 'Min Nombre', 'Max Nombre', 'DegLib Nombre']
    ads.DataSet.compareDataFrames(refRes.dfTransData('fr').sort_values(by='DossierExec'),
                                  res.dfTransData('fr').sort_values(by='DossierExec'),
                                  indexCols=['DossierExec'],
                                  subsetCols=relDiffCols,
                                  dropCloser=14)

    logger.info0('PASS testMcdsOptArsNonRegression: Constructor, fromFile, postComputeColumns')


###############################################################################
#                         Actions to be done after all tests                  #
###############################################################################
def testEnd():
    # if KFinalCleanup:
    #     uivu.cleanupWorkDir()
    uivu.logEnd(what=KWhat2Test)
