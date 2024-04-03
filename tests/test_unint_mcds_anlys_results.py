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

# To run : simply run "pytest" and check standard output + ./tmp/unt-ars.{datetime}.log for details

import copy

import numpy as np
import pandas as pd

import pytest

import pyaudisam as ads

import unintval_utils as uivu


# Setup local logger.
logger = uivu.setupLogger('unt.ars', level=ads.DEBUG,
                          otherLoggers={'ads.eng': ads.INFO2, 'ads.dat': ads.INFO,
                                        'ads.anr': ads.INFO5})

# Set to False to skip final cleanup (useful for debugging)
KFinalCleanup = True

KWhat2Test = 'MCDS analysis results set'


###############################################################################
#                         Actions to be done before any test                  #
###############################################################################
def testBegin():
    uivu.logBegin(what=KWhat2Test)
    # uivu.setupWorkDir('unt-mares')


###############################################################################
#                                Test Cases                                   #
###############################################################################

# 5. AnalysisResultsSet and ResultsSet classes (2/2, see test_unint_anlys_results.py for 1/2)
# i. Comparison
def testMcdsArsCompare():

    # MCDSAnalysisResultsSet objects and loading from files.
    modelIdCols = ['Model']
    # modelParamCols = ['LTrunc', 'RTrunc', 'FitDistCuts', 'DiscrDistCuts']
    sampleIdCols = ['Species', 'Periods', 'Prec.', 'Duration']
    caseIdCols = ['AnlysNum', 'SampNum'] + sampleIdCols + modelIdCols
    sampCols = [('sample', col, 'Value') for col in sampleIdCols]
    miSampCols = pd.MultiIndex.from_tuples(sampCols)
    custCols = [('sample', 'AnlysNum', 'Value'), ('sample', 'SampNum', 'Value')] \
               + sampCols + [('model', 'Model', 'Value')]
    miCustCols = pd.MultiIndex.from_tuples(custCols)
    dfCustColTrans = \
        pd.DataFrame(index=miCustCols,
                     data=dict(en=caseIdCols,
                               fr=['NumAnlys', 'NumSamp', 'Espèce', 'Périodes', 'Préc.', 'Durée', 'Modèle']))

    # Réference (built with Distance 7.3)
    rsDist = ads.MCDSAnalysisResultsSet(miSampleCols=miSampCols, sampleIndCol=('sample', 'SampNum', 'Value'),
                                        miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                        distanceUnit='Meter', areaUnit='Hectare',
                                        surveyType='Point', distanceType='Radial', clustering=False)
    rsDist.fromFile(uivu.pRefInDir / 'ACDC2019-Papyrus-ALAARV-TURMER-comp-dist-auto.ods', sheetName='RefDist73',
                    postComputed=True)  # Avoid re-computations, some columns are now missing, files are old actually !

    # Results obtained with via pyaudisam.
    rsAuto = ads.MCDSAnalysisResultsSet(miSampleCols=miSampCols, sampleIndCol=('sample', 'SampNum', 'Value'),
                                        miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                        distanceUnit='Meter', areaUnit='Hectare',
                                        surveyType='Point', distanceType='Radial', clustering=False)
    rsAuto.fromFile(uivu.pRefInDir / 'ACDC2019-Papyrus-ALAARV-TURMER-comp-dist-auto.ods', sheetName='ActAuto',
                    postComputed=True)  # Avoid re-computations, some columns are now missing, files are old actually !

    # Index columns
    indexCols = custCols + [('parameters', 'left truncation distance', 'Value'),
                            ('parameters', 'right truncation distance', 'Value'),
                            ('parameters', 'model fitting distance cut points', 'Value'),
                            ('parameters', 'distance discretisation cut points', 'Value')]

    # Columns to compare (after removing DeltaDCV and DeltaAIC because they depend on the whole analysis set,
    #                     which differs between the reference and the auto-produced,
    #                     and removing the execution time column).
    subsetCols = [col for col in rsDist.columns.to_list()
                  if col not in indexCols + [('run output', 'run time', 'Value'),
                                             ('density/abundance', 'density of animals', 'Delta Cv'),
                                             ('detection probability', 'Delta AIC', 'Value')]]
    logger.info('subsetCols: ' + str(subsetCols))

    dfRelDiff = rsDist.compare(rsAuto, subsetCols=subsetCols, indexCols=indexCols)
    logger.info('dfRelDiff: ' + dfRelDiff.to_string())
    assert len(dfRelDiff.columns) == 21
    assert len(dfRelDiff) == len(rsDist)

    dfRelDiff = rsDist.compare(rsAuto, subsetCols=subsetCols, indexCols=indexCols, dropCloser=16, dropNans=False)
    logger.info('dfRelDiff: ' + dfRelDiff.to_string())
    assert len(dfRelDiff.columns) == 21
    assert len(dfRelDiff) == 8

    dfRelDiff = rsDist.compare(rsAuto, subsetCols=subsetCols, indexCols=indexCols, dropCloser=16, dropNans=True)
    logger.info('dfRelDiff: ' + dfRelDiff.to_string())
    assert len(dfRelDiff.columns) == 21
    assert len(dfRelDiff) == 3

    dfRelDiff = rsDist.compare(rsAuto, subsetCols=subsetCols, indexCols=indexCols, dropCloser=5, dropNans=True)
    logger.info('dfRelDiff: ' + dfRelDiff.to_string())
    assert len(dfRelDiff.columns) == 21
    assert len(dfRelDiff) == 2

    # Drop also closer columns
    dfRelDiff = rsDist.compare(rsAuto, subsetCols=subsetCols, indexCols=indexCols, dropCloser=5, dropNans=True,
                               dropCloserCols=True)
    logger.info('dfRelDiff: ' + dfRelDiff.to_string())
    assert len(dfRelDiff.columns) == 19

    logger.info0('PASS testMcdsArsCompare: constructor, fromFile(ODS), compare')


# ### j. Post-computations
def testMcdsArsPostCompute():
    
    # MCDSAnalysisResultsSet object + loading from file
    modelIdCols = ['Model']
    # modelParamCols = ['LTrunc', 'RTrunc', 'FitDistCuts', 'DiscrDistCuts']
    sampleIdCols = ['Species', 'Periods', 'Prec.', 'Duration']
    caseIdCols = ['AnlysNum', 'SampNum'] + sampleIdCols + modelIdCols
    sampCols = [('sample', col, 'Value') for col in sampleIdCols]
    miSampCols = pd.MultiIndex.from_tuples(sampCols)
    custCols = [('sample', 'AnlysNum', 'Value'), ('sample', 'SampNum', 'Value')] \
               + sampCols + [('model', 'Model', 'Value')]
    miCustCols = pd.MultiIndex.from_tuples(custCols)
    dfCustColTrans = \
        pd.DataFrame(index=miCustCols,
                     data=dict(en=caseIdCols,
                               fr=['NumAnlys', 'NumSamp', 'Espèce', 'Périodes', 'Préc.', 'Durée', 'Modèle']))
    rsAuto = ads.MCDSAnalysisResultsSet(miSampleCols=miSampCols, sampleIndCol=('sample', 'SampNum', 'Value'),
                                        miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                        distanceUnit='Meter', areaUnit='Hectare',
                                        surveyType='Point', distanceType='Radial', clustering=False)
    rsAuto.fromFile(uivu.pRefInDir / 'ACDC2019-Papyrus-ALAARV-TURMER-comp-dist-auto.ods', sheetName='ActAuto')

    # Trigger post-computations
    logger.info('rsAuto.dfData: ' + rsAuto.dfData.to_string())

    # Load reference from file
    rsAutoRef = ads.MCDSAnalysisResultsSet(miSampleCols=miSampCols, sampleIndCol=('sample', 'SampNum', 'Value'),
                                           miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                           distanceUnit='Meter', areaUnit='Hectare',
                                           surveyType='Point', distanceType='Radial', clustering=False)
    rsAutoRef.fromFile(uivu.pRefOutDir / 'ACDC2019-Papyrus-ALAARV-TURMER-resultats-postcomp.ods')

    # Comparison of loaded results to reference
    # a. Index columns
    indexCols = custCols + [('parameters', 'left truncation distance', 'Value'),
                            ('parameters', 'right truncation distance', 'Value'),
                            ('parameters', 'model fitting distance cut points', 'Value'),
                            ('parameters', 'distance discretisation cut points', 'Value'),
                            ('parameters', 'estimator key function', 'Value'),
                            ('parameters', 'estimator adjustment series', 'Value'),
                            ('parameters', 'estimator selection criterion', 'Value')]
    # b. Colonnes to compare : we ignore ...
    # * DeltaDCV et DeltaAIC because they depend on the whole set of analyses actually done to get the results,
    #   that is possibly different sets in the 2 cases.
    # * other string-typed columns (comparison not implemented)
    subsetCols = [col for col in rsAutoRef.columns.to_list()
                  if col not in indexCols + [('run output', 'run time', 'Value'), ('run output', 'run folder', 'Value'),
                                             ('density/abundance', 'density of animals', 'Delta Cv'),
                                             ('detection probability', 'Delta AIC', 'Value'),
                                             ('detection probability', 'key function type', 'Value'),
                                             ('detection probability', 'adjustment series type', 'Value')]]
    # c. Comparison
    dfRelDiff = rsAuto.compare(rsAutoRef, subsetCols=subsetCols, indexCols=indexCols, dropCloser=15)
    logger.info('dfRelDiff: ' + dfRelDiff.to_string())
    assert len(dfRelDiff) == 0

    logger.info0('PASS testMcdsArsPostCompute: constructor, postCompute, columns, fromFile(ODS')


# ## 13. MCDSAnalysisResultsSet
RS = ads.MCDSAnalysisResultsSet


# ### a. _groupingIntervals
def testMcdsArsGroupingIntervals():

    # Old code that was slightly modified ... for non-regression tests.
    def intervalsOld(sDists, minIntrvDist, maxIntrvLen, intrvEpsilon):
        # For some reason, need for enforcing float dtype ... otherwise dtype='O' !?
        sSelDist = sDists.dropna().astype(float).sort_values()
        # List non-null differences between consecutive sorted distances
        dfIntrv = pd.DataFrame(dict(dist=sSelDist.values))
        if not dfIntrv.empty:
            dfIntrv['deltaDist'] = dfIntrv.dist.diff()
            dfIntrv.loc[dfIntrv.dist.idxmin(), 'deltaDist'] = np.inf
            dfIntrv.dropna(inplace=True)
            dfIntrv = dfIntrv[dfIntrv.deltaDist > 0].copy()
            # Deduce start (min) and end (sup) for each such interval (left-closed, right-open)
            dfIntrv['dMin'] = dfIntrv.loc[dfIntrv.deltaDist > minIntrvDist, 'dist']
            dfIntrv['dSup'] = dfIntrv.loc[dfIntrv.deltaDist > minIntrvDist, 'dist'].shift(-1).dropna()
            dfIntrv.loc[dfIntrv['dMin'].idxmax(), 'dSup'] = np.inf
            dfIntrv.dropna(inplace=True)
            dfIntrv['dSup'] = \
                dfIntrv['dSup'].apply(lambda supV: sSelDist[sSelDist < supV].max() + intrvEpsilon)
            dfIntrv = dfIntrv[['dMin', 'dSup']].reset_index(drop=True)
            # If these intervals are too wide, cut them up in equal sub-intervals and make them new intervals
            lsNewIntrvs = list()
            for _, sIntrv in dfIntrv.iterrows():
                if sIntrv.dSup - sIntrv.dMin > maxIntrvLen:
                    # logger.info(sIntrv.dSup, '-', sIntrv.dMin, '>', maxIntrvLen)
                    nSubIntrvs = (sIntrv.dSup - sIntrv.dMin) / maxIntrvLen
                    nSubIntrvs = int(nSubIntrvs) if nSubIntrvs - int(nSubIntrvs) < 0.5 else int(nSubIntrvs) + 1
                    subIntrvLen = (sIntrv.dSup - sIntrv.dMin) / nSubIntrvs
                    lsNewIntrvs += [pd.Series(dict(dMin=sIntrv.dMin + nInd * subIntrvLen,
                                                   dSup=min(sIntrv.dMin + (nInd + 1) * subIntrvLen,
                                                            sIntrv.dSup)))
                                    for nInd in range(nSubIntrvs)]
                else:
                    lsNewIntrvs.append(sIntrv)
            dfIntrv = pd.DataFrame(lsNewIntrvs).reset_index(drop=True)
            dfIntrv.sort_values(by='dMin', inplace=True)
            dfIntrv.rename(columns=dict(dMin='vmin', dSup='vsup'), inplace=True)
        return dfIntrv

    minDist = 5
    maxLen = 10
    eps = 1e-6
    v0 = 1
    KCases = [dict(values=[np.nan],  # empty series (after cleanup)
                   intervals=[]),
              dict(values=[v0],  # 1 isolated value in 1 shortest interval
                   intervals=[{'vmin': v0, 'vsup': v0 + eps}]),
              dict(values=[v0, 2, 3, 8, 9, v0 + maxLen, v0 + maxLen * 1.5 - 0.01,  # Don't cut 1st interval
                           22, 30, 35, np.nan, 44.9],  # 22 isolated inside 1 shortest interval
                   intervals=[{'vmin': v0, 'vsup': v0 + maxLen * 1.5 - 0.01},
                              {'vmin': 22, 'vsup': 22 + eps},
                              {'vmin': 30, 'vsup': 35 + eps},
                              {'vmin': 44.9, 'vsup': 44.9 + eps}]),
              dict(values=[v0, 2, np.nan, 3, 8, 9, v0 + maxLen, v0 + maxLen * 1.5,  # Cut 1st interval
                           22, 27, 35, 37, 39, 44.9],
                   intervals=[{'vmin': v0, 'vsup': 8.5},
                              {'vmin': 8.5 + eps, 'vsup': 16.0 + eps},
                              {'vmin': 22, 'vsup': 27 + eps},
                              {'vmin': 35, 'vsup': 39 + eps},
                              {'vmin': 44.9, 'vsup': 44.9 + eps}])]

    logger.info(f'{minDist=}, {maxLen=}, {eps=}:')
    for case in KCases:
        logger.info(f'* {case["values"]}: ')
        dfIntrvs = RS._groupingIntervals(pd.Series(case['values']), minDist=minDist, maxLen=maxLen, epsilon=eps)
        logger.info(' => ' + dfIntrvs.to_string())
        dfDiff = ads.DataSet.compareDataFrames(dfIntrvs.reset_index(), pd.DataFrame(case['intervals']).reset_index(),
                                               indexCols=['index'], dropCloser=4)
        assert dfDiff.empty, 'Oh, oh ... not what we expected ; diff to ref= ' + str(dfDiff.to_dict('index')) \
                             + '\nref= ' + str(case['intervals'])

        # Non regression, comparing to old code results
        dfIntrvsOld = intervalsOld(pd.Series(case['values']),
                                   minIntrvDist=minDist, maxIntrvLen=maxLen, intrvEpsilon=eps)
        dfDiffOld = ads.DataSet.compareDataFrames(dfIntrvs.reset_index(), dfIntrvsOld.reset_index(),
                                                  indexCols=['index'], dropCloser=6)
        assert dfDiff.empty, 'Oh, oh ... not what we expected ; diff to old= ' + str(dfDiffOld.to_dict('index'))

    logger.info0('PASS testMcdsArsGroupingIntervals: _groupingIntervals, compareDataFrames')


# ### b. _intervalIndex
def testMcdsArsIntervalIndex():

    eps = 1e-6

    dfIntrvs = pd.DataFrame([{'vmin': 1, 'vsup': 8.5 + eps},
                             {'vmin': 8.5 + eps, 'vsup': 16 + eps},
                             {'vmin': 22, 'vsup': 27 + eps},
                             {'vmin': 35, 'vsup': 39 + eps},
                             {'vmin': 44.9, 'vsup': 44.9 + eps}])
    logger.info('dfIntrvs: ' + dfIntrvs.to_string())

    sValues = pd.Series(
        [8.5, 39, 44.9, 1, 11, 16 + eps, 2, np.nan, 3, 8, 8.5 + eps, 9, 16, 22 - eps, 27, 27 + 2 * eps, 35, 22, 37])

    # Compute and show results
    sGroups = sValues.apply(RS._intervalIndex, dfIntervals=dfIntrvs)
    pd.DataFrame(dict(values=sValues, group=sGroups))

    # Auto-check
    assert sGroups.eq(pd.Series({0: 1, 1: 4, 2: 5, 3: 1, 4: 2, 5: -1, 6: 1, 7: 0, 8: 1, 9: 1, 10: 2,
                                 11: 2, 12: 2, 13: -1, 14: 3, 15: -1, 16: 4, 17: 3, 18: 4})).all()

    logger.info0('PASS testMcdsArsIntervalIndex: _intervalIndex')


# An MCDSAnalyser object for creating MCDSAnalysisResultsSet objects
def mcdsAnalyser():

    # Source / Results data
    transectPlaceCols = ['Point']
    passIdCol = 'Passage'
    effortCol = 'Effort'
    sampleDistCol = 'Distance'
    sampleDecCols = [effortCol, sampleDistCol]
    sampleNumCol = 'NumEchant'
    sampleSelCols = ['Espèce', passIdCol, 'Adulte', 'Durée']
    # sampleAbbrevCol = 'AbrevEchant'
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

    # Results post-computation parameters
    ldTruncIntrvSpecs = [dict(col='left', minDist=5.0, maxLen=5.0), dict(col='right', minDist=25.0, maxLen=25.0)]
    truncIntrvEpsilon = 1e-6

    # Load individualised observations and actual transects
    indivObsFile = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-ObsIndiv.ods'
    dfObsIndiv = ads.DataSet(indivObsFile, sheet='DonnéesIndiv').dfData
    dfTransects = ads.DataSet(indivObsFile, sheet='Inventaires').dfData
    dict(indivObs=len(dfObsIndiv), transects=len(dfTransects))

    # What's better to create an MCDSAnalysisResultsSet objets than a MCDSAnalyser instance ?
    return ads.MCDSAnalyser(dfObsIndiv, dfTransects=dfTransects, dSurveyArea=dSurveyArea,
                            transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                            sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols,
                            abbrevCol=anlysAbbrevCol, abbrevBuilder=uivu.analysisAbbrev,
                            anlysIndCol=varIndCol, sampleIndCol=sampleNumCol,
                            distanceUnit=distanceUnit, areaUnit=areaUnit,
                            surveyType=surveyType, distanceType=distanceType, clustering=clustering,
                            resultsHeadCols=dict(before=[varIndCol, sampleNumCol], sample=sampleSelCols,
                                                 after=anlysParamCols + [anlysAbbrevCol]),
                            ldTruncIntrvSpecs=ldTruncIntrvSpecs, truncIntrvEpsilon=truncIntrvEpsilon)


@pytest.fixture()
def mcdsAnalyser_fxt():
    return mcdsAnalyser()


# ### d. _postComputeChi2
def testMcdsArsPostComputeChi2(mcdsAnalyser_fxt):
    anlr = mcdsAnalyser_fxt

    # i. Create empty results
    results = anlr.setupResults()

    # ii. Fill it up for comprehensive code coverage
    assert len(RS.CLsChi2All) == 3
    clExpDetChi2 = ('detection probability', 'chi-square test probability determined expected', 'Value')
    preChi2NormalRange = [0, 0.2, 0.5, 0.8, 1]
    for preChi20 in [np.nan] + preChi2NormalRange:
        for preChi21 in [np.nan] + ([] if np.isnan(preChi20) else preChi2NormalRange):
            for preChi22 in [np.nan] + ([] if np.isnan(preChi21) else preChi2NormalRange):
                expDetChi2 = preChi22 if not np.isnan(preChi22) else preChi21 if not np.isnan(preChi21) else preChi20
                results.append(pd.Series({RS.CLsChi2All[2]: preChi20, RS.CLsChi2All[1]: preChi21,
                                          RS.CLsChi2All[0]: preChi22, clExpDetChi2: expDetChi2}))

    # iii. Post-compute Chi2
    results._postComputeChi2()

    # iv. Auto-check results
    assert results._dfData[RS.CLChi2].compare(results._dfData[clExpDetChi2]).empty

    # v. Done
    logger.info('results._dfData: ' + results._dfData.to_string())

    logger.info0('PASS testMcdsArsPostComputeChi2: constructor, _postComputeChi2, compare')


# ### e. _postComputeDeltaAicDCv
# 1/2: normal case
def testMcdsArsPostComputeDeltaAicDCv1(mcdsAnalyser_fxt):
    anlr = mcdsAnalyser_fxt

    # i. Create empty results
    results = anlr.setupResults()
    # ii. Fill it up for comprehensive code coverage
    clSampEsp = ('header (sample)', 'Espèce', 'Value')
    clSampPas = ('header (sample)', 'Passage', 'Value')
    clSampAdl = ('header (sample)', 'Adulte', 'Value')
    clSampDur = ('header (sample)', 'Durée', 'Value')
    ldSamples = [{clSampEsp: 'Wood troobidoo', clSampPas: '99', clSampAdl: 'Yes', clSampDur: '1 century'},
                 {clSampEsp: 'Garden screamer', clSampPas: '374', clSampAdl: 'May be', clSampDur: '2 micro-seconds'}]
    clExpDeltaAic = ('detection probability', 'Delta AIC expected', 'Value')
    clExpDeltaDcv = ('density/abundance', 'density of animals expected', 'Delta Cv')
    dExpDeltaAic = {np.nan: np.nan, 200: 0, 1000: 800}
    dExpDeltaDcv = {np.nan: np.nan, 0.1: 0, 1: 0.9}
    for dSample in ldSamples:
        for leftTrDist in [np.nan, 10]:
            dSample[RS.CLsTruncDist[0]] = leftTrDist
            for rightTrDist in [np.nan, 200]:
                dSample[RS.CLsTruncDist[1]] = rightTrDist
                for aic in [np.nan, 200, 1000]:
                    for dcv in [np.nan, 0.1, 1]:
                        results.append(pd.Series({RS.CLAic: aic, clExpDeltaAic: dExpDeltaAic[aic],
                                                  RS.CLDCv: dcv, clExpDeltaDcv: dExpDeltaDcv[dcv]}),
                                       sCustomHead=pd.Series(dSample))
    # iii. Post-compute Delta AIC and DeltaDCv
    results._postComputeDeltaAicDCv()
    # iv. Auto-check results
    assert results._dfData[RS.CLDeltaAic].compare(results._dfData[clExpDeltaAic]).empty
    assert results._dfData[RS.CLDeltaDCv].compare(results._dfData[clExpDeltaDcv]).empty

    # v. Done
    logger.info('results._dfData: ' + results._dfData.to_string())

    logger.info0('PASS testMcdsArsPostComputeDeltaAicDCv1: constructor, _postComputeDeltaAicDCv, compare')


# ### e. _postComputeDeltaAicDCv
# 2/2: special case with one all-NaN sample id column
def testMcdsArsPostComputeDeltaAicDCv2(mcdsAnalyser_fxt):
    anlr = mcdsAnalyser_fxt

    # i. Create empty results
    results = anlr.setupResults()
    # ii. Fill it up for comprehensive code coverage
    clSampEsp = ('header (sample)', 'Espèce', 'Value')
    clSampPas = ('header (sample)', 'Passage', 'Value')
    clSampAdl = ('header (sample)', 'Adulte', 'Value')
    clSampDur = ('header (sample)', 'Durée', 'Value')
    ldSamples = [{clSampEsp: 'Wood troobidoo', clSampPas: np.nan, clSampAdl: 'Yes', clSampDur: '1 century'},
                 {clSampEsp: 'Garden screamer', clSampPas: np.nan, clSampAdl: 'May be', clSampDur: '2 micro-seconds'}]
    clExpDeltaAic = ('detection probability', 'Delta AIC expected', 'Value')
    clExpDeltaDcv = ('density/abundance', 'density of animals expected', 'Delta Cv')
    dExpDeltaAic = {np.nan: np.nan, 200: 0, 1000: 800}
    dExpDeltaDcv = {np.nan: np.nan, 0.1: 0, 1: 0.9}
    for dSample in ldSamples:
        for leftTrDist in [np.nan, 10]:
            dSample[RS.CLsTruncDist[0]] = leftTrDist
            for rightTrDist in [np.nan, 200]:
                dSample[RS.CLsTruncDist[1]] = rightTrDist
                for aic in [np.nan, 200, 1000]:
                    for dcv in [np.nan, 0.1, 1]:
                        results.append(pd.Series({RS.CLAic: aic, clExpDeltaAic: dExpDeltaAic[aic],
                                                  RS.CLDCv: dcv, clExpDeltaDcv: dExpDeltaDcv[dcv]}),
                                       sCustomHead=pd.Series(dSample))
    # iii. Post-compute Delta AIC and DeltaDCv
    results._postComputeDeltaAicDCv()
    # iv. Auto-check results
    assert results._dfData[RS.CLDeltaAic].compare(results._dfData[clExpDeltaAic]).empty
    assert results._dfData[RS.CLDeltaDCv].compare(results._dfData[clExpDeltaDcv]).empty

    # v. Done
    logger.info('results._dfData: ' + results._dfData.to_string())

    logger.info0('PASS testMcdsArsPostComputeDeltaAicDCv2: constructor, _postComputeDeltaAicDCv, compare')


# ### f. _postComputeQualityIndicators (and callees)
# 1/2 Normal case
def testMcdsArsPostComputeQualityIndicators1(mcdsAnalyser_fxt):
    anlr = mcdsAnalyser_fxt

    # i. Create empty results
    results = anlr.setupResults()

    # ii. Fill it up for comprehensive code coverage
    statsRange = [np.nan, 0, 0.5, 1]  # Range for statistic tests
    densCvRange = [np.nan, 0.1, 0.3, 1]  # Range for density coef. of variation

    logger.info('Filling up results to post-compute')
    ldRes = list()

    # Try all key functions
    for keyFn in [np.nan, 'HNORMAL', 'UNIFORM', 'HAZARD', 'NEXPON']:
        # CLNTotPars and CLNAdjPars are used independently, no need for combinations
        for nPars in [np.nan, 0, 2, 10]:
            # CLNObs and CLNTotObs are always used together, no need for full combination stuff
            for nObs, nTotObs in [(np.nan, 50), (10, np.nan), (25, 50), (50, 50)]:
                for chi2 in statsRange:
                    for ks in statsRange:
                        for cvmUw in statsRange:
                            for cvmCw in statsRange:
                                for dcv in densCvRange:
                                    # Version 1 (1/2). results.append(pd.Series(...))
                                    # results.append(pd.Series(...))

                                    # Version 2 (1/2). Save dict to list for later
                                    ldRes.append({RS.CLKeyFn: keyFn, RS.CLNAdjPars: nPars, RS.CLNTotPars: nPars,
                                                  RS.CLNObs: nObs, RS.CLNTotObs: nTotObs,
                                                  RS.CLChi2: chi2, RS.CLKS: ks,
                                                  RS.CLCvMUw: cvmUw, RS.CLCvMCw: cvmCw, RS.CLDCv: dcv})

    # Version 1 (2/2). Very slow : 14mn for 65536 results
    # No more code needed.

    # Version 2 (2/2). Build final DataFrame in 1 fast operation
    # => very fast : total 260ms for 65536 results !!!
    results._dfData = pd.DataFrame(ldRes)
    results._dfData.columns = pd.MultiIndex.from_tuples(results._dfData.columns)

    logger.info('{} results'.format(len(results._dfData)))

    # iii. Post-compute Delta AIC and DeltaDCv
    results._postComputeQualityIndicators()

    # Note: For 65536 results, Windows 10, total elapsed :
    # * 2021-11-21 : i5-8365U : ~20-25mn (unoptimized pd.DataFrame.apply(axis='columns')-based code)
    # * 2021-11-28 : i7-10850H : ~370ms ! (numpy-optimized column-array-operation-based code)

    # iv. Auto-check results (statistically, not always value by value)
    df = results._dfData

    # SightRate
    assert df[RS.CLSightRate].isnull().sum() == 40960
    assert df[RS.CLSightRate].value_counts().eq(pd.Series({100: 20480, 50: 20480})).all()

    # New quality indicators that should be killed by at least 1 NaN in source results
    miCompCols = [RS.CLKeyFn, RS.CLNAdjPars, RS.CLNTotPars, RS.CLNObs, RS.CLNTotObs,
                  RS.CLChi2, RS.CLKS, RS.CLCvMUw, RS.CLCvMCw, RS.CLDCv]
    miNewQuaIndCols = [RS.CLCmbQuaBal2, RS.CLCmbQuaBal3, RS.CLCmbQuaChi2, RS.CLCmbQuaKS, RS.CLCmbQuaDCv]
    dfNaNKilledNewQuaIndics = df.loc[df[miCompCols].isnull().any(axis='columns'), miNewQuaIndCols]
    aHist = np.histogram(dfNaNKilledNewQuaIndics, bins=np.linspace(0, 0.15, 16))[0]
    logger.info('Histogram of NaN-killed new Qua Indics: ' + str(aHist))
    assert pd.Series(aHist).eq([378737, 423, 0, 2, 64, 189, 292, 245, 202, 120, 118, 40, 8, 0, 0]).all()

    # All quality indicators that were not killed by any NaN in source results
    miQuaIndCols = [RS.CLCmbQuaBal1] + miNewQuaIndCols
    dfNoNaNQuaIndics = df.loc[df[miCompCols].notnull().all(axis='columns'), miQuaIndCols]
    aHist = np.histogram(dfNoNaNQuaIndics, bins=np.linspace(0, 1, 11))[0]
    logger.info('Histogram of no-NaN Qua Indics: ' + str(aHist))
    assert pd.Series(aHist).eq([30788, 876, 0, 77, 403, 800, 918, 735, 339, 56]).all()

    # All quality indicators that were not killed by any NaN in source results
    # or 0 in source stat tests results or bad DCv or SightRate or NAdjustParams
    miStatestCols = [RS.CLChi2, RS.CLKS, RS.CLCvMUw, RS.CLCvMCw]
    dfNotSoBadQuaIndics = df.loc[df[miCompCols].notnull().all(axis='columns')
                                 & df[miStatestCols].gt(0.01).all(axis='columns')
                                 & df[RS.CLDCv].lt(0.3) & df[RS.CLNAdjPars].lt(4)
                                 & (df[RS.CLNObs] / df[RS.CLNTotObs]).gt(0.7), miQuaIndCols]
    aHist = np.histogram(dfNotSoBadQuaIndics, bins=np.linspace(0.3, 1, 8))[0]
    logger.info('Histogram of not-so-bad Qua Indics: ' + str(aHist))
    assert pd.Series(aHist).eq([0, 4, 60, 174, 256, 224, 50]).all()

    # v. Done
    logger.info('results._dfData: ' + results._dfData.to_string())

    logger.info0('PASS testMcdsArsPostComputeQualityIndicators1: constructor, _postComputeQualityIndicators')


# ### f. _postComputeQualityIndicators (and callees)
# 2/2 Special case, with 1 missing input column, let's say CLCvMUw
# Note: From far faster, only 72 results to compute
def testMcdsArsPostComputeQualityIndicators2(mcdsAnalyser_fxt):
    anlr = mcdsAnalyser_fxt

    # i. Create empty results
    results = anlr.setupResults()

    # ii. Fill it up
    statestRange = [0, 0.5, 1]  # Range for statistic tests
    densCvRange = [0.1, 0.3]  # Range for density coef. of variation
    logger.info('Filling up results to post-compute')
    ldRes = list()
    nObs, nTotObs = 45, 50  # Make tests last less (already tests before)
    # Try all key functions
    for keyFn in ['HNORMAL', 'UNIFORM', 'HAZARD', 'NEXPON']:
        # CLNTotPars and CLNAdjPars are used independently, no need for combinations
        for nPars in [0, 1, 2]:
            # CLNObs and CLNTotObs are always used together, no need for full combination stuff
            for chi2 in statestRange:
                ks = cvmCw = chi2  # Make tests last less (already tests before)
                for dcv in densCvRange:
                    ldRes.append({RS.CLKeyFn: keyFn, RS.CLNAdjPars: nPars, RS.CLNTotPars: nPars,
                                  RS.CLNObs: nObs, RS.CLNTotObs: nTotObs,
                                  RS.CLChi2: chi2, RS.CLKS: ks,
                                  RS.CLCvMCw: cvmCw, RS.CLDCv: dcv})  # Note: RS.CLCvMUw is missing
    results._dfData = pd.DataFrame(ldRes)
    results._dfData.columns = pd.MultiIndex.from_tuples(results._dfData.columns)
    logger.info('{} results'.format(len(results._dfData)))

    # iii. Post-compute Delta AIC and DeltaDCv
    results._postComputeQualityIndicators()

    # iv. Auto-check results (statistically, not value by value)
    df = results._dfData
    # New quality indicators that should be killed because of at least 1 NaN in source results (the missing CLCvMUw)
    miNewQuaIndCols = [RS.CLCmbQuaBal2, RS.CLCmbQuaBal3, RS.CLCmbQuaChi2, RS.CLCmbQuaKS, RS.CLCmbQuaDCv]
    assert df[miNewQuaIndCols].eq(0).all().all()

    # v. Done
    logger.info('results._dfData: ' + results._dfData.to_string())

    logger.info0('PASS testMcdsArsPostComputeQualityIndicators2: constructor, _postComputeQualityIndicators')


# ### g. _sampleDistTruncGroups
def testMcdsArsSampleDistTruncGroups(mcdsAnalyser_fxt):
    anlr = mcdsAnalyser_fxt

    # Load results to play with ...
    # Note: Okay, it's actually an MCDSTruncOptAnalysisResultsSet file ...
    #       but we'll ignore the extra columns, promised :-)
    resFileName = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-UnitestOptResultats.ods'
    logger.info(f'Loading results from {resFileName.as_posix()} ...')
    results = anlr.setupResults()
    results.fromOpenDoc(resFileName, postComputed=True)  # Prevent re-post-computation : not a problem here, but longer

    # Get results table and extract only needed columns (promise fulfilled ;-)
    # (Note: No post-computed column used here, so ... re-computation authorised, but not needed, and slower)
    CLSampNum = ('header (head)', 'NumEchant', 'Value')
    dfRes = results.getData(copy=True)[[CLSampNum, RS.CLParTruncLeft, RS.CLParTruncRight]]
    dfRes.head()

    # Test case: a hard-coded extract of once refin/ACDC2019-Naturalist-extrait-UnitestOptResultats.ods results file
    # CLSampNum = ('header (head)', 'NumEchant', 'Value')
    # dfRes = pd.DataFrame(
    #     index=[18, 17, 5, 16, 15, 0, 12, 8, 14, 9, 1, 19, 2, 22, 24, 21, 20, 26, 23, 27, 28, 30, 29, 32, 31,
    #            33, 35, 34, 37, 36, 40, 41, 39, 43, 44, 47, 48, 49, 50, 52, 51, 56, 53, 54, 57, 60, 62, 69,
    #            63, 65, 67, 68, 66, 64],
    #     columns=pd.MultiIndex.from_tuples([CLSampNum, RS.CLParTruncLeft, RS.CLParTruncRight]),
    #     data=[[0, np.nan, np.nan], [0, 10.0, np.nan], [0, np.nan, 280.0], [0, 15.0, 300.0], [0, 20.0, 300.0],
    #           [0, 12.9, 367.2], [0, np.nan, np.nan], [0, 10.0, np.nan], [0, np.nan, 280.0], [0, 15.0, 300.0],
    #           [0, 20.0, 300.0], [0, 24.1, 229.8],
    #           [1, np.nan, np.nan], [1, 10.0, np.nan], [1, np.nan, 280.0], [1, 15.0, 300.0], [1, 20.0, 300.0],
    #           [1, np.nan, np.nan], [1, 10.0, np.nan], [1, np.nan, 280.0], [1, 15.0, 300.0], [1, 20.0, 300.0],
    #           [2, np.nan, np.nan], [2, np.nan, 450.0], [2, np.nan, 500.0], [2, np.nan, np.nan],
    #           [2, np.nan, 450.0], [2, np.nan, 500.0],
    #           [3, np.nan, np.nan], [3, np.nan, 500.0], [3, np.nan, 600.0], [3, 17.2, 200.0], [3, np.nan, np.nan],
    #           [3, np.nan, 500.0], [3, np.nan, 600.0], [3, 4.4, 200.0],
    #           [4, np.nan, np.nan], [4, 10.0, np.nan], [4, np.nan, 280.0], [4, 15.0, 300.0], [4, np.nan, 731.1],
    #           [4, np.nan, np.nan], [4, 10.0, np.nan], [4, np.nan, 280.0], [4, 15.0, 300.0], [4, np.nan, 477.6],
    #           [5, np.nan, np.nan], [5, 10.0, np.nan], [5, np.nan, 350.0], [5, 15.0, 370.0], [5, np.nan, np.nan],
    #           [5, 10.0, np.nan], [5, np.nan, 350.0], [5, 15.0, 370.0]])
    # dfRes.head()

    # Expected results per sample.
    KDSampGroupNums = {
        0: dict(left={18: 0, 17: 1, 5: 0, 16: 2, 15: 3, 0: 1, 12: 0, 8: 1, 14: 0, 9: 2, 1: 3, 19: 3},
                right={18: 0, 17: 0, 5: 2, 16: 2, 15: 2, 0: 3, 12: 0, 8: 0, 14: 2, 9: 2, 1: 2, 19: 1}),
        1: dict(left={2: 0, 22: 1, 24: 0, 21: 1, 20: 2, 26: 0, 23: 1, 27: 0, 28: 1, 30: 2},
                right={2: 0, 22: 0, 24: 1, 21: 1, 20: 1, 26: 0, 23: 0, 27: 1, 28: 1, 30: 1}),
        2: dict(left={29: 0, 32: 0, 31: 0, 33: 0, 35: 0, 34: 0},
                right={29: 0, 32: 1, 31: 2, 33: 0, 35: 1, 34: 2}),
        3: dict(left={37: 0, 36: 0, 40: 0, 41: 2, 39: 0, 43: 0, 44: 0, 47: 1},
                right={37: 0, 36: 2, 40: 3, 41: 1, 39: 0, 43: 2, 44: 3, 47: 1}),
        4: dict(left={48: 0, 49: 1, 50: 0, 52: 1, 51: 0, 56: 0, 53: 1, 54: 0, 57: 1, 60: 0},
                right={48: 0, 49: 0, 50: 1, 52: 1, 51: 3, 56: 0, 53: 0, 54: 1, 57: 1, 60: 2}),
        5: dict(left={62: 0, 69: 1, 63: 0, 65: 1, 67: 0, 68: 1, 66: 0, 64: 1},
                right={62: 0, 69: 0, 63: 1, 65: 1, 67: 0, 68: 0, 66: 1, 64: 1})
    }

    ldTruncIntrvSpecs = [dict(col='left', minDist=5.0, maxLen=5.0), dict(col='right', minDist=25.0, maxLen=25.0)]
    truncIntrvEpsilon = 1e-6

    dRefGroupNums = dict()  # For building _distTruncGroups reference :-)
    for lblSamp in dfRes[CLSampNum].unique():

        logger.debug(f'* {lblSamp}')
        dSampGroupNums = RS._sampleDistTruncGroups(dfRes[dfRes[CLSampNum] == lblSamp],
                                                   ldIntrvSpecs=ldTruncIntrvSpecs, intrvEpsilon=truncIntrvEpsilon)
        assert all(sGroupNums.eq(pd.Series(KDSampGroupNums[lblSamp][colAlias])).all()
                   for colAlias, sGroupNums in dSampGroupNums.items())

        for colAlias, sGroupNums in dSampGroupNums.items():
            if colAlias not in dRefGroupNums:
                dRefGroupNums[colAlias] = sGroupNums
            else:
                dRefGroupNums[colAlias] = dRefGroupNums[colAlias].append(sGroupNums)

    logger.info0('PASS testMcdsArsSampleDistTruncGroups(1): constructor, getData, fromOpenDoc, _sampleDistTruncGroups')

    # Check and build results reference
    # lblSamp = 5
    #
    # df = dfRes[dfRes[CLSampNum] == lblSamp].copy()
    #
    # dGroupNums = RS._sampleDistTruncGroups(df, ldIntrvSpecs=ldIntrvSpecs, intrvEpsilon=intrvEpsilon)
    # for colAlias, sGroupNums in dGroupNums.items():
    #    df[RS.DCLGroupTruncDist[colAlias]] = sGroupNums
    #    logger.info(colAlias, '=', sGroupNums.to_dict(), ',', sep='')
    # df

    # ### h. _distTruncGroups
    dGroupNums = results._distTruncGroups()

    # Auto-check
    assert all(sGroupNums.eq(dRefGroupNums[colAlias]).all() for colAlias, sGroupNums in dGroupNums.items())

    logger.info0('PASS testMcdsArsSampleDistTruncGroups(2): _distTruncGroups')


# ### i. _filterSortKeySchemes
def testMcdsArsFilterSortKeySchemes(mcdsAnalyser_fxt):
    anlr = mcdsAnalyser_fxt

    # 1. Using predefined filter-sort key generation schemes
    # Load results to play with ...
    resFileName = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-UnitestOptResultats.ods'
    logger.info('Loading results from {} ...'.format(resFileName))
    results = anlr.setupResults(ldFilSorKeySchemes=None)  # Will use predefined filter-sort key generation schemes
    results.fromOpenDoc(resFileName, postComputed=True)  # Prevent re-post-computation : not a problem here, but longer

    assert results._filterSortKeySchemes() == ads.MCDSAnalysisResultsSet.AutoFilSorKeySchemes

    logger.info0('PASS testMcdsArsFilterSortKeySchemes(predef): constructor, fromOpenDoc, _filterSortKeySchemes')

    # 2. NOT using predefined filter-sort key generation schemes
    # Load results to play with ...
    resFileName = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-UnitestOptResultats.ods'
    logger.info('Loading results from {} ...'.format(resFileName))
    ldFilSorKeySchemes = \
        [dict(key=RS.CLGrpOrdClTrChi2KSDCv,
              sort=[RS.CLGroupTruncLeft, RS.CLGroupTruncRight,
                    RS.CLChi2, RS.CLKS, RS.CLDCv, RS.CLNObs, RS.CLRunStatus],
              ascend=[True, True, False, False, True, False, True],
              group=[RS.CLGroupTruncLeft, RS.CLGroupTruncRight]),
         dict(key=RS.CLGblOrdDAicChi2KSDCv,
              sort=[RS.CLParTruncLeft, RS.CLParTruncRight, RS.CLParModFitDistCuts,
                    RS.CLDeltaAic, RS.CLChi2, RS.CLKS, RS.CLDCv, RS.CLNObs, RS.CLRunStatus],
              ascend=[True, True, True, True, False, False, True, False, True], napos='first')]
    results = anlr.setupResults(ldFilSorKeySchemes=ldFilSorKeySchemes)  # Will not use predefined ones.
    results.fromOpenDoc(resFileName, postComputed=True)  # Prevent re-post-computation : not a problem here, but longer

    assert results._filterSortKeySchemes() == ldFilSorKeySchemes

    logger.info0('PASS testMcdsArsFilterSortKeySchemes(no predef): constructor, fromOpenDoc, _filterSortKeySchemes')


# ### j. _sampleFilterSortKeys
# ### k. _filterSortKeys
def testMcdsArsFilterSortKeys(mcdsAnalyser_fxt):
    anlr = mcdsAnalyser_fxt

    # Load results to play with ...
    # Note: Okay, it's actually an MCDSTruncOptAnalysisResultsSet file ...
    # but we'll ignore the extra columns, promised :-)
    resFileName = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-UnitestOptResultats.ods'
    logger.info('Loading results from {} ...'.format(resFileName))
    results = anlr.setupResults()
    results.fromOpenDoc(resFileName, postComputed=True)  # Prevent re-post-computation : not a problem here, but longer

    # Get results table and extract only needed columns (promise fulfilled ;-)
    # (Note: No post-computed column used here, so ... re-computation authorised, but not needed, and slower)
    CLSampNum = ('header (head)', 'NumEchant', 'Value')
    dfRes = results.getData(copy=True)[[CLSampNum, RS.CLParTruncLeft, RS.CLParTruncRight,
                                        RS.CLGroupTruncLeft, RS.CLGroupTruncRight, RS.CLChi2, RS.CLDCv]]
    dfRes.head()

    # Expected results per sample.
    CLGrpOrdClTrChi2 = (RS.CLCAutoFilSor, 'Chi2 (close trunc)', RS.CLTSortOrder)
    CLGblOrdDCv = (RS.CLCAutoFilSor, 'DCv (global)', RS.CLTSortOrder)
    KDSampFilSorKeys = \
        {
            0: {CLGrpOrdClTrChi2: {12: 0, 18: 1, 14: 0, 5: 1, 8: 0, 17: 1, 9: 0, 16: 1, 0: 0, 1: 0, 15: 1, 19: 0},
                CLGblOrdDCv: {18: 0, 12: 1, 5: 2, 14: 3, 17: 4, 8: 5, 0: 6, 16: 7, 9: 8, 15: 9, 1: 10, 19: 11}},
            1: {CLGrpOrdClTrChi2: {26: 0, 2: 1, 27: 0, 24: 1, 23: 0, 22: 1, 28: 0, 21: 1, 30: 0, 20: 1},
                CLGblOrdDCv: {2: 0, 26: 1, 27: 2, 24: 3, 22: 4, 23: 5, 21: 6, 28: 7, 20: 8, 30: 9}},
            2: {CLGrpOrdClTrChi2: {29: 0, 33: 1, 32: 0, 35: 1, 31: 0, 34: 1},
                CLGblOrdDCv: {29: 0, 33: 1, 32: 2, 35: 3, 31: 4, 34: 5}},
            3: {CLGrpOrdClTrChi2: {37: 0, 39: 1, 36: 0, 43: 1, 40: 0, 44: 1, 47: 0, 41: 0},
                CLGblOrdDCv: {37: 0, 39: 1, 36: 2, 43: 3, 40: 4, 44: 5, 47: 6, 41: 7}},
            4: {CLGrpOrdClTrChi2: {56: 0, 48: 1, 60: 0, 54: 1, 50: 2, 51: 0, 53: 0, 49: 1, 57: 0, 52: 1},
                CLGblOrdDCv: {48: 0, 56: 1, 50: 2, 54: 3, 60: 4, 51: 5, 49: 6, 53: 7, 52: 8, 57: 9}},
            5: {CLGrpOrdClTrChi2: {62: 0, 67: 1, 63: 0, 66: 1, 69: 0, 68: 1, 64: 0, 65: 1},
                CLGblOrdDCv: {62: 0, 67: 1, 63: 2, 66: 3, 69: 4, 68: 5, 65: 6, 64: 7}}
        }

    # Make test simpler : replace filter and sort key generation predefined scheme set by a shorter one.
    ldFilSorKeySchemes = \
        [dict(key=CLGrpOrdClTrChi2,
              sort=[RS.CLGroupTruncLeft, RS.CLGroupTruncRight, RS.CLChi2],
              ascend=[True, True, False],
              group=[RS.CLGroupTruncLeft, RS.CLGroupTruncRight]),
         dict(key=CLGblOrdDCv,
              sort=[RS.CLParTruncLeft, RS.CLParTruncRight, RS.CLDCv],
              ascend=[True, True, True], napos='first')]

    dRefFilSorKeys = dict()  # For building _filterSortKeys output reference :-)
    for lblSamp in dfRes[CLSampNum].unique():

        logger.debug(f'* {lblSamp}')
        dSampFSKeys = RS._sampleFilterSortKeys(dfRes[dfRes[CLSampNum] == lblSamp],
                                               ldFilSorKeySchemes=ldFilSorKeySchemes)
        assert all(sFSKeys.eq(pd.Series(KDSampFilSorKeys[lblSamp][colLbl])).all()
                   for colLbl, sFSKeys in dSampFSKeys.items())

        for colLbl, sFSKeys in dSampFSKeys.items():
            if colLbl not in dRefFilSorKeys:
                dRefFilSorKeys[colLbl] = sFSKeys
            else:
                dRefFilSorKeys[colLbl] = dRefFilSorKeys[colLbl].append(sFSKeys)

    logger.info0('PASS testMcdsArsFilterSortKeys(1): constructor, fromOpenDoc, fromOpenDoc, _sampleFilterSortKeys')

    # Check (by eyes) and build material for results reference
    # lblSamp = 5
    # df = dfRes[dfRes[CLSampNum] == lblSamp].copy()
    #
    # dSampFSKeys = RS._sampleFilterSortKeys(df, ldFilSorKeySchemes=ldFilSorKeySchemes)
    # for colLbl, sFSKeys in dSampFSKeys.items():
    #    df[colLbl] = sFSKeys
    #    logger.info(sFSKeys.to_dict(), ',', sep='')
    #
    # display(df.sort_values(by=[RS.CLGroupTruncLeft, RS.CLGroupTruncRight, RS.CLChi2],
    #                       ascending=[True, True, False]) \
    #         [[RS.CLGroupTruncLeft, RS.CLGroupTruncRight, RS.CLChi2, CLGrpOrdClTrChi2]])
    # display(df.sort_values(by=[RS.CLParTruncLeft, RS.CLParTruncRight, RS.CLDCv],
    #                       ascending=[True, True, True], na_position='first') \
    #         [[RS.CLParTruncLeft, RS.CLParTruncRight, RS.CLDCv, CLGblOrdDCv]])

    # ### k. _filterSortKeys

    # Load results to play with ...
    # Note: Okay, it's actually an MCDSTruncOptAnalysisResultsSet file ...
    #       but we'll ignore the extra columns, promised :-)
    resFileName = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-UnitestOptResultats.ods'
    logger.info('Loading results from {} ...'.format(resFileName))
    # Make test simpler : replace filter and sort key predefined generation scheme set by a shorter one.
    results = anlr.setupResults(ldFilSorKeySchemes=ldFilSorKeySchemes)
    results.fromOpenDoc(resFileName, postComputed=True)  # Prevent re-post-computation : not a problem here, but longer

    # Get results table and extract only needed columns (promise fulfilled ;-)
    # (Note: No post-computed column used here, so ... re-computation authorised, but not needed, and slower)
    CLSampNum = ('header (head)', 'NumEchant', 'Value')
    dfRes = results.getData(copy=True)[[CLSampNum, RS.CLParTruncLeft, RS.CLParTruncRight,
                                        RS.CLGroupTruncLeft, RS.CLGroupTruncRight, RS.CLChi2, RS.CLDCv]]
    dfRes.head()

    dFilSorKeys = results._filterSortKeys()

    # Auto-check
    assert all(sFSKeys.eq(dRefFilSorKeys[colLbl]).all() for colLbl, sFSKeys in dFilSorKeys.items())

    logger.info0('PASS testMcdsArsFilterSortKeys(2): fromOpenDoc, fromOpenDoc, _filterSortKeys')


# ### l. _indexOfDuplicates
def testMcdsArsIndexOfDuplicates():
    # Test cases
    df = pd.DataFrame(
        [dict(a=1.000, b=2.00, c=3.0, d='To be kept: first so as a.round(1) == 1.0, whatever c, b == 2'),
         dict(a=1.010, b=2.00, c=1.0, d='Duplicate: 2nd so as a.round(1) == 1.0, whatever c, b == 2'),
         dict(a=1.049, b=2.00, c=2.0, d='Duplicate: 3rd so as a.round(1) == 1.0, whatever c, b == 2'),
         dict(a=1.051, b=2.00, c=2.0, d='To be kept: first so as a.round(1) == 1.1, whatever c, b == 2'),
         dict(a=1.060, b=2.00, c=2.0, d='Duplicate: 2nd so as a.round(1) == 1.1, whatever c, b == 2'),
         dict(a=1.100, b=2.00, c=4.0, d='Duplicate: 3rd so as a.round(1) == 1.1, whatever c, b == 2'),
         dict(a=1.151, b=2.00, c=5.0, d='To be kept: first so as a.round(1) == 1.2, whatever c, b == 2'),
         dict(a=2.000, b=2.00, c=3.0, d='To be kept: first so as b == 2.0, whatever c, a == 2'),
         dict(a=2.000, b=2.00, c=5.0, d='Duplicate: 2nd so as b == 2.0, whatever c, a == 2'),
         dict(a=2.000, b=2.01, c=9.0, d='To be kept: first so as b == 2.0, whatever c, a == 2'),
         dict(a=2.000, b=1.9999999, c=3.0, d='To be kept: first so as b == 1.9999999, whatever c, a == 2')])

    # Compute filter
    iDupes = RS._indexOfDuplicates(df, keep='first', subset=['a', 'b'], round2decs=dict(a=1))
    logger.info('iDupes: ' + str(iDupes))

    # Apply filter
    df.drop(iDupes, inplace=True)
    logger.info('df: ' + df.to_string())

    # Auto-check
    assert all(iDupes == [1, 2, 4, 5, 8])
    assert all('Duplicate' not in s for s in df.d)

    logger.info0('PASS testMcdsArsIndexOfDuplicates: _indexOfDuplicates')


# ### m. _indexOfWorstOneCriterion
def testMcdsArsIndexOfWorstOneCriterion():
    # Test cases
    df = pd.DataFrame([dict(s=0, a=1.000), dict(s=0, a=0.010), dict(s=0, a=1.049), dict(s=0, a=1.051),
                       dict(s=0, a=0.060), dict(s=0, a=1.100), dict(s=0, a=1.151), dict(s=0, a=2.000),
                       dict(s=0, a=1.020), dict(s=0, a=1.500), dict(s=0, a=2.000), dict(s=0, a=1.010),
                       dict(s=0, a=1.049), dict(s=0, a=0.051), dict(s=1, a=1.060), dict(s=1, a=1.100),
                       dict(s=1, a=1.151), dict(s=2, a=3.000), dict(s=2, a=2.000), dict(s=2, a=6.000),
                       dict(s=2, a=0.060), dict(s=2, a=1.100), dict(s=2, a=3.010), dict(s=2, a=2.200),
                       dict(s=2, a=2.230), dict(s=3, a=1.100), dict(s=3, a=1.151), dict(s=3, a=2.000),
                       dict(s=3, a=2.000), dict(s=4, a=2.000), dict(s=4, a=2.000), dict(s=4, a=2.000),
                       dict(s=4, a=2.000), dict(s=4, a=2.000), dict(s=4, a=2.000)])
    s2filter = [0, 2, 3, 5]  # Ignore sample 1 and 4, add empty sample 5
    maxRes = 6  # Keep 6 best values at most.

    df.s.value_counts().sort_index()

    # df.sort_values(by=['s', 'a'])

    # Compute filter
    i2drop = RS._indexOfWorstOneCriterion(df, sampleIds=s2filter, sampleIdCol='s', critCol='a', ascendCrit=False,
                                          nTgtRes=maxRes)
    logger.info('i2drop: ' + str(i2drop))

    # Apply filter
    df.drop(i2drop, inplace=True)
    df.sort_values(by=['s', 'a'])

    df.s.value_counts().sort_index()

    # Auto-check
    assert all(i2drop == [2, 12, 8, 11, 0, 4, 13, 1, 21, 20])
    assert df[df.s.isin(s2filter)].s.value_counts().le(maxRes).all()
    assert df.loc[df.s.isin(s2filter)].groupby('s').a.max().le([2, 6, 2]).all()

    logger.info0('PASS testMcdsArsIndexOfWorstOneCriterion: _indexOfWorstOneCriterion')


# ### n. _indexOfWorstMultiOrderCriteria
def testMcdsArsIndexOfWorstMultiOrderCriteria():
    # Test cases
    df = pd.DataFrame([dict(s=0, a=1, b=1, c='Kept thanks to a and b'),
                       dict(s=0, a=0, b=1, c='Kept thanks to a and b'),
                       dict(s=0, a=2, b=2, c='Dropped because of a and b'),
                       dict(s=0, a=4, b=3, c='Dropped because of a and b'),
                       dict(s=0, a=3, b=2, c='Dropped because of a and b'),
                       dict(s=0, a=5, b=1, c='Kept thanks to b'),
                       dict(s=1, a=2, b=4, c='Dropped because of a and b'),
                       dict(s=1, a=1, b=3, c='Kept thanks to a'),
                       dict(s=1, a=4, b=0, c='Kept thanks to b')])
    critCols = ['a', 'b']
    supCrit = 2
    logger.info('df: ' + df.to_string())

    i2drop = RS._indexOfWorstMultiOrderCriteria(df, critCols=critCols, supCrit=supCrit)
    logger.info('i2drop: ' + str(i2drop))

    # Apply filter
    df.drop(i2drop, inplace=True)
    df.sort_values(by=['s', 'a'])

    # Auto-check
    assert all(i2drop == [2, 3, 4, 6])
    assert all('Dropped' not in s for s in df.c)

    logger.info0('PASS testMcdsArsIndexOfWorstMultiOrderCriteria: _indexOfWorstMultiOrderCriteria')


# ### o. filSorSchemeId
def testMcdsArsFilSorSchemeId():

    fsRes = ads.MCDSAnalysisResultsSet(sampleIndCol='Sample')

    dupSubset = [RS.CLNObs, RS.CLEffort, RS.CLDeltaAic, RS.CLChi2, RS.CLKS, RS.CLCvMUw, RS.CLCvMCw, RS.CLDCv,
                 RS.CLPDetec, RS.CLPDetecMin, RS.CLPDetecMax, RS.CLDensity, RS.CLDensityMin, RS.CLDensityMax]
    dDupRounds = {RS.CLDeltaAic: 1, RS.CLChi2: 2, RS.CLKS: 2, RS.CLCvMUw: 2, RS.CLCvMCw: 2, RS.CLDCv: 2,
                  RS.CLPDetec: 3, RS.CLPDetecMin: 3, RS.CLPDetecMax: 3, RS.CLDensity: 2, RS.CLDensityMin: 2,
                  RS.CLDensityMax: 2}
    schEx = dict(method=RS.filterSortOnExecCode,
                 deduplicate=dict(dupSubset=dupSubset, dDupRounds=dDupRounds),
                 filterSort=dict(whichFinalQua=RS.CLCmbQuaBal1, ascFinalQua=False))
    schACCQ1 = dict(method=RS.filterSortOnExCAicMulQua,
                    deduplicate=dict(dupSubset=dupSubset, dDupRounds=dDupRounds),
                    filterSort=dict(sightRate=92.5, nBestAIC=3, nBestQua=1,
                                    whichBestQua=[RS.CLGrpOrdClTrChi2KSDCv, RS.CLGrpOrdClTrDCv, RS.CLGrpOrdClTrQuaBal1,
                                                  RS.CLGrpOrdClTrQuaChi2, RS.CLGrpOrdClTrQuaKS, RS.CLGrpOrdClTrQuaDCv],
                                    nFinalRes=12, whichFinalQua=RS.CLCmbQuaBal1, ascFinalQua=False))

    schACCQ2 = copy.deepcopy(schACCQ1)
    schACCQ3 = copy.deepcopy(schACCQ1)
    schACCQ3['filterSort']['sightRate'] = 93.0
    schACCQ4 = copy.deepcopy(schACCQ1)
    schACCQ4['filterSort']['nBestAIC'] = 2
    schACCQ5 = copy.deepcopy(schACCQ2)
    schACCQ5['filterSort']['nBestQua'] = 5

    assert fsRes.filSorSchemeId(schEx) == 'ExCode'
    assert fsRes.filSorSchemeId(schACCQ1).startswith('ExAicMQua')
    assert fsRes.filSorSchemeId(schACCQ2) == fsRes.filSorSchemeId(schACCQ1)
    assert fsRes.filSorSchemeId(schACCQ3) != fsRes.filSorSchemeId(schACCQ1) \
           and fsRes.filSorSchemeId(schACCQ3).startswith('ExAicMQua')
    assert fsRes.filSorSchemeId(schACCQ4).startswith(fsRes.filSorSchemeId(schACCQ1))
    assert fsRes.filSorSchemeId(schACCQ5).startswith(fsRes.filSorSchemeId(schACCQ1))
    logger.info('Success !')

    dict(schEx=fsRes.filSorSchemeId(schEx), schACCQ1=fsRes.filSorSchemeId(schACCQ1),
         schACCQ2=fsRes.filSorSchemeId(schACCQ2), schACCQ3=fsRes.filSorSchemeId(schACCQ3),
         schACCQ4=fsRes.filSorSchemeId(schACCQ4), schACCQ5=fsRes.filSorSchemeId(schACCQ5))

    logger.info0('PASS testMcdsArsFilSorSchemeId: constructor, filSorSchemeId')


###############################################################################
#                         Actions to be done after all tests                  #
###############################################################################
def testEnd():
    # if KFinalCleanup:
    #     uivu.cleanupWorkDir()
    uivu.logEnd(what=KWhat2Test)
