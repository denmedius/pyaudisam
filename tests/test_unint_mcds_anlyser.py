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

# Automated unit and integration tests for "analyser" submodule (*Analyser class part)

# To run : simply run "pytest" and check standard output + ./tmp/unt-anr.{datetime}.log for details

import pandas as pd

import pytest

import pyaudisam as ads

import unintval_utils as uivu


# Setup local logger.
logger = uivu.setupLogger('unt.anr', level=ads.DEBUG,
                          otherLoggers={'ads.eng': ads.INFO2, 'ads.dat': ads.INFO, 'ads.ans': ads.INFO2})

what2Test = 'analyser'


###############################################################################
#                         Actions to be done before any test                  #
###############################################################################
def testBegin():
    uivu.logBegin(what=what2Test)


###############################################################################
#                                Test Cases                                   #
###############################################################################

# ## 8. Abstract class Analyser

# Short identification string for a sample.
def sampleAbbrev(sSample):
    abrvSpe = ''.join(word[:4].title() for word in sSample['Espèce'].split(' ')[:2])

    sampAbbrev = '{}-{}-{}-{}'.format(abrvSpe, sSample.Passage.replace('+', ''),
                                      sSample.Adulte.replace('+', ''), sSample['Durée'])

    return sampAbbrev


# Short identification string for an analysis.
def analysisAbbrev(sAnlys):
    # Sample abbreviation
    abbrevs = [sampleAbbrev(sAnlys)]
    # Model + Parameters abbreviation
    abbrevs += [sAnlys['FonctionClé'][:3].lower(), sAnlys['SérieAjust'][:3].lower()]
    dTroncAbrv = {'l': 'TrGche' if 'TrGche' in sAnlys.index else 'TroncGche',
                  'r': 'TrDrte' if 'TrDrte' in sAnlys.index else 'TroncDrte',
                  'm': 'NbTrches' if 'NbTrches' in sAnlys.index else 'NbTrModel'
                  if 'NbTrModel' in sAnlys.index else 'NbTrchMod',
                  'd': 'NbTrDiscr'}
    for abrv, name in dTroncAbrv.items():
        if name in sAnlys.index and not pd.isnull(sAnlys[name]):
            val = sAnlys[name][0].lower() if isinstance(sAnlys[name], str) else int(sAnlys[name])
            abbrevs.append(f'{abrv}{val}')

    return '-'.join(abbrevs)


def count2AdultCat(sCounts):
    return 'm' if 'Mal' in sCounts[sCounts > 0].index[0] else 'a'


def count2DurationCat(sCounts):
    return '5mn' if '5' in sCounts[sCounts > 0].index[0] else '10mn'


def indivdSightings():

    # Load FieldDataSet from file
    dfObs = pd.read_csv(uivu.pRefInDir / 'ACDC2019-Naturalist-ExtraitObsBrutesAvecDist.txt', sep='\t', decimal=',')
    countCols = ['nMalAd10', 'nAutAd10', 'nMalAd5', 'nAutAd5']
    fds = ads.FieldDataSet(source=dfObs, countCols=countCols,
                           addMonoCatCols={'Adulte': count2AdultCat, 'Durée': count2DurationCat})

    # Individualise
    dfObsIndiv = fds.individualise()

    logger.info('Individualised sightings:\n' + dfObsIndiv.to_string(min_rows=20, max_rows=20))

    return dfObsIndiv


@pytest.fixture()
def indivdSightings_fxt():
    return indivdSightings()


def testAnalyser(indivdSightings_fxt):

    dfObsIndiv = indivdSightings_fxt

    # a. Generate implicit partial variant combination table
    # Number of individuals per species
    dfIndivCounts = dfObsIndiv.loc[dfObsIndiv.Adulte == 'm', ['Espèce', 'Adulte']].groupby('Espèce').count()
    dfIndivCounts.rename(columns=dict(Adulte='Mâles'), inplace=True)
    dfIndivCounts.sort_values(by='Mâles', ascending=False, inplace=True)
    logger.info('Num. of indiv. par species:\n' + dfIndivCounts[dfIndivCounts['Mâles'] >= 20].to_string())

    nMaxMal10 = 30
    varEspeces = list(dfIndivCounts[dfIndivCounts['Mâles'] >= nMaxMal10].index)  # => 1 variant per species
    varPassages = ['']  # All passed together (no selection criteria) => 1 only variant
    varAdultes = ['m', 'm+a']  # Males alone, and then males and other adults => 2 variants
    varDurees = ['5mn', '10mn']  # 5 first mn, or all 10 => 2 variants
    dfImplSampSpecs = ads.Analyser.implicitPartialVariantSpecs({'Espèces': varEspeces, 'Passages': varPassages,
                                                                'Adultes': varAdultes, 'Durées': varDurees})
    logger.info('Implicit variant specs:\n' + dfImplSampSpecs.to_string())

    # ### b. Explicit partial variant combination generation
    dfExplSampSpecs = ads.Analyser.explicitPartialVariantSpecs(dfImplSampSpecs)
    logger.info('Explicit variant specs:\n' + dfExplSampSpecs.to_string())

    # c. Direct explicitation of all variants from user specs (implicit and explicit)
    # i. Specs from a dict of DataFrame
    wbpnUserVariantSpecs = uivu.pRefInDir / 'ACDC2019-Naturalist-ExtraitSpecsAnalyses.xlsx'
    ddfUserVariantSpecs = pd.read_excel(wbpnUserVariantSpecs, sheet_name=None)
    logger.info('Implicit user specs:')
    for spName, spData in ddfUserVariantSpecs.items():
        logger.info(f'* {spName}:\n' + spData.to_string())

    dfFinalExplSpecs = ads.Analyser.explicitVariantSpecs(ddfUserVariantSpecs, ignore=['Params3_expl'],
                                                         varIndCol='IndAnlys',
                                                         # convertCols={ 'Durée': int }, # float 'cause of Excel
                                                         computedCols=dict(AbrevAnlys=analysisAbbrev))
    logger.info('Explicitated dict[DataFrame] user specs:\n' + dfFinalExplSpecs.to_string())

    # ii. Specs from an Excel workbook
    dfFinalExplSpecs1 = ads.Analyser.explicitVariantSpecs(wbpnUserVariantSpecs, ignore=['Params3_expl'],
                                                          varIndCol='IndAnlys',
                                                          # convertCols={'Durée': int}, # float 'cause of Excel
                                                          computedCols=dict(AbrevAnlys=analysisAbbrev))
    logger.info('Explicitated Excel user specs:\n' + dfFinalExplSpecs1.to_string())

    # iii. Check that the result is the same
    assert dfFinalExplSpecs.compare(dfFinalExplSpecs1).empty

    # Just to see by eye
    # dfFinalExplSpecs.to_excel(uivu.pTmpDir / 'tools-unitests-final-expl-specs.xlsx', index=False)

    # iv. Computational checks
    nSamp1Vars = 1
    df = ddfUserVariantSpecs['Echant1_impl']
    for col in df.columns:
        nSamp1Vars *= len(df[col].dropna())

    nSamp2Vars = 1
    df = ddfUserVariantSpecs['Echant2_impl']
    for col in df.columns:
        nSamp2Vars *= len(df[col].dropna())

    nModVars = 1
    df = ddfUserVariantSpecs['Modl_impl']
    for col in df.columns:
        nModVars *= len(df[col].dropna())
    nSamp1ParWithVars = \
        len(ddfUserVariantSpecs['Params1_expl'].drop_duplicates(subset=ddfUserVariantSpecs['Echant1_impl'].columns))
    nSamp1Pars = len(ddfUserVariantSpecs['Params1_expl'])
    nSamp2ParWithVars = \
        len(ddfUserVariantSpecs['Params2_expl'].drop_duplicates(subset=ddfUserVariantSpecs['Echant2_impl'].columns))
    nSamp2Pars = len(ddfUserVariantSpecs['Params2_expl'])
    nExpdVars = nModVars * (nSamp1Pars + nSamp1Vars - nSamp1ParWithVars + nSamp2Pars + nSamp2Vars - nSamp2ParWithVars)

    logger.info(f'{nModVars=} * ({nSamp1Pars=} + {nSamp1Vars=} - {nSamp1ParWithVars=}'
                f' + {nSamp2Pars=} + {nSamp2Vars=} - {nSamp2ParWithVars=}) =?= {nExpdVars=}')

    assert nModVars == 2
    assert nSamp1Pars == 16
    assert nSamp1Vars == 4
    assert nSamp1ParWithVars == 4
    assert nSamp2Pars == 8
    assert nSamp2Vars == 2
    assert nSamp2ParWithVars == 2

    assert len(dfFinalExplSpecs) == nExpdVars

    logger.info0('PASS testAnalyser: implicitPartialVariantSpecs, explicitPartialVariantSpecs, explicitVariantSpecs')


# ## I.9. Abstract class DSAnalyser
def testDsAnalyser():

    # ### a. userSpec2ParamNames
    IntSpecEstimKeyFn = 'EstimKeyFn'
    IntSpecEstimAdjustFn = 'EstimAdjustFn'
    IntSpecEstimCriterion = 'EstimCriterion'
    IntSpecCVInterval = 'CvInterval'
    IntSpecMinDist = 'MinDist'  # Left truncation distance
    IntSpecMaxDist = 'MaxDist'  # Right truncation distance
    IntSpecFitDistCuts = 'FitDistCuts'
    IntSpecDiscrDistCuts = 'DiscrDistCuts'
    int2UserSpecREs = \
        {IntSpecEstimKeyFn: [r'ke[a-z]*[\.\-_ ]*f', r'f[o]?n[a-z]*[\.\-_ ]*cl'],
         IntSpecEstimAdjustFn: [r'ad[a-z]*[\.\-_ ]*s', r's[éa-z]*[\.\-_ ]*aj'],
         IntSpecEstimCriterion: [r'crit[èa-z]*[\.\-_ ]*'],
         IntSpecCVInterval: [r'conf[a-z]*[\.\-_ ]*[a-z]*[\.\-_ ]*int',
                             r'in[o]?n[a-z]*[\.\-_ ]*conf'],
         IntSpecMinDist: [r'min[a-z]*[\.\-_ ]*d', r'd[a-z]*[\.\-_ ]*min',
                          r'tr[a-z]*[\.\-_ ]*ga', r'tr[a-z]*[\.\-_ ]*gc', r'le[a-z]*[\.\-_ ]*tr'],
         IntSpecMaxDist: [r'max[a-z]*[\.\-_ ]*d', r'd[a-z]*[\.\-_ ]*max',
                          r'tr[a-z]*[\.\-_ ]*dr', r'tr[a-z]*[\.\-_ ]*dt', r'le[a-z]*[\.\-_ ]*tr'],
         IntSpecFitDistCuts: [r'fit[a-z]*[\.\-_ ]*d', r'tr[a-z]*[\.\-_ ]*[a-z]*[\.\-_ ]*mod'],
         IntSpecDiscrDistCuts: [r'dis[a-z]*[\.\-_ ]*d', r'tr[a-z]*[\.\-_ ]*[a-z]*[\.\-_ ]*dis']}

    assert ads.DSAnalyser.userSpec2ParamNames(['key fn', 'série-aj', 'est.crit.', 'ConfInt',
                                               'fit d', 'disc d', 'min dist', 'maxd'], int2UserSpecREs) \
           == [IntSpecEstimKeyFn, IntSpecEstimAdjustFn, IntSpecEstimCriterion, IntSpecCVInterval,
               IntSpecFitDistCuts, IntSpecDiscrDistCuts, IntSpecMinDist, IntSpecMaxDist]

    # ### b. _explicitParamSpecs
    sampleSelCols = ['Espèce', 'Passage', 'Adulte', 'Durée']
    sampleIndCol = 'IndSamp'
    varIndCol = 'IndAnlys'
    anlysAbbrevCol = 'AbrevAnlys'

    # i. Through file specified implicit combinations
    implParamSpecs = uivu.pRefInDir / 'ACDC2019-Naturalist-ExtraitSpecsAnalyses.xlsx'
    dfExplParamSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols = \
        ads.DSAnalyser._explicitParamSpecs(implParamSpecs=implParamSpecs,
                                           int2UserSpecREs=int2UserSpecREs,
                                           sampleSelCols=sampleSelCols, abbrevCol=anlysAbbrevCol,
                                           abbrevBuilder=analysisAbbrev, anlysIndCol=varIndCol,
                                           sampleIndCol=sampleIndCol, dropDupes=False)
    logger.info(f'_explicitParamSpecs results: {len(dfExplParamSpecs)}, {userParamSpecCols},'
                f' {intParamSpecCols}, {unmUserParamSpecCols}')

    assert len(dfExplParamSpecs) == 48
    assert userParamSpecCols == ['FonctionClé', 'SérieAjust', 'TrGche', 'TrDrte', 'NbTrchMod']
    assert intParamSpecCols == ['EstimKeyFn', 'EstimAdjustFn', 'MinDist', 'MaxDist', 'FitDistCuts']
    assert unmUserParamSpecCols == []

    logger.info('Explicitated Excel implicit parameter specs:\n' + dfExplParamSpecs.to_string())

    # ii. Through DataFrame-specified explicit combinations, with cleaned up duplicates and neutral traversing columns
    dfExplParamSpecs.drop(columns=[varIndCol, anlysAbbrevCol, sampleIndCol], inplace=True)
    # Add many duplicates
    dfExplParamSpecs = dfExplParamSpecs.append(dfExplParamSpecs, ignore_index=True)
    # Add 2 neutral pass-through columns
    dfExplParamSpecs['AvecTronc'] = \
        dfExplParamSpecs[['TrGche', 'TrDrte']].apply(lambda s: s.isnull().all(), axis='columns')
    dfExplParamSpecs['AbrevEsp'] = \
        dfExplParamSpecs['Espèce'].apply(lambda s: ''.join(m[:4] for m in s.split()))

    logger.info('DataFrame explicit parameter specs:\n' + dfExplParamSpecs.to_string())

    # * Neutral columns not specified + uncleaned duplicates
    dfExplParamSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols = \
        ads.DSAnalyser._explicitParamSpecs(dfExplParamSpecs=dfExplParamSpecs, int2UserSpecREs=int2UserSpecREs,
                                           sampleSelCols=sampleSelCols, abbrevCol=anlysAbbrevCol,
                                           abbrevBuilder=analysisAbbrev, anlysIndCol=varIndCol,
                                           sampleIndCol=sampleIndCol, dropDupes=False)
    logger.info(f'_explicitParamSpecs results: {len(dfExplParamSpecs)}, {userParamSpecCols},'
                f' {intParamSpecCols}, {unmUserParamSpecCols}')

    assert len(dfExplParamSpecs) == 2 * 48
    assert userParamSpecCols == ['FonctionClé', 'SérieAjust', 'TrGche', 'TrDrte', 'NbTrchMod']
    assert intParamSpecCols == ['EstimKeyFn', 'EstimAdjustFn', 'MinDist', 'MaxDist', 'FitDistCuts']
    assert unmUserParamSpecCols == ['AvecTronc', 'AbrevEsp']

    # * Neutral columns specified + cleaned up duplicates
    dfExplParamSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols = \
        ads.DSAnalyser._explicitParamSpecs(dfExplParamSpecs=dfExplParamSpecs, int2UserSpecREs=int2UserSpecREs,
                                           sampleSelCols=sampleSelCols, abbrevCol=anlysAbbrevCol,
                                           abbrevBuilder=analysisAbbrev, anlysIndCol=varIndCol,
                                           sampleIndCol=sampleIndCol, anlysSpecCustCols=['AvecTronc', 'AbrevEsp'],
                                           dropDupes=True)
    logger.info(f'_explicitParamSpecs results: {len(dfExplParamSpecs)}, {userParamSpecCols},'
                f' {intParamSpecCols}, {unmUserParamSpecCols}')

    assert len(dfExplParamSpecs) == 48
    assert userParamSpecCols == ['FonctionClé', 'SérieAjust', 'TrGche', 'TrDrte', 'NbTrchMod']
    assert intParamSpecCols == ['EstimKeyFn', 'EstimAdjustFn', 'MinDist', 'MaxDist', 'FitDistCuts']
    assert unmUserParamSpecCols == []

    logger.info('Explicitated DataFrame explicit parameter specs:\n' + dfExplParamSpecs.to_string())

    logger.info0('PASS testDSAnalyser: userSpec2ParamNames, _explicitParamSpecs')


# ## II.1&2.class MCDSAnalyser : Run multiple analyses with real-life data
def testMcdsAnalyser():

    logger.info0('SKIPPED testMcdsAnalyser: see val_mcds_analyser_test.py')


# ## II.3. MCDSPreAnalyser : Run multiple pre-analyses with real-life data
def testMcdsPreAnalyser():

    logger.info0('SKIPPED testMcdsPreAnalyser: see val_mcds_preanalyser_test.py')


###############################################################################
#                         Actions to be done after all tests                  #
###############################################################################
def testEnd():
    uivu.logEnd(what=what2Test)
