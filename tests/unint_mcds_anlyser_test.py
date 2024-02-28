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

# To run : simply run "pytest" or "python <this file>" in current folder
#          and check standard output ; and ./tmp/unt-ars.{datetime}.log for details

import sys

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
            abbrevs.append('{}{}'.format(abrv, sAnlys[name][0].lower() if isinstance(sAnlys[name], str)
            else int(sAnlys[name])))

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
    #dfFinalExplSpecs.to_excel(uivu.pTmpDir / 'tools-unitests-final-expl-specs.xlsx', index=False)

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
        {IntSpecEstimKeyFn: ['ke[a-z]*[\.\-_ ]*f', 'f[o]?n[a-z]*[\.\-_ ]*cl'],
         IntSpecEstimAdjustFn: ['ad[a-z]*[\.\-_ ]*s', 's[éa-z]*[\.\-_ ]*aj'],
         IntSpecEstimCriterion: ['crit[èa-z]*[\.\-_ ]*'],
         IntSpecCVInterval: ['conf[a-z]*[\.\-_ ]*[a-z]*[\.\-_ ]*int',
                             'in[o]?n[a-z]*[\.\-_ ]*conf'],
         IntSpecMinDist: ['min[a-z]*[\.\-_ ]*d', 'd[a-z]*[\.\-_ ]*min',
                          'tr[a-z]*[\.\-_ ]*ga', 'tr[a-z]*[\.\-_ ]*gc', 'le[a-z]*[\.\-_ ]*tr'],
         IntSpecMaxDist: ['max[a-z]*[\.\-_ ]*d', 'd[a-z]*[\.\-_ ]*max',
                          'tr[a-z]*[\.\-_ ]*dr', 'tr[a-z]*[\.\-_ ]*dt', 'le[a-z]*[\.\-_ ]*tr'],
         IntSpecFitDistCuts: ['fit[a-z]*[\.\-_ ]*d', 'tr[a-z]*[\.\-_ ]*[a-z]*[\.\-_ ]*mod'],
         IntSpecDiscrDistCuts: ['dis[a-z]*[\.\-_ ]*d', 'tr[a-z]*[\.\-_ ]*[a-z]*[\.\-_ ]*dis']}

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
    dfExplParamSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols = \
        ads.DSAnalyser._explicitParamSpecs(implParamSpecs=uivu.pRefInDir / 'ACDC2019-Naturalist-ExtraitSpecsAnalyses.xlsx',
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
    dfExplParamSpecs = dfExplParamSpecs.append(dfExplParamSpecs, ignore_index=True)  # Pleins de doublons !
    dfExplParamSpecs['AvecTronc'] = dfExplParamSpecs[['TrGche', 'TrDrte']].apply(lambda s: s.isnull().all(), axis='columns')  # Neutre 1
    dfExplParamSpecs['AbrevEsp'] = dfExplParamSpecs['Espèce'].apply(lambda s: ''.join(m[:4] for m in s.split()))  # Neutre 2

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


# ## II.1.class MCDSAnalyser
def testMcdsAnalyser():

    raise NotImplementedError('testMcdsAnalyser: TODO !')


def testMcdsPreAnalyser():

    raise NotImplementedError('testMcdsPreAnalyser: TODO !')


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

            # Tests for Analyser
            testAnalyser(indivdSightings())

            # Tests for DSAnalyser
            testDsAnalyser()

            # Tests for MCDSAnalyser
            testMcdsAnalyser()

            # Tests for MCDSPreAnalyser
            testMcdsPreAnalyser()

            # Done.
            testEnd()

            # Success !
            rc = 0

        except Exception as exc:
            logger.exception(f'Exception: {exc}')
            rc = 1

    uivu.logEnd(what=what2Test, rc=rc)

    sys.exit(rc)
