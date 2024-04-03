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

# Automated unit and integration tests for "optimiser" submodule (*Optimiser class part)

# To run : simply run "pytest" and check standard output + ./tmp/unt-opr.{datetime}.log for details

import pathlib as pl
import lzma
import pickle
import re

import numpy as np
import pandas as pd

import pytest

import pyaudisam as ads

import unintval_utils as uivu


# Setup local logger.
logger = uivu.setupLogger('unt.opr', level=ads.DEBUG,
                          otherLoggers={'ads.eng': ads.INFO2, 'ads.dat': ads.INFO, 'ads.ans': ads.INFO2})

# Set to False to skip final cleanup (useful for debugging)
KFinalCleanup = True

KWhat2Test = 'optimiser'


###############################################################################
#                         Actions to be done before any test                  #
###############################################################################
def testBegin():
    uivu.logBegin(what=KWhat2Test)
    uivu.setupWorkDir('unt-opr')


###############################################################################
#                                Test Cases                                   #
###############################################################################

# ## I.11. DSParamsOptimiser abstract class (class and static methods)
# Defs for param. spec. mini-language
adspo = ads.DSParamsOptimiser

auto = adspo.Auto()


def dist(min, max):
    return adspo.DistInterval(int(min), int(max))


def quant(pct):
    return adspo.OutliersMethod('quant', int(pct))


def tucquant(pct):
    return adspo.OutliersMethod('tucquant', float(pct))


def mult(min, max):
    return adspo.MultInterval(float(min), float(max))


def abs(min, max):
    return adspo.AbsInterval(int(min), int(max))


def min(expr):
    return dict(op='min', expr=expr)


# ### a. _parseUserSpec
def testDSOpterParseUserSpecs():

    # Parse spec : no error (note: look at case ;-).
    for spec in [5, 12.0, 'auto', 'Auto', 'dist(5, 12)', 'quant(8)', 'QUANT(12)', 'tucquant(5)', 'mult(1.4, 7.3)', 'Abs(4, 10)']:
        r = adspo._parseUserSpec(spec,
                                 globals=dict(Auto=adspo.Auto,
                                              DistInterval=adspo.DistInterval,
                                              AbsInterval=adspo.AbsInterval,
                                              MultInterval=adspo.MultInterval,
                                              OutliersMethod=adspo.OutliersMethod),
                                 locals=dict(auto=auto, dist=dist, quant=quant, tucquant=tucquant,
                                             mult=mult, abs=abs))
        assert r[0] is None
        logger.info(f'{spec} => ' + ', '.join(str(x) for x in r))

    # Parse spec : errors because of bad output types.
    for spec in [1, 6.0, 'auto', 'dist(5, 12)', 'quant(8)', 'tucquant(5)', 'mult(1.4, 7.3)', 'abs(4, 10)']:
        r = adspo._parseUserSpec(spec,
                                 globals=dict(Auto=adspo.Auto,
                                              DistInterval=adspo.DistInterval,
                                              AbsInterval=adspo.AbsInterval,
                                              MultInterval=adspo.MultInterval,
                                              OutliersMethod=adspo.OutliersMethod),
                                 locals=dict(auto=auto, dist=dist, quant=quant, tucquant=tucquant,
                                             mult=mult, abs=abs),
                                 errIfNotA=[dict])
        assert r[1] is None
        logger.info(f'{spec} => ' + ', '.join(str(x) for x in r))

    # Parse spec : empty and no error.
    for spec in [None, np.nan, '', '   ']:
        r = adspo._parseUserSpec(spec,
                                 globals=dict(Auto=adspo.Auto,
                                              DistInterval=adspo.DistInterval,
                                              AbsInterval=adspo.AbsInterval,
                                              MultInterval=adspo.MultInterval,
                                              OutliersMethod=adspo.OutliersMethod),
                                 locals=dict(auto=auto, dist=dist, quant=quant, tucquant=tucquant,
                                             mult=mult, abs=abs),
                                 nullOrEmpty='rien', errIfNotA=[dict])  # Note that errIfNotA is ignored (feature).
        assert r[0] is None and r[1] == 'rien'
        logger.info(f'{spec} => ' + ', '.join(str(x) for x in r))

    # Parse spec : oneStrArg and no error.
    for spec in ['min(ks*chi2/12)']:
        r = adspo._parseUserSpec(spec,
                                 globals=dict(),
                                 locals=dict(min=min),
                                 oneStrArg=True)
        assert r[0] is None and r[1] == dict(op='min', expr='ks*chi2/12')
        logger.info(f'{spec} => ' + ', '.join(str(x) for x in r))

    # Parse spec : errors.
    for spec in ['dist(5m, 12m)', 'quant(8%)', 'tucquant(t)', 'tuckey(5)', 'mult(1,4, 7.3)', 'abs(4, \'m\')']:
        r = adspo._parseUserSpec(spec,
                                 globals=dict(Auto=adspo.Auto,
                                              DistInterval=adspo.DistInterval,
                                              AbsInterval=adspo.AbsInterval,
                                              MultInterval=adspo.MultInterval,
                                              OutliersMethod=adspo.OutliersMethod),
                                 locals=dict(auto=auto, dist=dist, quant=quant, tucquant=tucquant,
                                             mult=mult, abs=abs))
        assert r[1] is None
        logger.info(f'{spec} => ' + ', '.join(str(x) for x in r))


# ### b. _parseDistTruncationUserSpec
def testDSOpterParseDistTruncationUserSpec():

    logger.info('No error')
    r = adspo._parseDistTruncationUserSpec(2.0, errIfNotA=[float])
    logger.info(f'* status={r[0]}, parsedValue={r[1]}')
    assert r == (None, 2.0)

    r = adspo._parseDistTruncationUserSpec(7, errIfNotA=[int])
    logger.info(f'* status={r[0]}, parsedValue={r[1]}')
    assert r == (None, 7)

    r = adspo._parseDistTruncationUserSpec('auto', errIfNotA=[adspo.Auto])
    logger.info(f'* status={r[0]}, parsedValue={r[1]}')
    assert r == (None, adspo.Auto())

    r = adspo._parseDistTruncationUserSpec('quant(5)', errIfNotA=[adspo.OutliersMethod])
    logger.info(f'* status={r[0]}, parsedValue={r[1]}')
    assert r == (None, adspo.OutliersMethod('quant', 5))
    r = adspo._parseDistTruncationUserSpec('abs(8, 12)', errIfNotA=[adspo.AbsInterval])
    logger.info(f'* status={r[0]}, parsedValue={r[1]}')
    assert r == (None, adspo.AbsInterval(8, 12))
    r = adspo._parseDistTruncationUserSpec('dist(0, 70)', errIfNotA=[adspo.DistInterval])
    logger.info(f'* status={r[0]}, parsedValue={r[1]}')
    assert r == (None, adspo.DistInterval(0, 70))
    r = adspo._parseDistTruncationUserSpec('mult(0.6, 1.2)', errIfNotA=[adspo.MultInterval])
    logger.info(f'* status={r[0]}, parsedValue={r[1]}')
    assert r == (None, adspo.MultInterval(0.6, 1.2))
    r = adspo._parseDistTruncationUserSpec('tucquant(2.5)')
    logger.info(f'* status={r[0]}, parsedValue={r[1]}')
    assert r == (None, adspo.OutliersMethod('tucquant', 2.5))

    # Bad type errors.
    r = adspo._parseDistTruncationUserSpec('auto', errIfNotA=(adspo.AbsInterval, adspo.MultInterval))
    logger.info(r[0])
    assert r[0] is not None and r[1] is None
    r = adspo._parseDistTruncationUserSpec('quant(5)', errIfNotA=[adspo.Auto])
    logger.info(r[0])
    assert r[0] is not None and r[1] is None
    r = adspo._parseDistTruncationUserSpec('abs(8, 12)', errIfNotA=(adspo.OutliersMethod,))
    logger.info(r[0])
    assert r[0] is not None and r[1] is None
    r = adspo._parseDistTruncationUserSpec('mult(0.6, 1.2)', errIfNotA=(adspo.DistInterval, adspo.OutliersMethod))
    logger.info(r[0])
    assert r[0] is not None and r[1] is None
    r = adspo._parseDistTruncationUserSpec('tucquant(2.5)', errIfNotA=(adspo.DistInterval, adspo.MultInterval))
    logger.info(r[0])
    assert r[0] is not None and r[1] is None

    # Parsing errors.
    r = adspo._parseDistTruncationUserSpec('autox')
    logger.info(r[0])
    assert r[0] is not None and r[1] is None
    r = adspo._parseDistTruncationUserSpec('tuckey(5)')
    logger.info(r[0])
    assert r[0] is not None and r[1] is None
    r = adspo._parseDistTruncationUserSpec('abs(12)')
    logger.info(r[0])
    assert r[0] is not None and r[1] is None
    r = adspo._parseDistTruncationUserSpec('mult(0.6, x)')
    logger.info(r[0])
    assert r[0] is not None and r[1] is None
    r = adspo._parseDistTruncationUserSpec('tucquant(2.5%)')
    logger.info(r[0])
    assert r[0] is not None and r[1] is None


# ## 12. MCDSTruncationOptimiser abstract class
adsto = ads.MCDSTruncationOptimiser


def count2AdultCat(sCounts):
    return 'm' if 'Mal' in sCounts[sCounts > 0].index[0] else 'a'


def count2DurationCat(sCounts):
    return '5mn' if '5' in sCounts[sCounts > 0].index[0] else '10mn'


KFdsCountCols = ['nMalAd10', 'nAutAd10', 'nMalAd5', 'nAutAd5']


# Create an individualised sightings data set
def indivSightings():
    
    fds = ads.FieldDataSet(source=uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-ObsBrutesAvecDist.txt',
                           importDecFields=['distMem'], countCols=KFdsCountCols,
                           addMonoCatCols={'Adulte': count2AdultCat, 'Durée': count2DurationCat})
    
    return fds.individualise()


@pytest.fixture
def indivSightings_fxt():
    return indivSightings()


def testMcdsTruncOpterCtorGetParams(indivSightings_fxt):

    # ### a. Individualised data set
    dfObsIndiv = indivSightings_fxt
    dfObsIndiv.drop(columns=KFdsCountCols, inplace=True)
    dfObsIndiv.tail()

    transectPlaceCols = ['Point']
    passIdCol = 'Passage'
    effortCol = 'Effort'
    sampleDistCol = 'distMem'
    sampleDecCols = [effortCol, sampleDistCol]
    sampleCols = [passIdCol, 'Adulte', 'Durée']
    sampleSelCols = ['Espèce'] + sampleCols
    varIndCol = 'IndAnlys'
    anlysAbbrevCol = 'AbrevAnlys'
    dSurveyArea = dict(Zone='ACDC', Surface='2400')

    # Show samples
    logger.info('Samples:\n' + dfObsIndiv[sampleCols].drop_duplicates().to_string())

    # ### b. Ctor
    logger.info0('testMcdsTruncOpterCtorGetXxxParams: Constructor ...')

    # Check run method and time-out support
    try:
        _ = ads.MCDSTruncationOptimiser(
             dfObsIndiv, effortConstVal=1, dSurveyArea=dSurveyArea,
             transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
             sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols, sampleDistCol=sampleDistCol,
             workDir=uivu.pWorkDir, runMethod='os.system', runTimeOut=120)
    except AssertionError as exc:
        if re.search("Can't care about .+s execution time limit", str(exc)):
            logger.info('Good: Expected refuse to work for incompatible params')
        else:
            raise

    # An operational one for checks below
    optr = ads.MCDSTruncationOptimiser(
         dfObsIndiv, effortConstVal=1, dSurveyArea=dSurveyArea,
         transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
         sampleDecCols=sampleDecCols, sampleDistCol=sampleDistCol,
         distanceUnit='Meter', areaUnit='Hectare',
         surveyType='Point', distanceType='Radial', clustering=False,
         resultsHeadCols=dict(before=[varIndCol], sample=sampleSelCols, after=[anlysAbbrevCol]),
         abbrevCol=anlysAbbrevCol, workDir=uivu.pWorkDir, logData=False,
         defEstimKeyFn='HNO', defEstimAdjustFn='COS',
         defEstimCriterion='AIC', defCVInterval=95,
         defExpr2Optimise='chi2', defMinimiseExpr=False,
         defOutliersMethod='tucquant', defOutliersQuantCutPct=5,
         defFitDistCutsFctr=dict(min=2 / 3, max=3 / 2),
         defDiscrDistCutsFctr=dict(min=1 / 3, max=1),
         defSubmitTimes=4, defSubmitOnlyBest=2,
         dDefOptimCoreParams=dict(core='zoopt'))

    logger.info0('PASS testMcdsTruncOpterCtorGetXxxParams: Constructor')

    # ### c. getAnalysisOptimExprParams
    logger.info0('testMcdsTruncOpterCtorGetXxxParams: getAnalysisOptimExprParams ...')

    # Spec is present
    sAnIntSpec = pd.Series({adsto.IntSpecExpr2Optimise: 'min(ks*chi2/12)'})
    r = optr.getAnalysisOptimExprParams(sAnIntSpec)
    logger.info(f'Spec is present: status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(minimiseExpr=True, expr2Optimise='ks*chi2/12')

    # Spec is null
    sAnIntSpec = pd.Series({adsto.IntSpecExpr2Optimise: None})
    r = optr.getAnalysisOptimExprParams(sAnIntSpec)
    logger.info(f'Spec is null: status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(minimiseExpr=False, expr2Optimise='chi2')

    # Spec is absent
    sAnIntSpec = pd.Series(dtype=float)
    r = optr.getAnalysisOptimExprParams(sAnIntSpec)
    logger.info(f'Spec is absent: status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(minimiseExpr=False, expr2Optimise='chi2')

    logger.info0('PASS testMcdsTruncOpterCtorGetXxxParams: getAnalysisOptimExprParams')

    # ### d. getAnalysisFixedParams
    logger.info0('testMcdsTruncOpterCtorGetXxxParams: getAnalysisFixedParams ...')

    # All specs present
    sAnIntSpec = pd.Series({adsto.IntSpecEstimKeyFn: 'HNO', adsto.IntSpecEstimAdjustFn: 'POLY',
                            adsto.IntSpecEstimCriterion: 'AIC', adsto.IntSpecCVInterval: 97})
    r = optr.getAnalysisFixedParams(sAnIntSpec)
    logger.info(f'All specs present: status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(estimKeyFn='HNO', estimAdjustFn='POLY', estimCriterion='AIC', cvInterval=97)

    # Some specs absent => default values
    sAnIntSpec = pd.Series({adsto.IntSpecEstimKeyFn: 'UNI', adsto.IntSpecEstimAdjustFn: 'POLY'})
    r = optr.getAnalysisFixedParams(sAnIntSpec)
    logger.info(f'Some specs absent => default values: status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(estimKeyFn='UNI', estimAdjustFn='POLY', estimCriterion='AIC', cvInterval=95)

    logger.info0('PASS testMcdsTruncOpterCtorGetXxxParams: getAnalysisFixedParams')

    # ### e. getAnalysisOptimedParams
    logger.info0('testMcdsTruncOpterCtorGetXxxParams: getAnalysisOptimedParams ...')

    #
    logger.info('* All present and variant (check computations) 1:')

    # Get a "random" sample from indiv. data set + compute some base figures for checking results
    sAnSpec = pd.Series({'Espèce': 'Alauda arvensis', 'Passage': 'a+b', 'Adulte': 'm+a', 'Durée': '10mn'})
    sds = optr._mcDataSet.sampleDataSet(sAnSpec[sampleCols])
    sSampleDistances = sds.dfData[sampleDistCol].dropna()

    sqd = np.sqrt(len(sSampleDistances.dropna()))
    dMin = sSampleDistances.min()
    dMax = sSampleDistances.max()

    logger.info(f'  - Sample distances: dMin={dMin}, dMax={dMax}, ...')
    logger.info(f'    {len(sSampleDistances)} sample distances =>\n{sorted(sSampleDistances.values)}')

    # i. Call method
    sAnIntSpec = pd.Series({adsto.IntSpecMinDist: 'auto', adsto.IntSpecMaxDist: 'quant(5)',
                            adsto.IntSpecFitDistCuts: 'abs(8, 12)', adsto.IntSpecDiscrDistCuts: 'mult(0.6, 1.2)',
                            adsto.IntSpecOutliersMethod: 'tucquant(2.5)'})
    e, r = optr.getAnalysisOptimedParams(sAnIntSpec, sSampleDistances)
    assert e is None
    sr = str({k: str(v) for k, v in r.items()})
    logger.info(f'  - Actual result     : {sr}')

    # ii. Compute theoretical result
    qLeft, qRight = np.percentile(a=sSampleDistances, q=[2.5, 95])
    logger.info('Base variables  :' + str(dict(sqd=sqd, dMin=dMin, dMax=dMax, qLeft=qLeft, qRight=qRight)))
    sol = dict(minDist=ads.Interval(dMin, qLeft), maxDist=ads.Interval(qRight, dMax),
               fitDistCuts=ads.Interval(8, 12),
               discrDistCuts=ads.Interval(int(round(sqd * 0.6)), int(round(sqd * 1.2))))
    ssol = str({k: str(v) for k, v in sol.items()})
    logger.info(f'  - Theoretical result: {ssol}')

    # iii. Check "equality" (for some reason, must use str repr for comparison ...)
    assert sr == ssol

    #
    logger.info('* All present and variant (check computations) 2:')

    # Get a new "random" sample from indiv. data set + compute some base figures for checking results
    sAnSpec = pd.Series({'Espèce': 'Alauda arvensis', 'Passage': 'a+b', 'Adulte': 'm+a', 'Durée': '5mn'})
    sds = optr._mcDataSet.sampleDataSet(sAnSpec[sampleCols])
    sSampleDistances = sds.dfData[sampleDistCol].dropna()

    sqd = np.sqrt(len(sSampleDistances.dropna()))
    dMin = sSampleDistances.min()
    dMax = sSampleDistances.max()

    logger.info(f'* Sample distances: dMin={dMin}, dMax={dMax}, ...')
    logger.info(f'  {len(sSampleDistances)} sample distances =>\n{sorted(sSampleDistances.values)}')

    # i. Call method
    sAnIntSpec = pd.Series({adsto.IntSpecMinDist: 'quant(5)', adsto.IntSpecMaxDist: 'auto',
                            adsto.IntSpecFitDistCuts: 'mult(3/4, 5/4)', adsto.IntSpecDiscrDistCuts: 'abs(4, 6)',
                            adsto.IntSpecOutliersMethod: 'tucquant(1)'})
    e, r = optr.getAnalysisOptimedParams(sAnIntSpec, sSampleDistances)
    assert e is None
    sr = str({k: str(v) for k, v in r.items()})
    logger.info(f'  - Actual result     : {sr}')

    # ii. Compute theoretical result
    qLeft, q25, q75, q95, qRight = np.percentile(a=sSampleDistances, q=[5, 25, 75, 95, 99])
    qSup = q75 + 1.5 * (q75 - q25)
    logger.info('Base variables  :' + str(dict(sqd=sqd, dMin=dMin, dMax=dMax, qLeft=qLeft,
                                               q75=q75, qSup=qSup, q95=q95, qRight=qRight)))
    sol = dict(minDist=ads.Interval(dMin, qLeft), maxDist=ads.Interval(qSup, dMax),
               fitDistCuts=ads.Interval(int(round(sqd * 3 / 4)), int(round(sqd * 5 / 4))),
               discrDistCuts=ads.Interval(4, 6))
    ssol = str({k: str(v) for k, v in sol.items()})
    logger.info(f'  - Theoretical result: {ssol}')

    # iii. Check "equality" (for some reason, must use str repr for comparison ...)
    assert sr == ssol

    #
    logger.info('* All present and variant (check computations) 3:')

    # Get a new "random" sample from indiv. data set + compute some base figures for checking results
    sAnSpec = pd.Series({'Espèce': 'Alauda arvensis', 'Passage': 'a+b', 'Adulte': 'm+a', 'Durée': '10mn'})
    sds = optr._mcDataSet.sampleDataSet(sAnSpec[sampleCols])
    sSampleDistances = sds.dfData[sampleDistCol].dropna()

    sqd = np.sqrt(len(sSampleDistances.dropna()))
    dMin = sSampleDistances.min()
    dMax = sSampleDistances.max()

    logger.info(f'* Sample distances: dMin={dMin}, dMax={dMax}, ...')
    logger.info(f'  {len(sSampleDistances)} sample distances =>\n{sorted(sSampleDistances.values)}')

    # i. Call method
    sAnIntSpec = pd.Series({adsto.IntSpecMinDist: 'auto', adsto.IntSpecMaxDist: 'auto',
                            adsto.IntSpecFitDistCuts: 'auto', adsto.IntSpecDiscrDistCuts: 'auto',
                            adsto.IntSpecOutliersMethod: 'tucquant(2)'})
    e, r = optr.getAnalysisOptimedParams(sAnIntSpec, sSampleDistances)
    assert e is None
    sr = str({k: str(v) for k, v in r.items()})
    logger.info(f'  - Actual result     : {sr}')
    
    # ii. Compute theoretical result
    qLeft, q95, qRight = np.percentile(a=sSampleDistances, q=[2, 95, 98])
    logger.info('Base variables  :' + str(dict(sqd=sqd, dMin=dMin, dMax=dMax, qLeft=qLeft, q95=q95, qRight=qRight)))
    sol = dict(minDist=ads.Interval(dMin, qLeft), maxDist=ads.Interval(q95, dMax),
               fitDistCuts=ads.Interval(int(round(sqd * 2 / 3)), int(round(sqd * 3 / 2))),
               discrDistCuts=ads.Interval(int(round(sqd / 3)), int(round(sqd))))
    ssol = str({k: str(v) for k, v in sol.items()})
    logger.info(f'  - Theoretical result: {ssol}')
    
    # iii. Check "equality" (for some reason, must use str repr for comparison ...)
    assert sr == ssol

    #
    logger.info('* All present and variant (check computations) 4:')
    
    # i. Call method
    sAnIntSpec = pd.Series({adsto.IntSpecMinDist: 'auto', adsto.IntSpecMaxDist: 'auto',
                            adsto.IntSpecFitDistCuts: 'auto', adsto.IntSpecDiscrDistCuts: 'auto',
                            adsto.IntSpecOutliersMethod: 'auto'})
    e, r = optr.getAnalysisOptimedParams(sAnIntSpec, sSampleDistances)
    assert e is None
    sr = str({k: str(v) for k, v in r.items()})
    logger.info(f'  - Actual result     : {sr}')
    
    # ii. Compute theoretical result
    qLeft, qRight = np.percentile(a=sSampleDistances, q=[5, 95])
    logger.info('Base variables  :' + str(dict(sqd=sqd, dMin=dMin, dMax=dMax, qLeft=qLeft, qRight=qRight)))
    sol = dict(minDist=ads.Interval(dMin, qLeft), maxDist=ads.Interval(qRight, dMax),
               fitDistCuts=ads.Interval(int(round(sqd * 2 / 3)), int(round(sqd * 3 / 2))),
               discrDistCuts=ads.Interval(int(round(sqd / 3)), int(round(sqd))))
    ssol = str({k: str(v) for k, v in sol.items()})
    logger.info(f'  - Theoretical result: {ssol}')
    
    # iii. Check "equality" (for some reason, must use str repr for comparison ...)
    assert sr == ssol

    #
    logger.info('* All present, some variant, some consts (check computations) 1:')
    
    # i. Call method
    sAnIntSpec = pd.Series({adsto.IntSpecMinDist: 12, adsto.IntSpecMaxDist: 'quant(5)',
                            adsto.IntSpecFitDistCuts: 'abs(8, 12)', adsto.IntSpecDiscrDistCuts: 'mult(0.6, 1.2)',
                            adsto.IntSpecOutliersMethod: 'tucquant(2.5)'})
    e, r = optr.getAnalysisOptimedParams(sAnIntSpec, sSampleDistances)
    assert e is None
    sr = str({k: str(v) for k, v in r.items()})
    logger.info(f'  - Actual result     : {sr}')
    
    # ii. Compute theoretical result
    qLeft, qRight = np.percentile(a=sSampleDistances, q=[2.5, 95])
    logger.info('Base variables  :' + str(dict(sqd=sqd, dMin=dMin, dMax=dMax, qLeft=qLeft, qRight=qRight)))
    sol = dict(minDist=12, maxDist=ads.Interval(qRight, dMax), fitDistCuts=ads.Interval(8, 12),
               discrDistCuts=ads.Interval(int(round(sqd * 0.6)), int(round(sqd * 1.2))))
    ssol = str({k: str(v) for k, v in sol.items()})
    logger.info(f'  - Theoretical result: {ssol}')
    
    # iii. Check "equality" (for some reason, must use str repr for comparison ...)
    assert sr == ssol

    #
    logger.info('* All present, some variant, some consts (check computations) 2:')
    
    # i. Call method
    sAnIntSpec = pd.Series({adsto.IntSpecMinDist: 'quant(5)', adsto.IntSpecMaxDist: 250.0,
                            adsto.IntSpecFitDistCuts: 'mult(3/4, 5/4)', adsto.IntSpecDiscrDistCuts: 'abs(4, 6)',
                            adsto.IntSpecOutliersMethod: 'tucquant(1)'})
    e, r = optr.getAnalysisOptimedParams(sAnIntSpec, sSampleDistances)
    assert e is None
    sr = str({k: str(v) for k, v in r.items()})
    logger.info(f'  - Actual result     : {sr}')
    
    # ii. Compute theoretical result
    qLeft, qRight = np.percentile(a=sSampleDistances, q=[5, 99])
    logger.info('Base variables  :' + str(dict(sqd=sqd, dMin=dMin, dMax=dMax, qLeft=qLeft, qRight=qRight)))
    sol = dict(minDist=ads.Interval(dMin, qLeft), maxDist=250.0,
               fitDistCuts=ads.Interval(int(round(sqd * 3 / 4)), int(round(sqd * 5 / 4))),
               discrDistCuts=ads.Interval(4, 6))
    ssol = str({k: str(v) for k, v in sol.items()})
    logger.info(f'  - Theoretical result: {ssol}')
    
    # iii. Check "equality" (for some reason, must use str repr for comparison ...)
    assert sr == ssol

    #
    logger.info('* All present, some variant, some consts (check computations) 3:')
    
    # i. Call method
    sAnIntSpec = pd.Series({adsto.IntSpecMinDist: 'auto', adsto.IntSpecMaxDist: 'auto',
                            adsto.IntSpecFitDistCuts: 17, adsto.IntSpecDiscrDistCuts: 'auto',
                            adsto.IntSpecOutliersMethod: 'tucquant(6)'})
    e, r = optr.getAnalysisOptimedParams(sAnIntSpec, sSampleDistances)
    assert e is None
    sr = str({k: str(v) for k, v in r.items()})
    logger.info(f'  - Actual result     : {sr}')
    
    # ii. Compute theoretical result
    qLeft, qRight = np.percentile(a=sSampleDistances, q=[6, 94])
    logger.info('Base variables  :' + str(dict(sqd=sqd, dMin=dMin, dMax=dMax, qLeft=qLeft, qRight=qRight)))
    sol = dict(minDist=ads.Interval(dMin, qLeft), maxDist=ads.Interval(qRight, dMax),
               fitDistCuts=17, discrDistCuts=ads.Interval(int(round(sqd / 3)), int(round(sqd))))
    ssol = str({k: str(v) for k, v in sol.items()})
    logger.info(f'  - Theoretical result: {ssol}')
    
    # iii. Check "equality" (for some reason, must use str repr for comparison ...)
    assert sr == ssol

    #
    logger.info('* All present, some variant, some consts (check computations) 4:')
    
    # i. Call method
    sAnIntSpec = pd.Series({adsto.IntSpecMinDist: 'auto', adsto.IntSpecMaxDist: 'auto',
                            adsto.IntSpecFitDistCuts: 'auto', adsto.IntSpecDiscrDistCuts: 6,
                            adsto.IntSpecOutliersMethod: 'auto'})
    e, r = optr.getAnalysisOptimedParams(sAnIntSpec, sSampleDistances)
    assert e is None
    sr = str({k: str(v) for k, v in r.items()})
    logger.info(f'  - Actual result     : {sr}')
    
    # ii. Compute theoretical result
    qLeft, qRight = np.percentile(a=sSampleDistances, q=[5, 95])
    logger.info('Base variables  :' + str(dict(sqd=sqd, dMin=dMin, dMax=dMax, qLeft=qLeft, qRight=qRight)))
    sol = dict(minDist=ads.Interval(dMin, qLeft), maxDist=ads.Interval(qRight, dMax),
               fitDistCuts=ads.Interval(int(round(sqd * 2 / 3)), int(round(sqd * 3 / 2))), discrDistCuts=6)
    ssol = str({k: str(v) for k, v in sol.items()})
    logger.info(f'  - Theoretical result: {ssol}')
    
    # iii. Check "equality" (for some reason, must use str repr for comparison ...)
    assert sr == ssol

    logger.info0('PASS testMcdsTruncOpterCtorGetXxxParams: getAnalysisOptimedParams')

    # ### f. getOptimisationCoreParams
    logger.info0('testMcdsTruncOpterCtorGetXxxParams: getOptimisationCoreParams ...')

    #
    logger.info('Specs not present => default from ctor')
    sAnIntSpec = pd.Series({adsto.IntSpecOptimisationCore: np.nan})
    r = optr.getOptimisationCoreParams(sAnIntSpec)
    logger.info(f'=> status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(core='zoopt')

    #
    logger.info('Specs null => default from ctor:')
    sAnIntSpec = pd.Series(dtype=float)
    r = optr.getOptimisationCoreParams(sAnIntSpec)
    logger.info(f'=> status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(core='zoopt')

    #
    logger.info('Some specs present, with all default values ; string as last param.:')
    sAnIntSpec = pd.Series({adsto.IntSpecOptimisationCore: 'zoopt(mxi=0,a=racos)'})
    r = optr.getOptimisationCoreParams(sAnIntSpec)
    logger.info(f'=> status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(core='zoopt')

    #
    logger.info('Some specs present, some with default values, some not, 1 non keyword param.:')
    sAnIntSpec = pd.Series({adsto.IntSpecOptimisationCore: 'zoopt(80, a=racos)'})
    r = optr.getOptimisationCoreParams(sAnIntSpec)
    logger.info(f'=> status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(core='zoopt', maxIters=80)

    #
    logger.info('All specs present, no default value:')
    sAnIntSpec = pd.Series({adsto.IntSpecOptimisationCore: 'zoopt(a=sracos,mxi=450,tv=1,mxr=5)'})
    r = optr.getOptimisationCoreParams(sAnIntSpec)
    logger.info(f'=> status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(core='zoopt', algorithm='sracos', maxIters=450, termExprValue=1, maxRetries=5)

    logger.info0('PASS testMcdsTruncOpterCtorGetXxxParams: getOptimisationCoreParams')

    # ### g. getOptimisationSubmitParams
    logger.info0('testMcdsTruncOpterCtorGetXxxParams: getOptimisationSubmitParams ...')

    #
    logger.info('Specs not present => default from ctor:')
    sAnIntSpec = pd.Series({adsto.IntSpecSubmitParams: np.nan})
    r = optr.getOptimisationSubmitParams(sAnIntSpec)
    logger.info(f'=> status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(times=4, onlyBest=2)

    #
    logger.info('Specs null => default from ctor:')
    sAnIntSpec = pd.Series(dtype=float)
    r = optr.getOptimisationSubmitParams(sAnIntSpec)
    logger.info(f'=> status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(times=4, onlyBest=2)

    #
    logger.info('Some specs present, with default values:')
    sAnIntSpec = pd.Series({adsto.IntSpecSubmitParams: 'times(n=9)'})
    r = optr.getOptimisationSubmitParams(sAnIntSpec)
    logger.info(f'=> status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(times=9, onlyBest=None)

    #
    logger.info('All specs present, no default value:')
    sAnIntSpec = pd.Series({adsto.IntSpecSubmitParams: 'times(100, b=22)'})
    r = optr.getOptimisationSubmitParams(sAnIntSpec)
    logger.info(f'=> status={r[0]}, parsedValue={r[1]}')
    assert r[0] is None and r[1] == dict(times=100, onlyBest=22)

    #
    logger.info('Bad times times (n):')
    sAnIntSpec = pd.Series({adsto.IntSpecSubmitParams: 'times(n=0, b=22)'})
    r = optr.getOptimisationSubmitParams(sAnIntSpec)
    logger.info(f'=> status={r[0]}, parsedValue={r[1]}')
    assert r[1] is None and str(r[0]).find('Run times must be > 0') >= 0

    #
    logger.info('Bad best kept values number (b):')
    sAnIntSpec = pd.Series({adsto.IntSpecSubmitParams: 'times(2, b=0)'})
    r = optr.getOptimisationSubmitParams(sAnIntSpec)
    logger.info(f'=> status={r[0]}, parsedValue={r[1]}')
    assert r[1] is None and str(r[0]).find('Number of best kept values must be > 0') >= 0

    logger.info0('PASS testMcdsTruncOpterCtorGetXxxParams: getOptimisationSubmitParams')


# II.4. MCDSZerothOrderTruncationOptimiser : Optimise truncation params on real-life data
# Note: Only from explicit specs here.
# Note: For MCDSTruncationOptAnalyser tests, see val_mcds_optanalyser_test

def testMcdsZerothOrderTruncationOptimiser(indivSightings_fxt):

    # a. Explicit analysis specs
    # i. Individualised data set
    dfObsIndiv = indivSightings_fxt
    dfObsIndiv.drop(columns=KFdsCountCols, inplace=True)

    logger.info(f'Individualised sightings: n={len(dfObsIndiv)} =>\n'
                + dfObsIndiv.to_string(min_rows=20, max_rows=20)) 

    # ii. Explicit analysis specs (through an analyser object)
    transectPlaceCol = 'Point'
    transectPlaceCols = [transectPlaceCol]
    passIdCol = 'Passage'
    effortCol = 'Effort'

    sampleDistCol = 'distMem'
    sampleDecCols = [effortCol, sampleDistCol]

    sampleSelCols = ['Espèce', passIdCol, 'Adulte', 'Durée']
    sampleIndCol = 'IndSamp'

    varIndCol = 'IndAnlys'
    anlysAbbrevCol = 'AbrevAnlys'

    dSurveyArea = dict(Zone='ACDC', Surface='2400')

    # Constant effort per point x passage (= 1) => no need to pass transect info (auto-generated)
    anlysr = ads.MCDSAnalyser(dfObsIndiv, effortConstVal=1, dSurveyArea=dSurveyArea,
                              transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                              sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols,
                              abbrevCol=anlysAbbrevCol, abbrevBuilder=uivu.analysisAbbrev,
                              anlysIndCol=varIndCol, sampleIndCol=sampleIndCol,
                              distanceUnit='Meter', areaUnit='Hectare',
                              surveyType='Point', distanceType='Radial', clustering=False,
                              resultsHeadCols=dict(before=[varIndCol], sample=sampleSelCols, after=[anlysAbbrevCol]),
                              workDir=uivu.pWorkDir, runMethod='subprocess.run', logProgressEvery=5)

    anlysSpecFile = uivu.pRefInDir / 'ACDC2019-Naturalist-extrait-SpecsAnalyses.xlsx'
    dfAnlysExplSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols, verdict, reasons = \
        anlysr.explicitParamSpecs(implParamSpecs=anlysSpecFile, dropDupes=True, check=True)

    # iii. Done.
    anlysr.shutdown()

    # b. Explicit optimisation specs
    # i. Left part = standard analysis params without truncation specs, from 4. above
    dfOptimExplSpecs = dfAnlysExplSpecs[sampleSelCols + ['FonctionClé', 'SérieAjust']].drop_duplicates().reset_index(drop=True)
    logger.info(f'Explicit specs (left part):\n' + dfOptimExplSpecs.to_string())

    # ii. Right part : as many as possible truncation optimisation params combinations
    dfMoreOptimCols = pd.DataFrame(dict(CritChx=[None, 'AIC']*6,
                                        IntervConf=[None, 95, 97]*4,
                                        TroncGche=['auto', None, 20, 'dist(5, 30)', 50.0, 'quant(3)']*2,
                                        TroncDrte=[None, 'auto', 'dist(150, 300)', 200.0, 'tucquant(2)', 250]*2,
                                        MethOutliers=[None, 'auto', None, None,
                                                      None, 'quant(6)', None, None,
                                                      None, 'tucquant(8)', None, None],
                                        NbTrModel=[None, 9.0, 'auto', 17, 'abs(5, 10)', 'mult(0.5,5/4)']*2,
                                        NbTrDiscr=[None, 'auto', 4, 'abs(5, 10)', 16.0, 'mult(0.5,5/4)']*2,
                                        ExprOpt=[None, 'max(chi2)', 'min(1-chi2)', 'max(chi2)',
                                                 'max(ks)', 'max(cvmuw*cvmcw)']*2,
                                        MoteurOpt=[None, 'zoopt', 'zoopt(mxi=20, a=racos)',
                                                   'zoopt(mxi=30, mxr=2, tv=0.5)']*3,
                                        ParExec=[None, 'times(2)', 'times(3, b=2)']*4))
    logger.info(f'Explicit specs (right part):\n' + dfOptimExplSpecs.to_string())

    # iii. Concat left and right parts
    dfOptimExplSpecs = pd.concat([dfOptimExplSpecs, dfMoreOptimCols], axis='columns')

    # iv. Add neutral and path-through columns (from specs to results) : no real use, but for testing this useful feature
    speAbbrevCol = 'AbrevEsp'
    dfOptimExplSpecs[speAbbrevCol] = dfOptimExplSpecs['Espèce'].apply(lambda s: ''.join(m[:4] for m in s.split()))

    # v. Artificially generate some duplicates (for testing auto-removal later :-)
    dfOptimExplSpecs = dfOptimExplSpecs.append(dfOptimExplSpecs, ignore_index=True)
    logger.info("Explicit specs (concat'd + neutral col. + artificial duplicates):"
                f" n={len(dfOptimExplSpecs)}\n" + dfOptimExplSpecs.to_string())

    # ### c. MCDSZerothOrderTruncationOptimiser object
    # i. dfOptimExplSpecs columns that give the analysis & optimisation parameters
    optimParamsSpecsCols = ['FonctionClé', 'SérieAjust', 'CritChx', 'IntervConf',
                            'TroncGche', 'TroncDrte', 'MethOutliers', 'NbTrModel', 'NbTrDiscr',
                            'ExprOpt', 'MoteurOpt', 'ParExec']

    # ii. Optimiser
    optIndCol = 'IndOptim'
    optAbbrevCol = 'AbrevOptim'
    zoptr = ads.MCDSZerothOrderTruncationOptimiser(
        dfObsIndiv, effortConstVal=1, dSurveyArea=dSurveyArea,
        transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
        sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols, sampleDistCol=sampleDistCol,
        anlysSpecCustCols=[speAbbrevCol], abbrevCol=optAbbrevCol, abbrevBuilder=uivu.analysisAbbrev,
        anlysIndCol=optIndCol, sampleIndCol=sampleIndCol,
        distanceUnit='Meter', areaUnit='Hectare',
        surveyType='Point', distanceType='Radial', clustering=False,
        resultsHeadCols=dict(before=[optIndCol], sample=sampleSelCols,
                             after=optimParamsSpecsCols + [speAbbrevCol]),
        workDir=uivu.pWorkDir, runMethod='subprocess.run', runTimeOut=None,
        logData=False, logProgressEvery=1, backupEvery=5,  # <= Need at least 6 runs for seeing a recovery file
        defEstimKeyFn='HAZ', defEstimAdjustFn='POLY', defEstimCriterion='AIC', defCVInterval=93,
        defExpr2Optimise='1-ks', defMinimiseExpr=True,
        defOutliersMethod='quant', defOutliersQuantCutPct=5.5,
        defFitDistCutsFctr=dict(min=1/2, max=4/3), defDiscrDistCutsFctr=dict(min=1/2, max=1.2),
        defSubmitTimes=4, defSubmitOnlyBest=1,
        defCoreMaxIters=45, defCoreTermExprValue=0.2, defCoreMaxRetries=1)

    # ### d. Check optimisation specs
    dfOptimExplSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols, verdict, reasons = \
        zoptr.explicitParamSpecs(dfExplParamSpecs=dfOptimExplSpecs, dropDupes=True, check=True)
    assert len(dfOptimExplSpecs) == 12
    assert userParamSpecCols == optimParamsSpecsCols
    assert intParamSpecCols == ['EstimKeyFn', 'EstimAdjustFn', 'EstimCriterion', 'CvInterval',
                                'MinDist', 'MaxDist', 'OutliersMethod', 'FitDistCuts', 'DiscrDistCuts',
                                'Expr2Optimise', 'OptimisationCore', 'SubmitParams']
    assert unmUserParamSpecCols == []
    assert verdict
    assert not reasons

    logger.info("Checked optim. explicit specs (contat'd + neutral col. + artificial duplicates):"
                f" n={len(dfOptimExplSpecs)}\n" + dfOptimExplSpecs.to_string())

    # ### e. Run optimisations (parallel)
    # Python 3.8, Windows 10, Core i5 8365U (4 HT cores, 1.6-4.1GHz, cache  6Mb, bus 4GT/s) + SSD 256Gb NVME + RAM 16Gb, "optimal performances" power scheme
    # * 2021-01: 12 optimisations, 1430 analyses, 12 threads : subprocess = 3mn13, system = 2mn35, 1mn54?
    # * 2021-10-02: idem : system 2mn15
    # Python 3.8, Windows 10, Core i7 10850H (6 HT cores, 2.7-5.1GHz, cache 12Mb, bus 8GT/s) + SSD 512Gb NVME + RAM 32Gb "optimal performances" power scheme
    # * 2023-11-02: 12 optimisations, 1434 analyses, 12 threads : system = 1mn44s, 1mn46s, subprocess = 1mn47s
    #               (N.B. not saturating the CPU: 6 max MCDS // processes observed => increase threads ? data set too small ?)

    results = zoptr.run(dfOptimExplSpecs, threads=12)
    assert len(results) == 20  # Given the ParExec column of dfOptimExplSpecs

    zoptr.shutdown()

    assert speAbbrevCol in results.dfTransData('fr').columns

    dfFrRes = results.dfTransData('fr')
    logger.info(f'Optim. results (fr): n={len(dfFrRes)} =>\n' + dfFrRes.to_string())

    results.toExcel(pl.Path(zoptr.workDir) / 'unintst-mcds-optimiser-results-fr.xlsx', lang='fr')

    # ### f. Recovery : Run again optimisations, but from the last backup
    # (use case: crash, or mandatory/auto reboot of computer in the middle of a long optimisation run)
    # i. Quickly check content of the recovery file
    optResBkpPath = pl.Path(zoptr.workDir / 'optr-resbak-0.pickle.xz')
    with lzma.open(optResBkpPath, 'rb') as file:
        dfData, specs = pickle.load(file)
    assert len(dfData) == 17
    exptdCols = [optIndCol] + sampleSelCols + optimParamsSpecsCols + [speAbbrevCol]
    exptdCols += ['OptAbbrev', 'KeyFn', 'AdjSer', 'EstCrit', 'CVInt', 'OptCrit',
                  'MinDist', 'MaxDist', 'FitDistCuts', 'DiscrDistCuts',
                  'SetupStatus', 'SubmitStatus', 'NFunEvals', 'MeanFunElapd',
                  '1-ks', 'maxDist', 'discrDistCuts', 'chi2', 'minDist',
                  'fitDistCuts', 'ks', '1-chi2', 'cvmuw*cvmcw']
    logger.info('Expected dfData.columns: ' + ', '.join(exptdCols))
    logger.info('Read     dfData.columns: ' + ', '.join(dfData.columns))
    assert dfData.columns.tolist() == exptdCols

    # ii. Create the optimiser object : have to be a clone of the one whose execution was backed up
    zoptr = ads.MCDSZerothOrderTruncationOptimiser(
        dfObsIndiv, effortConstVal=1, dSurveyArea=dSurveyArea,
        transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
        sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols, sampleDistCol=sampleDistCol,
        anlysSpecCustCols=[speAbbrevCol], abbrevCol=optAbbrevCol, abbrevBuilder=uivu.analysisAbbrev,
        anlysIndCol=optIndCol, sampleIndCol=sampleIndCol,
        distanceUnit='Meter', areaUnit='Hectare',
        surveyType='Point', distanceType='Radial', clustering=False,
        resultsHeadCols=dict(before=[optIndCol], sample=sampleSelCols,
                             after=optimParamsSpecsCols + [speAbbrevCol]),
        workDir=uivu.pWorkDir, logProgressEvery=1,
        defEstimKeyFn='HAZ', defEstimAdjustFn='POLY', defEstimCriterion='AIC', defCVInterval=93,
        defExpr2Optimise='1-ks', defMinimiseExpr=True,
        defOutliersMethod='quant', defOutliersQuantCutPct=5.5,
        defFitDistCutsFctr=dict(min=1/2, max=4/3), defDiscrDistCutsFctr=dict(min=1/2, max=1.2),
        defSubmitTimes=4, defSubmitOnlyBest=1,
        defCoreMaxIters=45, defCoreTermExprValue=0.2, defCoreMaxRetries=1)

    # iii. Run optimisation with recovery results ... using exact same optim. specs (MANDATORY)
    results2 = zoptr.run(dfOptimExplSpecs, recover=True, threads=12)

    zoptr.shutdown()

    # iv. Quickly check results
    assert len(results2) == 20  # Given the ParExec column of dfOptimExplSpecs

    dfFrRes2 = results2.dfTransData('fr')
    logger.info(f'Optim. results (fr, recovery run): n={len(dfFrRes2)} =>\n' + dfFrRes2.to_string())

    # results2.toExcel(pl.Path(zoptr.workDir) / 'unintst-mcds-optimiser-results2-fr.xlsx', lang='fr')

    # v. Check equality of 1st 17 results in `results` and `results2`, + added num of results
    # (20 results at the end, 1st 17 ones in backup file, so only reloaded => only 3 left to be recomputed)
    dfComp = results.dfTransData(lang='fr').compare(results2.dfTransData(lang='fr'))
    logger.info(f'First run vs recovery compared results: n={len(dfComp)} =>\n' + dfComp.to_string())

    assert len(dfComp) <= 3

    logger.info0('PASS testMcdsZerothOrderTruncationOptimiser: Constructor, explicitParamSpecs, run, recovery')


###############################################################################
#                         Actions to be done after all tests                  #
###############################################################################
def testEnd():
    if KFinalCleanup:
        uivu.cleanupWorkDir()
    uivu.logEnd(what=KWhat2Test)
