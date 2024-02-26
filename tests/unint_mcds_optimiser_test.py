# coding: utf-8
# PyAuDiSam: Automation of Distance Sampling analyses with Distance software (http://distancesampling.org/)
import re
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

import numpy as np
import pandas as pd

import pyaudisam as ads

import unintval_utils as uivu


# Setup local logger.
logger = uivu.setupLogger('unt.opr', level=ads.DEBUG,
                          otherLoggers={'ads.eng': ads.INFO2, 'ads.dat': ads.INFO, 'ads.ans': ads.INFO2})

what2Test = 'optimiser'


###############################################################################
#                         Actions to be done before any test                  #
###############################################################################
def testBegin():
    uivu.logBegin(what=what2Test)


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


def testMcdsTruncOpterCtorGetXxxParams():

    # ### a. Individualised data set
    countCols = ['nMalAd10', 'nAutAd10', 'nMalAd5', 'nAutAd5']

    fds = ads.FieldDataSet(source='refin/ACDC2019-Naturalist-ExtraitObsBrutesAvecDist.txt',
                           importDecFields=['distMem'], countCols=countCols,
                           addMonoCatCols={'Adulte': count2AdultCat, 'Durée': count2DurationCat})
    dfObsIndiv = fds.individualise()
    dfObsIndiv.drop(columns=countCols, inplace=True)
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
             workDir=uivu.pTmpDir/ 'mcds-optr', runMethod='os.system', runTimeOut=120)
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
         abbrevCol=anlysAbbrevCol, workDir=uivu.pTmpDir/ 'mcds-optr', logData=False,
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
def testMcdsZerothOrderTruncationOptimiser():

    raise NotImplementedError('testMcdsZerothOrderTruncationOptimiser: TODO !')


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

            testDSOpterParseUserSpecs()

            testDSOpterParseDistTruncationUserSpec()

            testMcdsTruncOpterCtorGetXxxParams()
            testMcdsZerothOrderTruncationOptimiser()

            # Done.
            testEnd()

            # Success !
            rc = 0

        except Exception as exc:
            logger.exception(f'Exception: {exc}')
            rc = 1

    uivu.logEnd(what=what2Test, rc=rc)

    sys.exit(rc)
