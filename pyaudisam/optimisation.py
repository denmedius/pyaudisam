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

# Submodule "optimisation": One layer above analysis, to optimise some of the parameters of one DS analysis.

# Warning: If you add/remove/change optimisation parameters to XXOptimisation class ctors,
#          remember to update ParmXX constants accordingly in XXOptimiser classes.

from collections import namedtuple as ntuple

import math
import numpy as np
import pandas as pd

import zoopt
import pkg_resources as pkgrsc  # zoopt has no standard __version__ !

from . import log, runtime
from .executor import Executor
from .engine import MCDSEngine
from .analysis import MCDSAnalysis
from .analyser import MCDSAnalysisResultsSet

runtime.update({'zoopt': pkgrsc.get_distribution('zoopt').version})

logger = log.logger('ads.opn')


class Interval(object):

    """A basic closed interval class for numbers"""
    
    def __init__(self, min=0, max=-1):
    
        """Ctor
        
        Parameters:
        :param min: min of interval if a number, 
                    or interval itself if a (min, max) tuple/list or a dict(min=, max=) or an Interval
        :param max: max of interval if min is a number, ignored otherwise
        
        Ex: Interval(min=3, max=6.9), Interval((2.3, 4.2)), Interval(dict(max=2.3, min=1.2))
        """
    
        if isinstance(min, Interval):
            self.min = min.min
            self.max = min.max
        elif isinstance(min, (tuple, list)):
            self.min = min[0]
            self.max = min[1]
        elif isinstance(min, dict):
            self.min = min['min']
            self.max = min['max']
        else:
            self.min = min
            self.max = max
        
    def check(self, order=False, minRange=(None, None), maxRange=(None, None)):
    
        errors = list()  # No error by default.
        
        if order and self.min > self.max:
            errors.append(f'min:{self.min} > max:{self.max}')
            
        if minRange[0] is not None and self.min < minRange[0]:
            errors.append(f'min:{self.min} < {minRange[0]}')
        if minRange[1] is not None and self.min > minRange[1]:
            errors.append(f'min:{self.min} > {minRange[1]}')
        
        if maxRange[0] is not None and self.max < maxRange[0]:
            errors.append(f'max:{self.max} < {maxRange[0]}')
        if maxRange[1] is not None and self.max > maxRange[1]:
            errors.append(f'max:{self.max} > {maxRange[1]}')
        
        return ', '.join(errors)
        
    def __repr__(self):
    
        return '[{}, {}]'.format(self.min, self.max)
        

class Error(object):

    """Error class for shipping error messages to the end user"""

    def __init__(self, error=None, head=''):
        
        """Ctor
        
        Parameters:
        :param error: string or Error
        :error head: string ; ignored if error is an Error
        """
    
        self.heads = list()
        self.errors = list()
        
        if head or error:
            self.append(error, head)
        
    def append(self, error, head=''):
    
        """Append an error to another
        
        Parameters:
        :param error: string or Error
        :param head: string ; ignored if error is an Error
        """
        
        if isinstance(error, self.__class__):
            self.heads += error.heads
            self.errors += error.errors
        else:
            self.heads.append(head)
            self.errors.append(error)
        
    def __repr__(self):
    
        msgs = list()
        prvHd = ''
        for hd, err in zip(self.heads, self.errors):
            msg = ''
            if hd != prvHd and hd:
                msg += hd + ' : '
            msg += err
            msgs.append(msg)
            prvHd = hd
        return ' & '.join(msgs)
        
    def __bool__(self):
    
        return any(err for err in self.errors)


class DSOptimisation(object):
    
    """DSOptimisation (abstract) : A distance sampling analysis optimisation
         possibly run in parallel with others, through an asynchronous "submit then getResults" scheme.
    """
    
    def __init__(self, engine, sampleDataSet, name=None,
                 executor=None, customData=None, error=None,
                 expr2Optimise='chi2', minimiseExpr=False, **optimCoreOptions):
        
        """Ctor

        Parameters:
        :param engine: DS engine to use
        :param sampleDataSet: data.SampleDataSet instance to use
        :param name: used for prefixing run folders (sure to be automatically unique anyway),
            analysis names, and so on, only for user-friendliness and easier debugging ;
            default: None => auto-generated from optimisation parameters
        :param executor: Executor object to use for parallel execution of multiple optimisation instances
             Note: Up to the caller to shut it down when no more needed (not owned).
        :param customData: custom data for the run analyses to ship through
        :param error: if an error occurred somewhere before, a string explaining it
                      in order to prevent real submission, but still for keeping trace
                      of unrun optimisations in optimiser results table :
                      the optimisation then always returns at least a 1-row (empty/null) result + errors.
        :param string expr2Optimise: Math. expression (python syntax) to optimise,
                      using analyses results var. names inside (see derived classes for details)
        :param minimiseExpr: True for minimisation of expr2Optimise, false for maximisation
        :param optimCoreOptions: dict of specific options for the optimising core below
        """
        
        self.engine = engine
        self.executor = executor if executor is not None else Executor()
        self.sampleDataSet = sampleDataSet
        self.name = name
        self.customData = customData
        self.setupError = error
        self.expr2Optimise = expr2Optimise
        self.minimiseExpr = minimiseExpr
        self.optimCoreOptions = optimCoreOptions
        
    def _run(self, times=1, onlyBest=None, *args, **kwargs):
        
        """Really do the optimisation work : run the optimiser for this
        (this method is called by the executor thread/process that takes it from the submit queue)

        Parameters:
        :param times: Number of times to auto-run the optimisation (> 0 ; default = 1)
        :param onlyBest: When multiple runs, number of best optimisations to retain (> 0 ; default = all runs)
        :param *args, **kwargs: other _run params
        :return: List of "solutions", each as a dict with target analysis params in the ctor order
                 and then { expr2Optimise: analysis value }
        """

        raise NotImplementedError('DSOptimisation is an abstract class : implement _run in a derived class')
    
    def submit(self, times=1, onlyBest=None, error=None, *args, **kwargs):
    
        """Submit the optimisation, to run it possibly in parallel with others (through self.executor)
        
        :param times: Number of times to auto-run the optimisation (> 0 ; default = 1)
        :param onlyBest: Number of best run results to keep (> 0 ; default None = all runs)
        :param error: if an error occurred somewhere before since construction, a string explaining it
                      in order to prevent real submission, but still for keeping trace
                      of unrun optimisations in optimiser results table :
                      the optimisation then always returns at least a 1-row (empty/null) result + errors.
        :param args, kwargs: other _run arguments
        """

        # Submit optimisation work and return a Future object to ask from and wait for its results.
        self.future = \
            self.executor.submit(self._run, *args, 
                                 **{'times': times, 'onlyBest': onlyBest, 'error': error, **kwargs})
        
        return self.future

    def functionValue(self, anlysValue):
    
        """Analysis value to function (to optimise) value
        
        Inverse of functionValue (functionValue(analysisValue(x)) = x).
        """
        if pd.isnull(anlysValue):
            funcValue = None
        else:
            funcValue = anlysValue if self.minimiseExpr else -anlysValue
            
        return funcValue
    
    def analysisValue(self, fnValue):
    
        """Function (to optimise) value to analysis value
        
        Inverse of functionValue (analysisValue(functionValue(x)) = x).
        """
    
        return fnValue if self.minimiseExpr else -fnValue 
    

# class MCDSOptimisation(DSOptimisation):
#
#     """Optimisation for MCDS analyses ... stuff
#     """
#
#     pass


class MCDSTruncationOptimisation(DSOptimisation):

    """Optimisation for MCDS analyses distance truncations
    """
    
    EngineClass = MCDSEngine
        
    # Names of possible solution dimensions (the truncation parameter values we are searching for).
    SolutionDimensionNames = ['minDist', 'maxDist', 'fitDistCuts', 'discrDistCuts']
    
    def __init__(self, engine, sampleDataSet, name=None, executor=None,
                 distanceField='Distance', customData=None,
                 logData=False, autoClean=True, error=None,
                 estimKeyFn=EngineClass.EstKeyFnDef, estimAdjustFn=EngineClass.EstAdjustFnDef,
                 estimCriterion=EngineClass.EstCriterionDef, cvInterval=EngineClass.EstCVIntervalDef,
                 minDist=None, maxDist=None, fitDistCutsFctr=None, discrDistCutsFctr=None,
                 fitDistCuts=None, discrDistCuts=None,
                 expr2Optimise='chi2', minimiseExpr=False, **optimCoreOptions):

        """Ctor
        
        Parameters (see base class, specific ones only here):
        :param distanceField: Name of distance data column in sampleDataSet table
        :param error: a string for explaining any pre-init error that'll prevent the optimisation from running,
                      but will keep producing results (null of course) ;
                      this is done for keeping trace of unrun optimisations in results table (1 empty/null row).

        Fixed analysis parameters (see MCDSAnalysis):
        :param estimKeyFn: 
        :param estimAdjustFn: 
        :param estimCriterion: 
        :param cvInterval:  

        Optimisation target parameters (at least one MUST be not None):
        :param minDist: Min, max Interval() or 2-item list or tuple,
                        or int / float constant for left truncation distance ;
                        default: None => not optimised
        :param maxDist: Idem, for right truncation distance ;
                        default: None => not optimised
        :param fitDistCutsFctr: Idem, for the mult. factor to apply to sqrt(nb of sightings)
                        to get the number of distance cut intervals (for _model_fitting_) ;
                        default: None => not optimised if not fitDistCuts
        :param discrDistCutsFctr: Idem for the mult. factor to apply to sqrt(nb of sightings)
                        for the number of distance cut intervals (for _distance_values_discretisation_) ;
                        default: None => not optimised if not discrDistCuts
        :param fitDistCuts: Idem for the absolute number of distance cut intervals
                        (for _model_fitting_) if fitDistCutsFctr is None ;
                        default: None => fitDistCuts not optimised if not fitDistCutsFctr
        :param discrDistCuts: Idem for the number of distance cut intervals
                        (for _distance_values_discretisation_) if discrDistCutsFctr is None ;
                        default: None => not optimised if not discrDistCutsFctr
        """

        # Check engine
        assert isinstance(engine, MCDSEngine), 'Engine must be an MCDSEngine'
        
        # Check and prepare analysis and optimisation params
        moreError = Error()
        if not (len(estimKeyFn) >= 2 and estimKeyFn in [kf[:len(estimKeyFn)] for kf in engine.EstKeyFns]):
            moreError.append('Invalid estimate key function {}: should be in {} or at least 2-char abbreviations'
                             .format(estimKeyFn, engine.EstKeyFns))
        if not (len(estimAdjustFn) >= 2 and estimAdjustFn in [kf[:len(estimAdjustFn)] for kf in engine.EstAdjustFns]):
            moreError.append('Invalid estimate adjust function {}: should be in {} or at least 2-char abbreviations'
                             .format(estimAdjustFn, engine.EstAdjustFns))
        if not (estimCriterion in engine.EstCriteria):
            moreError.append('Invalid estimate criterion {}: should be in {}'
                             .format(estimCriterion, engine.EstCriteria))
        if not (0 < cvInterval < 100):
            moreError.append('Invalid cvInterval {}% : should be in {}'.format(cvInterval, ']0%, 100%['))
               
        if not any(optPar is not None for optPar in [minDist, maxDist, fitDistCuts, discrDistCuts]):
            moreError.append('At least 1 analysis parameter has to be optimised')
            
        self.dConstParams = dict()  # Params that won't be optimised : constants.
        if minDist is not None and not isinstance(minDist, (int, float)):
            minDist = Interval(minDist)
        elif minDist is not None:
            self.dConstParams['minDist'] = minDist
            
        if maxDist is not None and not isinstance(maxDist, (int, float)):
            maxDist = Interval(maxDist)
        elif maxDist is not None:
            self.dConstParams['maxDist'] = maxDist
            
        if fitDistCutsFctr is not None:
            fitDistCutsFctr = Interval(fitDistCutsFctr)
        if fitDistCuts is not None and not isinstance(fitDistCuts, (int, float)):
            fitDistCuts = Interval(fitDistCuts)
        elif fitDistCuts is not None and fitDistCutsFctr is None:
            self.dConstParams['fitDistCuts'] = fitDistCuts
            
        if discrDistCutsFctr is not None:
            discrDistCutsFctr = Interval(discrDistCutsFctr)
        if discrDistCuts is not None and not isinstance(discrDistCuts, (int, float)):
            discrDistCuts = Interval(discrDistCuts)
        elif discrDistCuts is not None and discrDistCutsFctr is None:
            self.dConstParams['discrDistCuts'] = discrDistCuts

        if self.dConstParams:
            logger.info1(f'TrOptimisation({self.dConstParams})')
        
        if not (minDist is None or isinstance(minDist, (int, float)) or 0 <= minDist.min < minDist.max):
            moreError.append('Invalid left truncation distance {}'.format(minDist))
        if not (maxDist is None or isinstance(maxDist, (int, float)) or 0 <= maxDist.min < maxDist.max):
            moreError.append('Invalid right truncation distance {}'.format(maxDist))
        if not (minDist is None or isinstance(minDist, (int, float))
                or maxDist is None or isinstance(maxDist, (int, float)) or minDist.max < maxDist.min):
            moreError.append('Max left truncation distance {} must be lower than min right one {}'
                             .format(minDist.max, maxDist.min))
        
        if not (fitDistCutsFctr is None or 0 <= fitDistCutsFctr.min < fitDistCutsFctr.max):
            moreError.append('Invalid mult. factor for number of fitting distance cuts {}'
                             .format(fitDistCutsFctr))
        if not (discrDistCutsFctr is None or 0 <= discrDistCutsFctr.min < discrDistCutsFctr.max):
            moreError.append('Invalid mult. factor number for distance discretisation cuts {}'
                             .format(discrDistCutsFctr))
        
        if not (fitDistCutsFctr is None or fitDistCuts is None):
            moreError.append('Can\'t specify both absolute value and mult. factor'
                             ' for number of discretisation distance cuts')
        if not (fitDistCuts is None or isinstance(fitDistCuts, (int, float))
                or 2 <= fitDistCuts.min < fitDistCuts.max):
            moreError.append('Invalid number of fitting distance cuts {}'.format(fitDistCuts))
        if not (discrDistCutsFctr is None or discrDistCuts is None):
            moreError.append('Can\'t specify both absolute value and mult. factor'
                             ' for number of discretisation distance cuts')
        if not (discrDistCuts is None or isinstance(discrDistCuts, (int, float))
                or 2 <= discrDistCuts.min < discrDistCuts.max):
            moreError.append('Invalid number of distance discretisation cuts {}'.format(discrDistCuts))
        
        # Build name from main params if not specified
        if name is None:
            fields = ['mcds'] + [p[:3].lower() for p in [estimKeyFn, estimAdjustFn]]
            if estimCriterion != self.EngineClass.EstCriterionDef:
                fields.append(estimCriterion.lower())
            if cvInterval != self.EngineClass.EstCVIntervalDef:
                fields.append(str(cvInterval))
            name = '-'.join(fields)

        # Show and merge errors if any.
        if moreError:
            logger.error('Check failed for optimisation params: ' + str(moreError))
            if error:
                error.append(moreError)
            else:
                error = moreError
        
        # Initialise base.
        super().__init__(engine, sampleDataSet, name=name, 
                         executor=executor, customData=customData, error=error,
                         expr2Optimise=expr2Optimise, minimiseExpr=minimiseExpr, **optimCoreOptions)
                
        # Save / compute params.
        # a. Analysis
        self.logData = logData
        self.autoClean = autoClean
        self.estimKeyFn = estimKeyFn
        self.estimAdjustFn = estimAdjustFn
        self.estimCriterion = estimCriterion
        self.cvInterval = cvInterval

        # b. Analysis or optimisation (whether const or variant params)
        self.minDist = minDist
        self.maxDist = maxDist
        if fitDistCutsFctr is not None or discrDistCutsFctr is not None:
            sqrtNbSights = math.sqrt(len(sampleDataSet.dfData[distanceField].dropna()))
        if fitDistCutsFctr is not None:
            self.fitDistCuts = Interval(min=int(round(fitDistCutsFctr.min*sqrtNbSights)),
                                        max=int(round(fitDistCutsFctr.max*sqrtNbSights)))
        else:
            self.fitDistCuts = fitDistCuts
        if discrDistCutsFctr is not None:
            self.discrDistCuts = Interval(min=int(round(discrDistCutsFctr.min*sqrtNbSights)),
                                          max=int(round(discrDistCutsFctr.max*sqrtNbSights)))
        else:
            self.discrDistCuts = discrDistCuts
            
        # Other optimisation stuff.
        fltSup = float('inf')  # sys.float_info.max
        self.invalidFuncValue = fltSup if minimiseExpr else -fltSup

        # Where to store analyses elapsed times before computing stats at the end.
        self.elapsedTimes = list()

    # Post-process analysis results (adapted from MCDSAnalysisResultsSet.postComputeColumns)
    @staticmethod
    def _postProcessAnalysisResults(sResults):

        RSClass = MCDSAnalysisResultsSet  # ResultsSet class

        # Chi2 test probability.
        chi2AllColLbls = [col for col in RSClass.CLsChi2All if col in sResults.keys()]
        sResults[RSClass.CLChi2] = RSClass._determineChi2Value(sResults[chi2AllColLbls])

        # Combined quality indicators
        # a. Make sure requested columns are there, and add them if not (NaN value)
        miCompCols = RSClass.CLsQuaIndicSources
        for miCol in miCompCols:
            if miCol not in sResults.index:
                sResults[miCol] = np.nan

        # b. NaN value MUST kill down the indicators to compute => we have to enforce this
        sResults.fillna({RSClass.CLNObs: RSClass.KilrNObs,
                         RSClass.CLChi2: RSClass.KilrStaTest, RSClass.CLKS: RSClass.KilrStaTest,
                         RSClass.CLCvMUw: RSClass.KilrStaTest, RSClass.CLCvMCw: RSClass.KilrStaTest,
                         RSClass.CLDCv: RSClass.KilrDensCv,  # Usually considered good under 0.3
                         RSClass.CLNTotObs: RSClass.KilrNTotObs,  # Should slap down _normObs whatever NObs
                         RSClass.CLNAdjPars: RSClass.KilrNPars,  # Should slap down _normNAdjPars whatever NObs
                         RSClass.CLNTotPars: RSClass.KilrNPars},
                        inplace=True)

        # c. Compute indicators at last !
        aCombQuaData = np.expand_dims(sResults[miCompCols].values, axis=0)  # Series to 1 row results array.
        sResults[RSClass.CLCmbQuaBal1] = RSClass._combinedQualityBalanced1(aCombQuaData)
        for miCol, aIndic in zip(RSClass.CLsNewQuaIndics, RSClass._combinedQualityAll(aCombQuaData)):
            sResults[miCol] = aIndic[0]

        return sResults

    # Alias and name / index (in analysis results) of results values available
    # for use in analysis value computation expressions
    # And the worst possible value for each, for (bad) default value when not present in results for some reason.
    # Warning: Don't use np.nan for these worst values : at least zoopt doesn't like it !
    RSClass = MCDSAnalysisResultsSet
    AnlysResultsIndex = \
        dict(chi2=(RSClass.CLChi2, RSClass.KilrStaTest), ks=(RSClass.CLKS, RSClass.KilrStaTest),
             cvmuw=(RSClass.CLCvMUw, RSClass.KilrStaTest), cvmcw=(RSClass.CLCvMCw, RSClass.KilrStaTest),
             dcv=(RSClass.CLDCv, RSClass.KilrDensCv),
             balq1=(RSClass.CLCmbQuaBal1, RSClass.KilrBalQua), balq2=(RSClass.CLCmbQuaBal2, RSClass.KilrBalQua),
             balq3=(RSClass.CLCmbQuaBal3, RSClass.KilrBalQua),
             balqc2=(RSClass.CLCmbQuaChi2, RSClass.KilrBalQua), balqks=(RSClass.CLCmbQuaKS, RSClass.KilrBalQua),
             balqcv=(RSClass.CLCmbQuaDCv, RSClass.KilrBalQua))

    @classmethod
    def _getAnalysisResultValue(cls, resultExpr, sResults, invalidValue):
        
        dLocals = {alias: sResults.get(name, worst) for alias, (name, worst) in cls.AnlysResultsIndex.items()}
                                          
        # logger.debug3('_getAnalysisResultValue: locals={}'.format(dLocals))
        
        try:
            value = eval(resultExpr, None, dLocals)
            if np.isnan(value):
                value = invalidValue
        except Exception as exc:
            value = invalidValue
            logger.warning('Failed to evaluate {} : {}'.format(resultExpr, exc))
        
        return value

    def _runOneAnalysis(self, minDist=MCDSEngine.DistMinDef, maxDist=MCDSEngine.DistMaxDef, 
                        fitDistCuts=MCDSEngine.DistFitCutsDef, discrDistCuts=MCDSEngine.DistDiscrCutsDef,
                        valueExpr='chi2'):
                              
        """Run one analysis (among many others in the optimisation process) and compute its values to optimise
           See MCDSAnalysis.__init__ for most parameters
           :param string valueExpr: Math. expression (python syntax) for computing analysis value
               (using result names from AnlysResultsIndex) (ex: chi2, chi2*ks, ...)
        """

        # Run analysis (Submit, and wait for end of execution) : parallelism taken care elsewhere.
        startTime = pd.Timestamp.now()

        dNameFlds = dict(l=minDist, r=maxDist, f=fitDistCuts, d=discrDistCuts)
        nameSufx = ''.join(c+str(int(v)) for c, v in dNameFlds.items() if v is not None)

        logger.debug2(f'Running analysis (minDist={minDist}, maxDist={maxDist},'
                      f'fitDistCuts={fitDistCuts}, discrDistCuts={discrDistCuts}) ...')
                      
        anlys = MCDSAnalysis(engine=self.engine, sampleDataSet=self.sampleDataSet,
                             name=self.name + '-' + nameSufx, logData=self.logData,
                             estimKeyFn=self.estimKeyFn, estimAdjustFn=self.estimAdjustFn,
                             estimCriterion=self.estimCriterion, cvInterval=self.cvInterval,
                             minDist=minDist, maxDist=maxDist,
                             fitDistCuts=fitDistCuts, discrDistCuts=discrDistCuts)

        anlys.submit()
        
        sResults = anlys.getResults(postCleanup=self.autoClean)
        # logger.debug3('Analysis results: {}'.format(sResults.to_dict()))

        # Post-process results, and compute analysis values (_the_ values to optimise).
        if anlys.success() or anlys.warnings():

            sResults = self._postProcessAnalysisResults(sResults)

            value = self._getAnalysisResultValue(valueExpr, sResults, self.invalidFuncValue)
        
        else:

            value = self.invalidFuncValue

        logger.debug1('Analysis result value: {} = {}'.format(valueExpr, value))

        # Store elapsed time for this analysis, for later stats
        self.elapsedTimes.append((pd.Timestamp.now() - startTime).total_seconds())

        return value
    
    # Column names and translations
    RunColumns = ['OptAbbrev', 'KeyFn', 'AdjSer', 'EstCrit', 'CVInt', 'OptCrit',
                  'MinDist', 'MaxDist', 'FitDistCuts', 'DiscrDistCuts']
    
    DfRunColumnTrans = \
        pd.DataFrame(index=RunColumns,
                     data=dict(en=['Optim Abbrev', 'Mod Key Fn', 'Mod Adj Ser', 'Mod Chc Crit', 'Conf Interv',
                                   'Optim Crit', 'Left Trunc Dist', 'Right Trunc Dist',
                                   'Fit Dist Cuts', 'Discr Dist Cuts'],
                               fr=['Abrév Optim', 'Fn Clé Mod', 'Sér Ajust Mod', 'Crit Chx Mod', 'Interv Conf',
                                   'Crit Optim', 'Dist Tronc Gche', 'Dist Tronc Drte',
                                   'Tranch Dist Mod', 'Tranch Dist Discr']))

    def getResults(self):
        
        # Wait for availability (async) and get optimised results = target analysis parameters.
        ldOptimResults = self.future.result()
        
        # Build header columns for all the optimisation results (same for all):
        # actual (not default) analysis and optimisation params.
        sHead = pd.Series(data=[self.name, self.estimKeyFn, self.estimAdjustFn,
                                self.estimCriterion, self.cvInterval,
                                '{}({})'.format('min' if self.minimiseExpr else 'max', self.expr2Optimise),
                                str(self.minDist), str(self.maxDist),
                                str(self.fitDistCuts), str(self.discrDistCuts)],
                          index=self.RunColumns)
        
        # Build final table of optimisation results : header then results, for each optimisation.
        dfOptimResults = pd.DataFrame(data=[sHead.append(pd.Series(optRes)) for optRes in ldOptimResults])
        
        # Done
        return dfOptimResults


class MCDSZerothOrderTruncationOptimisation(MCDSTruncationOptimisation):
    
    """Zero-order optimisation (no derivation used) for MCDS analyses distance truncations
    """
    
    EngineClass = MCDSEngine
        
    # Column names and translations
# TODO
#    RunColumns = MCDSTruncationOptimisation.RunColumns + ['SetupStatus', 'SubmitStatus', 'NFunEvals', 'MeanFunElapd']
#    
#    DfRunColumnTrans = \
#        pd.DataFrame(index=RunColumns,
#                     data=dict(en=list(MCDSTruncationOptimisation.DfRunColumnTrans.en)
#                                  + ['Setup Status', 'Submit Status', 'Num Fun Evals', 'Mean Fun Elapsed'],
#                               fr=list(MCDSTruncationOptimisation.DfRunColumnTrans.fr)
#                                  + ['Setup Status', 'Submit Status', 'Num Fun Evals', 'Mean Fun Elapd']))

    @staticmethod
    def zoopt(mxi=100, tv=None, a='racos', mxr=0):
    
        """Function for parsing optimisation specs:
        * see optimiser.DSOptimiser._parseOptimCoreUserSpecs()
        * see zoopt module for details
        
        Parameters:
        :param mxi: max nb of iterations; default 100; 0 => no limit => use terminal value
        :param tv: terminal value; default None = no terminal value check
        :param a: algorithm to use: 'racos' (RACOS (AAAI'16) and Sequential RACOS (AAAI'17))
                                    or 'poss' (Pareto Optimisation Subset Selection (NIPS'15))
        :param mxr: max nb of retries on zoopt.Opt.min (exception); <= 0 => no retry
        """
    
        dParms = dict(core='zoopt')
        
        mxi = max(0, mxi)
        if mxi != 0:  # Default zoopt value
            dParms.update(maxIters=mxi)

        if tv is not None:  # Default zoopt value
            dParms.update(termExprValue=tv)
            
        if a != 'racos':  # Default zoopt value
            dParms.update(algorithm=a)
            
        mxr = max(0, mxr)
        if mxr != 0:  # Default value
            dParms.update(maxRetries=mxr)
            
        return dParms

    CoreName = 'zoopt'
    CoreParamNames = ['maxIters', 'termExprValue', 'algorithm', 'maxRetries']
    CoreUserSpecParser = zoopt
    
    Parameter = ntuple('Parameter', ['name', 'interval', 'continuous', 'ordered'],
                       defaults=['unknown', Interval(), True, True])
    
    def __init__(self, engine, sampleDataSet, name=None,
                 distanceField='Distance', customData=None,
                 executor=None, logData=False, autoClean=True, error=None,
                 estimKeyFn=EngineClass.EstKeyFnDef, estimAdjustFn=EngineClass.EstAdjustFnDef, 
                 estimCriterion=EngineClass.EstCriterionDef, cvInterval=EngineClass.EstCVIntervalDef,
                 minDist=None, maxDist=None, fitDistCutsFctr=None, discrDistCutsFctr=None,
                 fitDistCuts=None, discrDistCuts=None,
                 expr2Optimise='chi2', minimiseExpr=False, 
                 maxIters=100, termExprValue=None, algorithm='racos', maxRetries=0):  # CoreParamNames !

        """Ctor
        
        Other parameters: See base class
        
        ZOOpt specific parameters:
        :param algorithm: Zeroth Order optimisation algorithm to use
                          (only 'racos' is suitable here, 'poss' is not, don't use)
        :param maxRetries: Max number of retries on optim. core failure ; default: 0 => 0 retries = 1 try
        :param maxIters: Number of iterations that stop optimisation algorithm when reached ; default: 0 => no limit
        :param termExprValue: Value that stops optimisation algorithm when exceeded ;
                          default: None => no such check done
                          Note: when minimising, "exceeded" means that the function value becomes <= that termExprValue,
                                and the other way round when maximising.
        """
        
        # Initialise base.
        super().__init__(engine, sampleDataSet, name=name,
                         distanceField=distanceField, customData=customData,
                         executor=executor, logData=logData, autoClean=autoClean, error=error,
                         estimKeyFn=estimKeyFn, estimAdjustFn=estimAdjustFn,
                         estimCriterion=estimCriterion, cvInterval=cvInterval,
                         minDist=minDist, maxDist=maxDist,
                         fitDistCutsFctr=fitDistCutsFctr, discrDistCutsFctr=discrDistCutsFctr,
                         fitDistCuts=fitDistCuts, discrDistCuts=discrDistCuts,
                         expr2Optimise=expr2Optimise, minimiseExpr=minimiseExpr,
                         maxIters=maxIters, termExprValue=termExprValue, algorithm=algorithm, maxRetries=maxRetries)

        # Prepare optimisation parameters.
        self.dVariantParams = dict()  # Keys must be from SolutionDimensionNames
        if isinstance(self.minDist, Interval):
            self.dVariantParams.update(minDist=self.Parameter(name='MinDist', interval=self.minDist,
                                                              continuous=True, ordered=True))
        if isinstance(self.maxDist, Interval):
            self.dVariantParams.update(maxDist=self.Parameter(name='MaxDist', interval=self.maxDist,
                                                              continuous=True, ordered=True))
        if isinstance(self.fitDistCuts, Interval):
            self.dVariantParams.update(fitDistCuts=self.Parameter(name='FitDistCuts', interval=self.fitDistCuts,
                                                                  continuous=False, ordered=True))
        if isinstance(self.discrDistCuts, Interval):
            self.dVariantParams.update(discrDistCuts=self.Parameter(name='DiscrDistCuts', interval=self.discrDistCuts,
                                                                    continuous=False, ordered=True))
        
        assert all(name in self.SolutionDimensionNames for name in self.dVariantParams)
        
        logger.info1(f'ZOTrOptimisation({self.dVariantParams})')
        
        # Columns names for each optimisation result row (see _run).
        self.resultsCols = ['SetupStatus', 'SubmitStatus', 'NFunEvals', 'MeanFunElapd'] \
                           + list(self.dVariantParams.keys()) + [self.expr2Optimise]
        
        # zoopt optimiser initialisation.
        self.zooptDims = \
            zoopt.Dimension(size=len(self.dVariantParams),
                            regs=[[param.interval.min, param.interval.max] for param in self.dVariantParams.values()],
                            tys=[param.continuous for param in self.dVariantParams.values()],
                            order=[param.ordered for param in self.dVariantParams.values()])

        self.zooptObjtv = zoopt.Objective(func=self._function, dim=self.zooptDims)

        self.zooptParams = zoopt.Parameter(budget=maxIters, algorithm=algorithm,
                                           terminal_value=self.functionValue(termExprValue))

        # Retry-on-failure stuff
        self.maxRetries = max(maxRetries, 0)
        self.retries = 0  # Just to keep track ...

    def _function(self, solution):
    
        """The function to minimise : called as many times as needed by zoopt kernel.
        :param solution: the zoopt "possible Solution" object to try and check if good enough
        """

        # Retrieve input parameters (to optimise)
        dParams = dict(zip(self.dVariantParams.keys(), solution.get_x()))
        dParams.update(self.dConstParams)
        
        # Run analysis and get value.
        anlysValue = self._runOneAnalysis(valueExpr=self.expr2Optimise, **dParams)
        
        # One more function evaluated.
        self.nFunEvals += 1

        # Compute function value from analysis value.
        return self.functionValue(anlysValue)
    
    def _optimize(self):

        # Run the optimiser (for as many retries as requested in case of it fails)
        nTriesLeft = maxTries = self.maxRetries + 1
        while True:
            try:
                return zoopt.Opt.min(self.zooptObjtv, self.zooptParams)
                # Done !
            except Exception as exc:
                nTriesLeft -= 1
                if nTriesLeft > 0:
                    self.retries += 1  # Just to keep track.
                    logger.warning('zoopt.Opt.min retry #{} on {}'.format(maxTries - nTriesLeft, exc),
                                   exc_info=True)
                else:
                    logger.warning('zoopt.Opt.min failed after #{} tries on {}'.format(maxTries, exc),
                                   exc_info=True)
                    return None

    def _run(self, times=1, onlyBest=None, error=None, *args, **kwargs):
        
        """Really do the optimisation work (use the optimisation core for this).
        (this method is called by the executor thread/process that takes it from the submit queue)
        :return: List of "solutions", each as a dict with target analysis params in the ctor order
                 preceded by { expr2Optimise: analysis value }
        """
        
        # Number of evaluations of function to optimise.
        self.nFunEvals = 0
      
        # When self.setupError or (submit) error, simply return a well-formed but empty results.
        if self.setupError or error:
            return [dict(zip(self.resultsCols, 
                             [self.setupError, error, self.nFunEvals, 0.0] + [None]*(len(self.resultsCols) - 3)))]
            
        # Run the requested optimisations and get the solutions (ignore None ones).
        solutions = [self._optimize() for _ in range(times)]
        solutions = [sol for sol in solutions if sol is not None]
        
        # Keep only best solutions if requested.
        if onlyBest is not None and len(solutions) >= onlyBest:
            solutions = sorted(solutions, key=lambda sol: sol.get_value(), reverse=self.minimiseExpr)[:onlyBest]
        
        # Extract target results if any.
        if not solutions:
            return []
        
        nMeanFunEvals = int(round(self.nFunEvals / len(solutions)))
        meanFunElapd = np.nan if not self.nFunEvals or not self.elapsedTimes \
                              else sum(self.elapsedTimes) / self.nFunEvals
        return [dict(zip(self.resultsCols, [None, None, nMeanFunEvals, meanFunElapd]
                                           + sol.get_x() + [self.analysisValue(sol.get_value())]))
                for sol in solutions]
