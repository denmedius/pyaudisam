# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Optimisation : One layer above analysis, to optimise some of the parameters of one DS analysis.
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment

# Warning: If you add/remove/change optimisation parameters to XXOptimisation class ctors,
#          remember to update ParmXX constants accordingly in XXOptimiser classes.


import sys
import pathlib as pl
import shutil
import argparse

from collections import OrderedDict as odict, namedtuple as ntuple

import numpy as np
import pandas as pd

import logging

import zoopt

from autods.engine import MCDSEngine
from autods.data import MCDSAnalysisResultsSet
from autods.analysis import MCDSAnalysis
from autods.executor import Executor

# The local logger
logger = logging.getLogger('autods')

class Interval(object):

    """A basic closed interval class for numbers"""
    
    def __init__(self, min=0, max=-1):
    
        """Ctor
        
        Parameters:
        :param min: min of interval if a number, 
                    or interval itself if a (min, max) tuple/list or a dict(min=, max=) or a Interval
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
        
    def __str__(self):
        return '[{}, {}]'.format(self.min, self.max)
        

class DSOptimisation(object):
    
    """DSOptimisation (abstract) : A distance sampling analysis optimisation
         possibly run in parallel with others, through an asynchronous "submit then getResults" scheme.
    """
    
    #OptStatusCol = 'OptStatus'
    
    def __init__(self, engine, sampleDataSet, name=None,
                 executor=None, customData=None, error=None,
                 expr2optimise='chi2', minimiseExpr=False, **optimCoreOptions):
        
        """Ctor

        Parameters:
        :param engine: DS engine to use
        :param sampleDataSet: data.SampleDataSet instance to use
        :param name: used for prefixing run folders (sure to be automatically unique anyway),
            analysis names, and so on, only for user-friendliness and deasier debugging ;
            default: None => auto-generated from optimisation parameters
        :param executor: Executor object to use for parallel execution of multiple optimisations instances
             Note: Up to the caller to shut it down when no more needed (not owned).
        :param customData: custom data for the run analyses to ship through
        :param error: if an error occurred somewhere before, a string explaining it
                      in order to prevent real submission, but still for keeping trace
                      of unrun optimisations in optimiser results table :
                      the optimisation then always returns at least a 1-row (empty/null) result + errors.
        :param string expr2optimise: Math. expression (python syntax) to optimise,
                      using analyses results var. names inside (see derived classes for details)
        :param minimiseExpr: True for minimisation of expr2optimise, false for maximisation
        :param optimCoreOptions: dict of specific options for the optimising core below
        """
        
        self.engine = engine
        self.executor = executor if executor is not None else Executor(parallel=False)
        self.sampleDataSet = sampleDataSet
        self.name = name
        self.customData = customData
        self.setupError = error
        self.expr2optimise = expr2optimise
        self.minimiseExpr = minimiseExpr
        self.optimCoreOptions = optimCoreOptions

    def _run(self, repeats=1, onlyBest=None, *args, **kwargs):
        
        """Really do the optimisation work : run the optimiser for this
        :param repeats: Number of times to repeat the optimisation
        :param onlyBest: When repeating, number of best optimisations to retain (default = repeats)
        :param *args, **kwargs: other _run params
        :return: List of "solutions", each as an odict with target analysis params in the ctor order
                 and then { expr2optimise: analysis value }
        """

        raise NotImplementedError('DSOptimisation is an abstract class : implement _run in a derived class')
    
    def submit(self, repeats=1, onlyBest=None, error=None, *args, **kwargs):
    
        """Submit the optimisation, to run it possibly in parallel with others (through self.executor)
        
        :param repeats: Number of auto-repetitions of each optimisation (default 1 = no repetition)
        :param onlyBest: Number of best repetition results to keep (default None = all repetitions)
        :param error: if an error occurred somewhere before since construction, a string explaining it
                      in order to prevent real submission, but still for keeping trace
                      of unrun optimisations in optimiser results table :
                      the optimisation then always returns at least a 1-row (empty/null) result + errors.
        :param *args, **kwargs: other _run arguments
        """

        # Submit optimisation work and return a Future object to ask from and wait for its results.
        self.future = \
            self.executor.submit(self._run, *args, 
                                 **{ 'repeats': repeats, 'onlyBest': onlyBest, 'error': error, **kwargs })
        
        return self.future

    def functionValue(self, anlysValue):
    
        """Analysis value to function (to optimise) value
        
        Inverse of functionValue (functionValue(analysisValue(x)) = x).
        """
    
        return anlysValue if self.minimiseExpr else -anlysValue 
    
    def analysisValue(self, fnValue):
    
        """Function (to optimise) value to analysis value
        
        Inverse of functionValue (analysisValue(functionValue(x)) = x).
        """
    
        return fnValue if self.minimiseExpr else -fnValue 
    

#class MCDSOptimisation(DSOptimisation):
#
#    """Optimisation for MCDS analyses ... stuff
#    """
#
#    pass


class MCDSTruncationOptimisation(DSOptimisation):

    """Optimisation for MCDS analyses distance truncations
    """
    
    EngineClass = MCDSEngine
        
    def __init__(self, engine, sampleDataSet, name=None, executor=None,
                 distanceField='Distance', customData=None, logData=False, error=None,
                 estimKeyFn=EngineClass.EstKeyFnDef, estimAdjustFn=EngineClass.EstAdjustFnDef, 
                 estimCriterion=EngineClass.EstCriterionDef, cvInterval=EngineClass.EstCVIntervalDef,
                 minDist=None, maxDist=None, fitDistCutsFctr=None, discrDistCutsFctr=None,
                 fitDistCuts=None, discrDistCuts=None,
                 expr2optimise='chi2', minimiseExpr=False, **optimCoreOptions):

        """Ctor
        
        Parameters (see base class, specific ones only here):
        :param distanceField: Name of distance data column in sampleDataSet table

        Fixed analysis parameters (see MCDSAnalysis):
        :param estimKeyFn: 
        :param estimAdjustFn: 
        :param estimCriterion: 
        :param cvInterval:  

        Optimisation target parameters (at least one MUST be not None):
        :param minDist: Distance Interval() for left truncation distance ; default: None => not optimised
        :param maxDist: Distance Interval() for right truncation distance ; default: None => not optimised
        :param fitDistCutsFctr: Interval() for the mult. factor to apply to sqrt(nb of sightings)
            to get the number of distance cut intervals (for _model_fitting_) ;
            default: None => not optimised if not fitDistCuts
        :param discrDistCutsFctr: Interval() for the mult. factor to apply to sqrt(nb of sightings)
            for the number of distance cut intervals (for _distance_values_discretisation_) ;
            default: None => not optimised if not discrDistCuts
        :param fitDistCuts: Interval() for the absolute number of distance cut intervals
            (for _model_fitting_) if fitDistCutsFctr is None ;
            default: None => fitDistCuts not optimised if not fitDistCutsFctr
        :param discrDistCuts: Interval() for the number of distance cut intervals
            (for _distance_values_discretisation_) if discrDistCutsFctr is None ;
            default: None => not optimised if not discrDistCutsFctr
        """

        # Check engine
        assert isinstance(engine, MCDSEngine), 'Engine must be an MCDSEngine'
        
        # Check analysis params
        assert len(estimKeyFn) >= 2 and estimKeyFn in [kf[:len(estimKeyFn)] for kf in engine.EstKeyFns], \
               'Invalid estimate key function {}: should be in {} or at least 2-char abreviations' \
               .format(estimKeyFn, engine.EstKeyFns)
        assert len(estimAdjustFn) >= 2 \
               and estimAdjustFn in [kf[:len(estimAdjustFn)] for kf in engine.EstAdjustFns], \
               'Invalid estimate adjust function {}: should be in {} or at least 2-char abreviations' \
               .format(estimAdjustFn, engine.EstAdjustFns)
        assert estimCriterion in engine.EstCriterions, \
               'Invalid estimate criterion {}: should be in {}'.format(estimCriterion, engine.EstCriterions)
        assert cvInterval > 0 and cvInterval < 100, \
               'Invalid cvInterval {}% : should be in {}'.format(cvInterval, ']0%, 100%[')
               
        assert any(optPar is not None for optPar in [minDist, maxDist, fitDistCuts, discrDistCuts]), \
               'At least 1 analysis parameters have to be optimised'
               
        if minDist is not None:
            minDist = Interval(minDist)
        if maxDist is not None:
            maxDist = Interval(maxDist)
        if fitDistCutsFctr is not None:
            fitDistCutsFctr = Interval(fitDistCutsFctr)
        if fitDistCuts is not None:
            fitDistCuts = Interval(fitDistCuts)
        if discrDistCutsFctr is not None:
            discrDistCutsFctr = Interval(discrDistCutsFctr)
        if discrDistCuts is not None:
            discrDistCuts = Interval(discrDistCuts)

        assert minDist is None or 0 <= minDist.min < minDist.max, \
               'Invalid left truncation distance {}'.format(minDist)
        assert maxDist is None or maxDist == 'auto' or 0 <= maxDist.min < maxDist.max, \
               'Invalid right truncation distance {}'.format(maxDist)
        assert minDist is None or maxDist is None or minDist.max < maxDist.min, \
               'Max left truncation distance {} must be lower than min right one {}' \
               .format(minDist.max, maxDist.min)
        
        assert fitDistCutsFctr is None or 0 <= fitDistCutsFctr.min < fitDistCutsFctr.max, \
               'Invalid mult. factor for number of fitting distance cuts {}'.format(fitDistCutsFctr)
        assert discrDistCutsFctr is None or 0 <= discrDistCutsFctr.min < discrDistCutsFctr.max, \
               'Invalid mult. factor number of distance discretisation {}'.format(discrDistCutsFctr)
        
        assert not(fitDistCutsFctr is not None and fitDistCuts is not None), \
               'Can\'t specify both absolute value and mult. factor for number of discretisation distance cuts'
        assert fitDistCuts is None or 0 <= fitDistCuts.min < fitDistCuts.max, \
               'Invalid number of fitting distance cuts {}'.format(fitDistCuts)
        assert not(discrDistCutsFctr is not None and discrDistCuts is not None), \
               'Can\'t specify both absolute value and mult. factor for number of discretisation distance cuts'
        assert discrDistCuts is None or 0 <= discrDistCuts.min < discrDistCuts.max, \
               'Invalid number of distance discretisation {}'.format(discrDistCuts)
        
        # Build name from main params if not specified
        if name is None:
            fields = ['mcds'] + [p[:3].lower() for p in [estimKeyFn, estimAdjustFn]]
            if estimCriterion != self.EngineClass.EstCriterionDef:
                fields.append(estimCriterion.lower())
            if cvInterval != self.EngineClass.EstCVIntervalDef:
                fields.append(str(cvInterval))
            name = '-'.join(fields)

        # Initialise base.
        super().__init__(engine, sampleDataSet, name=name, 
                         executor=executor, customData=customData, error=error,
                         expr2optimise=expr2optimise, minimiseExpr=minimiseExpr, **optimCoreOptions)
        
        # Save / compute params.
        # a. Analysis
        self.logData = logData
        self.estimKeyFn = estimKeyFn
        self.estimAdjustFn = estimAdjustFn
        self.estimCriterion = estimCriterion
        self.cvInterval = cvInterval

        # b. Optimisation
        self.minDist = minDist
        self.maxDist = maxDist
        if fitDistCutsFctr is not None or discrDistCutsFctr is not None:
            sqrtNbSights = np.sqrt(len(sampleDataSet.dfData[distanceField].dropna()))
        if fitDistCutsFctr is not None:
            self.fitDistCuts = Interval(min=round(fitDistCutsFctr.min*sqrtNbSights),
                                        max=round(fitDistCutsFctr.max*sqrtNbSights))
        else:
            self.fitDistCuts = fitDistCuts
        if discrDistCutsFctr is not None:
            self.discrDistCuts = Interval(min=round(discrDistCutsFctr.min*sqrtNbSights),
                                          max=round(discrDistCutsFctr.max*sqrtNbSights))
        else:
            self.discrDistCuts = discrDistCuts

        # Other optimisation stuff.
        fltSup = float('inf') # sys.float_info.max
        self.invalidFuncValue = fltSup if minimiseExpr else -fltSup

    # Post-process analysis results (adapted from MCDSAnalysisResultsSet.postComputeColumns)
    @staticmethod
    def _postProcessAnalysisResults(sResults):

        # Compute determined Chi2 test probability (last value of all the Chi2 tests done).
        chi2AllColInds = [col for col in MCDSAnalysisResultsSet.Chi2AllColInds if col in sResults.keys()]
        
        sResults[MCDSAnalysisResultsSet.Chi2ColInd] = \
            MCDSAnalysisResultsSet.determineChi2Value(sResults[chi2AllColInds])

        return sResults

    # Alias and name / index (in analysis results) of results values available for analysis value computation
    AnlysResultIndex = \
        dict(chi2=('detection probability', 'chi-square test probability determined', 'Value'),
             ks=('detection probability', 'Kolmogorov-Smirnov test probability', 'Value'),
             cvmuw=('detection probability', 'Cramér-von Mises (uniform weighting) test probability', 'Value'),
             cvmcw=('detection probability', 'Cramér-von Mises (cosine weighting) test probability', 'Value'))

    @classmethod
    def _getAnalysisResultIndex(cls, resultName):
        
        if resultName in cls.AnlysResultIndex:
            resInd = cls.AnlysResultIndex[resultName]
        else:
            raise NotImplementedError(f'Don\'t know where to find {resultName}')
        
        return resInd

    @classmethod
    def _getAnalysisResultValue(cls, resultExpr, sResults):
        
        dLocals = { alias: sResults[name] for alias, name in cls.AnlysResultIndex.items() \
                                          if name in sResults.index }
        
        try:
            value = eval(resultExpr, None, dLocals)
        except Exception as exc:
            value = self.invalidFuncValue
            logger.warning('Failed to evaluate {} : {}'.format(resultExpr, exc))
        
        logger.debug('_getAnalysisResultValue: {} = {} (locals={})'.format(resultExpr, value, dLocals))
        
        return value

    def _runOneAnalysis(self, minDist=MCDSEngine.DistMinDef, maxDist=MCDSEngine.DistMaxDef, 
                        fitDistCuts=MCDSEngine.DistFitCutsDef, discrDistCuts=MCDSEngine.DistDiscrCutsDef,
                        valueExpr='chi2'):
                              
        """Run one analysis (among many others in the optimisation process) and compute its values to optimise
           See MCDSAnalysis.__init__ for most parameters
           :param string valueExpr: Math. expression (python syntax) for computing analysis value
               (using result names from AnlysResultIndex) (ex: chi2, chi2*ks, ...)
        """

        # Run analysis (Submit, and wait for end of execution) : parallelism taken care elsewhere.
        dNameFlds = dict(l=minDist, r=maxDist, f=fitDistCuts, d=discrDistCuts)
        nameSufx = ''.join(c+str(int(v)) for c, v in dNameFlds.items() if v is not None)

        logger.debug(f'runAnalysis(minDist={minDist}, maxDist={maxDist},' \
                      'fitDistCuts={fitDistCuts}, discrDistCuts={discrDistCuts}) ...')
        anlys = MCDSAnalysis(engine=self.engine, sampleDataSet=self.sampleDataSet,
                             name=self.name + '-' + nameSufx, logData=self.logData,
                             estimKeyFn=self.estimKeyFn, estimAdjustFn=self.estimAdjustFn,
                             estimCriterion=self.estimCriterion, cvInterval=self.cvInterval,
                             minDist=minDist, maxDist=maxDist,
                             fitDistCuts=fitDistCuts, discrDistCuts=discrDistCuts)

        anlys.submit()
        
        sResults = anlys.getResults(postCleanup=True)

        # Post-process results, and compute analysis values (_the_ values to optimise).
        if anlys.success() or anlys.warnings():

            sResults = self._postProcessAnalysisResults(sResults)

            #logger.debug('Analysis results = {}'.format(sResults.to_dict()))

            value = self._getAnalysisResultValue(valueExpr, sResults) 
        
        else:

            value = self.invalidFuncValue

        #logger.debug(f'=> {valueExpr} = {value}')

        return value
    
    RunColumns = ['OptAbbrev', 'KeyFn', 'AdjSer', 'EstCrit', 'CVInt', 'OptCrit',
                  'MinDist', 'MaxDist', 'FitDistCuts', 'DiscrDistCuts']
    
    # DataFrame for translating 3-level multi-index columns to 1 level lang-translated columns
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
        lodOptimResults = self.future.result()
        
        # Build header columns for all the optimisation results (same for all):
        # actual (not default) analysis and optimisation params.
        sHead = pd.Series(data=[self.name, self.estimKeyFn, self.estimAdjustFn,
                                self.estimCriterion, self.cvInterval,
                                '{}({})'.format('min' if self.minimiseExpr else 'max', self.expr2optimise),
                                str(self.minDist), str(self.maxDist),
                                str(self.fitDistCuts), str(self.discrDistCuts)],
                          index=self.RunColumns)
        
        # Build final table of optimisation results : header then results, for each optimisation.
        return pd.DataFrame(data=[sHead.append(pd.Series(optRes)) for optRes in lodOptimResults])


class MCDSZeroOrderTruncationOptimisation(MCDSTruncationOptimisation):
    
    """Zero-order optimisation (no derivation used) for MCDS analyses distance truncations
    """
    
    EngineClass = MCDSEngine
        
    # Function for parsing optimisation specs (see optimiser.DSOptimiser._parseOptimCoreUserSpecs)
    def zoopt(a='racos', mxi=0, tv=None):
    
        dParms = dict(core='zoopt')
        
        if a != 'racos':
            dParms.update(algorithm=a)
        if mxi != 0:
            dParms.update(maxIters=mxi)
        if tv is not None:
            dParms.update(termValue=tv)
            
        return dParms

    CoreName = 'zoopt'
    CoreParamNames = ['algorithm', 'maxIters', 'termValue']
    CoreUserSpecParser = zoopt
    
    Parameter = ntuple('Parameter', ['name', 'interval', 'continuous', 'ordered'],
                                    defaults=['unknown', Interval(), True, True])
    
    def __init__(self, engine, sampleDataSet, name=None, 
                 distanceField='Distance', customData=None, executor=None, logData=False, error=None,
                 estimKeyFn=EngineClass.EstKeyFnDef, estimAdjustFn=EngineClass.EstAdjustFnDef, 
                 estimCriterion=EngineClass.EstCriterionDef, cvInterval=EngineClass.EstCVIntervalDef,
                 minDist=None, maxDist=None, fitDistCutsFctr=None, discrDistCutsFctr=None,
                 fitDistCuts=None, discrDistCuts=None,
                 expr2optimise='chi2', minimiseExpr=False, 
                 algorithm='racos', maxIters=0, termValue=None): # CoreParamNames !

        """Ctor
        Other parameters: See base class
        
        ZOOpt specific parameters:
        :param algorithm: Zero Order optimisation algorithm to use (among ['racos', 'poss'])
        :param maxIters: Number of iterations that stop optimisation algorithm when reached ; default: 0 => no limit
        :param termValue: Value that stops optimisation algorithm when reached ; default: None => no such termination
        """
        
        # Initialise base.
        super().__init__(engine, sampleDataSet, name=name,
                         distanceField=distanceField, customData=customData,
                         executor=executor, logData=logData, error=error,
                         estimKeyFn=estimKeyFn, estimAdjustFn=estimAdjustFn,
                         estimCriterion=estimCriterion, cvInterval=cvInterval,
                         minDist=minDist, maxDist=maxDist,
                         fitDistCutsFctr=fitDistCutsFctr, discrDistCutsFctr=discrDistCutsFctr,
                         fitDistCuts=fitDistCuts, discrDistCuts=discrDistCuts,
                         expr2optimise=expr2optimise, minimiseExpr=minimiseExpr,
                         algorithm=algorithm, maxIters=maxIters, termValue=termValue)

        # Prepare optimisation parameters.
        self.parameters = odict()
        if self.minDist is not None:
            self.parameters.update(minDist=self.Parameter(name='MinDist', interval=self.minDist,
                                                          continuous=True, ordered=True))
        if self.maxDist is not None:
            self.parameters.update(maxDist=self.Parameter(name='MaxDist', interval=self.maxDist,
                                                          continuous=True, ordered=True))
        if self.fitDistCuts is not None:
            self.parameters.update(fitDistCuts=self.Parameter(name='FitDistCuts', interval=self.fitDistCuts,
                                                              continuous=False, ordered=True))
        if self.discrDistCuts is not None:
            self.parameters.update(discrDistCuts=self.Parameter(name='DiscrDistCuts', interval=self.discrDistCuts,
                                                                continuous=False, ordered=True))
        
        logger.debug('ZOOptimisation({})'.format(dict(self.parameters)))
        
        # Columns names for each optimisation result row (see _run).
        self.resultsCols = ['SetupStatus', 'SubmitStatus', self.expr2optimise] + list(self.parameters.keys())
        
        # zoopt optimiser initialisation.
        self.zooptDims = \
            zoopt.Dimension(size=len(self.parameters),
                            regs=[[param.interval.min, param.interval.max] for param in self.parameters.values()],
                            tys=[param.continuous for param in self.parameters.values()],
                            order=[param.ordered for param in self.parameters.values()])

        self.zooptObjtv = zoopt.Objective(func=self._function, dim=self.zooptDims)

        self.zooptParams = zoopt.Parameter(algorithm=algorithm, budget=maxIters, terminal_value=termValue)
        
    def _function(self, solution):
    
        """The function to minimise : called as many times as needed by zoopt kernel.
        :param solution: the zoop "possible Solution" object to try and check if good enough
        """

        # Retrieve input parameters (to optimise)
        params = dict(zip(self.parameters.keys(), solution.get_x()))
        
        # Run analysis and get value.
        anlysValue = self._runOneAnalysis(valueExpr=self.expr2optimise, **params)
        
        # Compute function value from analysis value.
        return self.functionValue(anlysValue)
    
    def _run(self, repeats=1, onlyBest=None, error=None, *args, **kwargs):
        
        """Really do the optimisation work (use the optimisation core for this).
        (this method is called by the executor thread/process that takes it from the submit queue)
        :return: List of "solutions", each as an odict with target analysis params in the ctor order
                 preceded by { expr2optimise: analysis value }
        """
        
        # When self.setupError or (submit) error, simply return an well-formed but empty results.
        if self.setupError or error:
            return [odict(zip(self.resultsCols, 
                              [self.setupError, error] + [None]*(len(self.resultsCols))))]
            
        # Run the requested optimisations and get solutions.
        solutions = [zoopt.Opt.min(self.zooptObjtv, self.zooptParams) for _ in range(repeats)]
        
        # Keep only best solutions if requested.
        if onlyBest is not None and len(solutions) >= onlyBest:
            solutions = sorted(solutions, key=lambda sol: sol.get_value(), reverse=self.minimiseExpr)[:onlyBest]
        
        # Extract target results.
        return [odict(zip(self.resultsCols,
                          [None, None, self.analysisValue(sol.get_value())] \
                          + sol.get_x())) for sol in solutions]


if __name__ == '__main__':

    raise NotImplementedError()

    from autods.data import SampleDataSet

    # Parse command line args.
    argser = argparse.ArgumentParser(description='Run a distance sampling analysis using a DS engine from Distance software')

    argser.add_argument('-g', '--debug', dest='debug', action='store_true', default=False, 
                        help='Generate input data files, but don\'t run analysis')
    argser.add_argument('-w', '--workdir', type=str, dest='workDir', default='.',
                        help='Folder where to store DS analyses subfolders and output files')
    argser.add_argument('-e', '--engine', type=str, dest='engineType', default='MCDS', choices=['MCDS'],
                        help='The Distance engine to use, among MCDS, ... and no other for the moment')
    argser.add_argument('-d', '--datafile', type=str, dest='dataFile',
                        help='tabular data file path-name (XLSX or CSV/tab format)' \
                             ' with at least region, surface, point, effort and distance columns')
    argser.add_argument('-k', '--keyfn', type=str, dest='keyFn', default='HNORMAL', choices=['UNIFORM', 'HNORMAL', 'HAZARD'],
                        help='Model key function')
    argser.add_argument('-a', '--adjustfn', type=str, dest='adjustFn', default='COSINE', choices=['COSINE', 'POLY', 'HERMITE'],
                        help='Model adjustment function')
    argser.add_argument('-c', '--criterion', type=str, dest='criterion', default='AIC', choices=['AIC', 'AICC', 'BIC', 'LR'],
                        help='Criterion to use for selecting number of adjustment terms of the model')
    argser.add_argument('-i', '--cvinter', type=int, dest='cvInterval', default=95,
                        help='Confidence value for estimated values interval (%%)')

    args = argser.parse_args()
    
    # Load data set.
    sampleDataSet = SampleDataSet(source=args.dataFile)

    # Create DS engine
    engine = MCDSEngine(workDir=args.workDir,
                        distanceUnit='Meter', areaUnit='Hectare',
                        surveyType='Point', distanceType='Radial')

    # Create and run analysis
    optimion = MCDSOptimisation(engine=engine, sampleDataSet=sampleDataSet, name=args.engineType,
                                estimKeyFn=args.keyFn, estimAdjustFn=args.adjustFn,
                                estimCriterion=args.criterion, cvInterval=args.cvInterval)

    dResults = optimion.submit(realRun=not args.debug)
    
    # Print results
    logger.debug('Results:')
    for k, v in dResults.items():
        logger.debug('* {}: {}'.format(k, v))

    sys.exit(analysis.runStatus)
