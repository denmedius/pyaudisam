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

# Submodule "analysis": One layer above engines, to run DS analyses from an input data set, and get computation results

import pathlib as pl
import shutil

import numpy as np
import pandas as pd

import concurrent.futures as cofu

from . import log
from .engine import DSEngine, MCDSEngine
from .executor import Executor

logger = log.logger('ads.ans')


# Analysis (abstract) : Gather input params, data set, results, debug and log files
class DSAnalysis:
    
    EngineClass = DSEngine
    
    # Run columns for output : root engine output (3-level multi-index)
    CLRunStatus = ('run output', 'run status', 'Value')
    CLRunStartTime = ('run output', 'start time', 'Value')
    CLRunElapsedTime = ('run output', 'elapsed time', 'Value')
    CLRunFolder = ('run output', 'run folder', 'Value')

    RunRunColumns = [CLRunStatus, CLRunStartTime, CLRunElapsedTime, CLRunFolder]
    RunFolderColumn = next(iter(col for col in RunRunColumns if col[1] == 'run folder'))
    
    # DataFrame for translating 3-level multi-index columns to 1 level lang-translated columns
    DRunRunColumnTrans = dict(en=['ExCod', 'StartTime', 'ElapsedTime', 'RunFolder'],
                              fr=['CodEx', 'HeureExec', 'DuréeExec', 'DossierExec'])
    
    # Ctor
    # * :param: engine : DS engine to use
    # * :param: sampleDataSet : data.SampleDataSet instance to use
    # * :param: name : name (maybe empty), used for prefixing run folders or so, only for user-friendliness
    # * :param: customData : any custom data to be transported with the analysis object
    #                        during run (left completely untouched)
    def __init__(self, engine, sampleDataSet, name, customData=None):
        
        self.engine = engine
        self.sampleDataSet = sampleDataSet
        self.name = name
        self.customData = customData
        

class MCDSAnalysis(DSAnalysis):
    
    EngineClass = MCDSEngine
    
    @staticmethod
    def checkDistCuts(distCuts, minDist, maxDist):
        if distCuts is None:
            return  # OK !
        if isinstance(distCuts, int) or isinstance(distCuts, float):
            assert distCuts > 1, 'Invalid number of distance cuts {}; should be None or > 1'.format(distCuts)
        elif isinstance(distCuts, list):
            assert len(distCuts) > 0, 'Invalid distance cut list {}; should not be empty'.format(distCuts)
            prevCut = -1e10 if minDist is None else minDist
            for cut in distCuts:
                assert cut > prevCut, 'Invalid distance cut list {}; should be made of strictly increasing values' \
                                      ' in ]{}, {}['.format(minDist, maxDist, distCuts)
                prevCut = cut
            if maxDist is not None:
                assert distCuts[-1] < maxDist, \
                       'Invalid last distance cut {}; should be empty < {}'.format(distCuts[-1], maxDist)
    
    def __init__(self, engine, sampleDataSet, name=None, customData=None, logData=False,
                 estimKeyFn=EngineClass.EstKeyFnDef, estimAdjustFn=EngineClass.EstAdjustFnDef, 
                 estimCriterion=EngineClass.EstCriterionDef, cvInterval=EngineClass.EstCVIntervalDef,
                 minDist=EngineClass.DistMinDef, maxDist=EngineClass.DistMaxDef, 
                 fitDistCuts=EngineClass.DistFitCutsDef, discrDistCuts=EngineClass.DistDiscrCutsDef):
        
        """Ctor
        
        Parameters:
        :param engine: DS engine to use
        :param sampleDataSet: SampleDataSet instance to use
        :param name: used for prefixing run folders (sure to be automatically unique anyway),
            analysis names, and so on, only for user-friendliness and easier debugging ;
            default: None => auto-generated from optimisation parameters
        :param customData: custom data for the run to ship through
        :param estimKeyFn: fitting estimator key function (see Distance documentation)
        :param estimAdjustFn: fitting estimator adjustment series
        :param estimCriterion: criterion for judging goodness of fit
        :param cvInterval:  confidence value interval (%)
        :param logData: if True, print input data in output log
        :param minDist: left truncation distance ; None or NaN or >=0 
        :param maxDist: right truncation distance ; None or NaN or > :param minDist:
        :param fitDistCuts: number of distance intervals for _model_fitting_
            None or NaN or int = number of equal sub-intervals of [:param minDist:, :param maxDist:] 
            or list of distance values inside [:param minDist:, :param maxDist:]
        :param discrDistCuts: number of distance intervals for _distance_values_discretisation_
            None or NaN or int = number of equal sub-intervals of [:param minDist:, :param maxDist:] 
            or list of distance values inside [:param minDist:, :param maxDist:]
        """
        
        # Check engine
        assert isinstance(engine, MCDSEngine), 'Engine must be an MCDSEngine'
        
        # Check analysis params
        assert len(estimKeyFn) >= 2 and estimKeyFn in [kf[:len(estimKeyFn)] for kf in engine.EstKeyFns], \
               'Invalid estimate key function {}: should be in {} or at least 2-char abbreviations' \
               .format(estimKeyFn, engine.EstKeyFns)
        assert len(estimAdjustFn) >= 2 and estimAdjustFn in [kf[:len(estimAdjustFn)] for kf in engine.EstAdjustFns], \
               'Invalid estimate adjust function {}: should be in {} or at least 2-char abbreviations' \
               .format(estimAdjustFn, engine.EstAdjustFns)
        assert estimCriterion in engine.EstCriteria, \
               'Invalid estimate criterion {}: should be in {}'.format(estimCriterion, engine.EstCriteria)
        assert 0 < cvInterval < 100, \
               'Invalid cvInterval {}% : should be in {}'.format(cvInterval, ']0%, 100%[')
        if isinstance(minDist, float) and np.isnan(minDist):
            minDist = None  # enforce minDist NaN => None for later
        assert minDist is None or minDist >= 0, \
               'Invalid left truncation distance {}: should be None/NaN or >= 0'.format(minDist)
        if isinstance(maxDist, float) and np.isnan(maxDist):
            maxDist = None  # enforce maxDist NaN => None for later
        assert maxDist is None or minDist is None or minDist <= maxDist, \
               'Invalid right truncation distance {}:' \
               ' should be None/NaN or >= left truncation distance if specified, or >= 0'.format(maxDist)
        if isinstance(fitDistCuts, float) and np.isnan(fitDistCuts):
            fitDistCuts = None  # enforce fitDistCuts NaN => None for later
        self.checkDistCuts(fitDistCuts, minDist, maxDist)
        if isinstance(discrDistCuts, float) and np.isnan(discrDistCuts):
            discrDistCuts = None  # enforce discrDistCuts NaN => None for later
        self.checkDistCuts(discrDistCuts, minDist, maxDist)
        
        # Build name from main params if not specified
        if name is None:
            name = '-'.join(['mcds'] + [p[:3].lower() for p in [estimKeyFn, estimAdjustFn]])
            if estimCriterion != self.EngineClass.EstCriterionDef:
                name += '-' + estimCriterion.lower()
            if cvInterval != self.EngineClass.EstCVIntervalDef:
                name += '-' + str(cvInterval)

        # Initialise base.
        super().__init__(engine, sampleDataSet, name, customData)

        # Compute sample stats.
        self.sSampleStats = self.engine.computeSampleStats(self.sampleDataSet)

        # Analysis run time-out implemented here if engine doesn't know how to do it
        # (but then, MCDS exe are not killed, only abandoned in "space")
        self.timeOut = engine.timeOut if engine.runMethod == 'os.system' else None
        if self.timeOut is not None:
            logger.info2(f"Will take care of {self.timeOut}s time limit because engine can't do this")

        # Save params.
        self.logData = logData
        self.estimKeyFn = estimKeyFn
        self.estimAdjustFn = estimAdjustFn
        self.estimCriterion = estimCriterion
        self.cvInterval = cvInterval
        self.minDist = minDist
        self.maxDist = maxDist
        self.fitDistCuts = fitDistCuts
        self.discrDistCuts = discrDistCuts
    
    # Run columns for output : analysis params + root engine output (3-level multi-index)
    CLParEstKeyFn = ('parameters', 'estimator key function', 'Value')
    CLParEstAdjSer = ('parameters', 'estimator adjustment series', 'Value')
    CLParEstSelCrit = ('parameters', 'estimator selection criterion', 'Value')
    CLParEstCVInt = ('parameters', 'CV interval', 'Value')
    CLParTruncLeft = ('parameters', 'left truncation distance', 'Value')
    CLParTruncRight = ('parameters', 'right truncation distance', 'Value')
    CLParModFitDistCuts = ('parameters', 'model fitting distance cut points', 'Value')
    CLParModDiscrDistCuts = ('parameters', 'distance discretisation cut points', 'Value')

    RunParColumns = [CLParEstKeyFn, CLParEstAdjSer, CLParEstSelCrit, CLParEstCVInt,
                     CLParTruncLeft, CLParTruncRight, CLParModFitDistCuts, CLParModDiscrDistCuts]
    MIRunColumns = pd.MultiIndex.from_tuples(RunParColumns + DSAnalysis.RunRunColumns)
    
    # DataFrame for translating 3-level multi-index columns to 1 level lang-translated columns
    DfRunColumnTrans = \
        pd.DataFrame(index=MIRunColumns,
                     data=dict(en=['Mod Key Fn', 'Mod Adj Ser', 'Mod Chc Crit', 'Conf Interv',
                                   'Left Trunc Dist', 'Right Trunc Dist', 'Fit Dist Cuts', 'Discr Dist Cuts']
                                  + DSAnalysis.DRunRunColumnTrans['en'],
                               fr=['Fn Clé Mod', 'Sér Ajust Mod', 'Crit Chx Mod', 'Interv Conf',
                                   'Dist Tronc Gche', 'Dist Tronc Drte', 'Tranch Dist Mod', 'Tranch Dist Discr']
                                  + DSAnalysis.DRunRunColumnTrans['fr']))
    
    # Start running the analysis, and return immediately (the associated cofu.Future object) :
    # this starts an async. run ; you'll need to call getResults to wait for the real end of execution.
    def submit(self, realRun=True):
        
        # Ask the engine to start running the analysis
        self.future = \
            self.engine.submitAnalysis(sampleDataSet=self.sampleDataSet, runPrefix=self.name,
                                       realRun=realRun, logData=self.logData,
                                       estimKeyFn=self.estimKeyFn, estimAdjustFn=self.estimAdjustFn,
                                       estimCriterion=self.estimCriterion, cvInterval=self.cvInterval,
                                       minDist=self.minDist, maxDist=self.maxDist,
                                       fitDistCuts=self.fitDistCuts, discrDistCuts=self.discrDistCuts)
        
        return self.future
        
    # Wait for the real end of analysis execution, and return its results.
    # This indicates the end of an async. run when returning.
    def _wait4Results(self):
        
        # Get analysis execution results, when the computation is finished (blocking)
        try:
            if self.timeOut is not None:
                startTime = pd.Timestamp.now()  # In case of cofu.TimeoutError
            self.runStatus, self.startTime, self.elapsedTime, self.runDir, self.sResults = \
                self.future.result(timeout=self.timeOut)
        except cofu.TimeoutError:
            logger.error('MCDS Analysis run timed-out after {}s'.format(self.timeOut))
            self.runStatus, self.startTime, self.elapsedTime, self.runDir, self.sResults = \
                self.engine.RCTimedOut, startTime, self.timeOut, None, None
        
    # Wait for the real end of analysis execution, and return its results.
    # This indicates the end of an async. run when returning.
    def getResults(self, postCleanup=False):
        
        # Get analysis execution results, when the computation is finished (blocking)
        self._wait4Results()
        
        # Build up run data (input parameters and run status) for output.
        sRunData = pd.Series(data=[self.estimKeyFn, self.estimAdjustFn, self.estimCriterion,
                                   self.cvInterval, self.minDist, self.maxDist, self.fitDistCuts, self.discrDistCuts,
                                   self.runStatus, self.startTime, self.elapsedTime, self.runDir],
                             index=self.MIRunColumns)
        
        # Append the run results (if any usable) to the analysis sample stats and run data for output.
        sSampStatsRunData = pd.concat([self.sSampleStats, sRunData])
        if self.engine.success(self.runStatus) or self.engine.warnings(self.runStatus):
            self.sResults = pd.concat([sSampStatsRunData, self.sResults])
        else:
            self.sResults = sSampStatsRunData
            
        # Post cleanup if requested.
        if postCleanup:
            self.cleanup()
        
        # Done: Return a result, even if not run or MCDS crashed or ...
        return self.sResults
        
    def cleanup(self):
    
        if 'runDir' in dir(self) and self.runDir is not None:
        
            runDir = pl.Path(self.runDir)
            if runDir.is_dir():
            
                # Take extra precautions before rm -fr :-) (at least 14 files inside after a report generation)
                if not runDir.is_symlink() and len(list(runDir.rglob('*'))) < 15:
                    logger.info3('Removing run folder "{}"'.format(runDir.as_posix()))
                    shutil.rmtree(runDir)
                else:
                    logger.warning('Cowardly refused to remove suspect analysis run folder "{}"'.format(runDir))
        
    def wasRun(self):
    
        self._wait4Results()  # First, wait for end of actual run !
        
        return self.engine.wasRun(self.runStatus)
    
    def success(self):
   
        self._wait4Results()  # First, wait for end of actual run !
        
        return self.engine.success(self.runStatus)
    
    def warnings(self):
    
        self._wait4Results()  # First, wait for end of actual run !
        
        return self.engine.warnings(self.runStatus)
    
    def errors(self):
   
        self._wait4Results()  # First, wait for end of actual run !
        
        return self.engine.errors(self.runStatus)


class MCDSPreAnalysis(MCDSAnalysis):

    """Note: Was implemented this strange way at the beginning, but made simpler later,
    and thus making it be fruitfully replaceable by a simple MCDSAnalysis through a simple MCDSAnalyser.
    """
    
    EngineClass = MCDSAnalysis.EngineClass
        
    # * modelStrategy: iterable of dict(keyFn=, adjSr=, estCrit=, cvInt=)
    # * executor: Executor object to use for parallel execution of multiple pre-analyses instances
    #             Note: Up to the caller to shut it down when no more needed (not owned).
    def __init__(self, engine, sampleDataSet, name=None, customData=None, logData=False, executor=None,
                 modelStrategy=[dict(keyFn='HNORMAL', adjSr='COSINE', estCrit='AIC', cvInt=95)],
                 minDist=EngineClass.DistMinDef, maxDist=EngineClass.DistMaxDef, 
                 fitDistCuts=EngineClass.DistFitCutsDef, discrDistCuts=EngineClass.DistDiscrCutsDef):

        assert len(modelStrategy) > 0, 'MCDSPreAnalysis: Empty modelStrategy !'
        
        super().__init__(engine, sampleDataSet, name, customData, logData,
                         minDist=minDist, maxDist=maxDist, fitDistCuts=fitDistCuts, discrDistCuts=discrDistCuts)

        self.modelStrategy = modelStrategy
        self.executor = executor if executor is not None else Executor()
    
    MIAicValue = ('detection probability', 'AIC value', 'Value')

    @classmethod
    def isAnalysisBetter(cls, left, right):

        """Return True if left is better than right, otherwise False"""

        if left.success() or left.warnings():
            if right.success() or right.warnings():
                answ = left.getResults().get(cls.MIAicValue, 1e9) < right.getResults().get(cls.MIAicValue, 1e9)
            else:
                answ = True
        else:
            answ = False

        return answ

    def _run(self):

        # Run models as planned in modelStrategy for best results
        bestAnlys = None
        for model in self.modelStrategy:

            modAbbrev = model['keyFn'][:3].lower() + '-' + model['adjSr'][:3].lower()

            # Create and run analysis for the new model
            anlys = MCDSAnalysis(engine=self.engine, sampleDataSet=self.sampleDataSet,
                                 name=self.name + '-' + modAbbrev, logData=False,
                                 estimKeyFn=model['keyFn'], estimAdjustFn=model['adjSr'],
                                 estimCriterion=model['estCrit'], cvInterval=model['cvInt'])
            anlys.submit()

            # Save analysis if better or first + cleanup no more needed analysis.
            if bestAnlys is None:
                bestAnlys = anlys
            elif self.isAnalysisBetter(anlys, bestAnlys):
                bestAnlys.cleanup()
                bestAnlys = anlys
            else:
                anlys.cleanup()

        return bestAnlys
    
    def submit(self, realRun=True):

        # Submit analysis work and return a Future object to ask from and wait for its results.
        self.future = self.executor.submit(self._run)
        
        return self.future
    
    def getResults(self, postCleanup=False):
        
        # Get self result : the best analysis.
        anlys = self.future.result()
        
        # Get execution results of this best analysis.
        sResults = anlys.getResults()
        
        # Store best analysis other outputs ... as self ones
        self.runStatus, self.startTime, self.elapsedTime, self.runDir = \
            anlys.runStatus, anlys.startTime, anlys.elapsedTime, anlys.runDir
        
        return sResults
