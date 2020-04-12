# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Analysis : One layer above engines, to run DS analyses from an imput data set, and get computation results
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment


import sys
import pathlib as pl
import shutil
import argparse

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger('autods')

#import engine as adse # Bad: double import of 'engine' averall.
from autods.engine import DSEngine, MCDSEngine # Good
#import autods.engine as adse # Also good.
from autods.executor import Executor


# Analysis (abstract) : Gather input params, data set, results, debug and log files
class DSAnalysis(object):
    
    EngineClass = DSEngine
    
    # Run columns for output : root engine output (3-level multi-index)
    RunRunColumns = [('run output', 'run status', 'Value'),
                     ('run output', 'run time',   'Value'),
                     ('run output', 'run folder', 'Value')]
    RunFolderColumn = RunRunColumns[2]
    
    # DataFrame for translating 3-level multi-index columns to 1 level lang-translated columns
    DRunRunColumnTrans = dict(en=['ExCod', 'RunTime', 'RunFolder'],
                              fr=['CodEx', 'HeureExec', 'DossierExec'])
    
    # Ctor
    # * :param: engine : DS engine to use
    # * :param: dataSet : data.SampleDataSet instance to use
    # * :param: name : name (may be empty), used for prefixing run folders or so, only for user-friendliness
    # * :param: customData : any custom data to be transported with the analysis object
    #                        during run (left completely untouched)
    def __init__(self, engine, dataSet, name, customData=None):
        
        self.engine = engine
        self.dataSet = dataSet
        self.name = name
        self.customData = customData
        
class MCDSAnalysis(DSAnalysis):
    
    EngineClass = MCDSEngine
    
    @staticmethod
    def checkDistCuts(distCuts, minDist, maxDist):
        if distCuts is None:
            return # OK !
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
                assert distCuts[-1] < maxDist, 'Invalid last distance cut {}; should be empty < {}'.format(distCuts[-1], maxDist)
    
    def __init__(self, engine, dataSet, name=None, customData=None, logData=False,
                 estimKeyFn=EngineClass.EstKeyFnDef, estimAdjustFn=EngineClass.EstAdjustFnDef, 
                 estimCriterion=EngineClass.EstCriterionDef, cvInterval=EngineClass.EstCVIntervalDef,
                 minDist=EngineClass.DistMinDef, maxDist=EngineClass.DistMaxDef, 
                 fitDistCuts=EngineClass.DistFitCutsDef, discrDistCuts=EngineClass.DistDiscrCutsDef):
        
        """
            :param logData: if True, print input data in output log
            :param minDist: None or NaN or >=0 
            :param maxDist: None or NaN or > :param minDist:
            :param fitDistCuts: None or NaN or int = number of equal sub-intervals of [:param minDist:, :param maxDist:] 
                                or list of distance values inside [:param minDist:, :param maxDist:]
                                ... for _model_fitting_
            :param discrDistCuts: None or NaN or int = number of equal sub-intervals of [:param minDist:, :param maxDist:] 
                                or list of distance values inside [:param minDist:, :param maxDist:]
                                ... for _distance_values_discretisation_
        """
        
        # Check engine
        assert isinstance(engine, MCDSEngine), 'Engine must be an MCDSEngine'
        
        # Check analysis params
        assert len(estimKeyFn) >= 2 and estimKeyFn in [kf[:len(estimKeyFn)] for kf in engine.EstKeyFns], \
               'Invalid estimate key function {}: should be in {} or at least 2-char abreviations' \
               .format(estimKeyFn, engine.EstKeyFns)
        assert len(estimAdjustFn) >= 2 and estimAdjustFn in [kf[:len(estimAdjustFn)] for kf in engine.EstAdjustFns], \
               'Invalid estimate adjust function {}: should be in {} or at least 2-char abreviations' \
               .format(estimAdjustFn, engine.EstAdjustFns)
        assert estimCriterion in engine.EstCriterions, \
               'Invalid estimate criterion {}: should be in {}'.format(estimCriterion, engine.EstCriterions)
        assert cvInterval > 0 and cvInterval < 100, \
               'Invalid cvInterval {}% : should be in {}'.format(cvInterval, ']0%, 100%[')
        if isinstance(minDist, float) and np.isnan(minDist):
            minDist = None # enforce minDist NaN => None for later
        assert minDist is None or minDist >= 0, \
               'Invalid left truncation distance {}: should be None/NaN or >= 0'.format(minDist)
        if isinstance(maxDist, float) and np.isnan(maxDist):
            maxDist = None # enforce maxDist NaN => None for later
        assert maxDist is None or minDist is None or minDist <= maxDist, \
               'Invalid right truncation distance {}:' \
               ' should be None/NaN or >= left truncation distance if specified, or >= 0'.format(maxDist)
        if isinstance(fitDistCuts, float) and np.isnan(fitDistCuts):
            fitDistCuts = None # enforce fitDistCuts NaN => None for later
        self.checkDistCuts(fitDistCuts, minDist, maxDist)
        if isinstance(discrDistCuts, float) and np.isnan(discrDistCuts):
            discrDistCuts = None # enforce discrDistCuts NaN => None for later
        self.checkDistCuts(discrDistCuts, minDist, maxDist)
        
        # Build name from main params if not specified
        if name is None:
            name = '-'.join(['mcds'] + [p[:3].lower() for p in [estimKeyFn, estimAdjustFn]])
            if estimCriterion != self.EngineClass.EstCriterionDef:
                name += '-' + estimCriterion.lower()
            if cvInterval != self.EngineClass.EstCVIntervalDef:
                name += '-' + str(cvInterval)

        # Initialise base.
        super().__init__(engine, dataSet, name, customData)
        
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
    MIRunColumns = pd.MultiIndex.from_tuples([('parameters', 'estimator key function', 'Value'),
                                              ('parameters', 'estimator adjustment series', 'Value'),
                                              ('parameters', 'estimator selection criterion', 'Value'),
                                              ('parameters', 'CV interval', 'Value'),
                                              ('parameters', 'left truncation distance', 'Value'),
                                              ('parameters', 'right truncation distance', 'Value'),
                                              ('parameters', 'model fitting distance cut points', 'Value'),
                                              ('parameters', 'distance discretisation cut points', 'Value')] \
                                             + DSAnalysis.RunRunColumns)
    
    # DataFrame for translating 3-level multi-index columns to 1 level lang-translated columns
    DfRunColumnTrans = \
        pd.DataFrame(index=MIRunColumns,
                     data=dict(en=['Mod Key Fn', 'Mod Adj Ser', 'Mod Chc Crit', 'Conf Interv',
                                   'Left Trunc Dist', 'Right Trunc Dist', 'Fit Dist Cuts', 'Discr Dist Cuts'] \
                                  + DSAnalysis.DRunRunColumnTrans['en'],
                               fr=['Fn Clé Mod', 'Sér Ajust Mod', 'Crit Chx Mod', 'Interv Conf',
                                   'Dist Tronc Gche', 'Dist Tronc Drte', 'Tranch Dist Mod', 'Tranch Dist Discr'] \
                                  + DSAnalysis.DRunRunColumnTrans['fr']))
    
     # Start running the analysis, and return immediately (the associated cofu.Future object) :
    # this begins an async. run ; you'll need to call getResults to wait for the real end of execution.
    def run(self, realRun=True, postCleanup=False):
        
        # Ask the engine to start running the analysis
        self.future = \
            self.engine.run(sampleDataSet=self.dataSet, runPrefix=self.name,
                            realRun=realRun, logData=self.logData,
                            estimKeyFn=self.estimKeyFn, estimAdjustFn=self.estimAdjustFn,
                            estimCriterion=self.estimCriterion, cvInterval=self.cvInterval,
                            minDist=self.minDist, maxDist=self.maxDist,
                            fitDistCuts=self.fitDistCuts, discrDistCuts=self.discrDistCuts)
        
        return self.future
        
    # Wait for the real end of analysis execution, and return its results.
    # This terminates an async. run when returning.
    def _wait4Results(self):
        
        # Get analysis execution results, when the computation is finished (blocking)
        self.runStatus, self.runTime, self.runDir, self.sResults = self.future.result()
        
    # Wait for the real end of analysis execution, and return its results.
    # This terminates an async. run when returning.
    def getResults(self, postCleanup=False):
        
        # Get analysis execution results, when the computation is finished (blocking)
        self._wait4Results()
        
        # Append the analysis stats (if any usable) to the input parameters.
        sParams = pd.Series(data=[self.estimKeyFn, self.estimAdjustFn, self.estimCriterion,
                                  self.cvInterval, self.minDist, self.maxDist, self.fitDistCuts, self.discrDistCuts,
                                  self.runStatus, self.runTime, self.runDir],
                            index=self.MIRunColumns)
        
        if self.engine.success(self.runStatus) or self.engine.warnings(self.runStatus):
            self.sResults = sParams.append(self.sResults)
        else:
            self.sResults = sParams
            
        # Post cleanup if requested.
        if postCleanup:
            self.cleanup()
        
        # Return a result, even if not run or MCDS crashed or ...
        return self.sResults
        
    def cleanup(self):
    
        if 'runDir' in dir(self):
        
            runDir = pl.Path(self.runDir)
            if runDir.is_dir():
            
                # Take extra precautions before rm -fr :-) (at least 14 files inside after a report generation)
                if not runDir.is_symlink() and len(list(runDir.rglob('*'))) < 15:
                    logger.debug('Removing run folder "{}"'.format(runDir))
                    shutil.rmtree(self.runDir)
                else:
                    logger.warning('Refused to remove suspect analysis run folder "{}"'.format(runDir))
        
    def wasRun(self):
    
        self._wait4Results() # First, wait for end of actual run !
        
        return self.engine.wasRun(self.runStatus)
    
    def success(self):
   
        self._wait4Results() # First, wait for end of actual run !
        
        return self.engine.success(self.runStatus)
    
    def warnings(self):
    
        self._wait4Results() # First, wait for end of actual run !
        
        return self.engine.warnings(self.runStatus)
    
    def errors(self):
   
        self._wait4Results() # First, wait for end of actual run !
        
        return self.engine.errors(self.runStatus)


class MCDSPreAnalysis(MCDSAnalysis):
    
    EngineClass = MCDSAnalysis.EngineClass
        
    # * modelStrategy: iterable of dict(keyFn=, adjSr=, estCrit=, cvInt=)
    # * executor: Executor object to use for parallel execution of multiple pre-analyses instances
    #             Note: Up to the caller to shut it down when no more needed (not owned).
    def __init__(self, engine, dataSet, name=None, customData=None, logData=False, executor=None,
                 modelStrategy=[dict(keyFn='HNORMAL', adjSr='COSINE', estCrit='AIC', cvInt=95)],
                 minDist=EngineClass.DistMinDef, maxDist=EngineClass.DistMaxDef, 
                 fitDistCuts=EngineClass.DistFitCutsDef, discrDistCuts=EngineClass.DistDiscrCutsDef):

        assert len(modelStrategy) > 0, 'MCDSPreAnalysis: Empty modelStrategy !'
        
        super().__init__(engine, dataSet, name, customData, logData,
                         minDist=minDist, maxDist=maxDist, fitDistCuts=fitDistCuts, discrDistCuts=discrDistCuts)

        self.modelStrategy = modelStrategy
        self.executor = executor if executor is not None else Executor(parallel=False)
    
    def _run(self):

        # Run models as planned in modelStrategy if something goes wrong
        dAnlyses = dict()
        for model in self.modelStrategy:

            modAbbrev = model['keyFn'][:3].lower() + '-' + model['adjSr'][:3].lower()

            # Create and run analysis for the new model
            anlys = MCDSAnalysis(engine=self.engine, dataSet=self.dataSet,
                                 name=self.name + '-' + modAbbrev, logData=False,
                                 estimKeyFn=model['keyFn'], estimAdjustFn=model['adjSr'],
                                 estimCriterion=model['estCrit'], cvInterval=model['cvInt'])
            anlys.run()

            # Stop here if run was OK, and save successful analysis.
            if anlys.success(): # Note that this call is blocking ... waiting for anlys end.
                dAnlyses['success'] = anlys
                logger.info(anlys.name + ' => success.')
                break

            # Otherwise, save 1st Warning and 1st error or "no" result analysis,
            # and then go on (may be the next will be an OK or warning one)
            elif anlys.warnings():
                if 'warning' not in dAnlyses:
                    dAnlyses['warning'] = anlys
            elif 'error' not in dAnlyses:
                dAnlyses['error'] = anlys

        # Notify the best obtained result and retrieve analysis of.
        if 'success' not in dAnlyses:
            if 'warning' in dAnlyses:
                anlys = dAnlyses['warning']
                logger.info(anlys.name + ' => warnings.')
            else:
                anlys = dAnlyses['error']
                logger.info(anlys.name + ' => errors.')
        else:
            pass # anlys is dAnlyses['success'] already (see break above)..
            
        return anlys # Return best analysis.
    
    def run(self):

        # Submit analysis work and return a Future object to ask from and wait for its results.
        self.future = self.executor.submit(self._run)
        
        return self.future
    
    def getResults(self):
        
        # Get self result : the best analysis.
        anlys = self.future.result()
        
        # Get execution results of this best analysis.
        sResults = anlys.getResults()
        
        # Store best analysis other outputs ... as self ones
        self.runStatus, self.runTime, self.runDir = anlys.runStatus, anlys.runTime, anlys.runDir
        
        return sResults


if __name__ == '__main__':

    #import data as adsd
    from autods.data import SampleDataSet

    # Parse command line args.
    argser = argparse.ArgumentParser(description='Run a distance sampling analysis using a DS engine from Distance software')

    argser.add_argument('-g', '--debug', dest='debug', action='store_true', default=False, 
                        help='Folder where to store DS analyses subfolders and output files')
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
    dataSet = SampleDataSet(source=args.dataFile)

    # Create DS engine
    engine = MCDSEngine(workDir=args.workDir,
                        distanceUnit='Meter', areaUnit='Hectare',
                        surveyType='Point', distanceType='Radial')

    # Create and run analysis
    analysis = MCDSAnalysis(engine=engine, dataSet=dataSet, name=args.engineType,
                            estimKeyFn=args.keyFn, estimAdjustFn=args.adjustFn,
                            estimCriterion=args.criterion, cvInterval=args.cvInterval)

    dResults = analysis.run(realRun=not args.debug)
    
    # Print results
    logger.debug('Results:')
    for k, v in dResults.items():
        logger.debug('* {}: {}'.format(k, v))

    sys.exit(analysis.runStatus)
