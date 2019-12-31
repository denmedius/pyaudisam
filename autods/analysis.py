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
import argparse

import numpy as np
import pandas as pd

#import engine as adse # Bad: double import of 'engine' averall.
from autods.engine import DSEngine, MCDSEngine # Good
#import autods.engine as adse # Also good.


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
    
    def __init__(self, engine, dataSet, name):
        
        self.engine = engine
        self.dataSet = dataSet
        self.name = name
        
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
    
    def __init__(self, engine, dataSet, name=None, logData=False,
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
        super().__init__(engine, dataSet, name)
        
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
    
    def run(self, realRun=True):
        
        self.runStatus, self.runTime, self.runDir = \
            self.engine.run(dataSet=self.dataSet, runPrefix=self.name, realRun=realRun, logData=self.logData,
                            estimKeyFn=self.estimKeyFn, estimAdjustFn=self.estimAdjustFn,
                            estimCriterion=self.estimCriterion, cvInterval=self.cvInterval,
                            minDist=self.minDist, maxDist=self.maxDist,
                            fitDistCuts=self.fitDistCuts, discrDistCuts=self.discrDistCuts)
        
        # Load and decode output stats.
        sResults = pd.Series(data=[self.estimKeyFn, self.estimAdjustFn, self.estimCriterion,
                                   self.cvInterval, self.minDist, self.maxDist, self.fitDistCuts, self.discrDistCuts,
                                   self.runStatus, self.runTime, self.runDir],
                             index=self.MIRunColumns)
        
        if self.success() or self.warnings():
            sResults = sResults.append(self.engine.decodeStats())
            
        print()
        
        # Return a result, event if not run or MCDS crashed or ...
        return sResults
    
    def wasRun(self):
        return self.engine.wasRun(self.runStatus)
    
    def success(self):
        return self.engine.success(self.runStatus)
    
    def warnings(self):
        return self.engine.warnings(self.runStatus)
    
    def errors(self):
        return self.engine.errors(self.runStatus)


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
    print('Results:\n' + '\n'.join('* {}: {}'.format(k, v) for k, v in dResults.items()))

    sys.exit(analysis.runStatus)
