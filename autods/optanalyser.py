# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Optanalyser: Run a bunch of DS analyses according to a user-friendly set of analysis specs
#              with possibly some undetermined analysis parameters in specs :
#              for these analyses, an auto-computation of these parameters will then be run first
#              through some optimisation engine specified in specs : zoopt only for now,
#              and for some kind of parameters : only distance truncations supported for now)
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment


import numpy as np
import pandas as pd

import autods.log as log

logger = log.logger('ads.onr')

from autods.engine import MCDSEngine
from autods.analyser import MCDSAnalyser
from autods.optimiser import MCDSTruncationOptimiser, MCDSZerothOrderTruncationOptimiser


class MCDSTruncationOptanalyser(MCDSAnalyser):

    """Run a bunch of MCDS analyses, with possibly automatic truncation parameter computation before"""

    # Name of the spec. column to hold the "is truncation stuff optimised" 0/1 flag.
    OptimTruncFlagCol = 'OptimTrunc'

    def __init__(self, dfMonoCatObs, dfTransects=None, effortConstVal=1, dSurveyArea=dict(), 
                       transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                       sampleSelCols=['Species', 'Pass', 'Adult', 'Duration'], 
                       sampleDecCols=['Effort', 'Distance'], sampleDistCol='Distance', anlysSpecCustCols=[],
                       abbrevCol='AnlysAbbrev', abbrevBuilder=None, anlysIndCol='AnlysNum', sampleIndCol='SampleNum',
                       distanceUnit='Meter', areaUnit='Hectare',
                       surveyType='Point', distanceType='Radial', clustering=False,
                       resultsHeadCols=dict(before=['AnlysNum', 'SampleNum'], after=['AnlysAbbrev'], 
                                            sample=['Species', 'Pass', 'Adult', 'Duration']),
                       workDir='.', runMethod='subprocess.run', runTimeOut=120, logData=False,
                       logAnlysProgressEvery=50, logOptimProgressEvery=5, autoClean=True,
                       defEstimKeyFn=MCDSEngine.EstKeyFnDef, defEstimAdjustFn=MCDSEngine.EstAdjustFnDef,
                       defEstimCriterion=MCDSEngine.EstCriterionDef, defCVInterval=MCDSEngine.EstCVIntervalDef,
                       defMinDist=MCDSEngine.DistMinDef, defMaxDist=MCDSEngine.DistMaxDef, 
                       defFitDistCuts=MCDSEngine.DistFitCutsDef, defDiscrDistCuts=MCDSEngine.DistDiscrCutsDef,
                       defExpr2Optimise='chi2', defMinimiseExpr=False,
                       defOutliersMethod='tucquant', defOutliersQuantCutPct=5,
                       defFitDistCutsFctr=dict(min=2/3, max=3/2),
                       defDiscrDistCutsFctr=dict(min=1/3, max=1),
                       defSubmitTimes=1, defSubmitOnlyBest=None, dDefSubmitOtherParams=dict(),
                       dDefOptimCoreParams=dict(core='zoopt', maxIters=100, termExprValue=None,
                                                algorithm='racos', maxRetries=0)):

        """Ctor
        
        Parameters (see base class for missing ones : only specific stuff here):
        :param anlysIndCol: Must not be None, needed for joining optimisation results.
        
        Parameters for auto-computing truncation:
        :param def*: default values for "auto" specs etc.

        Other parameters: See base class.
        """

        if MCDSTruncationOptanalyser.OptimTruncFlagCol not in anlysSpecCustCols:
            anlysSpecCustCols = anlysSpecCustCols + [MCDSTruncationOptanalyser.OptimTruncFlagCol]

        super().__init__(dfMonoCatObs, dfTransects=dfTransects,
                         effortConstVal=effortConstVal, dSurveyArea=dSurveyArea, 
                         transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                         sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols,
                         abbrevCol=abbrevCol, abbrevBuilder=abbrevBuilder,
                         anlysIndCol=anlysIndCol, sampleIndCol=sampleIndCol,
                         anlysSpecCustCols=anlysSpecCustCols,
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         resultsHeadCols=resultsHeadCols,
                         workDir=workDir, runMethod=runMethod, runTimeOut=runTimeOut,
                         logData=logData, logProgressEvery=logAnlysProgressEvery,
                         defEstimKeyFn=defEstimKeyFn, defEstimAdjustFn=defEstimAdjustFn,
                         defEstimCriterion=defEstimCriterion, defCVInterval=defCVInterval,
                         defMinDist=defMinDist, defMaxDist=defMaxDist, 
                         defFitDistCuts=defFitDistCuts, defDiscrDistCuts=defDiscrDistCuts)

        # Save Optimisation-specific args.
        self.sampleDistCol = sampleDistCol
        
        self.logOptimProgressEvery = logOptimProgressEvery
        self.autoClean = autoClean
        
        # Default values for optimisation parameters.
        self.defExpr2Optimise = defExpr2Optimise
        self.defMinimiseExpr = defMinimiseExpr
        self.dDefOptimCoreParams = dDefOptimCoreParams
        self.defSubmitTimes = defSubmitTimes
        self.defSubmitOnlyBest = defSubmitOnlyBest
        self.dDefSubmitOtherParams = dDefSubmitOtherParams
        self.defOutliersMethod = defOutliersMethod
        self.defOutliersQuantCutPct = defOutliersQuantCutPct
        self.defFitDistCutsFctr = defFitDistCutsFctr
        self.defDiscrDistCutsFctr = defDiscrDistCutsFctr

        # Specs.
        self.updateSpecs(**{name: getattr(self, name)
                            for name in ['defExpr2Optimise', 'defMinimiseExpr', 'dDefOptimCoreParams',
                                         'defSubmitTimes', 'defSubmitOnlyBest', 'dDefSubmitOtherParams',
                                         'defOutliersMethod', 'defOutliersQuantCutPct',
                                         'defFitDistCutsFctr', 'defDiscrDistCutsFctr']})

        # An optimiser instance only there for explicitParamSpecs() delegation.
        # Note: For the moment, only zoopt engine supported
        # TODO: Add support for other engines, thanks to the OptimCore columns spec (default = zoopt)
        #       provided the associated MCDSXXXTruncationOptimiser classes derive from MCDSTruncationOptimiser.
        self.zoptr4Specs = \
            MCDSZerothOrderTruncationOptimiser(
                self.dfMonoCatObs, dfTransects=self._mcDataSet.dfTransects,
                dSurveyArea=self._mcDataSet.dSurveyArea, transectPlaceCols=self._mcDataSet.transectPlaceCols,
                passIdCol=self._mcDataSet.passIdCol, effortCol=self._mcDataSet.effortCol,
                sampleSelCols=self.sampleSelCols, sampleDecCols=self._mcDataSet.sampleDecFields,
                sampleDistCol=self.sampleDistCol, abbrevCol=self.abbrevCol, abbrevBuilder=self.abbrevBuilder,
                anlysIndCol=self.anlysIndCol, sampleIndCol=self.sampleIndCol, anlysSpecCustCols=anlysSpecCustCols,
                distanceUnit=self.distanceUnit, areaUnit=self.areaUnit,
                surveyType=self.surveyType, distanceType=self.distanceType, clustering=self.clustering,
                resultsHeadCols=dict(),
                workDir=self.workDir, runMethod=self.runMethod, runTimeOut=self.runTimeOut, logData=self.logData,
                logProgressEvery=self.logOptimProgressEvery, autoClean=self.autoClean,
                defEstimKeyFn=self.defEstimKeyFn, defEstimAdjustFn=self.defEstimAdjustFn,
                defEstimCriterion=self.defEstimCriterion, defCVInterval=self.defCVInterval,
                defExpr2Optimise=self.defExpr2Optimise, defMinimiseExpr=self.defMinimiseExpr,
                defOutliersMethod=self.defOutliersMethod, defOutliersQuantCutPct=self.defOutliersQuantCutPct,
                defFitDistCutsFctr=self.defFitDistCutsFctr, defDiscrDistCutsFctr=self.defDiscrDistCutsFctr,
                defSubmitTimes=self.defSubmitTimes, defSubmitOnlyBest=self.defSubmitOnlyBest,
                defCoreMaxIters=self.dDefOptimCoreParams['maxIters'],
                defCoreTermExprValue=self.dDefOptimCoreParams['termExprValue'],
                defCoreAlgorithm=self.dDefOptimCoreParams['algorithm'],
                defCoreMaxRetries=self.dDefOptimCoreParams['maxRetries'])

        # Optimiser really used for optimisations, create in run() (debug use only).
        self.zoptr = None

    def explicitParamSpecs(self, implParamSpecs=None, dfExplParamSpecs=None, dropDupes=True, check=False):
    
        return self.zoptr4Specs.explicitParamSpecs(implParamSpecs=implParamSpecs, dfExplParamSpecs=dfExplParamSpecs,
                                                   dropDupes=dropDupes, check=check)

    def computeUndeterminedParamSpecs(self, dfExplParamSpecs=None, implParamSpecs=None, threads=None):
    
        """Run truncation optimisation for analyses with undetermined truncation param. specs
        and merge the computed specs to the ones of analyses with already determined truncation param. specs.
        
        Call explicitParamSpecs(..., check=True) before this to make sure user specs are OK

        Parameters:
        :param dfExplParamSpecs: Explicit MCDS analysis param specs, as a DataFrame
          (generated through explicitVariantSpecs, as an example),
        :param implParamSpecs: Implicit MCDS analysis param specs, suitable for explicitation
          through explicitVariantSpecs
        :param threads: Number of parallel threads to use (default None: no parallelism, no asynchronism)
           
        :return: the merged explicit param. specs for all the analyses (with optimised or not truncation param. specs)
        """
        
        # 1. Explicitate, complete and check analysis specs (for usability).
        # (should be also done before calling run, to avoid failure).
        dfExplParamSpecs, userParamSpecCols, intParamSpecCols, _, checkVerdict, checkErrors = \
            self.explicitParamSpecs(implParamSpecs, dfExplParamSpecs, dropDupes=True, check=True)
        assert checkVerdict, 'Error: Analysis & optimisation params check failed: {}'.format('; '.join(checkErrors))        
    
        # 2. Extract optimisation specs (params specified as optimisation specs).
        # a. Get spec. columns to examine for deciding if analyses specs imply prior optimisation or not.
        optimUserParamSpecCols = \
            self.zoptr4Specs.optimisationParamSpecUserNames(userParamSpecCols, intParamSpecCols)
        
        # b. Search for (possibly) optimisation. specs with string data (const params are numbers or lists)
        def analysisNeedsOptimisationFirst(sAnlysSpec):
            return any(isinstance(v, str) for v in sAnlysSpec.values)
        dfExplOptimParamSpecs = \
            dfExplParamSpecs[dfExplParamSpecs[optimUserParamSpecCols] \
                                .apply(analysisNeedsOptimisationFirst, axis='columns')]
         
        # 3. Run optimisations if needed and replace computed truncation params in analysis specs
        logger.info('Found {}/{} analysis specs implying some prior optimisation'
                    .format(len(dfExplOptimParamSpecs), len(dfExplParamSpecs)))
        if not dfExplOptimParamSpecs.empty:
        
            # Note: For the moment, only zoopt engine supported
            # TODO: Add support for other engines, thanks to the OptimCore columns spec (default = zoopt)
            #       provided the associated MCDSXXXTruncationOptimiser classes derive from MCDSTruncationOptimiser.
            # a. Create optimiser object
            self.zoptr = MCDSZerothOrderTruncationOptimiser(
                        self.dfMonoCatObs, dfTransects=self._mcDataSet.dfTransects,
                        dSurveyArea=self._mcDataSet.dSurveyArea, transectPlaceCols=self._mcDataSet.transectPlaceCols,
                        passIdCol=self._mcDataSet.passIdCol, effortCol=self._mcDataSet.effortCol,
                        sampleSelCols=self.sampleSelCols, sampleDecCols=self._mcDataSet.sampleDecFields,
                        sampleDistCol=self.sampleDistCol, abbrevCol=self.abbrevCol, abbrevBuilder=self.abbrevBuilder,
                        anlysIndCol=self.anlysIndCol, sampleIndCol=self.sampleIndCol, 
                        anlysSpecCustCols=[self.OptimTruncFlagCol],
                        distanceUnit=self.distanceUnit, areaUnit=self.areaUnit,
                        surveyType=self.surveyType, distanceType=self.distanceType,
                        clustering=self.clustering,
                        resultsHeadCols=dict(before=[self.anlysIndCol, self.sampleIndCol],
                                             sample=self.sampleSelCols,
                                             after=userParamSpecCols),
                        workDir=self.workDir, runMethod=self.runMethod, runTimeOut=self.runTimeOut,
                        logData=self.logData, logProgressEvery=self.logOptimProgressEvery, autoClean=self.autoClean,
                        defEstimKeyFn=self.defEstimKeyFn, defEstimAdjustFn=self.defEstimAdjustFn,
                        defEstimCriterion=self.defEstimCriterion, defCVInterval=self.defCVInterval,
                        defExpr2Optimise=self.defExpr2Optimise, defMinimiseExpr=self.defMinimiseExpr,
                        defOutliersMethod=self.defOutliersMethod, defOutliersQuantCutPct=self.defOutliersQuantCutPct,
                        defFitDistCutsFctr=self.defFitDistCutsFctr, defDiscrDistCutsFctr=self.defDiscrDistCutsFctr,
                        defSubmitTimes=self.defSubmitTimes, defSubmitOnlyBest=self.defSubmitOnlyBest,
                        defCoreMaxIters=self.dDefOptimCoreParams['maxIters'],
                        defCoreTermExprValue=self.dDefOptimCoreParams['termExprValue'],
                        defCoreAlgorithm=self.dDefOptimCoreParams['algorithm'],
                        defCoreMaxRetries=self.dDefOptimCoreParams['maxRetries'])

            # b. Run optimisations
            optimResults = self.zoptr.run(dfExplOptimParamSpecs, threads=threads)
            self.zoptr.shutdown()
            
            # c. Merge optimisation results into param. specs.
            # * Extract computed (= optimised) analysis params.
            dfOptimRes = \
                optimResults.dfSubData(subset=[self.anlysIndCol] + optimResults.optimisationTargetColumns())
            dfOptimRes.set_index(self.anlysIndCol, inplace=True)
            dfOptimRes.sort_index(inplace=True)
            
            # * Replicate optimisation specs as much as there are associated results
            #   (optimisation may keep more than 1 "best" result row)
            dfExplCompdParamSpecs = dfExplOptimParamSpecs.copy()
            dfExplCompdParamSpecs.set_index(self.anlysIndCol, inplace=True)
            dfExplCompdParamSpecs.sort_index(inplace=True)
            dfExplCompdParamSpecs = \
                dfExplCompdParamSpecs.loc[np.repeat(dfExplCompdParamSpecs.index, #.to_numpy(),
                                                    np.unique(dfOptimRes.index, return_counts=True)[1])]

            # * Replace optim. specs by optim. results (optim. target columns = truncation param. ones)
            optimTgtUserSpecCols = self.zoptr.optimisationTargetColumnUserNames()
            dfOptimRes.rename(columns=dict(zip(optimResults.optimisationTargetColumns(),
                                               optimTgtUserSpecCols)), inplace=True)
            bdfToBeKeptSpecCells = ~dfExplCompdParamSpecs[optimTgtUserSpecCols].applymap(lambda v: isinstance(v, str))
            dfExplCompdParamSpecs[optimTgtUserSpecCols] = \
                dfExplCompdParamSpecs[optimTgtUserSpecCols].where(bdfToBeKeptSpecCells,
                                                                  other=dfOptimRes[optimTgtUserSpecCols])

            # * Concat const specs to computed ones.
            dfExplParamSpecs.set_index(self.anlysIndCol, inplace=True)

            dfExplConstParamSpecs = \
                dfExplParamSpecs.loc[~dfExplParamSpecs.index.isin(dfExplOptimParamSpecs[self.anlysIndCol])].copy()
            dfExplConstParamSpecs.reset_index(inplace=True)
            dfExplConstParamSpecs[self.OptimTruncFlagCol] = 0  # Const = "unoptimised" truncation params.

            dfExplCompdParamSpecs.reset_index(inplace=True)
            dfExplCompdParamSpecs[self.OptimTruncFlagCol] = 1  # Computed = "optimised" truncation params.

            dfExplParamSpecs = dfExplConstParamSpecs.append(dfExplCompdParamSpecs, ignore_index=True)
            dfExplParamSpecs.sort_values(by=self.anlysIndCol, inplace=True)

        else:

            # Add non optimise(d) truncations analysis flag.
            dfExplParamSpecs[self.OptimTruncFlagCol] = 0  # Const = "unoptimised" truncation params.

        # Done.
        return dfExplParamSpecs

    def run(self, dfExplParamSpecs=None, implParamSpecs=None, threads=None):
    
        """Run specified analyses, after automatic computing of truncation parameter if needed
        
        Call explicitParamSpecs(..., check=True) before this to make sure user specs are OK

        Parameters:
        :param dfExplParamSpecs: Explicit MCDS analysis param specs, as a DataFrame
          (generated through explicitVariantSpecs, as an example),
        :param implParamSpecs: Implicit MCDS analysis param specs, suitable for explicitation
          through explicitVariantSpecs
        :param threads: Number of parallel threads to use (default None: no parallelism, no asynchronism)
           
        :return: the MCDSAnalysisResultsSet holding the analyses results
        """
        
        # 1. Run optimisations when needed and replace computed truncation params in analysis specs
        #    (warning: as some optimisation specs may specify to keep more than 1 "best" result,
        #              dfExplParamSpecs may grow accordingly, as well as the final number of result rows).
        dfExplParamSpecs = \
            self.computeUndeterminedParamSpecs(dfExplParamSpecs, implParamSpecs, threads=threads)

        # 2. Run all analyses, now all parameters are there.
        return super().run(dfExplParamSpecs, threads=threads)

    def shutdown(self):
    
        self.zoptr4Specs.shutdown()

        super().shutdown()


if __name__ == '__main__':

    import sys

    sys.exit(0)
