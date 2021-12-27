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

# Submodule "optanalyser": Run a bunch of DS analyses according to a user-friendly set of analysis specs
#  with possibly some undetermined analysis parameters in specs :
#  for these analyses, an auto-computation of these parameters will then be run first
#  through some optimisation engine specified in specs (only "zoopt" supported as for now),
#  and for some kind of parameters (only distance truncations supported as for now).

import numpy as np
import pandas as pd

from . import log
from .engine import MCDSEngine
from .analysis import MCDSAnalysis
from .analyser import MCDSAnalyser, MCDSAnalysisResultsSet
from .optimiser import MCDSZerothOrderTruncationOptimiser

logger = log.logger('ads.onr')


class MCDSTruncOptanalysisResultsSet(MCDSAnalysisResultsSet):

    """A specialized results set for MCDS optanalysers"""

    Super = MCDSAnalysisResultsSet

    # Name of the spec. column to hold the "is truncation stuff optimised" 0/1 flag.
    OptimTruncFlagCol = 'OptimTrunc'
    CLOptimTruncFlag = ('header (tail)', OptimTruncFlagCol, 'Value')
    
    def __init__(self, miCustomCols=None, dfCustomColTrans=None, miSampleCols=None, sampleIndCol=None,
                 sortCols=[], sortAscend=[], distanceUnit='Meter', areaUnit='Hectare',
                 surveyType='Point', distanceType='Radial', clustering=False,
                 ldTruncIntrvSpecs=[dict(col='left', minDist=5.0, maxLen=5.0),
                                    dict(col='right', minDist=25.0, maxLen=25.0)],
                 truncIntrvEpsilon=1e-6, ldFilSorKeySchemes=None):
        
        """Ctor

        Parameters:
        :param miSampleCols: columns to use for grouping by sample ; defaults to miCustomCols if None
        :param sampleIndCol: multi-column index for the sample Id column ; no default, must be there !
        :param ldFilSorKeySchemes: Replacement for predefined filter-sort key schemes
                                   None => use predefined ones AutoFilSorKeySchemes.
        """

        super().__init__(miCustomCols=miCustomCols, dfCustomColTrans=dfCustomColTrans,
                         miSampleCols=miSampleCols, sampleIndCol=sampleIndCol,
                         sortCols=sortCols, sortAscend=sortAscend, distanceUnit=distanceUnit, areaUnit=areaUnit,
                         surveyType=surveyType, distanceType=distanceType, clustering=clustering,
                         ldTruncIntrvSpecs=ldTruncIntrvSpecs, truncIntrvEpsilon=truncIntrvEpsilon,
                         ldFilSorKeySchemes=ldFilSorKeySchemes)

        assert self.CLOptimTruncFlag in self.miCustomCols

    def copy(self, withData=True):
    
        """Clone function, with optional data copy"""
    
        # Create new instance with same ctor params.
        clone = MCDSTruncOptanalysisResultsSet(miCustomCols=self.miCustomCols, dfCustomColTrans=self.dfCustomColTrans,
                                               miSampleCols=self.miSampleCols, sampleIndCol=self.sampleIndCol,
                                               sortCols=self.sortCols, sortAscend=self.sortAscend,
                                               distanceUnit=self.distanceUnit, areaUnit=self.areaUnit,
                                               surveyType=self.surveyType, distanceType=self.distanceType,
                                               clustering=self.clustering,
                                               ldTruncIntrvSpecs=self.ldTruncIntrvSpecs,
                                               truncIntrvEpsilon=self.truncIntrvEpsilon)

        # Copy data if needed.
        if withData:
            clone._dfData = self._dfData.copy()
            clone.rightColOrder = self.rightColOrder
            clone.postComputed = self.postComputed

        return clone
    
    # Post computations : Truncations groups.
    @classmethod
    def _sampleDistTruncGroups(cls, dfSampRes, ldIntrvSpecs, intrvEpsilon=1e-6):

        """Compute distance truncation groups for 1 sample, for each target distance truncation column"""

        # For each optimised or not case,
        dTruncGroups = dict()  # key=ldIntrvSpecs[*]['col'], value=list(Series of group nums)
        for isOpt in sorted(dfSampRes[cls.CLOptimTruncFlag].unique()):

            logger.debug3('* {}optimised'.format('' if isOpt else 'un').title())

            # Compute truncation groups for this case and sample
            dOptTruncGroups = \
                super()._sampleDistTruncGroups(dfSampRes=dfSampRes[dfSampRes[cls.CLOptimTruncFlag] == isOpt],
                                               ldIntrvSpecs=ldIntrvSpecs, intrvEpsilon=intrvEpsilon)

            # Store them for later concatenation
            for colAlias, sGrpNums in dOptTruncGroups.items():
                if colAlias not in dTruncGroups:
                    dTruncGroups[colAlias] = list()
                dTruncGroups[colAlias].append(sGrpNums)

        # Concat series of computed group nums (opt or not) for each target distance column to group
        return {colAlias: pd.concat(lsGrpNums) for colAlias, lsGrpNums in dTruncGroups.items()}

    # Post computations : Schemes for computing filtering and sorting keys (see inherited _postComputeFilterSortKeys).
    AutoFilSorKeySchemes = \
        [  # Orders inside groups with identical truncation params.
         dict(key=Super.CLGrpOrdSmTrAic,  # Best AIC, for same left & right truncations (but variable nb of cut points)
              sort=[MCDSAnalysis.CLParTruncLeft, MCDSAnalysis.CLParTruncRight,
                    Super.CLDeltaAic, Super.CLChi2, Super.CLKS, Super.CLDCv, Super.CLNObs, MCDSAnalysis.CLRunStatus],
              ascend=[True, True, True, False, False, True, False, True],
              group=[MCDSAnalysis.CLParTruncLeft, MCDSAnalysis.CLParTruncRight, MCDSAnalysis.CLParModFitDistCuts]),
      
         # Orders inside groups of close truncation params.
         # dict(key=Super.CLGrpOrdClTrChi2,  # Best Chi2 inside groups of close truncation params
         #      sort=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight,
         #            Super.CLChi2],
         #      ascend=[True, True, True, False],
         #      group=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight]),
         # dict(key=Super.CLGrpOrdClTrKS,  # Best KS inside groups of close truncation params
         #      sort=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight,
         #            Super.CLKS],
         #      ascend=[True, True, True, False],
         #      group=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight]),
         dict(key=Super.CLGrpOrdClTrDCv,  # Best DCv inside groups of close truncation params
              sort=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight,
                    Super.CLDCv],
              ascend=[True, True, True, True],
              group=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight]),
         dict(key=Super.CLGrpOrdClTrChi2KSDCv,  # Best Chi2 & KS & DCv inside groups of close truncation params
              sort=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight,
                    Super.CLChi2, Super.CLKS, Super.CLDCv, Super.CLNObs, MCDSAnalysis.CLRunStatus],
              ascend=[True, True, True, False, False, True, False, True],
              group=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight]),

         dict(key=Super.CLGrpOrdClTrQuaBal1,  # Best Combined Quality 1 inside groups of close truncation params
              sort=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight,
                    Super.CLCmbQuaBal1],
              ascend=[True, True, True, False],
              group=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight]),
         dict(key=Super.CLGrpOrdClTrQuaBal2,  # Best Combined Quality 2 inside groups of close truncation params
              sort=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight,
                    Super.CLCmbQuaBal2],
              ascend=[True, True, True, False],
              group=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight]),
         dict(key=Super.CLGrpOrdClTrQuaBal3,  # Best Combined Quality 3 inside groups of close truncation params
              sort=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight,
                    Super.CLCmbQuaBal3],
              ascend=[True, True, True, False],
              group=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight]),
         dict(key=Super.CLGrpOrdClTrQuaChi2,  # Best Combined Quality Chi2+ inside groups of close truncation params
              sort=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight,
                    Super.CLCmbQuaChi2],
              ascend=[True, True, True, False],
              group=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight]),
         dict(key=Super.CLGrpOrdClTrQuaKS,  # Best Combined Quality KS+ inside groups of close truncation params
              sort=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight,
                    Super.CLCmbQuaKS],
              ascend=[True, True, True, False],
              group=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight]),
         dict(key=Super.CLGrpOrdClTrQuaDCv,  # Best Combined Quality DCv+ inside groups of close truncation params
              sort=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight,
                    Super.CLCmbQuaDCv],
              ascend=[True, True, True, False],
              group=[CLOptimTruncFlag, Super.CLGroupTruncLeft, Super.CLGroupTruncRight]),

         # Global orders (no grouping by close or identical truncations)
         dict(key=Super.CLGblOrdChi2KSDCv,
              sort=[Super.CLChi2, Super.CLKS, Super.CLDCv, Super.CLNObs, MCDSAnalysis.CLRunStatus],
              ascend=[False, False, True, False, True]),
         dict(key=Super.CLGblOrdQuaBal1,
              sort=[Super.CLCmbQuaBal1], ascend=False),
         dict(key=Super.CLGblOrdQuaBal2,
              sort=[Super.CLCmbQuaBal2], ascend=False),
         dict(key=Super.CLGblOrdQuaBal3,
              sort=[Super.CLCmbQuaBal3], ascend=False),
         dict(key=Super.CLGblOrdQuaChi2,
              sort=[Super.CLCmbQuaChi2], ascend=False),
         dict(key=Super.CLGblOrdQuaKS,
              sort=[Super.CLCmbQuaKS], ascend=False),
         dict(key=Super.CLGblOrdQuaDCv,
              sort=[Super.CLCmbQuaDCv], ascend=False),

         dict(key=Super.CLGblOrdDAicChi2KSDCv,
              sort=[MCDSAnalysis.CLParTruncLeft, MCDSAnalysis.CLParTruncRight, MCDSAnalysis.CLParModFitDistCuts,
                    Super.CLDeltaAic, Super.CLChi2, Super.CLKS, Super.CLDCv, Super.CLNObs, MCDSAnalysis.CLRunStatus],
              ascend=[True, True, True,
                      True, False, False, True, False, True], napos='first')]

    # Enforce uniqueness of keys in filter and sort key specs.
    assert len(AutoFilSorKeySchemes) == len(set(scheme['key'] for scheme in AutoFilSorKeySchemes)), \
           'Duplicated scheme key in MCDSTruncOptanalysisResultsSet.AutoFilSorKeySchemes'

    # Enforce uniqueness of sort and group column in filter and sort key specs.
    assert all(len(scheme['sort']) == len(set(scheme['sort'])) for scheme in AutoFilSorKeySchemes), \
           'Duplicated sort column spec in some scheme of MCDSTruncOptanalysisResultsSet.AutoFilSorKeySchemes'
    assert all(len(scheme.get('group', [])) == len(set(scheme.get('group', []))) for scheme in AutoFilSorKeySchemes), \
           'Duplicated group column spec in some scheme of MCDSTruncOptanalysisResultsSet.AutoFilSorKeySchemes'

    # Check sort vs ascend list lengths in filter and sort key specs.
    assert all(isinstance(scheme['ascend'], bool) or len(scheme['ascend']) == len(scheme['sort'])
               for scheme in AutoFilSorKeySchemes), \
           'Inconsistent ascend vs sort specs in some scheme of MCDSTruncOptanalysisResultsSet.AutoFilSorKeySchemes'


class MCDSTruncationOptanalyser(MCDSAnalyser):

    """Run a bunch of MCDS analyses, with possibly automatic truncation parameter computation before"""

    # Name of the spec. column to hold the "is truncation stuff optimised" 0/1 flag.
    OptimTruncFlagCol = MCDSTruncOptanalysisResultsSet.OptimTruncFlagCol

    def __init__(self, dfMonoCatObs, dfTransects=None, effortConstVal=1, dSurveyArea=dict(), 
                 transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                 sampleSelCols=['Species', 'Pass', 'Adult', 'Duration'], 
                 sampleDecCols=['Effort', 'Distance'], sampleDistCol='Distance', anlysSpecCustCols=[],
                 abbrevCol='AnlysAbbrev', abbrevBuilder=None, anlysIndCol='AnlysNum', sampleIndCol='SampleNum',
                 distanceUnit='Meter', areaUnit='Hectare',
                 surveyType='Point', distanceType='Radial', clustering=False,
                 resultsHeadCols=dict(before=['AnlysNum', 'SampleNum'], after=['AnlysAbbrev'], 
                                      sample=['Species', 'Pass', 'Adult', 'Duration']),
                 ldTruncIntrvSpecs=[dict(col='left', minDist=5.0, maxLen=5.0),
                                    dict(col='right', minDist=25.0, maxLen=25.0)], truncIntrvEpsilon=1e-6,
                 workDir='.', runMethod='subprocess.run', runTimeOut=300, logData=False,
                 logAnlysProgressEvery=50, logOptimProgressEvery=5, backupOptimEvery=50, autoClean=True,
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

        assert anlysIndCol, 'anlysIndCol must not be None, needed for analysis identification'

        # Add OptimTruncFlag column to anlysSpecCustCols if not there
        if self.OptimTruncFlagCol not in anlysSpecCustCols:
            anlysSpecCustCols = anlysSpecCustCols + [self.OptimTruncFlagCol]

        # Add OptimTruncFlag column to resultsHeadCols['after']
        resultsHeadCols['after'].append(self.OptimTruncFlagCol)
        
        # Initialise base.
        super().__init__(dfMonoCatObs, dfTransects=dfTransects,
                         effortConstVal=effortConstVal, dSurveyArea=dSurveyArea, 
                         transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                         sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols,
                         abbrevCol=abbrevCol, abbrevBuilder=abbrevBuilder,
                         anlysIndCol=anlysIndCol, sampleIndCol=sampleIndCol,
                         anlysSpecCustCols=anlysSpecCustCols,
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         surveyType=surveyType, distanceType=distanceType, clustering=clustering,
                         resultsHeadCols=resultsHeadCols,
                         ldTruncIntrvSpecs=ldTruncIntrvSpecs, truncIntrvEpsilon=truncIntrvEpsilon,
                         workDir=workDir, runMethod=runMethod, runTimeOut=runTimeOut,
                         logData=logData, logProgressEvery=logAnlysProgressEvery,
                         defEstimKeyFn=defEstimKeyFn, defEstimAdjustFn=defEstimAdjustFn,
                         defEstimCriterion=defEstimCriterion, defCVInterval=defCVInterval,
                         defMinDist=defMinDist, defMaxDist=defMaxDist, 
                         defFitDistCuts=defFitDistCuts, defDiscrDistCuts=defDiscrDistCuts)

        # Save Optimisation-specific args.
        self.sampleDistCol = sampleDistCol
        
        self.logOptimProgressEvery = logOptimProgressEvery
        self.backupOptimEvery = backupOptimEvery
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
        # TODO: Make this instantiation simpler (most params unused => default values would then fit well !)
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
                workDir=self.workDir, runMethod=self.runMethod, runTimeOut=self.runTimeOut,
                logData=self.logData, logProgressEvery=self.logOptimProgressEvery,
                backupEvery=self.backupOptimEvery, autoClean=self.autoClean,
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

    def computeUndeterminedParamSpecs(self, dfExplParamSpecs=None, implParamSpecs=None,
                                      threads=None, recover=False):
    
        """Run truncation optimisation for analyses with undetermined truncation param. specs
        and merge the computed specs to the ones of analyses with already determined truncation param. specs.
        
        Call explicitParamSpecs(..., check=True) before this to make sure user specs are OK

        Parameters:
        :param dfExplParamSpecs: Explicit MCDS analysis param specs, as a DataFrame
          (generated through explicitVariantSpecs, as an example),
        :param implParamSpecs: Implicit MCDS analysis param specs, suitable for explicitation
          through explicitVariantSpecs
        :param threads: Number of parallel threads to use (default None: no parallelism, no asynchronism)
        :param recover: Recover a previous run interrupted during optimisations ; using last available backup file
           
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
        
        # b. Search for (possibly) optimisation specs with string data (const params are numbers or lists)
        def analysisNeedsOptimisationFirst(sAnlysSpec):
            return any(isinstance(v, str) for v in sAnlysSpec.values)
        sbNeed4Opt1st = dfExplParamSpecs[optimUserParamSpecCols].apply(analysisNeedsOptimisationFirst, axis='columns')
        dfExplOptimParamSpecs = dfExplParamSpecs[sbNeed4Opt1st]
         
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
                                             sample=self.sampleSelCols, after=userParamSpecCols),
                        workDir=self.workDir, runMethod=self.runMethod, runTimeOut=self.runTimeOut,
                        logData=self.logData, logProgressEvery=self.logOptimProgressEvery,
                        backupEvery=self.backupOptimEvery, autoClean=self.autoClean,
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
            optimResults = self.zoptr.run(dfExplOptimParamSpecs, threads=threads, recover=recover)
            self.zoptr.shutdown()
            
            # c. Merge optimisation results into param. specs.
            # * Extract computed (= optimised) analysis params.
            dfOptimRes = \
                optimResults.dfSubData(columns=[self.anlysIndCol] + optimResults.optimisationTargetColumns())
            dfOptimRes.set_index(self.anlysIndCol, inplace=True)
            dfOptimRes.sort_index(inplace=True)
            
            # * Replicate optimisation specs as much as there are associated results
            #   (optimisation may keep more than 1 "best" result row)
            dfExplCompdParamSpecs = dfExplOptimParamSpecs.copy()
            dfExplCompdParamSpecs.set_index(self.anlysIndCol, inplace=True)
            dfExplCompdParamSpecs.sort_index(inplace=True)
            dfExplCompdParamSpecs = \
                dfExplCompdParamSpecs.loc[np.repeat(dfExplCompdParamSpecs.index,
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

    def setupResults(self, ldFilSorKeySchemes=None):
    
        """Build an empty results objects.

        Parameters:
        :param ldFilSorKeySchemes: Replacement for MCDSTruncOptanalysisResultsSet predefined filter-sort key schemes
                                   None => use predefined ones MCDSTruncOptanalysisResultsSet.AutoFilSorKeySchemes.
        """

        miCustCols, dfCustColTrans, miSampCols, sampIndMCol, sortCols, sortAscend = \
            self.prepareResultsColumns()

        return MCDSTruncOptanalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                              miSampleCols=miSampCols, sampleIndCol=sampIndMCol,
                                              sortCols=sortCols, sortAscend=sortAscend,
                                              distanceUnit=self.distanceUnit, areaUnit=self.areaUnit,
                                              surveyType=self.surveyType, distanceType=self.distanceType,
                                              clustering=self.clustering,
                                              ldTruncIntrvSpecs=self.ldTruncIntrvSpecs,
                                              truncIntrvEpsilon=self.truncIntrvEpsilon,
                                              ldFilSorKeySchemes=ldFilSorKeySchemes)
    
    def run(self, dfExplParamSpecs=None, implParamSpecs=None, threads=None, recoverOptims=False):
    
        """Run specified analyses, after automatic computing of truncation parameter if needed
        
        Call explicitParamSpecs(..., check=True) before this to make sure user specs are OK

        Parameters:
        :param dfExplParamSpecs: Explicit MCDS analysis param specs, as a DataFrame
          (generated through explicitVariantSpecs, as an example),
        :param implParamSpecs: Implicit MCDS analysis param specs, suitable for explicitation
          through explicitVariantSpecs
        :param threads: Number of parallel threads to use (default None: no parallelism, no asynchronism)
        :param recoverOptims: Recover a previous run interrupted during optimisations ; using last available backup file
           
        :return: the MCDSTruncOptanalysisResultsSet holding the analyses results
        """
        
        # 1. Run optimisations when needed and replace computed truncation params in analysis specs
        #    (warning: as some optimisation specs may specify to keep more than 1 "best" result,
        #              dfExplParamSpecs may grow accordingly, as well as the final number of result rows).
        dfExplParamSpecs = \
            self.computeUndeterminedParamSpecs(dfExplParamSpecs, implParamSpecs,
                                               threads=threads, recover=recoverOptims)

        # 2. Run all analyses, now all parameters are there.
        return super().run(dfExplParamSpecs, threads=threads)

    def shutdown(self):
    
        self.zoptr4Specs.shutdown()

        super().shutdown()


if __name__ == '__main__':

    import sys

    sys.exit(0)
