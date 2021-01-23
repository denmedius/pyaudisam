# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Parameters: Auto-determine analysis parameters values based on quality criteria
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment


import re

import pathlib as pl
from packaging import version

from collections import namedtuple as ntuple

import numpy as np
import pandas as pd

import pickle
import lzma

import autods.log as log

logger = log.logger('ads.opr')

from autods.data import MonoCategoryDataSet, ResultsSet
from autods.executor import Executor
from autods.engine import MCDSEngine
from autods.analyser import DSAnalyser, MCDSAnalyser
from autods.optimisation import Interval, Error, DSOptimisation, \
                                MCDSTruncationOptimisation, MCDSZerothOrderTruncationOptimisation


class OptimisationResultsSet(ResultsSet):
    
    """A specialized results set for DS analyses optimisations"""
    
    def __init__(self, optimisationClass, miCustomCols=None, dfCustomColTrans=None,
                       dComputedCols=None, dfComputedColTrans=None, sortCols=[], sortAscend=[]):
        
        assert issubclass(optimisationClass, DSOptimisation), \
               'optimisationClass must derive from DSOptimisation'
        assert miCustomCols is None \
               or (isinstance(miCustomCols, list) and len(miCustomCols) > 0 \
                   and all(isinstance(col, str) for col in miCustomCols)), \
               'customCols must be None or a list of strings'
        
        self.optimisationClass = optimisationClass
        
        # Optimisation results columns
        miCols = optimisationClass.RunColumns
        
        # DataFrame for translating column names
        dfColTrans = optimisationClass.DfRunColumnTrans
        
        # Initialize base class.
        super().__init__(miCols=miCols, dfColTrans=dfColTrans,
                         miCustomCols=miCustomCols, dfCustomColTrans=dfCustomColTrans,
                         sortCols=sortCols, sortAscend=sortAscend, dropNACols=False)
    
    def copy(self, withData=True):
    
        """Clone function (shallow), with optional (deep) data copy"""
    
        # 1. Call ctor without computed columns stuff (we no more have initial data)
        clone = OptimisationResultsSet(optimisationClass=self.optimisationClass,
                                       miCustomCols=self.miCustomCols.copy(),
                                       dfCustomColTrans=self.dfCustomColTrans.copy(),
                                       sortCols=self.sortCols.copy(), sortAscend=self.sortAscend.copy())
    
        # 2. Complete clone initialisation.
        # 3-level multi-index columns (module, statistic, figure)
        clone.miCols = self.miCols.copy()
        clone.computedCols = self.computedCols.copy()
        
        # DataFrames for translating columns names
        clone.dfColTrans = self.dfColTrans.copy()
        
        # Copy data if needed.
        if withData:
            clone._dfData = self._dfData.copy()
            clone.rightColOrder = self.rightColOrder
            clone.postComputed = self.postComputed

        return clone

    def fromExcel(self, fileName, sheetName=None, specs=True, specSheetsPrfx='sp-'):

        """Load (overwrite) data and optionnaly specs from an Excel worksheet (XLSX format),
        assuming ctor params match with Excel sheet column names and list,
        which can well be ensured by using the same ctor params as used for saving !
        """
        
        super().fromExcel(fileName, sheetName=sheetName, header=0, skipRows=None, indexCol=None,
                          specs=specs, specSheetsPrfx=specSheetsPrfx)

    def fromOpenDoc(self, fileName, sheetName=None, specs=True, specSheetsPrfx='sp-'):

        """Load (overwrite) data and optionnaly specs from an Open Document worksheet (ODS format),
        assuming ctor params match with ODF sheet column names and list,
        which can well be ensured by using the same ctor params as used for saving !
        Notes: Needs odfpy module and pandas.version >= 0.25.1
        """
        
        super().fromOpenDoc(fileName, sheetName=sheetName, header=0, skipRows=None, indexCol=None,
                            specs=specs, specSheetsPrfx=specSheetsPrfx)

    def fromFile(self, fileName, sheetName=None, specs=True, specSheetsPrfx='sp-'):

        """Load (overwrite) data data and optionnaly specs from a given file,
        (see ResultsSet.fromFile for supported formats, autodetected from file extensions)
        assuming ctor params match with the file contents,
        which can well be ensured by using the same ctor params as used for saving !
        """

        super().fromFile(fileName, sheetName=sheetName, header=0, skipRows=None, indexCol=None,
                         specs=specs, specSheetsPrfx=specSheetsPrfx)

    def optimisationTargetColumns(self):
    
        """The names of the columns holding the real optimisation results
        """
    
        return [col for col in self.optimisationClass.SolutionDimensionNames if col in self.columns]


class DSParamsOptimiser(object):

    """Run a bunch of DS analyses on samples extracted from a mono-category sightings data set,
       according to a user-friendly set of analysis specs (first set of fixed params),
       in order to determine best values for a second set of analysis params.
       Abstract class.
    """

    def __init__(self, dfMonoCatObs, dfTransects=None, effortConstVal=1, dSurveyArea=dict(), 
                 transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                 sampleSelCols=['Species', 'Pass', 'Adult', 'Duration'], 
                 sampleDecCols=['Effort', 'Distance'], sampleDistCol='Distance', anlysSpecCustCols=[],
                 abbrevCol='AnlysAbbrev', abbrevBuilder=None, anlysIndCol='AnlysNum', sampleIndCol='SampleNum', 
                 distanceUnit='Meter', areaUnit='Hectare', 
                 resultsHeadCols=dict(before=['AnlysNum', 'SampleNum'], after=['AnlysAbbrev'], 
                                      sample=['Species', 'Pass', 'Adult', 'Duration']),
                 workDir='.',
                 defExpr2Optimise='chi2', defMinimiseExpr=False,
                 defSubmitTimes=1, defSubmitOnlyBest=None, dDefSubmitOtherParams=dict(),
                 dDefOptimCoreParams=dict()):
                       
        """Ctor (don't use directly, abstract class)
        
        Parameters for input data to analyse:
        :param pd.DataFrame dfMonoCatObs: mono-category data from FieldDataSet.monoCategorise() or individualise()
        :param pd.DataFrame dfTransects: Transects infos with columns : transectPlaceCols (n), passIdCol (1),
            effortCol (1) ; if None, auto generated from input sightings
        :param effortConstVal: if dfTransects is None and effortCol not in source table, use this constant value
        :param transectPlaceCols:  See above dfTransects description
        :param passIdCol:  See above dfTransects description
        :param effortCol: See above dfTransects description
        :param sampleSelCols: sample identification = selection columns
        :param sampleDecCols: Decimal columns among sighting columns
        :param sampleDistCol: name of distance data column in run specs table
        :param anlysSpecCustCols: Special columns from analysis specs to simply pass through in results
        :param abbrevCol: Name of column to generate for abbreviating optimisation params, not sure really useful ...
        :param abbrevBuilder: Function of explicit analysis params (as a Series) to generate abbreviated name
        :param anlysIndCol: Name of column to generate for identifying analyses, unless already there in input data.
        :param sampleIndCol: Name of column to generate for identifying samples, unless already there in input data.
        
        Parameters for DS engine:
        :param distanceUnit: See DSEngine
        :param areaUnit: See DSEngine

        Parameters for optimisation output data:
        :param resultsHeadCols: dict of list of column names (from dfMonoCatObs) to use in order
            to build results header columns (left to results cols) ; 'sample' columns are sample selection columns.
        
        Parameters for evaluating analysis value to optimise
            (when not fully specified in each optimisation parameters):
        :param string defExpr2Optimise: Math. expression (python syntax) to optimise,
               using analyses results var. names inside (see derived classes for details)
        :param defMinimiseExpr: True for minimisation of expr2Optimise, false for maximisation

        Parameters for optimisation submissions
            (when not fully specified in each optimisation parameters):
        :param defSubmitTimes: Number of times to auto-run each optimisation (> 0 ; default 1)
        :param defSubmitOnlyBest: Number of best repetition results to keep
                                  (> 0 ; default None = all repetitions)
        :param dDefSubmitOtherParams: Other submission parameters

        Other parameters:
        :param dSurveyArea: dict of info about survey area (mainly name and surface) for DS analyses
        :param workDir: target folder for intermediate computations and results files
        :param dDefOptimCoreParams: Optimisation core specific params
        """

        # Save data.
        self.dfMonoCatObs = dfMonoCatObs
        self.sampleDistCol = sampleDistCol

        self.resultsHeadCols = resultsHeadCols
        self.abbrevCol = abbrevCol
        self.abbrevBuilder = abbrevBuilder
        self.anlysIndCol = anlysIndCol
        self.sampleSelCols = sampleSelCols
        self.sampleIndCol = sampleIndCol
        self.anlysSpecCustCols = anlysSpecCustCols
            
        self.distanceUnit = distanceUnit
        self.areaUnit = areaUnit

        self.workDir = workDir
        
        # Default values for optimisation parameters.
        # a. Expression to optimise
        self.defExpr2Optimise = defExpr2Optimise
        self.defMinimiseExpr = defMinimiseExpr
        
        # b. Optimisation core parameters.
        self.dDefOptimCoreParams = dDefOptimCoreParams

        # c. Optimisations submission (=run) parameters
        self.defSubmitTimes = defSubmitTimes
        self.defSubmitOnlyBest = defSubmitOnlyBest
        self.dDefSubmitOtherParams = dDefSubmitOtherParams

        # Mono-categorised data (all samples)
        self._mcDataSet = \
            MonoCategoryDataSet(dfMonoCatObs, dfTransects=dfTransects, effortConstVal=effortConstVal,
                                dSurveyArea=dSurveyArea, transectPlaceCols=transectPlaceCols,
                                passIdCol=passIdCol, effortCol=effortCol, sampleDecFields=sampleDecCols)
                                
        # Analysis engine and executor.
        self._executor = None
        self._engine = None

        # Results.
        self.results = None

        # Computation specifications, for traceability only.
        # For gathering copies of computations default parameter values, and stuff like that.
        self.specs = dict()
        self.updateSpecs(**{name: getattr(self, name) for name in ['distanceUnit', 'areaUnit']})
        self.updateSpecs(**{name: getattr(self, name)
                            for name in ['defExpr2Optimise', 'defMinimiseExpr', 'dDefOptimCoreParams',
                                         'defSubmitTimes', 'defSubmitOnlyBest', 'dDefSubmitOtherParams']})

    def updateSpecs(self, reset=False, overwrite=False, **specs):

        if reset:
            self.specs.clear()

        if not overwrite:
            assert all(name not in self.specs for name in specs), \
                   "Unless specified, won't overwrite already present specs {}" \
                   .format(', '.join(name for name in specs if name in self.specs))

        self.specs.update(specs)

    def flatSpecs(self):

        # Flatten "in-line" 2nd level dicts if any (with 1st level name prefixing).
        dFlatSpecs = dict()
        for name, value in self.specs.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    dFlatSpecs[name + n[0].upper() + n[1:]] = v
            else:
                dFlatSpecs[name] = value

        # Done.
        return dFlatSpecs

    # Optimiser internal parameter spec names, for which a match should be found (when one is needed)
    # with user explicit optimisation specs used in run() calls.
    IntSpecExpr2Optimise = 'Expr2Optimise'
    IntSpecOptimisationCore = 'OptimisationCore'
    IntSpecSubmitParams = 'SubmitParams'

    # Possible regexps (values) for auto-detection of optimiser _internal_ parameter spec names (keys)
    # from explicit _user_ spec columns
    # (regexps are re.search'ed : any match _anywhere_inside_ the column name is OK;
    #  and case is ignored during searching).
    Int2UserSpecREs = \
        {IntSpecExpr2Optimise:    ['opt[a-z]*[\.\-_ ]*exp', 'exp[a-z2]*[\.\-_ ]*opt',
                                   'opt[a-z]*[\.\-_ ]*cri', 'cri[a-z]*[\.\-_ ]*opt'],
         IntSpecOptimisationCore: ['opt[a-z]*[\.\-_ ]*core', 'mot[a-z]*[\.\-_ ]*opt',
                                   'noy[a-z]*[\.\-_ ]*opt'],
         IntSpecSubmitParams:     ['sub[a-z]*[\.\-_ ]*par', 'par[a-z]*[\.\-_ ]*sou',
                                   'run[a-z]*[\.\-_ ]*par', 'par[a-z]*[\.\-_ ]*ex',
                                   'mul[a-z]*[\.\-_ ]*opt', 'opt[a-z]*[\.\-_ ]*mul']}

    # Types for user specs parsing (see usage below)
    class Auto(object):
        def __repr__(self):
            return 'Auto()'
        def __eq__(self, other):
            return isinstance(other, self.__class__)
    
    @classmethod
    def _parseUserSpec(cls, spec, globals=dict(), locals=dict(),
                       oneStrArg=False, nullOrEmpty=Error, errIfNotA=[]):
                                  
        """Parse parameter(s) user spec with python-like simple expression syntax from given rules 
        
        :param spec: None or np.nan or string spec to parse
        :param globals: dict of globals for case-insensitive rules (we use eval function for parsing !)
        :param locals: dict of locals for case-insensitive rules (we use eval function for parsing !)
        :param oneStrArg: assume function call syntax with 1 single string argument
                (ex: input "f(x,y)" is tranformed to "f('x,y')" before calling eval)
        :param nullOrEmpty: return value for null of empty spec ; not checked against errIfNotA
            (default: Error => an instance of Error with error description inside)
        :param errIfNotA: list of autorised output types ; empty => any type
        
        :return: a 2-value tuple : (None or an Error instance, parsed value or None in case of error)
        """
    
        # Empty or Null cases
        if pd.isnull(spec) or (isinstance(spec, str) and not spec.strip()):
        
            if nullOrEmpty is Error:
                parsedValue = None
                parseError = Error('Should be specified ; did you mean "auto" ?')
            else:
                parsedValue = nullOrEmpty
                parseError = None
        
            return parseError, parsedValue

        # Other cases.
        spec = str(spec).lower()  # int and float cases, + lower case
        if oneStrArg:
            if "('" not in spec:
                spec = spec.replace('(', "('")
            if "')" not in spec:
                spec = spec.replace(')', "')")
        
        try:
            parsedValue = eval(spec, globals, locals)
            if errIfNotA and not isinstance(parsedValue, tuple(errIfNotA)):
                error = 'Not a {}'.format(', '.join(t.__name__ for t in errIfNotA))
                parseError = Error(head=spec, error=error)
                parsedValue = None
            else:
                parseError = None
        except Exception as exc:
            parsedValue = None
            parseError = Error(head=spec, error=str(exc))
        
        return parseError, parsedValue
    
    # Types for parsing user spec
    DistInterval = ntuple('DistInterval', ['dmin', 'dmax'], defaults=[0, -1])  # Distance interval
    AbsInterval = ntuple('AbsInterval', ['min', 'max'], defaults=[0, -1])  # Interval for actual values
    MultInterval = ntuple('MultInterval', ['kmin', 'kmax'])  # Interval for multipliers
    OutliersMethod = ntuple('OutliersMethod', ['method', 'percent'])

    @classmethod
    def _parseDistTruncationUserSpec(cls, spec, errIfNotA=[]):
    
        """Parse user spec for one analysis optimised parameter
        
        Parameters:
        :param spec: None or np.nan or string spec to parse
        :param errIfNotA: list of autorised output types ; empty => any type
        
        :return: a 2-value tuple : (None or an Error instance, parsed value or None in case of error)
                 Parsed value may result None or an instance of AbsInterval, MultInterval, OutliersMethod, Auto
        """

        # Defs for optimisation param. spec. mini-language
        auto = cls.Auto()
        def dist(dmin, dmax):
            return cls.DistInterval(dmin, dmax)
        def quant(pct):
            return cls.OutliersMethod('quant', pct)
        def tucquant(pct):
            return cls.OutliersMethod('tucquant', pct)
        def mult(kmin, kmax):
            return cls.MultInterval(kmin, kmax)
        def abs(min, max):
            return cls.AbsInterval(min, max)
            
        # Parse spec.
        return cls._parseUserSpec(spec, nullOrEmpty=None, errIfNotA=errIfNotA,
                                  globals=dict(Auto=cls.Auto, DistInterval=cls.DistInterval,
                                               AbsInterval=cls.AbsInterval, MultInterval=cls.MultInterval,
                                               OutliersMethod=cls.OutliersMethod),
                                  locals=dict(auto=auto, dist=dist, quant=quant, tucquant=tucquant,
                                              mult=mult, abs=abs))

    @classmethod
    def _parseOptimCoreUserSpecs(cls, spec, globals=dict(), locals=dict(),
                                 nullOrEmpty=Error, errIfNotA=[]):
    
        """Parse user spec for optimisation core parameters
        
        Praameters:
        :param spec: the spec to parse
        :param globals: dict of globals for rules (we use eval function for parsing !)
        :param locals: dict of locals for rules (we use eval function for parsing !)
        :param nullOrEmpty: return value for null of empty spec ; not checked against errIfNotA
            (default: Error => an instance of Error with error description inside)
        :param errIfNotA: list of autorised output types ; empty => any type
        
        :return: a 2-value tuple : (None or an Error instance, parsed value or None in case of error)
                 Parsed value may result None or an instance of AbsInterval, MultInterval, OutliersMethod, Auto
        """

        if isinstance(spec, str) and spec.strip():
        
            # We don't care about case
            spec = spec.lower()
            
            # A single string for the engine name is enough => add () at the end if so.
            if '(' not in spec:
                spec = spec + '()'
                
            # String parameters don't need quoting in spec, but python needs it : add it if needed
            spec = re.sub('([a-z_]+) *= *([^=0-9,; ][^=,;\) ]*)', r"\1='\2'", spec)
    
        # Parse pythonised spec.
        return cls._parseUserSpec(spec, nullOrEmpty=nullOrEmpty, errIfNotA=errIfNotA, 
                                  oneStrArg=False, globals=globals, locals=locals)
        

    # Optimisation object ctor parameter names (MUST match exactly: check in optimisation submodule !).
    ParmExpr2Optimise = 'expr2Optimise'
    ParmMinimiseExpr = 'minimiseExpr'

    def getAnalysisOptimExprParams(self, sAnIntSpec):
                                 
        """Retrieve optimisation expression parameters, from user specs and default values.
        
        :param sAnIntSpec: analysis parameter user specs with internal names (indexed with IntSpecXXX)
                           syntax: IntSpecExpr2Optimise => <min|max>(math. expr)

        :return: None or an Error instance, dict(expr2Optimise=..., minimiseExpr=...) or None
        
        Ex: max(aic), min(1/aic/ks)"""

        def _buildParsedValue(expr2Optimise, minimiseExpr):
            return { self.ParmExpr2Optimise: expr2Optimise, self.ParmMinimiseExpr: minimiseExpr }

        # Parse expression to optimise in sAnIntSpec if present.
        if self.IntSpecExpr2Optimise in sAnIntSpec:
        
            # Retrieve
            userOptExpr = sAnIntSpec[self.IntSpecExpr2Optimise]
                
            # Parse
            def min(expr):
                return _buildParsedValue(expr, True)
            def max(expr):
                return _buildParsedValue(expr, False)

            parseError, parsedValue = \
                 self._parseUserSpec(userOptExpr, globals=None, locals=dict(min=min, max=max),
                                     oneStrArg=True, errIfNotA=[dict],
                                     nullOrEmpty=_buildParsedValue(self.defExpr2Optimise, self.defMinimiseExpr))

        # No way: fall back to default values.
        else:
            
            parseError, parsedValue = \
                None, _buildParsedValue(self.defExpr2Optimise, self.defMinimiseExpr)
                 
        # Done.
        return parseError, parsedValue
    
    def getOptimisationCoreParams(self, sAnIntSpec):

        """Retrieve optimisation core parameters, from user specs and default values.
        
        Ex: zoopt(mxi=50, tv=1, a=racos, mxr=2)

        :param sAnIntSpec: analysis parameter user specs with internal names (indexed with IntSpecXXX)
                           syntax: IntSpecExpr2Optimise => <opt. core name>(**{k:v})
                           
        :return: None or an Error instance, dict(core=..., **{key:value}) or None
        """
        
        raise NotImplementedError('Abstract class: implement in a derived class')
    
    # Optimisation object ctor parameter names (MUST match exactly: check in optimisation submodule !).
    ParmSubmTimes = 'times'
    ParmSubmOnlyBest = 'onlyBest'

    def getOptimisationSubmitParams(self, sAnIntSpec):
    
        """Retrieve optimisation submission parameters from user specs and default values.
        
        :param sAnIntSpec: analysis parameter user specs with internal names (indexed with IntSpecXXX)
                           syntax: IntSpecSubmitParams => <times>([n=]<num>[, [b=]<num>])
                           
        :return: None or an Error instance, dict(=..., **{k:v}) or None
        
        Ex: dict(times=, onlyBest=, ...)"""

        def _buildParsedValue(times, onlyBest):
            return { self.ParmSubmTimes: times, self.ParmSubmOnlyBest: onlyBest }

        # Parse expression to optimise in sAnIntSpec if present.
        if self.IntSpecSubmitParams in sAnIntSpec:
        
            # Retrieve
            userOptExpr = sAnIntSpec[self.IntSpecSubmitParams]
                
            # Parse
            def times(n=1, b=None):
                assert n > 0, 'Run times must be > 0'
                assert b is None or b > 0, 'Number of best kept values must be > 0'
                return _buildParsedValue(n, b)

            parseError, parsedValue = \
                 self._parseUserSpec(userOptExpr, globals=None, locals=dict(times=times),
                                     nullOrEmpty=_buildParsedValue(self.defSubmitTimes, self.defSubmitOnlyBest),
                                     errIfNotA=[dict])

        # No way: fall back to default values.
        else:
            
            parseError, parsedValue = \
                None, _buildParsedValue(self.defSubmitTimes, self.defSubmitOnlyBest)
                 
        # Done.
        return parseError, parsedValue
        
    def setupOptimisation(self, sampleDataSet, name=None, customData=None, setupError=None, **otherParams):

        """Factory method for DSOptimisation derived classes
        
        Parameters:
        :param sampleDataSet: SampleDataSet to analyse
        :param name: of the optimisation (for clearer logging only, auto-generated and auto-unique anyway)
        :param customData: pd.Series of custom data for heading optimisation results columns
        :param error: if an error occurred during optimisation submit params user specs parsing,
                 a string is here for explaining it, and to prevent any submit call of course !
                 This is done for keeping trace of unrun optimisations in results table (1 empty/null row)
        :param otherParams: some room for derived classes
        """

        raise NotImplementedError('DSParamsOptimiser: Abstract class, implement setupOptimisation in derived one')

    def explicitParamSpecs(self, implParamSpecs=None, dfExplParamSpecs=None, dropDupes=True, check=False):
    
        """Explicitate analysis and optimisation param. specs if not already done, and complete columns if needed ;
        also automatically extract (regexps) columns which are really analysis parameters,
        with their analyser-internal name, and also their "user" name.
        
        Can moreover check params specs for usability, if check is True :
        * use it before calling analyser.run(implParamSpecs=..., dfExplParamSpecs=..., ...)
          to check that everythings OK,
        * or be sure that run() will fail at startup (because it itself will do it).
        
        Parameters:
        :param implParamSpecs: Implicit analysis param specs, suitable for explicitation
           through explicitVariantSpecs()
        :param dfExplParamSpecs: Explicit analysis param specs, as a DataFrame
           (generated through explicitVariantSpecs, as an example)
        :param dropDupes: if True, drop duplicates (keep first)
        :param check: if True, checks params for usability by run(),
           and return a bool verdict and a list of strings explaining the negative (False) verdict

        :return: a 3 or 5-item tuple :
           * explicit specs as a DataFrame (input dfExplParamSpecs not modified: a new updated one is returned),
           * list of analysis and optimisationparam. columns internal names,
           * list of analysis and optimisationparam. columns user names,
           if ckeck, 2 more items in the return tuple :
           * check verdict : True if everything went well, False otherwise,
             * some columns from paramSpecCols could not be found in dfExplParamSpecs columns,
             * some user columns could not be matched with some of the expected internal parameter names,
             * some rows are not suitable for DS analysis (empty sample identification columns, ...).
           * check failure reasons : list of strings explaining things that went bad.
        """
        
        # Explicitate and complete
        tplRslt = DSAnalyser._explicitParamSpecs(implParamSpecs, dfExplParamSpecs, self.Int2UserSpecREs,
                                                 sampleSelCols=self.sampleSelCols, abbrevCol=self.abbrevCol,
                                                 abbrevBuilder=self.abbrevBuilder, anlysIndCol=self.anlysIndCol,
                                                 sampleIndCol=self.sampleIndCol,
                                                 anlysSpecCustCols=self.anlysSpecCustCols, dropDupes=dropDupes)
        
        # Check if requested
        if check:
        
            verdict = True
            reasons = []
     
            dfExplParamSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols = tplRslt
            
            # Check that an internal column name was found for every user spec column.
            if len(unmUserParamSpecCols):
                verdict = False
                reasons.append('Failed to match some user spec. names with internal ones: {}'
                               .format(', '.join(unmUserParamSpecCols)))

            # Check that all rows are suitable for DS analysis (non empty sample identification columns, ...).
            if dfExplParamSpecs[self.sampleSelCols].isnull().all(axis='columns').any():
                verdict = False
                reasons.append('Some rows have some null sample selection columns')

            # Check done.
            tplRslt += verdict, reasons

        return tplRslt

    def shutdown(self):
    
        """Shutdown engine and executor (only usefull if run() raises an exception and so fails to do it),
        but keep the remainder of the object state as is.
        """
    
        if self._engine:
            self._engine.shutdown(executor=True)
            self._engine = None
        if self._executor:
            self._executor.shutdown()
            self._executor = None
            
#    def __del__(self):
#    
#        self.shutdown()


class MCDSTruncationOptimiser(DSParamsOptimiser):

    """Abstract class ; Run a bunch of MCDS truncation optimisations"""

    def __init__(self, dfMonoCatObs, dfTransects=None, effortConstVal=1, dSurveyArea=dict(), 
                       transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                       sampleSelCols=['Species', 'Pass', 'Adult', 'Duration'], 
                       sampleDecCols=['Effort', 'Distance'], sampleDistCol='Distance', anlysSpecCustCols=[],
                       abbrevCol='AnlysAbbrev', abbrevBuilder=None, anlysIndCol='AnlysNum', sampleIndCol='SampleNum',
                       distanceUnit='Meter', areaUnit='Hectare',
                       surveyType='Point', distanceType='Radial', clustering=False,
                       resultsHeadCols=dict(before=['AnlysNum', 'SampleNum'], after=['AnlysAbbrev'], 
                                            sample=['Species', 'Pass', 'Adult', 'Duration']),
                       workDir='.', runMethod='subprocess.run', runTimeOut=120,
                       logData=False, logProgressEvery=5, backupEvery=50, autoClean=True,
                       defEstimKeyFn=MCDSEngine.EstKeyFnDef, defEstimAdjustFn=MCDSEngine.EstAdjustFnDef,
                       defEstimCriterion=MCDSEngine.EstCriterionDef, defCVInterval=MCDSEngine.EstCVIntervalDef,
                       defExpr2Optimise='chi2', defMinimiseExpr=False,
                       defOutliersMethod='tucquant', defOutliersQuantCutPct=5,
                       defFitDistCutsFctr=dict(min=2/3, max=3/2), defDiscrDistCutsFctr=dict(min=1/3, max=1),
                       defSubmitTimes=1, defSubmitOnlyBest=None, dDefSubmitOtherParams=dict(),
                       dDefOptimCoreParams=dict()):

        """Ctor (don't use directly, abstract class)
        
        Parameters for MCDS engine:
        :param surveyType: See MCDSEngine
        :param distanceType: See MCDSEngine
        :param clustering: See MCDSEngine

        Parameters for auto-computing target intervals (when not fully specified in each optimisation parameters):
        :param defOutliersMethod: Outliers estimation method when min/maxDist=auto
               or auto(pct) but not auto(min, max): 
                * 'quant' : Pure [P, 100-P] % quantiles
                * 'tucquant' : Mixed P% quantiles & Tuckey method
        :param defOutliersQuantCutPct: Outliers cut % value (= P in outliersMethod description above) 
        :param defFitDistCutsFctr: Default factor multiplied to sqrt(nb of sightings)
               to get min/max FitDistCuts when fitDistCutsFctr is auto with no parameters (min and max)
        :param defDiscrDistCutsFctr: Factor multiplied to sqrt(nb of sightings)
               to get min/max DiscrDistCuts when discrDistCutsFctr is auto with no parameters (min and max)

        :param runMethod: for calling MCDS engine executable : 'os.system' or 'subprocess.run'
        :param runTimeOut: engine call time limit (s) ; None => no limit ;
                           WARNING: Not implemented (no way) for 'os.system' run method (no solution found)

        Other parameters: See base class.
        """

        super().__init__(dfMonoCatObs, dfTransects=dfTransects,
                         effortConstVal=effortConstVal, dSurveyArea=dSurveyArea, 
                         transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                         sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols,
                         sampleDistCol=sampleDistCol, anlysSpecCustCols=anlysSpecCustCols,
                         abbrevCol=abbrevCol, abbrevBuilder=abbrevBuilder,
                         anlysIndCol=anlysIndCol, sampleIndCol=sampleIndCol,
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         resultsHeadCols=resultsHeadCols, workDir=workDir,
                         defExpr2Optimise=defExpr2Optimise, defMinimiseExpr=defMinimiseExpr,
                         defSubmitTimes=defSubmitTimes, defSubmitOnlyBest=defSubmitOnlyBest,
                         dDefSubmitOtherParams=dDefSubmitOtherParams, dDefOptimCoreParams=dDefOptimCoreParams)

        assert runTimeOut is None or runMethod != 'os.system', \
               f"Can't care about {runTimeOut}s execution time limit with os.system run method (not implemented)"

        self.surveyType = surveyType
        self.distanceType = distanceType
        self.clustering = clustering
        
        self.runMethod = runMethod
        self.runTimeOut = runTimeOut
        self.logData = logData
        self.logProgressEvery = logProgressEvery
        self.backupEvery = backupEvery
        self.autoClean = autoClean
        
        self.defEstimKeyFn = defEstimKeyFn
        self.defEstimAdjustFn = defEstimAdjustFn
        self.defEstimCriterion = defEstimCriterion
        self.defCVInterval = defCVInterval
        self.defOutliersMethod = defOutliersMethod
        self.defOutliersQuantCutPct = defOutliersQuantCutPct
        self.defFitDistCutsFctr = \
            defFitDistCutsFctr if defFitDistCutsFctr is None else Interval(defFitDistCutsFctr)
        self.defDiscrDistCutsFctr = \
            defDiscrDistCutsFctr if defDiscrDistCutsFctr is None else Interval(defDiscrDistCutsFctr)
                         
        self.updateSpecs(**{name: getattr(self, name)
                            for name in ['runMethod', 'runTimeOut', 'surveyType', 'distanceType', 'clustering',
                                         'defEstimKeyFn', 'defEstimAdjustFn', 'defEstimCriterion', 'defCVInterval']})
        self.updateSpecs(**{name: getattr(self, name)
                            for name in ['defOutliersMethod', 'defOutliersQuantCutPct',
                                         'defFitDistCutsFctr', 'defDiscrDistCutsFctr']})

    # Optimiser internal parameter spec names, for which a match should be found (when one is needed)
    # with user explicit optimisation specs used in run() calls.
    IntSpecEstimKeyFn = MCDSAnalyser.IntSpecEstimKeyFn
    IntSpecEstimAdjustFn = MCDSAnalyser.IntSpecEstimAdjustFn
    IntSpecEstimCriterion = MCDSAnalyser.IntSpecEstimCriterion
    IntSpecCVInterval = MCDSAnalyser.IntSpecCVInterval
    IntSpecMinDist = MCDSAnalyser.IntSpecMinDist # Left truncation distance
    IntSpecMaxDist = MCDSAnalyser.IntSpecMaxDist # Right truncation distance
    IntSpecFitDistCuts = MCDSAnalyser.IntSpecFitDistCuts
    IntSpecDiscrDistCuts = MCDSAnalyser.IntSpecDiscrDistCuts
    IntSpecOutliersMethod = 'OutliersMethod'

    # Possible regexps (values) for auto-detection of optimiser _internal_ parameter spec names (keys)
    # from explicit _user_ spec columns
    # (regexps are re.search'ed : any match _anywhere_inside_ the column name is OK;
    #  and case is ignored during searching).
    Int2UserSpecREs = \
      dict(list(DSParamsOptimiser.Int2UserSpecREs.items()) \
           + list(MCDSAnalyser.Int2UserSpecREs.items()) \
           + [(IntSpecOutliersMethod, ['outl[a-z]*[\.\-_ ]*', 'me[a-z]*[\.\-_ ]*outl'])])

    # Names of internal parameters which can be used as settings for optimising truncations
    # and not only as const=fixed=pre-determined analysis parameters.
    IntOptimParamSpecNames = \
        [DSParamsOptimiser.IntSpecExpr2Optimise, DSParamsOptimiser.IntSpecOptimisationCore,
         DSParamsOptimiser.IntSpecSubmitParams,
         IntSpecMinDist, IntSpecMaxDist, IntSpecFitDistCuts, IntSpecDiscrDistCuts,
         IntSpecOutliersMethod]
         
    @classmethod
    def optimisationParamSpecUserNames(cls, userParamSpecCols, intParamSpecCols):
    
        """Extract user names of params spec. columns that may contain truncation optimisation parameters

        Parameters:
        :param userParamSpecCols: all user params spec column names to explore,
        :param intParamSpecCols: the matching (1 by 1, same order) internal user params spec column names.
        """
    
        # Internal column names to consider.
        optimIntCols = [intCol for intCol in cls.IntOptimParamSpecNames if intCol in intParamSpecCols]
        
        # And the corresponding user names.
        return [userCol for intCol, userCol in zip(intParamSpecCols, userParamSpecCols) if intCol in optimIntCols]

    # Optimisation object ctor parameter names (MUST match exactly: check in optimisation submodule !).
    ParmEstimKeyFn = 'estimKeyFn'
    ParmEstimAdjustFn = 'estimAdjustFn'
    ParmEstimCriterion = 'estimCriterion'
    ParmCVInterval = 'cvInterval'
    
    def getAnalysisFixedParams(self, sAnIntSpec):
    
        """Retrieve analysis fixed parameters of an optimisation, from user specs and default values
        
        :param sAnIntSpec: analysis parameter user specs with internal names (indexed with IntSpecXXX)
        
        :return: a 2-value tuple (None or an Error instance,
                                  dict(estimKeyFn=, estimAdjustFn=, estimCriterion=, cvInterval=) or None)
        """
        
        dParams = dict()
        
        estimKeyFn = sAnIntSpec.get(self.IntSpecEstimKeyFn, self.defEstimKeyFn)
        if pd.isnull(estimKeyFn):
            estimKeyFn = self.defEstimKeyFn
        
        estimAdjFn = sAnIntSpec.get(self.IntSpecEstimAdjustFn, self.defEstimAdjustFn)
        if pd.isnull(estimAdjFn):
            estimAdjFn = self.defEstimAdjustFn
        
        estimCrit = sAnIntSpec.get(self.IntSpecEstimCriterion, self.defEstimCriterion)
        if pd.isnull(estimCrit):
            estimCrit = self.defEstimCriterion
        
        cvInterv = sAnIntSpec.get(self.IntSpecCVInterval, self.defCVInterval)
        if pd.isnull(cvInterv):
            cvInterv = self.defCVInterval
        
        return None, { self.ParmEstimKeyFn: estimKeyFn, self.ParmEstimAdjustFn: estimAdjFn,
                       self.ParmEstimCriterion: estimCrit, self.ParmCVInterval: cvInterv }

    # From / to: Optimisation object ctor parameter names = Solution dimension names
    # To / From: Optimiser internal param. spec names.
    SolDim2IntSpecOptimTargetParamNames = \
        dict(zip(MCDSTruncationOptimisation.SolutionDimensionNames,
                 [IntSpecMinDist, IntSpecMaxDist, IntSpecFitDistCuts, IntSpecDiscrDistCuts]))
    IntSpec2SolDimOptimTargetParamNames = {v:k for k,v in SolDim2IntSpecOptimTargetParamNames.items()}

    def getAnalysisOptimedParams(self, sAnIntSpec, sSampleDists):
    
        """Compute optimisation intervals of an optimisation, from user specs and default parameters

        Some checks for final values are done, may be resulting in an Error.
        
        Parameters:
        :param sAnIntSpec: analysis parameter user specs with internal names (indexed with IntSpecXXX)
                           syntax: sequence of <key>=<value> separated by ','
        :param sSampleDists: sample sightings recorded distances
        
        :return: a 2-value tuple (None or an Error instance in case any parsing / check failed,
                                  dict(minDist=, maxDist=, fitDistCuts=, discrDistCuts=) or None)
        """
        
        # Parse found specs from strings (or so) to real objects
        errMinDist, minDistSpec = \
            self._parseDistTruncationUserSpec(sAnIntSpec.get(self.IntSpecMinDist, None),
                                              errIfNotA=[int, float, self.Auto,
                                                         self.DistInterval, self.OutliersMethod])
        errMaxDist, maxDistSpec =  \
            self._parseDistTruncationUserSpec(sAnIntSpec.get(self.IntSpecMaxDist, None),
                                              errIfNotA=[int, float, self.Auto,
                                                         self.DistInterval, self.OutliersMethod])
        errFitDistCuts, fitDistCutsSpec = \
            self._parseDistTruncationUserSpec(sAnIntSpec.get(self.IntSpecFitDistCuts, None),
                                              errIfNotA=[int, float, self.Auto, 
                                                         self.MultInterval, self.AbsInterval])
        errDiscrDistCuts, discrDistCutsSpec = \
            self._parseDistTruncationUserSpec(sAnIntSpec.get(self.IntSpecDiscrDistCuts, None),
                                              errIfNotA=[int, float, self.Auto,
                                                         self.MultInterval, self.AbsInterval])
        errOutliersMethod, outliersMethodSpec = \
            self._parseDistTruncationUserSpec(sAnIntSpec.get(self.IntSpecOutliersMethod, None),
                                              errIfNotA=[self.Auto, self.OutliersMethod])

        logger.debug('OptimedParams specs:' + str(dict(minDist=minDistSpec, maxDist=maxDistSpec,
                                                 fitDistCuts=fitDistCutsSpec, discrDistCuts=discrDistCutsSpec)))

        # Stop here if any parsing error.
        finalErr = Error()
        for err in [errMinDist, errMaxDist, errFitDistCuts, errDiscrDistCuts, errOutliersMethod]:
            if err:
                finalErr.append(err)
        if finalErr:
            return finalErr, None

        # Compute or translate parameter values from objects.
        
        # A. outliers method: method ('quant' or 'tucquant') and cut % (% of outliers on _each_ side)
        # - None or auto : use defOutliersMethod et defOutliersQuantCutPct,
        # - <meth>(<cut pct>) : use <meth> method, and <cut pct> cut %
        if outliersMethodSpec is None or isinstance(outliersMethodSpec, self.Auto):
            outliersMethod = self.defOutliersMethod
            outliersQuantCutPct = self.defOutliersQuantCutPct
        elif isinstance(outliersMethodSpec, self.OutliersMethod):
            outliersMethod = outliersMethodSpec.method
            outliersQuantCutPct = outliersMethodSpec.percent
        else:
            raise Exception('MCDSTruncationOptimiser.getAnalysisOptimedParams:'
                            'Should not fall there (outliersMethod specs)')

        # B. minDist, maxDist :
        # - None,
        # - auto : mode full auto via colonne OutliersMethod, 
        #                             ou alors defOutliersMethod et defOutliersQuantCutPct,
        # - <meth>(5) : mode auto tq OutliersMethod (<meth> quant(5) ou tucquant) et outliersQuantCutPct(ici 5%) fixés
        # - dist(20, 250) : mode fixé, dist min (20) et max (250) du domaine de variations fournies en dur

        sDist = sSampleDists.dropna()
        sqrNSights = np.sqrt(len(sDist))

        # minDist specs
        if minDistSpec is None or isinstance(minDistSpec, (int, float)):
            minDist = minDistSpec
        elif isinstance(minDistSpec, self.DistInterval):
            minDist = Interval(min=minDistSpec.dmin, max=minDistSpec.dmax)
        elif isinstance(minDistSpec, self.Auto):
            maxMinDist = np.percentile(a=sDist, q=outliersQuantCutPct)
            minDist = Interval(min=sDist.min(), max=maxMinDist)
        elif isinstance(minDistSpec, self.OutliersMethod):
            maxMinDist = np.percentile(a=sDist, q=minDistSpec.percent)
            minDist = Interval(min=sDist.min(), max=maxMinDist)
        else:
            raise Exception('MCDSTruncationOptimiser.getAnalysisOptimedParams:'
                            'Should not fall there (minDist specs)')
        
        if isinstance(minDist, Interval) and minDist.min == minDist.max:
            minDist = minDist.min

        # maxDist specs
        if maxDistSpec is None or isinstance(maxDistSpec, (int, float)):
            maxDist = maxDistSpec
        elif isinstance(maxDistSpec, self.DistInterval):
            maxDist = Interval(min=maxDistSpec.dmin, max=maxDistSpec.dmax)
        elif isinstance(maxDistSpec, (self.Auto, self.OutliersMethod)):
            if isinstance(maxDistSpec, self.Auto):
                d25, d75, d95, dPct = np.percentile(a=sDist, q=[25, 75, 95, 100-outliersQuantCutPct])
            else: # self.OutliersMethod
                d25, d75, d95, dPct = np.percentile(a=sDist, q=[25, 75, 95, 100-maxDistSpec.percent])
            if outliersMethod == 'quant':
                minMaxDist = dPct
            elif outliersMethod == 'tucquant':
                minMaxDist = min(max(d95, d75 + 1.5*(d75 - d25)), dPct)
            maxDist = Interval(min=minMaxDist, max=sDist.max())
        else:
            raise Exception('MCDSTruncationOptimiser.getAnalysisOptimedParams:'
                            'Should not fall there (maxDist specs)')

        if isinstance(maxDist, Interval) and maxDist.min == maxDist.max:
            minDist = maxDist.min

        # C. fitDistCuts, discrDistCuts :
        # - None,
        # - auto : mode full auto via colonne outliersMethod, 
        #                             ou alors defOutliersMethod et defOutliersQuantCutPct,
        # - mult(1/3, 3/2) : mode fixé, facteurs mult. min et max de sqrt(nb données) fournies en dur
        # - abs(5, 10) : mode fixé, facteurs mult. min et max de sqrt(nb données) fournies en dur

        # 1. fitDistCuts specs
        if fitDistCutsSpec is None or isinstance(fitDistCutsSpec, (int, float)):
            fitDistCuts = fitDistCutsSpec
        elif isinstance(fitDistCutsSpec, self.AbsInterval):
            fitDistCuts = Interval(min=max(2, fitDistCutsSpec.min), max=fitDistCutsSpec.max)
        elif isinstance(fitDistCutsSpec, self.Auto):
            fitDistCuts = Interval(min=max(2, int(round(self.defFitDistCutsFctr.min*sqrNSights))),
                                   max=int(round(self.defFitDistCutsFctr.max*sqrNSights)))
        elif isinstance(fitDistCutsSpec, self.MultInterval):
            fitDistCuts = Interval(min=max(2, int(round(fitDistCutsSpec.kmin*sqrNSights))),
                                   max=int(round(fitDistCutsSpec.kmax*sqrNSights)))
        else:
            raise Exception('MCDSTruncationOptimiser.getAnalysisOptimedParams:'
                            'Should not fall there (fitDistCuts specs)')
            
        if isinstance(fitDistCuts, Interval) and fitDistCuts.min == fitDistCuts.max:
            fitDistCuts = fitDistCuts.min

        # 2. discrDistCuts specs
        if discrDistCutsSpec is None or isinstance(discrDistCutsSpec, (int, float)):
            discrDistCuts = discrDistCutsSpec
        elif isinstance(discrDistCutsSpec, self.AbsInterval):
            discrDistCuts = Interval(min=max(2, discrDistCutsSpec.min), max=discrDistCutsSpec.max)
        elif isinstance(discrDistCutsSpec, self.Auto):
            discrDistCuts = Interval(min=max(2, int(round(self.defDiscrDistCutsFctr.min*sqrNSights))),
                                     max=int(round(self.defDiscrDistCutsFctr.max*sqrNSights)))
        elif isinstance(discrDistCutsSpec, self.MultInterval):
            discrDistCuts = Interval(min=max(2, int(round(discrDistCutsSpec.kmin*sqrNSights))),
                                     max=int(round(discrDistCutsSpec.kmax*sqrNSights)))
        else:
            raise Exception('MCDSTruncationOptimiser.getAnalysisOptimedParams:'
                            'Should not fall there (discrDistCuts specs)')
            
        if isinstance(discrDistCuts, Interval) and discrDistCuts.min == discrDistCuts.max:
            discrDistCuts = discrDistCuts.min

        # Final checks
        finalErr = Error()
        if minDist is not None:
            minDistChk = minDist if isinstance(minDist, Interval) else Interval(minDist, minDist)
            msg = minDistChk.check(order=True, minRange=(0, None), maxRange=(None, sDist.max()))
            if msg:
                finalErr.append(head='minDist', error=msg)

        if maxDist is not None:
            maxDistChk = maxDist if isinstance(maxDist, Interval) else Interval(maxDist, maxDist)
            minMax = None if minDist is None else (minDist.max if isinstance(minDist, Interval) else minDist)
            msg = maxDistChk.check(order=True, minRange=(minMax, None), maxRange=(None, sDist.max()))
            if msg:
                finalErr.append(head='maxDist', error=msg)

        if fitDistCuts is not None:
            fitDistCutsChk = \
                fitDistCuts if isinstance(fitDistCuts, Interval) else Interval(fitDistCuts, fitDistCuts)
            msg = fitDistCutsChk.check(order=True, minRange=(2, None))
            if msg:
                finalErr.append(head='fitDistCuts', error=msg)

        if discrDistCuts is not None:
            discrDistCutsChk = \
                discrDistCuts if isinstance(discrDistCuts, Interval) else Interval(discrDistCuts, discrDistCuts)
            msg = discrDistCutsChk.check(order=True, minRange=(2, None))
            if msg:
                finalErr.append(head='discrDistCuts', error=msg)

        logger.debug(f'OptimedParams: {minDist=}, {maxDist=}, {fitDistCuts=}, {discrDistCuts=}')

        return finalErr or None, \
               {self.IntSpec2SolDimOptimTargetParamNames[self.IntSpecMinDist]: minDist,
                self.IntSpec2SolDimOptimTargetParamNames[self.IntSpecMaxDist]: maxDist,
                self.IntSpec2SolDimOptimTargetParamNames[self.IntSpecFitDistCuts]: fitDistCuts,
                self.IntSpec2SolDimOptimTargetParamNames[self.IntSpecDiscrDistCuts]: discrDistCuts}
                                      
    # Supported truncation optimisation classes (=> engines = cores) (see submodule optimisation),
    # all must be subclasses of MCDSTruncationOptimisation.
    OptimisationClasses = [MCDSZerothOrderTruncationOptimisation] #, MCDSGridBruteTruncationOptimisation]
            
    def getOptimisationCoreParams(self, sAnIntSpec):

        """Retrieve optimisation core parameters, from user specs and default values.
        
        Ex: zoopt(mxi=0, tv=1, a=racos, mrx=3)

        :param sAnIntSpec: analysis parameter user specs with internal names (indexed with IntSpecXXX)
                           syntax: sequence of <key>=<value> separated by ','
                           
        :return: None or an Error instance, dict(core=..., **{key:value}) or None
        """
        
        # Parse optimisation core params in sAnIntSpec if present.
        if self.IntSpecOptimisationCore in sAnIntSpec:
        
            # Retrieve
            userSpec = sAnIntSpec[self.IntSpecOptimisationCore]
                
            # Parse
            parsers = { cls.CoreName: cls.CoreUserSpecParser for cls in self.OptimisationClasses }
            parseError, parsedValue = \
                 self._parseOptimCoreUserSpecs(userSpec, globals=None, locals=parsers,
                                               nullOrEmpty=self.dDefOptimCoreParams,
                                               errIfNotA=[dict])

        # No way: fallback to default values.
        else:
            
            parseError, parsedValue = None, self.dDefOptimCoreParams
                 
        # Done.
        return parseError, parsedValue
    
    def getOptimisationSetupParams(self, sAnIntSpec, sSampleDists):
                               
        """Compute optimisation setup parameters from user specs and default values.
        
        :param sAnIntSpec: analysis parameter user specs with internal names (indexed with IntSpecXXX)
        :param sSampleDists: sample sightings recorded distances
       
        :return: a 2-value tuple (None or an Error instance,
                          dict(minDist=, maxDist=, fitDistCuts=, discrDistCuts=) or None)

        """
        
        # Get params from each of these sets.
        dFinalParms = dict()
        finalError = Error()
        for err, dParms in [self.getAnalysisFixedParams(sAnIntSpec),
                            self.getAnalysisOptimExprParams(sAnIntSpec),
                            self.getAnalysisOptimedParams(sAnIntSpec, sSampleDists),
                            self.getOptimisationCoreParams(sAnIntSpec)]:
            if err is None:
                dFinalParms.update(dParms)
            else:
                finalError.append(err)

        # Any error => empty output params
        if finalError:
            logger.warning('Error(s) while parsing and computing setup params specs: {}'.format(finalError))
         
        return finalError, dFinalParms
        
    def setupOptimisation(self, sampleDataSet, name=None, customData=None, error=None,
                          estimKeyFn=MCDSEngine.EstKeyFnDef, estimAdjustFn=MCDSEngine.EstAdjustFnDef, 
                          estimCriterion=MCDSEngine.EstCriterionDef, cvInterval=MCDSEngine.EstCVIntervalDef,
                          expr2Optimise='chi2', minimiseExpr=False,
                          minDist=None, maxDist=None, fitDistCuts=None, discrDistCuts=None,
                          **dCoreParams):
                          
        """Factory method for MCDSXXXTruncationOptimisation classes"""

        # Search for optimisation class from core name dCoreParams['core']
        # (default to 'zoopt' in case of any error parsing optim core params)
        try:
            OptimionClass = next(iter(cls for cls in self.OptimisationClasses
                                          if cls.CoreName == dCoreParams.get('core', 'zoopt')))
        except StopIteration:
            raise NotImplementedError('No such optimisation core "{}" in house'.format(optimCore))
        
        # Check core params.
        invalidParams = [k for k in dCoreParams if k != 'core' and k not in OptimionClass.CoreParamNames]
        assert not invalidParams, \
               'No such parameter(s) {} for {} ctor'.format(','.join(invalidParams), OptimionClass.__name__)

        # Build remainder of optimisation ctor params.
        dOtherOptimParms = { k: v for k, v in dCoreParams.items() if k in OptimionClass.CoreParamNames }
        
        # Instanciate optimisation.
        optimion = OptimionClass(self._engine, sampleDataSet, name=name,
                                 customData=customData, error=error,
                                 executor=self._executor, logData=self.logData, autoClean=self.autoClean,
                                 estimKeyFn=estimKeyFn, estimAdjustFn=estimAdjustFn,
                                 estimCriterion=estimCriterion, cvInterval=cvInterval,
                                 minDist=minDist, maxDist=maxDist,
                                 fitDistCuts=fitDistCuts, discrDistCuts=discrDistCuts,
                                 expr2Optimise=expr2Optimise, minimiseExpr=minimiseExpr,
                                 **dOtherOptimParms)
                                 
        return optimion

    # Number of optimisation auto-backup files (alternating scheme)
    BackupAltNum = 2

    def _backupFileName(self, index=0):

        return pl.Path(self.workDir) / f'optr-resbak-{index}.pickle.xz'

    def setupResults(self, dfOptimParamSpecs=None, recover=False, loadFrom=None, sheetName=None):
    
        """Build an empty results objects and optionnaly pre-load results
        * from the last auto-backup file, after a crash / interruption of the optimisations,
        * or from the given file name.

        Parameters:
        :param dfOptimParamSpecs: if not None, optim. param. specs for the result object
                                  (overwrite any specs loaded from file; see blow)
        :param recover: if True, try and load the last backup file (see completeResults)
        :param loadFrom: if specified, and not recover, try and load the specified file
                         (see OptimisationResultsSet.fromFile for supported formats, autodetected from file extensions)
        :param sheetName: when loadFrom targets a workbook format file, the name of the sheet to load.
        """
    
        # Build the empty result object
        customCols = \
            self.resultsHeadCols['before'] + self.resultsHeadCols['sample'] + self.resultsHeadCols['after']
        dfCustColTrans = pd.DataFrame(index=customCols, data={ lang: customCols for lang in ['fr', 'en'] })

        self.results = OptimisationResultsSet(optimisationClass=MCDSTruncationOptimisation,
                                                                # TODO: MCDSZerothOrderTruncationOptimisation,
                                              miCustomCols=customCols, dfCustomColTrans=dfCustColTrans)

        # e. Load data from file if resquested to.
        if recover:

            # Check that there will be an analysis index = Id column in results
            assert self.anlysIndCol, "Can't recover optimisation results if no analysis index column specified !"

            # Search for backup files and sort them by modification time in descending order.
            bkupFileNames = [self._backupFileName(ind) for ind in range(self.BackupAltNum)]
            bkupFileNames = [fpn for fpn in bkupFileNames if fpn.is_file()]
            sBkupFileMTimes = pd.Series(index=[fpn.as_posix() for fpn in bkupFileNames], 
                                        data=[fpn.stat().st_mtime for fpn in bkupFileNames])
            assert not sBkupFileMTimes.empty, 'No such backup file found: {}'.format(self._backupFileName('*'))

            sBkupFileMTimes.sort_values(inplace=True, ascending=False)

            for bkupFpn in sBkupFileMTimes.index:
                logger.info('Recovering optimisation results from auto-backup {} ...'.format(bkupFpn))
                try:
                    self.results.fromPickle(bkupFpn, acceptNewCols=True)
                    break
                except Exception as exc:
                    if bkupFpn == sBkupFileMTimes.index[-1]:
                        logger.error('... failed ; no other usable backup file, sorry.')
                        raise
                    else:
                        logger.info('... failed ; trying next ...')

            logger.info('... success !')

            # Check that there IS really an analysis index = Id column in results
            assert self.anlysIndCol in self.results.columns, \
                   "Can't recover from backup results file because no analysis index/Id column found !"

        elif loadFrom:

            self.results.fromFile(fileName=loadFrom, sheetName=sheetName)

        if dfOptimParamSpecs is not None:

            # Set results specs for traceability.
            self.results.updateSpecs(overwrite=True, optimiser=self.flatSpecs(), optimisations=dfOptimParamSpecs)
        
    def optimisationTargetColumnUserNames(self):
    
        """Determine the user names of the results optimisation target columns
        """
   
        # Should not be called before run !
        assert self.results is not None
   
        # From results column names to internal spec name to user spec name.
        return [self.dInt2UserParamSpecNames[self.SolDim2IntSpecOptimTargetParamNames[solDimName]]
                for solDimName in self.results.optimisationTargetColumns()]

    def completeResults(self, dOptims):
    
        """Wait for and gather dOptims (MCDSOptimisation futures) results into a ResultsSet
        
        There should be no reason to specialise this for specific truncation optimisers, but ...
        """
    
        # Start of elapsed time measurement (yes, starting the optimisations may take some time, but it is
        # neglectable when compared to optimisation time ; and better here for evaluating mean per optimisation).
        optimStart = pd.Timestamp.now()
        
        # For each optimisation as it gets completed (first completed => first yielded)
        nDone = 0
        for optimFut in self._executor.asCompleted(dOptims):
            
            # Retrieve optimisation object from its associated future object
            optim = dOptims[optimFut]
            
            # Get analysis results and optimisation target column names in results
            dfResults = optim.getResults()

            # Get custom header values, and set target index (= columns) for results
            sCustomHead = optim.customData
            sCustomHead.index = self.results.miCustomCols

            # Save results (exact column list changes among optimisations objects so we must acceptNewCols)
            self.results.append(dfResults, sCustomHead=sCustomHead, acceptNewCols=True)

            # Backup raw results (with an alternate file scheme for safety) in case of unavoidable crash (or reboot).
            # See setupResults for how these files can be reused later for recovery.
            nDone += 1
            if nDone % self.backupEvery == 0:
                nBackupInd = (nDone // self.backupEvery) % self.BackupAltNum
                self.results.toPickle(fileName=self._backupFileName(nBackupInd), raw=True)

            # Report elapsed time and number of optimisations completed until now.
            if nDone % self.logProgressEvery == 0 or nDone == len(dOptims):
                now = pd.Timestamp.now()
                elapsedTilNow = now - optimStart
                if nDone < len(dOptims):
                    expectedEnd = \
                        now + pd.Timedelta(elapsedTilNow.value * (len(dOptims) - nDone) / nDone)
                    expectedEnd = expectedEnd.strftime('%Y-%m-%d %H:%M:%S').replace(now.strftime('%Y-%m-%d '), '')
                    endOfMsg = 'should end around ' + expectedEnd
                else:
                    endOfMsg = 'done'
                logger.info1('{}/{} optimisations in {} (mean {:.2f}s): {}.'
                             .format(nDone, len(dOptims), str(elapsedTilNow.round('S')).replace('0 days ', ''),
                                     elapsedTilNow.total_seconds() / nDone, endOfMsg))

        # Terminate analysis executor
        self._executor.shutdown()

        # Terminate analysis engine
        self._engine.shutdown()
        
    def run(self, dfExplParamSpecs=None, implParamSpecs=None, threads=None, recover=False):
   
        """Optimise specified analyses
        
        Call checkUserSpecs(...) before this to make sure user specs are OK
        
        Parameters:
        :param dfExplParamSpecs: optimisation params specs table, as a DataFrame
        :param implParamSpecs: Implicit pd.DataFrame and optimisation param specs, suitable for explicitation
                               through Analyser.explicitVariantSpecs
        :param threads: Number of parallel threads to use (default None: no parallelism, no asynchronism)
        :param recover: Recover a previously interrupted run using last available auto-backup file
        """
    
        # Executor for optimisations.
        self._executor = Executor(threads=threads)

        # MCDS analysis engine (a sequential one, because MCDSOptimisation does the parallel stuff itself,
        # but an asynchronous one if execution time limit is to be enforced with os.system run method).

        # Failed try: Seems we can't stack ThreadPoolExecutors, as optimisations get run sequentially
        #             when using an Executor(threads=1) (means async) for self._engine ... 
        #engineExor = None if self.runMethod != 'os.system' or self.runTimeOut is None else Executor(threads=1)
        self._engine = MCDSEngine(workDir=self.workDir, #executor=engineExor,
                                  runMethod=self.runMethod, timeOut=self.runTimeOut,
                                  distanceUnit=self.distanceUnit, areaUnit=self.areaUnit,
                                  surveyType=self.surveyType, distanceType=self.distanceType,
                                  clustering=self.clustering)

        # Custom columns for results.
        customCols = \
            self.resultsHeadCols['before'] + self.resultsHeadCols['sample'] + self.resultsHeadCols['after']
        
        # Explicitate and complete analysis and optimisation specs, and check for usability.
        # (should be also done before calling run, to avoid failure).
        dfExplParamSpecs, userParamSpecCols, intParamSpecCols, _, checkVerdict, checkErrors = \
            self.explicitParamSpecs(implParamSpecs, dfExplParamSpecs, dropDupes=True, check=True)
        assert checkVerdict, 'Error: Analysis / Optimisation params check failed: {}'.format('; '.join(checkErrors))
        
        # Build internal name => user name converter for spec. columns
        self.dInt2UserParamSpecNames = dict(zip(intParamSpecCols, userParamSpecCols))
        
        # Results object construction
        self.setupResults(recover=recover, dfOptimParamSpecs=dfExplParamSpecs)
        recoveredOptims = [] if self.results.empty else list(self.results.dfRawData[self.anlysIndCol].unique())

        # For each optimisation to run :
        runHow = 'in sequence' if threads <= 1 else f'{threads} parallel threads'
        logger.info('Running MCDS truncation optimisations for {} analyses specs ({}) ...'
                    .format(len(dfExplParamSpecs), runHow))
        if recoveredOptims:
            logger.info("... but {} already done and recovered won't be again ...".format(len(recoveredOptims)))
            logger.debug(f'{recoveredOptims=}')

        dOptims = dict()
        for optimInd, (_, sOptimSpec) in enumerate(dfExplParamSpecs.iterrows()):
            
            logger.info('#{}/{}: {} (Id {})'.format(optimInd+1, len(dfExplParamSpecs),
                                                    sOptimSpec[self.abbrevCol], sOptimSpec[self.anlysIndCol]))

            # Skip optimisation if already in results (recovered).
            if sOptimSpec[self.anlysIndCol] in recoveredOptims:
                logger.info('Skipping this one: already present in recovered results')
                continue

            # Select data sample to process (and skip if empty)
            sds = self._mcDataSet.sampleDataSet(sOptimSpec[self.sampleSelCols])
            if not sds:
                continue

            # Build optimisation params specs series with parameters internal names.
            sOptimIntSpec = sOptimSpec[userParamSpecCols].set_axis(intParamSpecCols, inplace=False)
            
            # Compute optimisation setup parameters from user specs and default values.
            setupError, dSetupParams = \
                self.getOptimisationSetupParams(sOptimIntSpec, sds.dfData[self.sampleDistCol])
            
            # Create optimisation object
            logger.debug('Optim. params: {}'.format(', '.join(f'{k}:{v}' for k,v in dSetupParams.items())))
            optim = self.setupOptimisation(sampleDataSet=sds, name=sOptimSpec[self.abbrevCol],
                                           customData=sOptimSpec[customCols].copy(),
                                           error=setupError, **dSetupParams)

            # Compute optimisation submission parameters from user specs and default values.
            submitError, dSubmitParams = self.getOptimisationSubmitParams(sOptimIntSpec)
                                               
            # Submit optimisation (but don't wait for it's finished, go on with next, may run in parallel)
            logger.debug('Submit params: {}'.format(', '.join([f'{k}:{v}' for k,v in dSubmitParams.items()])))
            optimFut = optim.submit(error=submitError, **dSubmitParams)
            
            # Store optimisation object and associated "future" for later use (should be running soon or later).
            dOptims[optimFut] = optim
            
            # Next analysis (loop).

        if self._executor.isAsync():
            logger.info('All optimisations started; now waiting for their end, and results ...')
        else:
            logger.info('All optimisations done; now collecting their results ...')

        # Wait for and gather results of all analyses.
        self.completeResults(dOptims)
        
        # Done.
        logger.info('Optimisations completed ({} analyses => {} results).'
                    .format(int(self.results.dfData.NFunEvals.sum()), len(self.results)))

        return self.results


class MCDSZerothOrderTruncationOptimiser(MCDSTruncationOptimiser):

    """Run a bunch of Zeroth Order MCDS truncation optimisations"""

    def __init__(self, dfMonoCatObs, dfTransects=None, effortConstVal=1, dSurveyArea=dict(), 
                       transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                       sampleSelCols=['Species', 'Pass', 'Adult', 'Duration'], 
                       sampleDecCols=['Effort', 'Distance'], sampleDistCol='Distance', anlysSpecCustCols=[],
                       abbrevCol='AnlysAbbrev', abbrevBuilder=None, anlysIndCol='AnlysNum', sampleIndCol='SampleNum',
                       distanceUnit='Meter', areaUnit='Hectare',
                       surveyType='Point', distanceType='Radial', clustering=False,
                       resultsHeadCols=dict(before=['AnlysNum', 'SampleNum'], after=['AnlysAbbrev'], 
                                            sample=['Species', 'Pass', 'Adult', 'Duration']),
                       workDir='.', runMethod='subprocess.run', runTimeOut=120,
                       logData=False, logProgressEvery=5, backupEvery=50, autoClean=True,
                       defEstimKeyFn=MCDSEngine.EstKeyFnDef, defEstimAdjustFn=MCDSEngine.EstAdjustFnDef,
                       defEstimCriterion=MCDSEngine.EstCriterionDef, defCVInterval=MCDSEngine.EstCVIntervalDef,
                       defExpr2Optimise='chi2', defMinimiseExpr=False,
                       defOutliersMethod='tucquant', defOutliersQuantCutPct=5,
                       defFitDistCutsFctr=Interval(min=2/3, max=3/2),
                       defDiscrDistCutsFctr=Interval(min=1/3, max=1),
                       defSubmitTimes=1, defSubmitOnlyBest=None,
                       defCoreMaxIters=100, defCoreTermExprValue=None, defCoreAlgorithm='racos', defCoreMaxRetries=0):

        super().__init__(dfMonoCatObs=dfMonoCatObs, dfTransects=dfTransects, 
                         effortConstVal=effortConstVal, dSurveyArea=dSurveyArea, 
                         transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                         sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols,
                         sampleDistCol=sampleDistCol, anlysSpecCustCols=anlysSpecCustCols,
                         abbrevCol=abbrevCol, abbrevBuilder=abbrevBuilder,
                         anlysIndCol=anlysIndCol, sampleIndCol=sampleIndCol,
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         surveyType=surveyType, distanceType=distanceType, clustering=clustering,
                         resultsHeadCols=resultsHeadCols,
                         workDir=workDir, runMethod=runMethod, runTimeOut=runTimeOut, logData=logData,
                         logProgressEvery=logProgressEvery, backupEvery=backupEvery, autoClean=autoClean,
                         defExpr2Optimise=defExpr2Optimise, defMinimiseExpr=defMinimiseExpr,
                         defOutliersMethod=defOutliersMethod, defOutliersQuantCutPct=defOutliersQuantCutPct,
                         defFitDistCutsFctr=defFitDistCutsFctr, defDiscrDistCutsFctr=defDiscrDistCutsFctr,
                         defSubmitTimes=defSubmitTimes, defSubmitOnlyBest=defSubmitOnlyBest,
                         dDefOptimCoreParams=dict(core='zoopt', maxIters=defCoreMaxIters, termExprValue=defCoreTermExprValue,
                                                  algorithm=defCoreAlgorithm, maxRetries=defCoreMaxRetries))
                         

if __name__ == '__main__':

    import sys

    print('Nothing done here.')
    
    sys.exit(0)
