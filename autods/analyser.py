# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Analyser: Run a bunch of DS analyses according to a user-friendly set of analysis specs
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment


import re
import pathlib as pl
from packaging import version

from collections import namedtuple as ntuple

import math
import numpy as np
import pandas as pd

import autods

logger = autods.log.logger('ads.anr')

from autods.data import MonoCategoryDataSet, ResultsSet
from autods.executor import Executor
from autods.engine import MCDSEngine
from autods.analysis import DSAnalysis, MCDSAnalysis, MCDSPreAnalysis


class AnalysisResultsSet(ResultsSet):
    
    """
    A result set for multiple analyses from the same engine.
    
    Internal columns index is a 3-level multi-index.
    """
    
    def __init__(self, analysisClass, miCustomCols=None, dfCustomColTrans=None,
                       dComputedCols=None, dfComputedColTrans=None, sortCols=[], sortAscend=[]):
        
        assert issubclass(analysisClass, DSAnalysis), 'analysisClass must derive from DSAnalysis'
        assert miCustomCols is None \
               or (isinstance(miCustomCols, pd.MultiIndex) and len(miCustomCols.levels) == 3), \
               'customCols must be None or a 3 level pd.MultiIndex'
        
        self.analysisClass = analysisClass
        self.engineClass = analysisClass.EngineClass

        # 3-level multi-index columns (module, statistic, figure) for analyses output
        miCols = \
            self.engineClass.statSampCols().append(analysisClass.MIRunColumns).append(self.engineClass.statModCols())
        
        # DataFrame for translating 3-level multi-index columns to 1 level lang-translated columns
        dfColTrans = pd.concat([self.engineClass.statSampColTrans(), analysisClass.DfRunColumnTrans,
                                self.engineClass.statModColTrans()])
        
        super().__init__(miCols=miCols, dfColTrans=dfColTrans,
                         miCustomCols=miCustomCols, dfCustomColTrans=dfCustomColTrans,
                         dComputedCols=dComputedCols, dfComputedColTrans=dfComputedColTrans,
                         sortCols=sortCols, sortAscend=sortAscend)
    
    def copy(self, withData=True):
    
        """Clone function (shallow), with optional (deep) data copy"""
    
        # 1. Call ctor without computed columns stuff (we no more have initial data)
        clone = AnalysisResultsSet(analysisClass=self.analysisClass,
                                   miCustomCols=self.miCustomCols, dfCustomColTrans=self.dfCustomColTrans,
                                   sortCols=[], sortAscend=[])
    
        # 2. Complete clone initialisation.
        # 3-level multi-index columns (module, statistic, figure)
        clone.miCols = self.miCols
        clone.computedCols = self.computedCols
        
        # DataFrames for translating 3-level multi-index columns to 1 level lang-translated columns
        clone.dfColTrans = self.dfColTrans
        
        # Copy data if needed.
        if withData:
            clone._dfData = self._dfData.copy()
            clone.rightColOrder = self.rightColOrder
            clone.postComputed = self.postComputed

        return clone
                

class Analyser(object):

    """Tools for building analysis variants specifications and explicitating them.
    
    Abstract base class for DS analysers.
    """

    def __init__(self):

        # Computation specifications, for traceability only.
        # For gathering copies of computations default parameter values, and stuff like that.
        self.specs = dict()

    def updateSpecs(self, reset=False, overwrite=False, **specs):

        if reset:
            self.specs.clear()

        if not overwrite:
            assert all(name not in self.specs for name in specs), \
                   "Won't overwrite already present specs {}" \
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
        return pd.Series(dFlatSpecs, name='Value')

    # Generation of a table of implicit "partial variant" specification,
    # from a list of possible data selection criteria for each variable.
    # "Partial variant" because its only about a sub-set of variants
    # * dVariants : { target columns name: list of possibles criteria for data selection }
    @staticmethod
    def implicitPartialVariantSpecs(dVariants):
        
        def fixedLengthList(toComplete, length):
            return toComplete + [np.nan]*(length - len(toComplete))

        nRows = max(len(l) for l in dVariants.values())

        return pd.DataFrame({ colName : fixedLengthList(variants, nRows) for colName, variants in dVariants.items() })

    @staticmethod
    def _dropCommentColumns(dfSpecs):
    
        """Drop (in-place) comment columns from a spec dataframe (implicit or explicit)
        """
        
        cols2drop = [col for col in dfSpecs.columns
                     if not col.strip() or col.startswith('Unnamed:')
                        or any(col.strip().lower().startswith(start)
                               for start in ['comm', 'rem', '#'])]

        dfSpecs.drop(columns=cols2drop, inplace=True)        

    @staticmethod
    def explicitPartialVariantSpecs(implSpecs, convertCols=dict()):

        """Generation of a table of explicit "partial variant" specifications, from an implicit one
        (= generate all combinations of variants)
        
        Parameters:
        :param:implSpecs: implicit partial specs object, as a DataFrame, taken as is,
           or a dict, preprocessed through implicitPartialVariantSpecs
        :param:convertCols Name and conversion method for explicit columns to convert 
          (each column is converted through :
           dfExplSpecs[colName] = dfExplSpecs[colName].apply(convertCol)
           for colName, convertCol in convertCols.items()) 
        """
        
        # Convert spec from dict to DataFrame if needed.
        if isinstance(implSpecs, dict):
            dfImplSpecs = Analyser.implicitPartialVariantSpecs(implSpecs)
        else:
            assert isinstance(implSpecs, pd.DataFrame)
            dfImplSpecs = implSpecs.copy()
        
        # Drop any comment / no header ... = useless column
        Analyser._dropCommentColumns(dfImplSpecs)
        
        # First columns kept (nearly) as is (actually an explicit one !) :
        # keep 1 heading NaN if any, drop trailing ones, and drop duplicates if any.
        def cleanup(s):
            return s if s.empty else s.iloc[0:1].append(s.iloc[1:].dropna()).drop_duplicates()
        dfExplSpecs = cleanup(dfImplSpecs[dfImplSpecs.columns[0]]).to_frame()

        # For each implicit specs column (but the first)
        for col in dfImplSpecs.columns[1:]:

            # Get variants :
            # keep 1 heading NaN if any, drop trailing ones, and drop duplicates if any.
            sVariants = cleanup(dfImplSpecs[col])
            
            # Duplicate current explicit table as much as variants are many
            dfExplSpecs = dfExplSpecs.loc[np.repeat(dfExplSpecs.index.to_numpy(), len(sVariants))]
            
            # Add the new columns by tiling the variants along the whole index range
            dfExplSpecs[col] = np.tile(sVariants.to_numpy(), len(dfExplSpecs) // len(sVariants))
            
            # Reset index for easy next duplication
            dfExplSpecs.reset_index(inplace=True, drop=True)

        # Convert specified columns if any
        assert all(colName in dfExplSpecs.columns for colName in convertCols), \
               'Could not find some column to convert of {} in explicit table columns {}' \
               .format(list(convertCols.keys()), list(dfExplSpecs.columns))
        for colName, convertCol in convertCols.items():
            dfExplSpecs[colName] = dfExplSpecs[colName].apply(convertCol)
        
        # Done.
        return dfExplSpecs

    SupportedFileExts = \
        ['.xlsx'] + (['.ods'] if version.parse(pd.__version__).release >= (0, 25) else [])
    
    @classmethod
    def _loadPartSpecsFromFile(cls, sourceFpn):
        
        if isinstance(sourceFpn, str):
            sourceFpn = pl.Path(sourceFpn)
    
        assert sourceFpn.exists(), 'Source file for partial analysis specs not found : {}'.format(sourceFpn)

        ext = sourceFpn.suffix.lower()
        assert ext in cls.SupportedFileExts, \
               'Unsupported source file type {}: not from {{{}}}' \
               .format(ext, ','.join(cls.SupportedFileExts))
        if ext in ['.xlsx']:
            dfData = pd.read_excel(sourceFpn, sheet_name=None)
        elif ext in ['.ods']:
            dfData = pd.read_excel(sourceFpn, sheet_name=None, engine='odf' if ext in ['.ods'] else 'openpyxml')
            
        return dfData
    
    @classmethod
    def explicitVariantSpecs(cls, partSpecs, keep=None, ignore=None,
                             varIndCol=None, convertCols=dict(), computedCols=dict()):
        
        """Generation of a table of explicit variant specifications,
        from a set of implicit and explicit partial variant specs objects
           
        Parameters
        :param partSpecs: The (ordered) dict of name => partial specs objects,
           each as a DataFrame, taken as is, or a dict, preprocessed through implicitPartialVariantSpecs.
           Or: pathname of an Excel (.xlsx) or Open Doc. (.ods) worksheet file (1 sheet per specs table)
           Warning: implicit tables are only found by their name containing "_impl"
        :param keep: List of names of specs to consider from partSpecs (default None => consider all)
        :param ignore: List of names of specs to ignore from partSpecs (default None => ignore none) ;
                       Warning: names in keep and ignore are ... ignored.
        :param varIndCol: Name of the autogenerated variant index column (defaut: None = no such column added)
        :param convertCols: Name and conversion method for explicit columns to convert 
          (each column is converted through :
           dfExplSpecs[colName] = dfExplSpecs[colName].apply(convertCol)
           for colName, convertCol in convertCols.items()) 
        :param:computedCols: Name and computing method for explicit columns to add (after appending :param varIndCol)
          (each column to add is computed through :
           dfExplSpecs[colName] = dfExplSpecs.apply(computeCol, axis='columns')
           for colName, computeCol in computedCols.items()) 

        TODO: Translate to english
        
        partSpecs est donc un dictionnaire ordonné de tables de specs partielles.
        * chaque table donne un sous-ensemble (ou la totalité) des colonnes de variantes d'analyses,
        * chaque table peut être implicite ou explicite :
            * explicite : toutes les combinaisons désirées sont données pour les colonnes de la table,
            * implicite : dans chaque colonne, verticalement, on donne la liste des variantes possibles
              (chaque colonne n'aura donc pas la même longueur) ; l'explicitation consistera à générer
              automatiquement la totalité des combinaisons possibles des valeurs données dans les colonnes,
            * le type des tables est déterminé par leur nom : implicite s'il contient "_impl",
              explicite sinon.
        * plusieurs tables peuvent avoir les mêmes colonnes :
            * chacune peut être soit implicite, soit explicite, peu importe,
            * une fois explicitées, elles doivent cependant fournir des lignes ne se recouvrant pas,
            * cela permet de spécifier facilement des combinaisons différentes pour des sous-ensembles
              distincts de valeurs d'un sous-ensemble donné de colonnes,
            * avant de les combiner avec les autres tables, on les concatène verticalement en 1 seule
              après explicitation individuelle,
        * les tables qui n'ont aucune colonne en commun avec les autres produiront
          l'ensemble des combinaisons possibles avec celles obtenues par explicitation des colonnes la précédant,
        * celles qui ont certaines (mais pas toutes) colonnes en communs avec celles qui la précèdent
          permettent de produire des variantes spécifiques pour ces colonnes : elles feront l'objet
          d'une jointure à gauche avec ces tables précédentes,
        * car l'algorithme d'explicitation - fusion des tables suit leur ordre dans le classeur,
          une fois faite l'explicitation - concaténation verticale des tables de même jeux de colonnes.
          
        N.B. Pas prévu, mais ça marche : pour imposer des valeurs de paramètres vides,
             il suffit de fournit une table vide, avec les entêtes correspondants (exemple avec ACDC).
        """
        
        # Load partial variant specs from source (trivial if given as a dict).
        assert isinstance(partSpecs, (dict, str, pl.Path)), 'Can explicit only worksheet files or alike dicts'
        dPartSpecs = partSpecs if isinstance(partSpecs, dict) else cls._loadPartSpecsFromFile(partSpecs)
        assert len(dPartSpecs) > 0, "Can't explicit variants with no partial variant"
        
        # Setup filters
        keep = keep or list(dPartSpecs.keys())
        ignore = ignore or []
        
        # Filter specs as requested and convert any implicit partial variant spec
        # from dict to DataFrame if needed.
        ddfPartSpecs = dict()
        for psName, psValues in dPartSpecs.items():
            if psName in keep and psName not in ignore:
                if isinstance(psValues, dict):
                    ddfPartSpecs[psName] = cls.implicitPartialVariantSpecs(psValues)
                else:
                    assert isinstance(psValues, pd.DataFrame)
                    ddfPartSpecs[psName] = psValues.copy()
                    # Drop any comment / no header ... = useless column
                    cls._dropCommentColumns(ddfPartSpecs[psName])
                
        # Group partial specs tables with same column sets (according to column names)
        dSameColsPsNames = dict() # { cols: [table names] }
        
        for psName, dfPsValues in ddfPartSpecs.items():
            
            colSetId = ':'.join(sorted(dfPsValues.columns))
            if colSetId not in dSameColsPsNames:
                dSameColsPsNames[colSetId] = list()
                
            dSameColsPsNames[colSetId].append(psName)

        # For each group, concat. tables into one, after expliciting if needed
        ldfExplPartSpecs = list()

        for lPsNames in dSameColsPsNames.values():

            ldfSameColsPartSpecs= list()
            for psName in lPsNames:

                dfPartSpecs = ddfPartSpecs[psName]

                # Implicit specs case:
                if '_impl' in psName:

                    dfPartSpecs = cls.explicitPartialVariantSpecs(dfPartSpecs)

                # Now, specs are explicit.
                ldfSameColsPartSpecs.append(dfPartSpecs)

            # Concat groups of same columns set explicit specs
            ldfExplPartSpecs.append(pd.concat(ldfSameColsPartSpecs, ignore_index=True))
        
        # Combine explicit specs (following in order)
        dfExplSpecs = ldfExplPartSpecs[0]

        for dfExplPartSpecs in ldfExplPartSpecs[1:]:

            commCols = [col for col in dfExplSpecs.columns if col in dfExplPartSpecs.columns]

            if commCols: # Any column in common : left join each left row to each matching right row

                dfExplSpecs = dfExplSpecs.join(dfExplPartSpecs.set_index(commCols), on=commCols)

            else: # No columns in common : combine each left row with all right rows

                nInitSpecs = len(dfExplSpecs)

                dfExplSpecs = dfExplSpecs.loc[np.repeat(dfExplSpecs.index, len(dfExplPartSpecs))]
                dfExplSpecs.reset_index(drop=True, inplace=True)

                dfExplPartSpecs = pd.DataFrame(data=np.tile(dfExplPartSpecs, [nInitSpecs, 1]),
                                               columns=dfExplPartSpecs.columns)

                dfExplSpecs = pd.concat([dfExplSpecs, dfExplPartSpecs], axis='columns')

            dfExplSpecs.reset_index(drop=True, inplace=True)
        
        # Convert specified columns if any
        assert all(colName in dfExplSpecs.columns for colName in convertCols), \
               'Could not find some column to convert of {} in explicit table columns {}' \
               .format(list(convertCols.keys()), list(dfExplSpecs.columns))
        for colName, convertCol in convertCols.items():
            dfExplSpecs[colName] = dfExplSpecs[colName].apply(convertCol)
        
        # Generate explicit variant index column if specified
        if varIndCol:
            dfExplSpecs.reset_index(drop=False, inplace=True)
            dfExplSpecs.rename(columns=dict(index=varIndCol), inplace=True)
            
        # Compute and add supplementary columns if any
        for colName, computeCol in computedCols.items():
            dfExplSpecs[colName] = dfExplSpecs.apply(computeCol, axis='columns')
                
        # Done.
        return dfExplSpecs


class DSAnalyser(Analyser):

    """Run a bunch of DS analyses on samples extracted from an individualised sightings data set,
    according to a user-friendly set of analysis specs,
    + Tools for building analysis variants specifications and explicitating them.
    
    Abstract class.
    """

    def __init__(self, dfMonoCatObs, dfTransects=None, effortConstVal=1, dSurveyArea=dict(), 
                       transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                       sampleSelCols=['Species', 'Pass', 'Adult', 'Duration'],
                       sampleDecCols=['Effort', 'Distance'], anlysSpecCustCols=[],
                       distanceUnit='Meter', areaUnit='Hectare',
                       resultsHeadCols=dict(before=['AnlysNum', 'SampleNum'], after=['AnlysAbbrev'], 
                                            sample=['Species', 'Pass', 'Adult', 'Duration']),
                       abbrevCol='AnlysAbbrev', abbrevBuilder=None,
                       anlysIndCol='AnlysNum', sampleIndCol='SampleNum',
                       workDir='.'):
                       
        """Ctor
        
        Parameters:
        :param pd.DataFrame dfMonoCatObs: mono-category sighting from FieldDataSet.monoCategorise() or individualise()
        :param pd.DataFrame dfTransects: Transects infos with columns : transectPlaceCols (n), passIdCol (1),
            effortCol (1) ; if None, auto generated from input sightings
        :param effortConstVal: if dfTransects is None and effortCol not in source table, use this constant value
        :param dSurveyArea: 
        :param transectPlaceCols: 
        :param passIdCol: 
        :param effortCol: 
        :param sampleSelCols: sample identification = selection columns
        :param sampleDecCols: Decimal columns among sighting columns
        :param anlysSpecCustCols: Special columns from analysis specs to simply pass through in results
        :param distanceUnit: see MCDSEngine
        :param areaUnit: see MCDSEngine
        :param resultsHeadCols: dict of list of column names (from dfMonoCatObs) to use in order
            to build results (right) header columns ; 'sample' columns are sample selection columns ;
            sampleIndCol is added to resultsHeadCols['before'] if not elswehere in resultsHeadCols ;
            same for anlysIndCol, right before sampleIndCol
        :param abbrevCol: Name of column to generate for abbreviating analyses params, not sure really useful ...
        :param abbrevBuilder: Function of explicit analysis params (as a Series) to generate abbreviated name
        :param anlysIndCol: Name of column to generate for identifying analyses, unless already there in input data.
        :param sampleIndCol: Name of column to generate for identifying samples, unless already there in input data.
        :param workDir: Folder where to generate analysis and results files
        """

        assert all(col in resultsHeadCols for col in ['before', 'sample', 'after'])
        assert sampleIndCol

        super().__init__()

        self.dfMonoCatObs = dfMonoCatObs

        self.resultsHeadCols = resultsHeadCols.copy()
        self.abbrevCol = abbrevCol
        self.abbrevBuilder = abbrevBuilder
        self.anlysIndCol = anlysIndCol
        self.sampleSelCols = sampleSelCols
        self.sampleIndCol = sampleIndCol
        self.anlysSpecCustCols = anlysSpecCustCols
        
        # sampleIndCol is added to resultsHeadCols['before'] if not already somewhere in resultsHeadCols
        self.sampIndResHChap = None
        for chap, cols in self.resultsHeadCols.items():
            for col in cols:
                if col == sampleIndCol:
                    self.sampIndResHChap = chap
                    break
        if not self.sampIndResHChap:
            self.sampIndResHChap = 'before'
            self.resultsHeadCols[self.sampIndResHChap] = [sampleIndCol] + self.resultsHeadCols[self.sampIndResHChap]

        # anlysIndCol is added to resultsHeadCols['before'], right at the beginning, if not None
        # and not already somewhere in resultsHeadCols
        if anlysIndCol and not any(anlysIndCol in cols for cols in self.resultsHeadCols.values()):
            self.resultsHeadCols['before'] = [anlysIndCol] + self.resultsHeadCols['before']
        
        self.workDir = workDir

        self.distanceUnit = distanceUnit
        self.areaUnit = areaUnit
        
        # Individualised data (all samples)
        self._mcDataSet = \
            MonoCategoryDataSet(dfMonoCatObs, dfTransects=dfTransects, effortConstVal=effortConstVal,
                                dSurveyArea=dSurveyArea, transectPlaceCols=transectPlaceCols,
                                passIdCol=passIdCol, effortCol=effortCol, sampleDecFields=sampleDecCols)
                                
        # Analysis engine and executor.
        self._executor = None
        self._engine = None
        
        # Results.
        self.results = None

        # Specs.
        self.updateSpecs(**dSurveyArea)
        self.updateSpecs(**{name: getattr(self, name) for name in ['distanceUnit', 'areaUnit']})

    # Possible regexps (values) for auto-detection of analyser _internal_ parameter spec names (keys)
    # from explicit _user_ spec columns
    # (regexps are re.search'ed : any match _anywhere_inside_ the column name is OK;
    #  and case is ignored during searching).
    Int2UserSpecREs = dict()

    @staticmethod
    def userSpec2ParamNames(userSpecCols, int2UserSpecREs, strict=True):

        """
        Retrieve the internal param. names matching with the "user specified" ones
        according to the given regexp dictionary.
        
        Parameters:
        :param userSpecCols: list of user spec. columns
        :param int2UserSpecREs: Possible regexps for internal param. names
        :param strict: if True, raise KeyError if any name in userSpecCols cannot be matched ;
                       if False, will return None for each unmatched internal param. name

        Return: List of internal param. names, same order as userSpecCols,
                with Nones when not found (if not strict)
        """
        logger.debug('Matching user spec. columns:')

        parNames = list()
        for specName in userSpecCols:
            try:
                parName = next(iter(parName for parName in int2UserSpecREs \
                                    if any(re.search(pat, specName, flags=re.IGNORECASE) \
                                           for pat in int2UserSpecREs[parName])))
                logger.debug(f' * "{specName}" => {parName}')
                parNames.append(parName)
            except StopIteration:
                if strict:
                    raise KeyError('Could not match user spec. column "{}" in sample data set columns [{}]'
                                   .format(specName, ', '.join(int2UserSpecREs.keys())))
                else:
                    logger.debug(f' * "{specName}" => Not found')
                    parNames.append(None)

        logger.debug('... success{}.'.format('' if strict else ' but {} mismatches.'.format(parNames.count(None))))

        return parNames

    @staticmethod
    def _explicitParamSpecs(implParamSpecs=None, dfExplParamSpecs=None, int2UserSpecREs=dict(),
                            sampleSelCols=['Species', 'Pass', 'Adult', 'Duration'],
                            abbrevCol='AnlysAbbrev', abbrevBuilder=None, anlysIndCol='AnlysNum',
                            sampleIndCol='SampleNum', anlysSpecCustCols=[], dropDupes=True):
                           
        """Explicitate analysis param. specs if not already done, and complete columns if needed ;
        also automatically extract (regexps) columns which are really analysis parameters,
        with their analyser-internal name, and also their "user" name.
        
        Parameters:
        :param implParamSpecs: Implicit analysis param specs, suitable for explicitation
           through explicitVariantSpecs()
        :param dfExplParamSpecs: Explicit analysis param specs, as a DataFrame
           (generated through explicitVariantSpecs, as an example)
        :param int2UserSpecREs: Possible regexps for internal param. names
        :param sampleSelCols: sample identification = selection columns
        :param abbrevCol: Name of column to generate for abbreviating analyses params, not sure really useful ...
        :param abbrevBuilder: Function of explicit analysis params (as a Series) to generate abbreviated name
        :param anlysIndCol: Name of column to generate for identifying analyses, unless already there in input data.
        :param sampleIndCol: Name of column to generate for identifying samples, unless already there in input data.
        :param anlysSpecCustCols: special columns from analysis specs to simply pass through and ignore
        :param dropDupes: if True, drop duplicates (keep first) in the final explicit DataFrame
           
        :return: Explicit specs as a DataFrame (input dfExplParamSpecs not modified : a new one is returned),
                 list of matched analysis param. columns user names,
                 list of matched analysis param. columns internal names,
                 list of unmatched analysis param. columns user names.
        """
    
        # Explicitate analysis specs if needed (and add computed columns if any and not already there).
        assert dfExplParamSpecs is None or implParamSpecs is None, \
               'Only one of dfExplParamSpecs and paramSpecCols can be specified'
        
        dCompdCols = {abbrevCol: abbrevBuilder} if abbrevCol and abbrevBuilder else {}
        if implParamSpecs is not None:
            dfExplParamSpecs = \
                Analyser.explicitVariantSpecs(implParamSpecs, varIndCol=anlysIndCol,
                                              computedCols=dCompdCols)
        else:
            dfExplParamSpecs = dfExplParamSpecs.copy()
            for colName, computeCol in dCompdCols.items():
                if colName not in dfExplParamSpecs.columns:
                    dfExplParamSpecs[colName] = dfExplParamSpecs.apply(computeCol, axis='columns')
            if anlysIndCol and anlysIndCol not in dfExplParamSpecs.columns:
                dfExplParamSpecs[anlysIndCol] = np.arange(len(dfExplParamSpecs))

        # Add sample index column if requested and not already there
        if sampleIndCol and sampleIndCol not in dfExplParamSpecs.columns:
            # Drop all-NaN sample selection columns (sometimes, it happens) for a working groupby()
            dfSampInd = dfExplParamSpecs[sampleSelCols].dropna(axis='columns', how='all')
            dfExplParamSpecs[sampleIndCol] = \
                dfSampInd.groupby(list(dfSampInd.columns), sort=False).ngroup()

        # Check for columns duplicates : a killer (to avoid weird error later).
        iCols = dfExplParamSpecs.columns
        assert not iCols.duplicated().any(), \
               'Some duplicate column(s) in parameter specs: ' + ' ,'.join(iCols[iCols.duplicated()])

        # Convert explicit. analysis spec. columns to the internal parameter names,
        # and extract the real analysis parameters (None entries when not matched).
        intParamSpecCols = \
            DSAnalyser.userSpec2ParamNames(dfExplParamSpecs.columns, int2UserSpecREs, strict=False)

        # Get back to associated column user names
        # a. matched with internal param. names
        userParamSpecCols = [usp for inp, usp in zip(intParamSpecCols, dfExplParamSpecs.columns) if inp]
        
        # b. unmatched with internal param. names, and assumed to be real user analysis params.
        ignUserParamSpeCols = sampleSelCols.copy()
        if abbrevCol:
            ignUserParamSpeCols += [abbrevCol]
        if sampleIndCol:
            ignUserParamSpeCols += [sampleIndCol]
        if anlysIndCol:
            ignUserParamSpeCols += [anlysIndCol]
        if anlysSpecCustCols:
            ignUserParamSpeCols += anlysSpecCustCols
        unmUserParamSpecCols = [usp for inp, usp in zip(intParamSpecCols, dfExplParamSpecs.columns)
                                if not inp and usp not in ignUserParamSpeCols]

        # Cleanup implicit name list from Nones (strict=False)
        intParamSpecCols = [inp for inp in intParamSpecCols if inp]

        # Drop duplicate specs if specified.
        if dropDupes:
            nBefore = len(dfExplParamSpecs)
            dupDetCols = sampleSelCols + userParamSpecCols
            dfExplParamSpecs.drop_duplicates(subset=dupDetCols, inplace=True)
            dfExplParamSpecs.reset_index(drop=True, inplace=True)
            logger.info('Dropped {} last duplicate specs of {}, on [{}] columns'
                        .format(nBefore - len(dfExplParamSpecs), nBefore, ', '.join(dupDetCols)))

        # Done.
        return dfExplParamSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols

    def explicitParamSpecs(self, implParamSpecs=None, dfExplParamSpecs=None, dropDupes=True, check=False):
    
        """Explicitate analysis param. specs if not already done, and complete columns if needed ;
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
           * list of analysis param. columns internal names,
           * list of analysis param. columns user names,
           if ckeck, 2 more items in the return tuple :
           * check verdict : True if everything went well, False otherwise,
             * some columns from paramSpecCols could not be found in dfExplParamSpecs columns,
             * some user columns could not be matched with some of the expected internal parameter names,
             * some rows are not suitable for DS analysis (empty sample identification columns, ...).
           * check failure reasons : list of strings explaining things that went bad.
        """
        
        # Explicitate and complete
        tplRslt = self._explicitParamSpecs(implParamSpecs, dfExplParamSpecs, self.Int2UserSpecREs,
                                           sampleSelCols=self.sampleSelCols, abbrevCol=self.abbrevCol,
                                           abbrevBuilder=self.abbrevBuilder, anlysIndCol=self.anlysIndCol,
                                           sampleIndCol=self.sampleIndCol, anlysSpecCustCols=self.anlysSpecCustCols,
                                           dropDupes=dropDupes)
        
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
            self._engine.shutdown()
            self._engine = None
        if self._executor:
            self._executor.shutdown()
            self._executor = None
        
#    def __del__(self):
#    
#        self.shutdown()


class MCDSAnalysisResultsSet(AnalysisResultsSet):

    """A specialized results set for MCDS analyses, with extra. post-computed columns : Delta AIC, Chi2 P"""
    
    # Computed Column Labels
    # a. Chi2 determined, Delta AIC, delta DCv
    CLChi2     = ('detection probability', 'chi-square test probability determined', 'Value')
    CLDeltaAic = ('detection probability', 'Delta AIC', 'Value')
    CLDeltaDCv = ('density/abundance', 'density of animals', 'Delta Cv')

    # b. Observation rate and combined quality indicators
    CLSightRate  = ('encounter rate', 'observation rate', 'Value')
    CLCmbQuaBal1 = ('combined quality', 'balanced 1', 'Value')
    CLCmbQuaBal2 = ('combined quality', 'balanced 2', 'Value')
    CLCmbQuaBal3 = ('combined quality', 'balanced 3', 'Value')
    CLCmbQuaChi2 = ('combined quality', 'more Chi2', 'Value')
    CLCmbQuaKS   = ('combined quality', 'more KS', 'Value')
    CLCmbQuaDCv  = ('combined quality', 'more DCv', 'Value')

    # c. Automated filtering and grouping + sorting
    CLCAutoFilSor = 'auto filter sort'  # Label "Chapter" (1st level)
    CLTTruncGroup = 'Group'  # Label "Type" (3rd level)
    CLTSortOrder = 'Order'  # Label "Type" (3rd level)
    CLTPreSelection = 'Pre-selection'  # Label "Type" (3rd level)

    #   i. close truncation group identification
    CLGroupTruncLeft  = (CLCAutoFilSor, MCDSAnalysis.CLParTruncLeft[1], CLTTruncGroup)
    CLGroupTruncRight = (CLCAutoFilSor, MCDSAnalysis.CLParTruncRight[1], CLTTruncGroup)

    #   ii. Order inside groups with same = identical truncation parameters (distances and model cut points)
    CLGrpOrdSmTrAic = (CLCAutoFilSor, 'AIC (same trunc)', CLTSortOrder)

    #   iii. Order inside groups of close truncation distances
    CLGrpOrdClTrChi2KSDCv = (CLCAutoFilSor, 'Chi2 KS DCv (close trunc)', CLTSortOrder)
    #CLGrpOrdClTrChi2 = (CLCAutoFilSor, 'Chi2 (close trunc)', CLTSortOrder)
    CLGrpOrdClTrDCv = (CLCAutoFilSor, 'DCv (close trunc)', CLTSortOrder)
    
    CLGrpOrdClTrQuaBal1 = (CLCAutoFilSor, 'Bal. quality 1 (close trunc)', CLTSortOrder)
    CLGrpOrdClTrQuaBal2 = (CLCAutoFilSor, 'Bal. quality 2 (close trunc)', CLTSortOrder)
    CLGrpOrdClTrQuaBal3 = (CLCAutoFilSor, 'Bal. quality 3 (close trunc)', CLTSortOrder)
    CLGrpOrdClTrQuaChi2 = (CLCAutoFilSor, 'Bal. quality Chi2+ (close trunc)', CLTSortOrder)
    CLGrpOrdClTrQuaKS   = (CLCAutoFilSor, 'Bal. quality KS+ (close trunc)', CLTSortOrder)
    CLGrpOrdClTrQuaDCv  = (CLCAutoFilSor, 'Bal. quality DCv+ (close trunc)', CLTSortOrder)
    
    #   iv. Global order
    CLGblOrdChi2KSDCv = (CLCAutoFilSor, 'Chi2 KS DCv (global)', CLTSortOrder)

    CLGblOrdQuaBal1 = (CLCAutoFilSor, 'Bal. quality 1 (global)', CLTSortOrder)
    CLGblOrdQuaBal2 = (CLCAutoFilSor, 'Bal. quality 2 (global)', CLTSortOrder)
    CLGblOrdQuaBal3 = (CLCAutoFilSor, 'Bal. quality 3 (global)', CLTSortOrder)
    CLGblOrdQuaChi2 = (CLCAutoFilSor, 'Bal. quality Chi2+ (global)', CLTSortOrder)
    CLGblOrdQuaKS   = (CLCAutoFilSor, 'Bal. quality KS+ (global)', CLTSortOrder)
    CLGblOrdQuaDCv  = (CLCAutoFilSor, 'Bal. quality DCv+ (global)', CLTSortOrder)

    CLGblOrdDAicChi2KSDCv = (CLCAutoFilSor, 'DeltaAIC Chi2 KS DCv (global)', CLTSortOrder)

    # Computed columns specs (name translation + position).
    _firstResColInd = len(MCDSEngine.statSampCols()) + len(MCDSAnalysis.MIRunColumns)
    DComputedCols = {CLSightRate: _firstResColInd + 10, # After Encounter Rate / Left|Right Trunc. Dist.
                     CLDeltaAic: _firstResColInd + 12, # Before AIC
                     CLChi2: _firstResColInd + 14, # Before all Chi2 tests 
                     CLDeltaDCv: _firstResColInd + 72, # Before Density of animals / Cv 
                     # And, at the end ...
                     **{cl: -1 for cl in [CLCmbQuaBal1, CLCmbQuaBal2, CLCmbQuaBal3,
                                          CLCmbQuaChi2, CLCmbQuaKS, CLCmbQuaDCv,
                                          CLGroupTruncLeft, CLGroupTruncRight,
                                          CLGrpOrdSmTrAic,
                                          CLGrpOrdClTrChi2KSDCv,  # CLGrpOrdClTrChi2,
                                          CLGrpOrdClTrDCv,
                                          CLGrpOrdClTrQuaBal1, CLGrpOrdClTrQuaBal2, CLGrpOrdClTrQuaBal3,
                                          CLGrpOrdClTrQuaChi2, CLGrpOrdClTrQuaKS, CLGrpOrdClTrQuaDCv,
                                          CLGblOrdChi2KSDCv,
                                          CLGblOrdQuaBal1, CLGblOrdQuaBal2, CLGblOrdQuaBal3,
                                          CLGblOrdQuaChi2, CLGblOrdQuaKS, CLGblOrdQuaDCv,
                                          CLGblOrdDAicChi2KSDCv]}}

    DfComputedColTrans = \
        pd.DataFrame(index=DComputedCols.keys(),
                     data=dict(en=['Obs Rate', 'Delta AIC', 'Chi2 P', 'Delta CoefVar Density',
                                   'Qual Bal 1', 'Qual Bal 2', 'Qual Bal 3',
                                   'Qual Chi2+', 'Qual KS+', 'Qual DCv+',
                                   'Group Left Trunc', 'Group Right Trunc',
                                   'Order Same Trunc AIC',
                                   'Order Close Trunc Chi2 KS DCv', #'Order Close Trunc Chi2',
                                   'Order Close Trunc DCv', 'Order Close Trunc Bal 1 Qual',
                                   'Order Close Trunc Bal 2 Qual', 'Order Close Trunc Bal 3 Qual',
                                   'Order Close Trunc Bal Chi2+ Qual', 'Order Close Trunc Bal KS+ Qual',
                                   'Order Close Trunc Bal DCv+ Qual',
                                   'Order Global Chi2 KS DCv', 'Order Global Bal 1 Qual',
                                   'Order Global Bal 2 Qual', 'Order Global Bal 3 Qual',
                                   'Order Global Bal Chi2+ Qual', 'Order Global Bal KS+ Qual',
                                   'Order Global Bal DCv+ Qual',
                                   'Order Global DeltaAIC Chi2 KS DCv'],
                               fr=['Taux Obs', 'Delta AIC', 'Chi2 P', 'Delta CoefVar Densité',
                                   'Qual Equi 1', 'Qual Equi 2', 'Qual Equi 3',
                                   'Qual Chi2+', 'Qual KS+', 'Qual DCv+',
                                   'Groupe Tronc Gche', 'Groupe Tronc Drte',
                                   'Ordre Tronc Ident AIC',
                                   'Ordre Tronc Proch Chi2 KS DCv', #'Ordre Tronc Proch Chi2',
                                   'Ordre Tronc Proch DCv', 'Ordre Tronc Proch Qual Equi 1',
                                   'Ordre Tronc Proch Qual Equi 2', 'Ordre Tronc Proch Qual Equi 3',
                                   'Ordre Tronc Proch Qual Equi Chi2+', 'Ordre Tronc Proch Qual Equi KS+',
                                   'Ordre Tronc Proch Qual Equi DCv+',
                                   'Ordre Global Chi2 KS DCv', 'Ordre Global Qual Equi 1',
                                   'Ordre Global Qual Equi 2', 'Ordre Global Qual Equi 3',
                                   'Ordre Global Qual Equi Chi2+', 'Ordre Global Qual Equi KS+',
                                   'Ordre Global Qual Equi DCv+',
                                   'Ordre Global DeltaAIC Chi2 KS DCv']))

    # Final-selection column label (empty, for user decision)
    CLFinalSelection = (CLCAutoFilSor, 'Final selection', 'Value')
    DFinalSelColTrans = dict(fr='Sélection finale', en='Final selection')

    # Shortcut to real truncation params columns names.
    DCLParTruncDist = dict(left=MCDSAnalysis.CLParTruncLeft, right=MCDSAnalysis.CLParTruncRight)

    def __init__(self, miCustomCols=None, dfCustomColTrans=None, miSampleCols=None, sampleIndCol=None,
                       sortCols=[], sortAscend=[], distanceUnit='Meter', areaUnit='Hectare',
                       surveyType='Point', distanceType='Radial', clustering=False,
                       ldTruncIntrvSpecs=[dict(col='left', minDist=5.0, maxLen=5.0),
                                          dict(col='right', minDist=25.0, maxLen=25.0)],
                       truncIntrvEpsilon=1e-6):
        
        """
        Parameters:
        :param miSampleCols: columns to use for grouping by sample ; defaults to miCustomCols if None
        :param sampleIndCol: multi-column index for the sample Id column ; no default, must be there !
        """

        assert all(len(self.DfComputedColTrans[lang].dropna()) == len(self.DComputedCols)
                   for lang in self.DfComputedColTrans.columns)

        assert sampleIndCol is not None

        assert all(spec['col'] in self.DCLParTruncDist for spec in ldTruncIntrvSpecs)

        # Initialise base.
        super().__init__(MCDSAnalysis, miCustomCols=miCustomCols, dfCustomColTrans=dfCustomColTrans,
                                       dComputedCols=self.DComputedCols, dfComputedColTrans=self.DfComputedColTrans,
                                       sortCols=sortCols, sortAscend=sortAscend)
        
        if self.CLFinalSelection:
            self.addColumnsTrans({self.CLFinalSelection: self.DFinalSelColTrans})

        # Sample columns
        self.miSampleCols = miSampleCols if miSampleCols is not None else self.miCustomCols
        self.sampleIndCol = sampleIndCol

        # Sample table (auto computed, see listSamples below)
        self.dfSamples = None
    
        # Descriptive parameters, not used in computations actually.
        self.distanceUnit = distanceUnit
        self.areaUnit = areaUnit
        self.surveyType = surveyType
        self.distanceType = distanceType
        self.clustering = clustering

        # Parameters for truncation group intervals post-computations
        self.ldTruncIntrvSpecs = ldTruncIntrvSpecs
        self.truncIntrvEpsilon = truncIntrvEpsilon

        # Cache for filter and sort applied schemes
        self.dFilSorViews = dict()  # Filter and sort scheme name => index of selected rows, scheme steps

    def copy(self, withData=True):
    
        """Clone function, with optional data copy"""
    
        # Create new instance with same ctor params.
        clone = MCDSAnalysisResultsSet(miCustomCols=self.miCustomCols, dfCustomColTrans=self.dfCustomColTrans,
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
            clone.dfSamples = None if dfSample is None else self.dfSamples.copy()
            clone.dFilSorViews = copy.deepcopy(self.dFilSorViews)

        return clone
    
    def onDataChanged(self):

        """React to results data (_dfData) changes that invalidate calculated data,
        but only when the calculus is coded in this class ; other calculi impacts taken care in base classes"""

        self.dfSamples = None
        self.dFilSorViews = dict()

    def dropRows(self, sbSelRows):
    
        super().dropRows(sbSelRows)

        self.onDataChanged()
        
    def setData(self, dfData, postComputed=False):
        
        super().setData(dfData, postComputed=postComputed)

        self.onDataChanged()

    # Get translate names of custom columns
    def transSampleColumns(self, lang):
        
        return self.dfCustomColTrans.loc[self.miSampleCols, lang].to_list()

    # Post-computations : Actual Chi2 value, from multiple tests done.
    MaxChi2Tests = 3 # TODO: Really a constant, or actually depends on some analysis params ?
    CLsChi2All = [('detection probability', 'chi-square test probability (distance set {})'.format(i), 'Value') \
                  for i in range(MaxChi2Tests, 0, -1)]
    
    # Pre-selection column label (from source column) and translation
    def preselectionColumn(self, srcCol):
        return ((srcCol[0], srcCol[1], self.CLTPreSelection),
                dict(fr='Pré-sélection ' + self.transColumn(srcCol, 'fr'),
                     en='Pre-selection ' + self.transColumn(srcCol, 'en')))

    @staticmethod
    def determineChi2Value(sChi2AllDists):
        for chi2 in sChi2AllDists:
            if not np.isnan(chi2):
                return chi2
        return np.nan

    def _postComputeChi2(self):
        
        logger.debug(f'Post-computing actual Chi2:{self.CLChi2}')
        
        # Last value of all the tests done.
        chi2AllColLbls = [col for col in self.CLsChi2All if col in self._dfData.columns]
        if chi2AllColLbls:
            self._dfData[self.CLChi2] = self._dfData[chi2AllColLbls].apply(self.determineChi2Value, axis='columns')

    # Post-computations : Delta AIC/DCv per sampleCols + truncation param. cols group => AIC - min(group).
    CLAic = ('detection probability', 'AIC value', 'Value')
    CLDCv = ('density/abundance', 'density of animals', 'Cv')
    CLsTruncDist = [('encounter rate', 'left truncation distance', 'Value'),
                    ('encounter rate', 'right truncation distance (w)', 'Value')]

    def _postComputeDeltaAicDcv(self):
        
        logger.debug(f'Post-computing Delta AIC/DCv: {self.CLDeltaAic} / {self.CLDeltaDCv}')
        
        # a. Minimum AIC & DCv per group
        #    (drop all-NaN sample selection columns (sometimes, it happens) for a working groupby())
        groupColLbls = self.miSampleCols.append(pd.MultiIndex.from_tuples(self.CLsTruncDist))
        groupColLbls = [col for col in groupColLbls
                        if col in self._dfData.columns and not self._dfData[col].isna().all()]
        df2Join = self._dfData.groupby(groupColLbls)[[self.CLAic, self.CLDCv]].min()
        
        # b. Rename computed columns to target 'Delta XXX'
        df2Join.columns = pd.MultiIndex.from_tuples([self.CLDeltaAic, self.CLDeltaDCv])

        # c. Join the column to the target data-frame
        self._dfData = self._dfData.join(df2Join, on=groupColLbls)

        # d. Compute delta-AIC & DCv in-place
        self._dfData[self.CLDeltaAic] = self._dfData[self.CLAic] - self._dfData[self.CLDeltaAic]
        self._dfData[self.CLDeltaDCv] = self._dfData[self.CLDCv] - self._dfData[self.CLDeltaDCv]

    # Post computations : Usefull columns for quality indicators.
    CLNObs = ('encounter rate', 'number of observations (n)', 'Value')
    CLNTotObs = MCDSEngine.MIStatSampCols[0]
    CLNTotPars = ('detection probability', 'total number of parameters (m)', 'Value')
    CLKS = ('detection probability', 'Kolmogorov-Smirnov test probability', 'Value')
    CLCvMUw = ('detection probability', 'Cramér-von Mises (uniform weighting) test probability', 'Value')
    CLCvMCw = ('detection probability', 'Cramér-von Mises (cosine weighting) test probability', 'Value')
    CLDCv = ('density/abundance', 'density of animals', 'Cv')

    # Post computations : Quality indicators.
    @classmethod
    def _normNObs(cls, sRes):
        return sRes[cls.CLNObs] / sRes[cls.CLNTotObs]

    @classmethod
    def _normNTotPars(cls, sRes, a=0.2, b=0.6, c=2):  #, d=1):
        #return 1 / (a * sRes[cls.CLNTotPars] + b)  # Trop pénalisant: a=0.2, b=1
        return 1 / (a * max(c, sRes[cls.CLNTotPars]) + b)  # Mieux: a=0.2, b=0.6, c=2 / a=0.2, b=0.8, c=1
        #return 1 / (a * max(c, sRes[cls.CLNTotPars])**d + b)  # Idem (d=1)

    @classmethod
    def _normCVDens(cls, sRes, a=12):
        #return max(0, 1 - a * sRes[cls.CLDCv]) # Pas très pénalisant: a=1
        return math.exp(-a * sRes[cls.CLDCv] * sRes[cls.CLDCv]) # Mieux : déjà ~0.33 à 30% (a=12)

    @classmethod
    def _combinedQualityBalanced1(cls, sRes):  # The one used for ACDC 2019 filtering & sorting in jan/feb 2021"""
        return (sRes[[cls.CLChi2, cls.CLKS, cls.CLCvMUw, cls.CLCvMCw]].prod()
                * cls._normNObs(sRes) * cls._normNTotPars(sRes, a=0.2, b=0.6)
                * cls._normCVDens(sRes, a=12)) ** (1.0/7)

    @classmethod
    def _combinedQualityBalanced2(cls, sRes):  # A more devaluating version for NTotPars and CVDens
        return (sRes[[cls.CLChi2, cls.CLKS, cls.CLCvMUw, cls.CLCvMCw]].prod()
                * cls._normNObs(sRes) * cls._normNTotPars(sRes, a=0.2, b=0.8, c=1)
                * cls._normCVDens(sRes, a=16)) ** (1.0/7)

    @classmethod
    def _combinedQualityBalanced3(cls, sRes):  # An even more devaluating version for NTotPars and CVDens
        return (sRes[[cls.CLChi2, cls.CLKS, cls.CLCvMUw, cls.CLCvMCw]].prod()
                * cls._normNObs(sRes) * cls._normNTotPars(sRes, a=0.3, b=0.7, c=1)
                * cls._normCVDens(sRes, a=20)) ** (1.0/7)

    @classmethod
    def _combinedQualityMoreChi2(cls, sRes):
        return (sRes[[cls.CLChi2, cls.CLChi2, cls.CLKS, cls.CLCvMUw, cls.CLCvMCw]].prod()
                * cls._normNObs(sRes) * cls._normNTotPars(sRes, a=0.2, b=0.6)
                * cls._normCVDens(sRes, a=12)) ** (1.0/8)

    @classmethod
    def _combinedQualityMoreKS(cls, sRes):
        return (sRes[[cls.CLChi2, cls.CLKS, cls.CLKS, cls.CLCvMUw, cls.CLCvMCw]].prod()
                * cls._normNObs(sRes) * cls._normNTotPars(sRes, a=0.2, b=0.6)
                * cls._normCVDens(sRes, a=12)) ** (1.0/8)

    @classmethod
    def _combinedQualityMoreDCv(cls, sRes):
        return (sRes[[cls.CLChi2, cls.CLKS, cls.CLCvMUw, cls.CLCvMCw]].prod()
                * cls._normNObs(sRes) * cls._normNTotPars(sRes, a=0.2, b=0.6)
                * cls._normCVDens(sRes, a=12) ** 2) ** (1.0/8)

    def _postComputeQualityIndicators(self):
        
        cls = self

        logger.debug('Post-computing Quality Indicators')

        self._dfData[self.CLSightRate] = 100 * self._dfData.apply(cls._normNObs, axis='columns') # [0,1] => %

        # Prepare data for computations
        logger.debug1('* Pre-processing source data')

        # a. extract the useful columns
        miCompCols = [cls.CLNObs, cls.CLNTotObs, cls.CLNTotPars, 
                      cls.CLChi2, cls.CLKS, cls.CLCvMUw, cls.CLCvMCw, cls.CLDCv]
        dfCompData = self._dfData[miCompCols].copy()

        # b. NaN should kill down these indicators => we have to enforce this !
        # TODO: Activate !
        #dfCompData[[cls.CLNObs, cls.CLChi2, cls.CLKS, cls.CLCvMUw, cls.CLCvMCw]].fillna(0, inplace=True)
        #dfCompData[cls.CLDCv].fillna(1e5, inplace=True)  # Usually considered good under 0.3 ...
        #dfCompData[cls.CLNTotObs].fillna(1e9, inplace=True)  # Should slap down normObs whatever NObs ...
        #dfCompData[cls.CLNTotPars].fillna(1e3, inplace=True)  # Should slap down normNTotPars whatever NObs ...

        logger.debug1('* Balanced quality 1')
        self._dfData[cls.CLCmbQuaBal1] = dfCompData.apply(cls._combinedQualityBalanced1, axis='columns')

        logger.debug1('* Balanced quality 2')
        self._dfData[cls.CLCmbQuaBal2] = dfCompData.apply(cls._combinedQualityBalanced2, axis='columns')

        logger.debug1('* Balanced quality 3')
        self._dfData[cls.CLCmbQuaBal3] = dfCompData.apply(cls._combinedQualityBalanced3, axis='columns')

        logger.debug1('* Balanced quality Chi2+')
        self._dfData[cls.CLCmbQuaChi2] = dfCompData.apply(cls._combinedQualityMoreChi2, axis='columns')

        logger.debug1('* Balanced quality KS+')
        self._dfData[cls.CLCmbQuaKS]   = dfCompData.apply(cls._combinedQualityMoreKS, axis='columns')

        logger.debug1('* Balanced quality DCv+')
        self._dfData[cls.CLCmbQuaDCv]  = dfCompData.apply(cls._combinedQualityMoreDCv, axis='columns')
        
    # List result samples
    def listSamples(self, rebuild=False):

        if rebuild or self.dfSamples is None:

            miSampleCols = self.miSampleCols
            if self.sampleIndCol not in miSampleCols:
                miSampleCols = pd.MultiIndex.from_tuples([self.sampleIndCol]).append(miSampleCols)
            self.dfSamples = self._dfData[miSampleCols]
            self.dfSamples = self.dfSamples.drop_duplicates()
            self.dfSamples.set_index(self.sampleIndCol, inplace=True)
            self.dfSamples.sort_index(inplace=True)
            assert len(self.dfSamples) == self.dfSamples.index.nunique()

        return self.dfSamples

    # Post computations : Quality indicators.
    def _postComputeTruncationGroups(self):
        
        logger.debug('Post-computing Truncation Groups (WARNING: not tested)')

        # For each sample,
        for lblSamp, sSamp in self.listSamples().iterrows():
            
            # Select sample results
            dfSampRes = self._dfData[self._dfData[self.sampleIndCol] == lblSamp]
            logger.debug1('#{} {} : {} rows '
                          .format(lblSamp, ', '.join([f'{k[1]}={v}' for k, v in sSamp.items()]), len(dfSampRes)))

            # For each truncation "method" (left or right)
            for dTrunc in self.ldTruncIntrvSpecs:

                truncCol = self.DCLParTruncDist[dTrunc['col']]
                minIntrvDist = dTrunc['minDist']
                maxIntrvLen = dTrunc['maxLen']

                logger.debug3(f'  - {truncCol[1]}')

                # For some reason, need for enforcing float dtype ... otherwise dtype='O' !?
                sSelDist = dfSampRes[truncCol].dropna().astype(float).sort_values()

                # List non-null differences between consecutive sorted distances
                dfIntrv = pd.DataFrame(dict(dist=sSelDist.values))
                if not dfIntrv.empty:

                    try:
                        dfIntrv['deltaDist'] = dfIntrv.dist.diff()
                        dfIntrv.loc[dfIntrv.dist.idxmin(), 'deltaDist'] = np.inf
                        dfIntrv.dropna(inplace=True)
                        dfIntrv = dfIntrv[dfIntrv.deltaDist > 0].copy()
                    except Exception:  # TODO: Remove this debugging try/except code
                        logger.error(f'_postComputeTruncationGroups(dfIntrv1): dfIntrv.dtypes={dfIntrv.dtypes}')
                        logger.error(f'_postComputeTruncationGroups(dfIntrv1): dfIntrv.dist.dtype={dfIntrv.dist.dtype}')
                        dfIntrv.to_pickle('tmp/anlr-dfIntrv1.pickle')
                        logger.error(f'_postComputeTruncationGroups(dfIntrv1): {dfIntrv:}')
                        raise

                    # Deduce start (min) and end (sup) for each such interval (left-closed, right-open)
                    try:
                        dfIntrv['dMin'] = dfIntrv.loc[dfIntrv.deltaDist > minIntrvDist, 'dist']
                        dfIntrv['dSup'] = dfIntrv.loc[dfIntrv.deltaDist > minIntrvDist, 'dist'].shift(-1).dropna()
                        dfIntrv.loc[dfIntrv['dMin'].idxmax(), 'dSup'] = np.inf
                        dfIntrv.dropna(inplace=True)

                        dfIntrv['dSup'] = \
                            dfIntrv['dSup'].apply(lambda supV: sSelDist[sSelDist < supV].max() + self.truncIntrvEpsilon)
                    except Exception:  # TODO: Remove this debugging try/except code
                        logger.error(f'_postComputeTruncationGroups(dfIntrv2): dfIntrv.dtypes={dfIntrv.dtypes}')
                        logger.error(f'_postComputeTruncationGroups(dfIntrv2): dfIntrv.dist.dtype={dfIntrv.dist.dtype}')
                        dfIntrv.to_pickle('tmp/anlr-dfIntrv2.pickle')
                        logger.error(f'_postComputeTruncationGroups(dfIntrv2): {dfIntrv:}')
                        raise

                    dfIntrv = dfIntrv[['dMin', 'dSup']].reset_index(drop=True)

                    # If these intervals are two wide, cut them up in equal sub-intervals and make them new intervals
                    lsNewIntrvs = list()
                    for _, sIntrv in dfIntrv.iterrows():

                        if sIntrv.dSup - sIntrv.dMin > maxIntrvLen:
                            nSubIntrvs = (sIntrv.dSup - sIntrv.dMin) / maxIntrvLen
                            nSubIntrvs = int(nSubIntrvs) if nSubIntrvs - int(nSubIntrvs) < 0.5 else int(nSubIntrvs) + 1
                            subIntrvLen = (sIntrv.dSup - sIntrv.dMin) / nSubIntrvs
                            lsNewIntrvs += [pd.Series(dict(dMin=sIntrv.dMin + nInd * subIntrvLen, 
                                                           dSup=min(sIntrv.dMin + (nInd + 1) * subIntrvLen,
                                                                    sIntrv.dSup)))
                                            for nInd in range(nSubIntrvs)]
                        else:
                            lsNewIntrvs.append(sIntrv)

                    dfIntrv = pd.DataFrame(lsNewIntrvs).reset_index(drop=True)
                    dfIntrv.sort_values(by='dMin', inplace=True)

                # Update result table : Assign positive interval = "truncation group" number
                # to each truncation distance (special case when no truncation: num=0 if NaN truncation distance)
                sb = (self._dfData[self.sampleIndCol] == lblSamp)
                def truncationIntervalNumber(d):
                    return 0 if pd.isnull(d) else 1 + dfIntrv[(dfIntrv.dMin <= d) & (dfIntrv.dSup > d)].index[0]
                self._dfData.loc[sb, (self.CLCAutoFilSor, truncCol[1], self.CLTTruncGroup)] = \
                    self._dfData.loc[sb, truncCol].apply(truncationIntervalNumber) 

            logger.debug1(f'  => {len(dfSampRes)} rows')
        
    # Post computations : filtering and sorting.
    # a. Column Labels for computed group and sort orders
    #    N.B. "close truncations" means "identical = same truncations" here (but see MCDSOptanalyserResultsSet)
    # See above

    # b. Schemes for computing filtering and sorting keys (see inherited _postComputeFilterSortKeys).
    AutoFilSorKeySchemes = \
    [  # Ordre dans groupe.
       dict(key=CLGrpOrdSmTrAic,  # Meilleur AIC, à troncatures D et G identiques (avec variantes de nb tranches)
            sort=[MCDSAnalysis.CLParTruncLeft, MCDSAnalysis.CLParTruncRight,
                  CLDeltaAic, CLChi2, CLKS, CLDCv, CLNObs, MCDSAnalysis.CLRunStatus],
            ascend=[True, True, True, False, False, True, False, True],
            group=[MCDSAnalysis.CLParTruncLeft, MCDSAnalysis.CLParTruncRight, MCDSAnalysis.CLParModFitDistCuts]),
        
#       dict(key=CLGrpOrdClTrChi2,  # Meilleur Chi2 par groupe de troncatures proches
#            sort=[CLGroupTruncLeft, CLGroupTruncRight,
#                  CLChi2],
#            ascend=[True, True, False],
#            group=[CLGroupTruncLeft, CLGroupTruncRight]),
#       dict(key=CLGrpOrdClTrKS,  # Meilleur KS par groupe de troncatures proches
#            sort=[CLGroupTruncLeft, CLGroupTruncRight,
#                  CLKS],
#            ascend=[True, True, False],
#            group=[CLGroupTruncLeft, CLGroupTruncRight]),
       dict(key=CLGrpOrdClTrDCv,  # Meilleur DCv par groupe de troncatures proches
            sort=[CLGroupTruncLeft, CLGroupTruncRight,
                  CLDCv],
            ascend=[True, True, True],
            group=[CLGroupTruncLeft, CLGroupTruncRight]),
       dict(key=CLGrpOrdClTrChi2KSDCv,  # Meilleur Chi2 & KS & DCv par groupe de troncatures proches
            sort=[CLGroupTruncLeft, CLGroupTruncRight,
                  CLChi2, CLKS, CLDCv, CLNObs, MCDSAnalysis.CLRunStatus],
            ascend=[True, True, False, False, True, False, True],
            group=[CLGroupTruncLeft, CLGroupTruncRight]),
        
       dict(key=CLGrpOrdClTrQuaBal1,  # Meilleur Qualité combinée équilibrée 1 par groupe de troncatures proches
            sort=[CLGroupTruncLeft, CLGroupTruncRight,
                  CLCmbQuaBal1],
            ascend=[True, True, False],
            group=[CLGroupTruncLeft, CLGroupTruncRight]),
       dict(key=CLGrpOrdClTrQuaBal2,  # Meilleur Qualité combinée équilibrée 2 par groupe de troncatures proches
            sort=[CLGroupTruncLeft, CLGroupTruncRight,
                  CLCmbQuaBal2],
            ascend=[True, True, False],
            group=[CLGroupTruncLeft, CLGroupTruncRight]),        
       dict(key=CLGrpOrdClTrQuaBal3,  # Meilleur Qualité combinée équilibrée 3 par groupe de troncatures proches
            sort=[CLGroupTruncLeft, CLGroupTruncRight,
                  CLCmbQuaBal3],
            ascend=[True, True, False],
            group=[CLGroupTruncLeft, CLGroupTruncRight]),
       dict(key=CLGrpOrdClTrQuaChi2,  # Meilleur Qualité combinée Chi2+ par groupe de troncatures proches
            sort=[CLGroupTruncLeft, CLGroupTruncRight,
                  CLCmbQuaChi2],
            ascend=[True, True, False],
            group=[CLGroupTruncLeft, CLGroupTruncRight]),
       dict(key=CLGrpOrdClTrQuaKS,  # Meilleur Qualité combinée KS+ par groupe de troncatures proches
            sort=[CLGroupTruncLeft, CLGroupTruncRight,
                  CLCmbQuaKS],
            ascend=[True, True, False],
            group=[CLGroupTruncLeft, CLGroupTruncRight]),
       dict(key=CLGrpOrdClTrQuaDCv,  # Meilleur Qualité combinée DCv+ par groupe de troncatures proches
            sort=[CLGroupTruncLeft, CLGroupTruncRight,
                  CLCmbQuaDCv],
            ascend=[True, True, False],
            group=[CLGroupTruncLeft, CLGroupTruncRight]),
        
       # Ordres globaux (sans groupage par troncatures id. ou proches)
       dict(key=CLGblOrdChi2KSDCv,
            sort=[CLChi2, CLKS, CLDCv, CLNObs, MCDSAnalysis.CLRunStatus],
            ascend=[False, False, True, False, True]),
       dict(key=CLGblOrdQuaBal1,
            sort=[CLCmbQuaBal1], ascend=False),
       dict(key=CLGblOrdQuaBal2,
            sort=[CLCmbQuaBal2], ascend=False),
       dict(key=CLGblOrdQuaBal3,
            sort=[CLCmbQuaBal3], ascend=False),
       dict(key=CLGblOrdQuaChi2,
            sort=[CLCmbQuaChi2], ascend=False),
       dict(key=CLGblOrdQuaKS,
            sort=[CLCmbQuaKS], ascend=False),
       dict(key=CLGblOrdQuaDCv,
            sort=[CLCmbQuaDCv], ascend=False),

       dict(key=CLGblOrdDAicChi2KSDCv,
            sort=[MCDSAnalysis.CLParTruncLeft, MCDSAnalysis.CLParTruncRight, MCDSAnalysis.CLParModFitDistCuts,
                  CLDeltaAic, CLChi2, CLKS, CLDCv, CLNObs, MCDSAnalysis.CLRunStatus],
            ascend=[True, True, True, True, False, False, True, False, True], napos='first'),
    ]

    # Enforce unicity of keys in filter and sort key specs.
    assert len(AutoFilSorKeySchemes) == len(set(scheme['key'] for scheme in AutoFilSorKeySchemes)), \
           'Duplicated scheme key in MCDSAnalysisResultsSet.AutoFilSorKeySchemes'

    # Enforce unicity of sort and group column in filter and sort key specs.
    assert all(len(scheme['sort']) == len(set(scheme['sort'])) for scheme in AutoFilSorKeySchemes), \
           'Duplicated sort column spec in some scheme of MCDSAnalysisResultsSet.AutoFilSorKeySchemes'
    assert all(len(scheme.get('group', [])) == len(set(scheme.get('group', []))) for scheme in AutoFilSorKeySchemes), \
           'Duplicated group column spec in some scheme of MCDSAnalysisResultsSet.AutoFilSorKeySchemes'

    # Check sort vs ascend list lengths in filter and sort key specs.
    assert all(isinstance(scheme['ascend'], bool) or len(scheme['ascend']) == len(scheme['sort'])
               for scheme in AutoFilSorKeySchemes), \
           'Unconsistent ascend vs sort specs in some scheme of MCDSAnalysisResultsSet.AutoFilSorKeySchemes'

    # c. Computation of filter and sort keys

    # Make results cell values hashable (needed for sorting in _postComputeFilterSortKeys)
    @staticmethod
    def _toSortable(value):

        if isinstance(value, list):
            return len(value)
        elif isinstance(value, (int, float, str)) or pd.isnull(value):
            return value
        else:
            raise NotImplementedError

    DCLUnsortableCols = \
        {MCDSAnalysis.CLParModFitDistCuts:
            (MCDSAnalysis.CLParModFitDistCuts[0], MCDSAnalysis.CLParModFitDistCuts[1], 'Sortable'),
         MCDSAnalysis.CLParModDiscrDistCuts:
            (MCDSAnalysis.CLParModDiscrDistCuts[0], MCDSAnalysis.CLParModDiscrDistCuts[1], 'Sortable')}

    # Make results cell values hashable (needed for grouping in _postComputeFilterSortKeys)
    @staticmethod
    def _toHashable(value):
    
        if isinstance(value, list):
            return ','.join(str(v) for v in value)
        elif isinstance(value, (int, float, str)) or pd.isnull(value):
            return value
        else:
            return str(value)

    DCLUnhashableCols = \
        {MCDSAnalysis.CLParModFitDistCuts:
            (MCDSAnalysis.CLParModFitDistCuts[0], MCDSAnalysis.CLParModFitDistCuts[1], 'Hashable'),
         MCDSAnalysis.CLParModDiscrDistCuts:
            (MCDSAnalysis.CLParModDiscrDistCuts[0], MCDSAnalysis.CLParModDiscrDistCuts[1], 'Hashable')}

    def _postComputeFilterSortKeys(self):
        
        """Compute and add partial or global order columns for later filtering and sorting"""

        cls = self

        logger.debug('Post-computing Filter and Sort keys')

        for lblSamp, sSamp in self.listSamples().iterrows():

            # Select sample data
            dfSampRes = self._dfData[self._dfData[self.sampleIndCol] == lblSamp].copy()
            logger.debug1('#{} {} : {} rows '
                          .format(lblSamp, ', '.join([f'{k[1]}={v}' for k, v in sSamp.items()]), len(dfSampRes)))

            # Apply each filter and sort scheme
            for scheme in cls.AutoFilSorKeySchemes:

                logger.debug3('* {}'.format(scheme))
                dfSampRes.to_pickle('tmp/_.pickle')

                # Workaround to-be-sorted problematic columns.
                sortCols = list()
                for col in scheme['sort']:
                    if col in cls.DCLUnsortableCols:
                        wkrndSortCol = cls.DCLUnsortableCols[col]
                        logger.debug4('{} => {}'.format(col, wkrndSortCol))
                        dfSampRes[wkrndSortCol] = dfSampRes[col].apply(cls._toSortable)
                        col = wkrndSortCol  # Will rather sort with this one !
                    sortCols.append(col)
                #print(sortCols)

                # Sort results
                dfSampRes.sort_values(by=sortCols, ascending=scheme['ascend'], 
                                      na_position=scheme.get('napos', 'last'), inplace=True)

                # Compute order (target series is indexed like dfSampRes).
                if 'group' in scheme:  # Partial = 'group' order.

                    # Workaround to-be-grouped problematic columns.
                    groupCols = list()
                    for col in scheme['group']:
                        if col in cls.DCLUnhashableCols:
                            wkrndGroupCol = cls.DCLUnhashableCols[col]
                            logger.debug4('{} => {}'.format(col, wkrndGroupCol))
                            dfSampRes[wkrndGroupCol] = dfSampRes[col].apply(cls._toHashable)
                            col = wkrndGroupCol  # Will rather group with this one !
                        groupCols.append(col)
                    #print(groupCols)

                    sSampOrder = dfSampRes.groupby(groupCols, dropna=False).cumcount()

                else:  # Global order.
                    sSampOrder = pd.Series(data=range(len(dfSampRes)), index=dfSampRes.index)

                # Update result table sample rows (new order column)
                self._dfData.loc[self._dfData[self.sampleIndCol] == lblSamp, scheme['key']] = sSampOrder
    
    # Post-computations : All of them.
    def postComputeColumns(self):
        
        self._postComputeChi2()
        self._postComputeDeltaAicDcv()
        self._postComputeQualityIndicators()
        self._postComputeTruncationGroups()
        self._postComputeFilterSortKeys()

    # Tools for actually filtering results
    @classmethod
    def indexOfDuplicates(cls, dfRes, keep='first', subset=list(), round2decs=dict()):
        
        if round2decs:
            #dfRes = dfRes.round(round2decs) # Buggy (pandas 1.0.x up to 1.1.2): forgets columns !?!?!?
            if len(subset) > 0:
                pass  # TODO: Optimise = only copy subset cols
            dfRes = dfRes.copy()
            for col, dec in round2decs.items():
                if len(subset) == 0 or col in subset:
                    dfRes[col] = dfRes[col].apply(lambda x: x if pd.isnull(x) else round(x, ndigits=dec))

            # Don't use df.round ... because it does not work, at least with pandas 1.0.x up to 1.1.2 !?!?!?
            #df = df.round(decimals={ col: dec for col, dec in self.trEnColNames(dColDecimals).items() \
            #                                  if col in df.columns })
            
        return dfRes[dfRes.duplicated(keep=keep, subset=subset)].index

    @classmethod
    def filterDichotScheme(cls, dfRes, sampleIds, sampleIdCol, critCol=CLCmbQuaBal1, ascendCrit=True,
                            minCritStep=0.001, nMinRes=10, verbose=False):
        
        """Fonction générique de filtrage avec stratégie de contrôle du nombre de résultats conservés
        via un schéma adaptatif dichotomique de seuillage sur 1 critère (fonction de son domaine réel de valeurs)
        """
        
        # For each sample ...
        i2Drop = []
        for sampId in sampleIds:

            # Extract results.
            dfSampRes = dfRes[dfRes[sampleIdCol] == sampId]
            if verbose: print('#{}: {} results'.format(sampId, len(dfSampRes)), end=' => ')

            # Compute criteria threshold variation scheme from actual value domain
            start = dfSampRes[critCol].max() if ascendCrit else dfSampRes[critCol].min()
            stop = dfSampRes[critCol].min() if ascendCrit else dfSampRes[critCol].max()
            if verbose: print(f'{critCol} [{start:.3f},{stop:.3f}]', end=': ')

            # No need for tweeking criteria thresholds, we won't get more results.
            if len(dfSampRes) <= nMinRes:
                if verbose: print('t={:.3f}/k={}'.format(stop, len(dfSampRes)), end=', ')
                if verbose: print('done, no more possible.')
                continue
            
            # For each step of the scheme ...
            i2DropSamp, thresh = [], start
            while True:
                
                # Next try : middle of the interval to explore.
                threshTry = (start + stop) / 2

                # Try and apply the threshold step : number of dropped results if ...
                if ascendCrit:
                    i2DropSampTry = dfSampRes[dfSampRes[critCol] < threshTry].index
                else:
                    i2DropSampTry = dfSampRes[dfSampRes[critCol] > threshTry].index

                if verbose: print('t={:.3f}/k={}'.format(threshTry, len(dfSampRes) - len(i2DropSampTry)), end=', ')

                # Stop here if the min number expected of results would be reached
                if len(dfSampRes) - len(i2DropSampTry) == nMinRes:
                    i2DropSamp, thresh = i2DropSampTry, threshTry
                    if verbose: print('done, target reached.')
                    break
                    
                # Stop when no change in list to drop and above the min number expected of results.
                elif len(i2DropSampTry) == len(i2DropSamp) and abs(start - stop) < minCritStep:
                    if verbose: print('done, no more change.')
                    break
                                
                # Update criteria interval to explore according to whether we would be
                #  below or above the min number expected of results if ...
                if len(dfSampRes) - len(i2DropSampTry) > nMinRes:
                    if ascendCrit:
                        stop = threshTry
                    else:
                        start = threshTry
                else:
                    if ascendCrit:
                        start = threshTry
                    else:
                        stop = threshTry
                        
                # Or else, save current try, and go on.
                i2DropSamp, thresh = i2DropSampTry, threshTry

            # Append index to drop for sample to the final one
            i2Drop = i2DropSamp if not len(i2Drop) else i2Drop.append(i2DropSamp)
     
        return i2Drop

    CLEffort = ('encounter rate', 'effort (L or K or T)', 'Value')
    CLPDetec = ('detection probability', 'probability of detection (Pw)', 'Value')
    CLPDetecMin = ('detection probability', 'probability of detection (Pw)', 'Lcl')
    CLPDetecMax = ('detection probability', 'probability of detection (Pw)', 'Ucl')
    CLDensity = ('density/abundance', 'density of animals', 'Value')
    CLDensityMin = ('density/abundance', 'density of animals', 'Lcl')
    CLDensityMax = ('density/abundance', 'density of animals', 'Ucl')

    LDupSubsetDef = [CLNObs, CLEffort, CLDeltaAic, CLChi2, CLKS, CLCvMUw, CLCvMCw, CLDCv, 
                     CLPDetec, CLPDetecMin, CLPDetecMax, CLDensity, CLDensityMin, CLDensityMax]
    DDupRoundsDef = {CLDeltaAic: 1, CLChi2: 2, CLKS: 2, CLCvMUw: 2, CLCvMCw: 2, CLDCv: 2, 
                     CLPDetec: 3, CLPDetecMin: 3, CLPDetecMax: 3, CLDensity: 2, CLDensityMin: 2, CLDensityMax: 2}

    @classmethod
    def _logFilterSortStep(cls, filSorSteps, scheme, step, propName, propValue):

        filSorSteps.append([scheme, step, propName, propValue])
        logger.debug2(f'* {step}: {propName} = {propValue}')

    @staticmethod
    def filterSortSchemeId(scheme=None, idFmt=None, **filterSort):
        """Unique (in memory) identification of a filter and sort scheme, with a human readable part :-)"""
        if scheme:
            name = scheme['idFmt'].format_map(scheme.get('filterSort', {})).replace('.', '')
        else:
            name = idFmt.format_map(filterSort).replace('.', '')
        return name + '@' + np.base_repr(id(scheme), 36).lower()

    def filterSortOnExecCode(self, schemeId, dupSubset=LDupSubsetDef, dDupRounds=DDupRoundsDef):

        """Minimal filter (drop ExecCode >= 3) and truncation distances + balanced quality 1 sorting ;
        doesn't actually modifies a single bit of the results set, but returns the resulting filtered and sorted index,
        suitable for indexing on self.dfData / dfTransData ...

        Parameters:
        :param schemeId: Scheme identification, for traceability
        :param dupSubset: Subset of (3-level multi-index) columns for detecting duplicates (as a list of tuples)
                          Warning: self.sampleIndCol is automatically prepended to this list if not already inside
        :param dDupRounds: {col: nb decimals} => number of decimals to keep (after rounding)
                           for a sub-set or all of dupSubset columns

        :return: tuple(index of selected and sorted results, log of filter & sort steps accomplished)
        """

        cls = self

        logger.debug(f'Filter and sort scheme "{schemeId}": Applying.')

        filSorSteps = list()

        dfFilSorRes = self.getData(copy=True)
        cls._logFilterSortStep(filSorSteps, schemeId, 'before', 'results', len(dfFilSorRes))

        dfFilSorRes.drop(dfFilSorRes[dfFilSorRes[MCDSAnalysis.CLRunStatus] > 2].index,
                         inplace=True)
        cls._logFilterSortStep(filSorSteps, schemeId, MCDSAnalysis.CLRunStatus[1], 'max', 2)
        cls._logFilterSortStep(filSorSteps, schemeId, MCDSAnalysis.CLRunStatus[1], 'results', len(dfFilSorRes))

        dfFilSorRes.sort_values(by=[self.sampleIndCol, MCDSAnalysis.CLParTruncLeft, MCDSAnalysis.CLParTruncRight,
                                    MCDSAnalysis.CLRunStatus],
                                ascending=True, na_position='first', inplace=True)
        if self.sampleIndCol not in dupSubset:
            dupSubset = [self.sampleIndCol] + dupSubset
        dfFilSorRes.drop(cls.indexOfDuplicates(dfFilSorRes, keep='first', subset=dupSubset, round2decs=dDupRounds),
                         inplace=True)
        cls._logFilterSortStep(filSorSteps, schemeId, 'duplicates', 'results', len(dfFilSorRes))

        sortCols = [MCDSAnalysis.CLParTruncLeft, MCDSAnalysis.CLParTruncRight, cls.CLGrpOrdClTrQuaBal1]
        dfFilSorRes.sort_values(by=[self.sampleIndCol] + sortCols,
                                ascending=True, na_position='first', inplace=True)
        cls._logFilterSortStep(filSorSteps, schemeId, 'sorting', 'columns', ', '.join(miCol[1] for miCol in sortCols))

        return dfFilSorRes.index, filSorSteps

    def filterSortOnAicCKCvQua(self, schemeId, sightRate=95, nBestAIC=2, nBestQua=1, nResults=10,
                               dupSubset=LDupSubsetDef, dDupRounds=DDupRoundsDef):

        """Filtrage et tri proche de 1 mais moins méchant, pour action manuelles de filtrage a posteriori
        doesn't actually modifies a single bit of the results set, but returns the resulting filtered and sorted index,
        suitable for indexing on self.dfData / dfTransData ...

        TODO: translate + document parameters

        1. Eliminer CodEx 3 et +,
        2. Par groupe de troncatures Gche et Drte et nb tranches fitting identiques,
           garder les <nBestAIC> meilleurs AIC & Chi2 & KS & DCv & NObs & CodEx,
        3. Par groupe de troncatures Gche et Drte proches
          (algo. de groupage à seuils, analyses optim / non optim séparées), garder :
            * les <nBestQua> meilleur Chi2 & KS & DCv & NObs & CodEx,
            * les <nBestQua> meilleur DCv & Chi2 & KS & NObs & CodEx,
            * les <nBestQua> meilleur indicateurQualitéCombiné(Chi2, KS, DCv, NObs, CodEx),
        4. Garder les Taux d'obs conservés >= <sightRate> %,
        5. Garder les <nResults> meilleurs résultats selon indicateurQualitéCombiné(Chi2, KS, DCv, NObs, CodEx),
        6. Trier par absence / simplicité des troncatures (sans < sans gche < sans drte < avec gche et dte)
           et ce même indicateur.

        Parameters:
        :param schemeId: Scheme identification, for traceability
        :param sightRate: Minimal observation rate (ratio of NTot Obs / NObs, not 1 because of dist. truncations)
        :param nBestAIC: Nb of best AIC results to keep per sample and group of close truncations
        :param nBestQua: Nb of best quality to keep per sample and group of close truncations
                         (quality: see above ???)
        :param nResults: Nb of best Bal Qua1 results to keep per sample
        :param dupSubset: Subset of (3-level multi-index) columns for detecting duplicates (as a list of tuples)
                          Warning: self.sampleIndCol is automatically prepended to this list if not already inside
        :param dDupRounds: {col: nb decimals} => number of decimals to keep (after rounding)
                           for a sub-set or all of dupSubset columns

        :return: tuple(index of selected and sorted results, log of filter & sort steps accomplished)
        """

        cls = self

        logger.debug(f'Filter and sort scheme "{schemeId}": Applying.')

        filSorSteps = list()

        dfFilSorRes = self.getData(copy=True)
        cls._logFilterSortStep(filSorSteps, schemeId, 'before', 'results', len(dfFilSorRes))

        dfFilSorRes.drop(dfFilSorRes[dfFilSorRes[MCDSAnalysis.CLRunStatus] > 2].index, inplace=True)
        cls._logFilterSortStep(filSorSteps, schemeId, MCDSAnalysis.CLRunStatus[1], 'max', 2)
        cls._logFilterSortStep(filSorSteps, schemeId, MCDSAnalysis.CLRunStatus[1], 'results', len(dfFilSorRes))

        dfFilSorRes.sort_values(by=[self.sampleIndCol, MCDSAnalysis.CLParTruncLeft, MCDSAnalysis.CLParTruncRight,
                                    MCDSAnalysis.CLRunStatus],
                                ascending=True, na_position='first', inplace=True)
        if self.sampleIndCol not in dupSubset:
            dupSubset = [self.sampleIndCol] + dupSubset
        dfFilSorRes.drop(cls.indexOfDuplicates(dfFilSorRes, keep='first', subset=dupSubset, round2decs=dDupRounds),
                         inplace=True)
        cls._logFilterSortStep(filSorSteps, schemeId, 'duplicates', 'results', len(dfFilSorRes))

        dfFilSorRes.drop(dfFilSorRes[dfFilSorRes[cls.CLGrpOrdSmTrAic] >= nBestAIC].index,
                         inplace=True)
        cls._logFilterSortStep(filSorSteps, schemeId, cls.CLGrpOrdSmTrAic[1], 'nb best', nBestAIC)
        cls._logFilterSortStep(filSorSteps, schemeId, cls.CLGrpOrdSmTrAic[1], 'results', len(dfFilSorRes))

        # TODO: Make used list of indicators customisable
        dfFilSorRes.drop(dfFilSorRes[(dfFilSorRes[cls.CLGrpOrdClTrChi2KSDCv] >= nBestQua)
                                     & (dfFilSorRes[cls.CLGrpOrdClTrDCv] >= nBestQua)
                                     & (dfFilSorRes[cls.CLGrpOrdClTrQuaBal1] >= nBestQua)
                                     & (dfFilSorRes[cls.CLGrpOrdClTrQuaChi2] >= nBestQua)
                                     & (dfFilSorRes[cls.CLGrpOrdClTrQuaKS] >= nBestQua)
                                     & (dfFilSorRes[cls.CLGrpOrdClTrQuaDCv] >= nBestQua)].index,
                                   # & (dfFilSorRes[cls.CLGrpOrdClTrChi2] > 0)].index,
                         inplace=True)
        cls._logFilterSortStep(filSorSteps, schemeId, 'best CKCv+CVDens+QualEqui+Chi2+KS+DCv (close trunc)',
                               'max (sup) number', nBestQua)
        cls._logFilterSortStep(filSorSteps, schemeId, 'best CKCv+CVDens+QualEqui+Chi2+KS+DCv (close trunc)',
                               'results', len(dfFilSorRes))

        dfFilSorRes.drop(dfFilSorRes[dfFilSorRes[cls.CLSightRate] < sightRate].index, inplace=True)
        cls._logFilterSortStep(filSorSteps, schemeId, 'non-outlier sightings', 'min %', sightRate)
        cls._logFilterSortStep(filSorSteps, schemeId, 'non-outlier sightings', 'actual number', len(dfFilSorRes))

        dfFilSorRes.drop(cls.filterDichotScheme(dfFilSorRes, critCol=cls.CLCmbQuaBal1, ascendCrit=True, nMinRes=nResults,
                                                sampleIds=dfFilSorRes[self.sampleIndCol].unique(),
                                                sampleIdCol=self.sampleIndCol),
                         inplace=True)
        cls._logFilterSortStep(filSorSteps, schemeId, 'best Bal1 Quality results', 'target per sample', nResults)
        cls._logFilterSortStep(filSorSteps, schemeId, 'best Bal1 Quality results',
                               'actual total number', len(dfFilSorRes))

        sortCols = [MCDSAnalysis.CLParTruncLeft, MCDSAnalysis.CLParTruncRight, cls.CLGrpOrdClTrQuaBal1]
        dfFilSorRes.sort_values(by=[self.sampleIndCol] + sortCols, ascending=True, na_position='first', inplace=True)
        cls._logFilterSortStep(filSorSteps, schemeId, 'sorting', 'columns', ', '.join(miCol[1] for miCol in sortCols))

        return dfFilSorRes.index, filSorSteps

    def _addPreselColumns(self, dfFilSorRes, filSorSteps, filSorSchId,
                          preselCols=[CLCmbQuaBal1], preselAscend=True, nSamplePreSels=5):

        """Add (in-place) a pre-selection column to a filtered and sorted translated results table

        Parameters:
        :param dfFilSorRes: the filtered and sorted table to update
        :param filSorSteps: the filtered and sorted step log to update
        :param filSorSchId: the filter and sort scheme Id (step-logging only)
        :param nSamplePreSels: Number of generated pre-selections per sample
        :param preselCols: Results columns to use for generating auto-preselection indices (in [1, nSamplePreSels])
        :param preselAscend: Order to use for each column (list), or all (single bool)
        """

        if isinstance(preselAscend, bool):
            preselAscend = [preselAscend for i in range(len(preselCols))]
        assert len(preselCols) == len(preselAscend), \
               'preselAscend must be a single bool or a list(bool) with len(preselCols)'

        self._logFilterSortStep(filSorSteps, filSorSchId, 'Auto-preselection',
                                'Nb of preselections', nSamplePreSels)

        # Create each pre-selection column: rank per sample in preselCol/ascending (or not) order
        # up to nSamplePreSels.
        for srcCol, srcColAscend in zip(preselCols, preselAscend):

            # Determine label and translation.
            tgtPreSelCol, dTgtPreSelColTrans = self.preselectionColumn(srcCol)
            self.addColumnsTrans({tgtPreSelCol: dTgtPreSelColTrans})

            # Compute contents and add to table
            dfFilSorRes.insert(dfFilSorRes.columns.get_loc(srcCol), tgtPreSelCol,
                               dfFilSorRes.groupby(self.miSampleCols.to_list())[[srcCol]] \
                                          .transform(lambda s: s.rank(ascending=srcColAscend,
                                                                      method='dense', na_option='keep'))[srcCol])
            dfFilSorRes.loc[dfFilSorRes[tgtPreSelCol] > nSamplePreSels, tgtPreSelCol] = np.nan

            self._logFilterSortStep(filSorSteps, filSorSchId, 'Auto-preselection',
                                    'Pre-selection column', srcCol)

        # Create final empty selection column (for the user to self-decide at the end)
        # (right before the first added pre-selection column, no choice)
        if len(preselCols) > 0:
            dfFilSorRes.insert(dfFilSorRes.columns.get_loc(preselCols[0]) - 1, self.CLFinalSelection, np.nan)

        return dfFilSorRes

    def dfFilSorData(self, scheme=[dict(idFmt='ExecCode', method=filterSortOnExecCode,
                                        preselCols=[CLCmbQuaBal1], preselAscs=False, preselNum=5)],
                     columns=None, lang=None, rebuild=False):

        """Extract filtered and sorted data following the given scheme

        Note: Let R be MCDSAnalysisResultsSet, or a subclass (needed below).
        
        Parameters:
        :param filSorSchemes: filter and sort scheme to apply
                 as a list of dict(idFmt= format string for generating the Id of the report
                                   method= ResClass.filterSortOnXXX method to use,
                                   deduplicate= dict(dupSubset=, dDupRounds=) of deduplication params
                                       (if not or partially given, see RCLS.filterSortOnXXX defaults)
                                   filterSort= dict of other <method> params,
                                   preselCols= target columns for generating auto-preselection ones,
                                               containing [1, preselNum] ranks ; default: []
                                   preselAscs= Rank direction to use for each column (list),
                                                  or all (single bool) ; default: True
                                   preselNum= number of (best) preselections to keep for each sample) ;
                                              default: 5
                 examples: dict(idFmt='ExecCode', => format string to generate the name of the report
                                method=R.filterSortOnExecCode,
                                preselCols=[R.CLCmbQuaBal1, R.CLCmbQuaBal2], preselAscs=False, preselNum=5),
                           dict(idFmt='AicCKCvQua-r{sightRate:.1f}d{nResults}', 
                                method=R.filterSortOnAicCKCvQua,
                                deduplicate=dict(dupSubset=[R.CLNObs, R.CLEffort, R.CLDeltaAic, R.CLChi2,
                                                            R.CLKS, R.CLCvMUw, R.CLCvMCw, R.CLDCv]),
                                                 dDupRounds={R.CLDeltaAic: 1, R.CLChi2: 2, R.CLKS: 2,
                                                             R.CLCvMUw: 2, R.CLCvMCw: 2, R.CLDCv: 2})
                                filterSort=dict(sightRate=92.5, nBestAIC=3, nBestQua=1, nResults=12),
                                preselCols=[R.CLCmbQuaBal1, R.R.CLDCv], preselAscs=[False, True], preselNum=3)
        :param columns: Subset and order of columns to keep at the end (before translation) (None = [] = all)
                       Warning: No need to specify here pre-selection and final selection columns,
                                as they'll be added automatically, and relocated at a non-customisable place.
        :param lang: Target language for column name translation (if None, no translation => keep original names)
        :param rebuild: If True, rebuild filtered and sorted table ; otherwise, simply reuse cached data
               if the results set didn't change enough meanwhile.

        :return: scheme id, result table, log of completed filter and sort steps 
        """

        # Apply filter and sort scheme if not already done or needed or user-required
        # => index of filtered and sorted rows.
        filSorSchId = self.filterSortSchemeId(scheme)
        if rebuild or not self.postComputed or filSorSchId not in self.dFilSorViews:
        
            # Apply scheme method => index of filtered and sorted results + log of steps
            iFilSor, filSorSteps = \
                scheme['method'](self, schemeId=filSorSchId,
                                 **scheme.get('filterSort', {}), **scheme.get('deduplicate', {}))
            self.dFilSorViews[filSorSchId] = iFilSor, filSorSteps

        else:

            # Get from cache.
            iFilSor, filSorSteps = self.dFilSorViews[filSorSchId]

            logger.debug(f'Filter and sort scheme "{filSorSchId}": Re-using cached results.')

        # Actually extract filtered and sorted rows and selected columns.
        dfFilSorRes = self.dfSubData(index=iFilSor, columns=columns, copy=True)

        # Add the preselection column (and update filter and sort log)
        dfFilSorRes = self._addPreselColumns(dfFilSorRes, filSorSteps, filSorSchId,
                                             nSamplePreSels=scheme.get('preselNum', 5),
                                             preselCols=scheme.get('preselCols', []), 
                                             preselAscend=scheme.get('preselAscs', True))

        # Final translation if specified.
        if lang:
            dfFilSorRes.columns = self.transColumns(dfFilSorRes.columns, lang)

        # Done.
        return filSorSchId, dfFilSorRes, filSorSteps

class MCDSAnalyser(DSAnalyser):

    """Run a bunch of MCDS analyses
    """

    def __init__(self, dfMonoCatObs, dfTransects=None, effortConstVal=1, dSurveyArea=dict(), 
                 transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                 sampleSelCols=['Species', 'Pass', 'Adult', 'Duration'],
                 sampleDecCols=['Effort', 'Distance'], anlysSpecCustCols=[],
                 distanceUnit='Meter', areaUnit='Hectare',
                 surveyType='Point', distanceType='Radial', clustering=False,
                 resultsHeadCols=dict(before=['AnlysNum', 'SampleNum'], after=['AnlysAbbrev'], 
                                      sample=['Species', 'Pass', 'Adult', 'Duration']),
                 ldTruncIntrvSpecs=[dict(col='left', minDist=5.0, maxLen=5.0),
                                    dict(col='right', minDist=25.0, maxLen=25.0)], truncIntrvEpsilon=1e-6,
                 abbrevCol='AnlysAbbrev', abbrevBuilder=None, anlysIndCol='AnlysNum', sampleIndCol='SampleNum',
                 workDir='.', runMethod='subprocess.run', runTimeOut=300, logData=False, logProgressEvery=50,
                 defEstimKeyFn=MCDSEngine.EstKeyFnDef, defEstimAdjustFn=MCDSEngine.EstAdjustFnDef,
                 defEstimCriterion=MCDSEngine.EstCriterionDef, defCVInterval=MCDSEngine.EstCVIntervalDef,
                 defMinDist=MCDSEngine.DistMinDef, defMaxDist=MCDSEngine.DistMaxDef, 
                 defFitDistCuts=MCDSEngine.DistFitCutsDef, defDiscrDistCuts=MCDSEngine.DistDiscrCutsDef):

        """Ctor

        Parameters:
        :param anlysSpecCustCols: Special columns from analysis specs to simply pass through in results
        :param runTimeOut: time-out for every analysis run (s) (None => no limit)
        :param runMethod: for calling MCDS engine executable : 'os.system' or 'subprocess.run'
        :param timeOut: engine call time limit (s) ; None => no limit ;
        :param ldTruncIntrvSpecs: separation and length specs for truncation group intervals computation
                                  (postComputations for automated results filtering and sorting)
        :param truncIntrvEpsilon: epsilon for truncation group intervals computation (idem)
        """

        assert distanceUnit == 'Meter', 'Not implemented: Only "Meter" distance unit supported for the moment'

        super().__init__(dfMonoCatObs=dfMonoCatObs, dfTransects=dfTransects, 
                         effortConstVal=effortConstVal, dSurveyArea=dSurveyArea, 
                         transectPlaceCols=transectPlaceCols, passIdCol=passIdCol,
                         effortCol=effortCol, sampleSelCols=sampleSelCols,
                         sampleDecCols=sampleDecCols, anlysSpecCustCols=anlysSpecCustCols,
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         resultsHeadCols=resultsHeadCols, abbrevCol=abbrevCol, abbrevBuilder=abbrevBuilder,
                         anlysIndCol=anlysIndCol, sampleIndCol=sampleIndCol, workDir=workDir)
                         
        assert logProgressEvery > 0, 'logProgressEvery must be positive'

        self.surveyType = surveyType
        self.distanceType = distanceType
        self.clustering = clustering
        
        self.runMethod = runMethod
        self.runTimeOut = runTimeOut
        self.logData = logData
        self.logProgressEvery = logProgressEvery

        self.ldTruncIntrvSpecs = ldTruncIntrvSpecs
        self.truncIntrvEpsilon = truncIntrvEpsilon
        
        self.defEstimKeyFn = defEstimKeyFn
        self.defEstimAdjustFn = defEstimAdjustFn
        self.defEstimCriterion = defEstimCriterion
        self.defCVInterval = defCVInterval
        self.defMinDist = defMinDist
        self.defMaxDist = defMaxDist
        self.defFitDistCuts = defFitDistCuts
        self.defDiscrDistCuts = defDiscrDistCuts
                         
        # Specs.
        self.updateSpecs(**{name: getattr(self, name)
                            for name in ['runMethod', 'runTimeOut', 'surveyType', 'distanceType', 'clustering',
                                         'defEstimKeyFn', 'defEstimAdjustFn', 'defEstimCriterion', 'defCVInterval',
                                         'defMinDist', 'defMaxDist', 'defFitDistCuts', 'defDiscrDistCuts']})

    # Analyser internal parameter spec names, for which a match should be found (when one is needed)
    # with user explicit optimisation specs used in run() calls.
    IntSpecEstimKeyFn = 'EstimKeyFn'
    IntSpecEstimAdjustFn = 'EstimAdjustFn'
    IntSpecEstimCriterion = 'EstimCriterion'
    IntSpecCVInterval = 'CvInterval'
    IntSpecMinDist = 'MinDist' # Left truncation distance
    IntSpecMaxDist = 'MaxDist' # Right truncation distance
    IntSpecFitDistCuts = 'FitDistCuts'
    IntSpecDiscrDistCuts = 'DiscrDistCuts'

    # Possible regexps (values) for auto-detection of analyser _internal_ parameter spec names (keys)
    # from explicit _user_ spec columns
    # (regexps are re.search'ed : any match _anywhere_inside_ the column name is OK;
    #  and case is ignored during searching).
    Int2UserSpecREs = \
      {IntSpecEstimKeyFn:     ['ke[a-z]*[\.\-_ ]*f', 'f[o]?n[a-z]*[\.\-_ ]*cl'],
       IntSpecEstimAdjustFn:  ['ad[a-z]*[\.\-_ ]*s', 's[éa-z]*[\.\-_ ]*aj'],
       IntSpecEstimCriterion: ['crit[èa-z]*[\.\-_ ]*'],
       IntSpecCVInterval:     ['conf[a-z]*[\.\-_ ]*[a-z]*[\.\-_ ]*int',
                               'int[a-z]*[\.\-_ ]*conf'],
       IntSpecMinDist:        ['min[a-z]*[\.\-_ ]*d', 'd[a-z]*[\.\-_ ]*min',
                               'tr[a-z]*[\.\-_ ]*g[ca]', 'le[a-z]*[\.\-_ ]*tr'],
       IntSpecMaxDist:        ['max[a-z]*[\.\-_ ]*d', 'd[a-z]*[\.\-_ ]*max',
                               'tr[a-z]*[\.\-_ ]*d[rt]', 'le[a-z]*[\.\-_ ]*tr'],
       IntSpecFitDistCuts:    ['fit[a-z]*[\.\-_ ]*d', 'tr[a-z]*[\.\-_ ]*[a-z]*[\.\-_ ]*mod'],
       IntSpecDiscrDistCuts:  ['disc[a-z]*[\.\-_ ]*d', 'tr[a-z]*[\.\-_ ]*[a-z]*[\.\-_ ]*disc']}

    # Analysis object ctor parameter names (MUST match exactly: check in analysis submodule !).
    ParmEstimKeyFn = 'estimKeyFn'
    ParmEstimAdjustFn = 'estimAdjustFn'
    ParmEstimCriterion = 'estimCriterion'
    ParmCVInterval = 'cvInterval'
    ParmMinDist = 'minDist'
    ParmMaxDist = 'maxDist'
    ParmFitDistCuts = 'fitDistCuts'
    ParmDiscrDistCuts = 'discrDistCuts'

    def _getAnalysisParams(self, sAnIntSpec):
    
        """Retrieve analysis parameters, from user specs and default values
        
        :param sAnIntSpec: analysis parameter user specs with internal names (indexed with IntSpecXXX)
        
        :return: dict(estimKeyFn=, estimAdjustFn=, estimCriterion=, cvInterval=,
                      minDist=, maxDist=, fitDistCuts=, discrDistCuts=)
        """
        return {self.ParmEstimKeyFn: sAnIntSpec.get(self.IntSpecEstimKeyFn, self.defEstimKeyFn),
                self.ParmEstimAdjustFn: sAnIntSpec.get(self.IntSpecEstimAdjustFn, self.defEstimAdjustFn),
                self.ParmEstimCriterion: sAnIntSpec.get(self.IntSpecEstimCriterion, self.defEstimCriterion),
                self.ParmCVInterval: sAnIntSpec.get(self.IntSpecCVInterval, self.defCVInterval),
                self.ParmMinDist: sAnIntSpec.get(self.IntSpecMinDist, self.defMinDist),
                self.ParmMaxDist: sAnIntSpec.get(self.IntSpecMaxDist, self.defMaxDist),
                self.ParmFitDistCuts: sAnIntSpec.get(self.IntSpecFitDistCuts, self.defFitDistCuts),
                self.ParmDiscrDistCuts: sAnIntSpec.get(self.IntSpecDiscrDistCuts, self.defDiscrDistCuts)}

    DAnlr2ResChapName = dict(before='header (head)', sample='header (sample)', after='header (tail)')

    def prepareResultsColumns(self):
        
        # a. Sample multi-index columns
        sampleSelCols = self.resultsHeadCols['sample']
        sampMCols = [(self.DAnlr2ResChapName['sample'], col, 'Value') for col in sampleSelCols]
        miSampCols = pd.MultiIndex.from_tuples(sampMCols)

        # b. Full custom multi-index columns to append and prepend to raw analysis results
        beforeCols = self.resultsHeadCols['before']
        custMCols = [(self.DAnlr2ResChapName['before'], col, 'Value') for col in beforeCols]
        custMCols += sampMCols
        
        afterCols = self.resultsHeadCols['after']
        custMCols += [(self.DAnlr2ResChapName['after'], col, 'Value') for col in afterCols]

        customCols = beforeCols + sampleSelCols + afterCols
        miCustCols = pd.MultiIndex.from_tuples(custMCols)

        # c. Translation for it (well, no translation actually ... only one language forced for all !)
        dfCustColTrans = pd.DataFrame(index=miCustCols, data={lang: customCols for lang in ['fr', 'en']})

        # d. The 3-columns index for the sample index column
        sampIndMCol = (self.DAnlr2ResChapName[self.sampIndResHChap], self.sampleIndCol, 'Value')

        # e. And finally, the result object (sorted at the end by the analysis or else sample index column)
        if self.anlysIndCol or self.sampleIndCol:
            sortCols = [next(mCol for mCol in custMCols if mCol[1] == self.anlysIndCol or self.sampleIndCol)]
        else:
            sortCols = []
        sortAscend = [True for col in sortCols]

        return miCustCols, dfCustColTrans, miSampCols, sampIndMCol, sortCols, sortAscend

    def setupResults(self):
    
        """Build an empty results objects.
        """
    
        miCustCols, dfCustColTrans, miSampCols, sampIndMCol, sortCols, sortAscend = \
            self.prepareResultsColumns()
        
        return MCDSAnalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                      miSampleCols=miSampCols, sampleIndCol=sampIndMCol,
                                      sortCols=sortCols, sortAscend=sortAscend,
                                      distanceUnit=self.distanceUnit, areaUnit=self.areaUnit,
                                      surveyType=self.surveyType, distanceType=self.distanceType,
                                      clustering=self.clustering,
                                      ldTruncIntrvSpecs=self.ldTruncIntrvSpecs,
                                      truncIntrvEpsilon=self.truncIntrvEpsilon)
    
    def _getResults(self, dAnlyses):
    
        """Wait for and gather dAnalyses (MCDSAnalysis futures) results into a MCDSAnalysisResultsSet 
        """
    
        # Start of elapsed time measurement (yes, starting the analyses may take some time, but it is
        # neglectable when compared to analysis time ; and better here for evaluating mean per analysis).
        anlysStart = pd.Timestamp.now()
        
        # Create results container.
        results = self.setupResults()

        # For each analysis as it gets completed (first completed => first yielded)
        nDone = 0
        for anlysFut in self._executor.asCompleted(dAnlyses):
            
            # Retrieve analysis object from its associated future object
            anlys = dAnlyses[anlysFut]
            
            # Get analysis results
            sResult = anlys.getResults()

            # Get custom header values, and set target index (= columns) for results
            sCustomHead = anlys.customData
            sCustomHead.index = results.miCustomCols

            # Save results
            results.append(sResult, sCustomHead=sCustomHead)

            # Report elapsed time and number of analyses completed until now (once per self.logProgressEvery analyses though).
            nDone += 1
            if nDone % self.logProgressEvery == 0 or nDone == len(dAnlyses):
                now = pd.Timestamp.now()
                elapsedTilNow = now - anlysStart
                if nDone < len(dAnlyses):
                    expectedEnd = \
                        now + pd.Timedelta(elapsedTilNow.value * (len(dAnlyses) - nDone) / nDone)
                    expectedEnd = expectedEnd.strftime('%Y-%m-%d %H:%M:%S').replace(now.strftime('%Y-%m-%d '), '')
                    endOfMsg = 'should end around ' + expectedEnd
                else:
                    endOfMsg = 'done'
                logger.info1('{}/{} analyses in {} (mean {:.3f}s): {}.'
                             .format(nDone, len(dAnlyses), str(elapsedTilNow.round('S')).replace('0 days ', ''),
                                     elapsedTilNow.total_seconds() / nDone, endOfMsg))

        # Terminate analysis executor
        self._executor.shutdown()

        # Terminate analysis engine
        self._engine.shutdown()

        return results

    def run(self, dfExplParamSpecs=None, implParamSpecs=None, threads=None):
    
        """Run specified analyses
        
        Call explicitParamSpecs(..., check=True) before this to make sure user specs are OK
        
        Parameters:
        :param dfExplParamSpecs: Explicit MCDS analysis param specs, as a DataFrame
          (generated through explicitVariantSpecs, as an example),
        :param implParamSpecs: Implicit MCDS analysis param specs, suitable for explicitation
          through explicitVariantSpecs
        :param threads: Number of parallel threads to use (default None: no parallelism, no asynchronism)
        """
    
        # Executor (parallel or sequential).
        self._executor = Executor(threads=threads)

        # MCDS analysis engine
        self._engine = MCDSEngine(workDir=self.workDir, executor=self._executor,
                                  runMethod=self.runMethod, timeOut=self.runTimeOut,
                                  distanceUnit=self.distanceUnit, areaUnit=self.areaUnit,
                                  surveyType=self.surveyType, distanceType=self.distanceType,
                                  clustering=self.clustering)

        # Custom columns for results.
        customCols = \
            self.resultsHeadCols['before'] + self.resultsHeadCols['sample'] + self.resultsHeadCols['after']
        
        # Explicitate and complete analysis specs, and check for usability
        # (should be also done before calling run, to avoid failure).
        dfExplParamSpecs, userParamSpecCols, intParamSpecCols, _, checkVerdict, checkErrors = \
            self.explicitParamSpecs(implParamSpecs, dfExplParamSpecs, dropDupes=True, check=True)
        assert checkVerdict, 'Analysis params check failed: {}'.format('; '.join(checkErrors))
        
        # For each analysis to run :
        runHow = 'in sequence' if threads <= 1 else f'{threads} parallel threads'
        logger.info('Running {} MCDS analyses ({}) ...'.format(len(dfExplParamSpecs), runHow))
        dAnlyses = dict()
        for anInd, sAnSpec in dfExplParamSpecs.iterrows():
            
            logger.info(f'#{anInd+1}/{len(dfExplParamSpecs)} : {sAnSpec[self.abbrevCol]}')

            # Select data sample to process
            sds = self._mcDataSet.sampleDataSet(sAnSpec[self.sampleSelCols])
            if not sds:
                continue

            # Build analysis params specs series with parameters internal names.
            sAnIntSpec = sAnSpec[userParamSpecCols].set_axis(intParamSpecCols, inplace=False)
            
            # Get analysis parameters from user specs and default values.
            dAnlysParams = self._getAnalysisParams(sAnIntSpec)
            
            # Analysis object
            logger.debug('Anlys params: {}'.format(', '.join(f'{k}:{v}' for k,v in dAnlysParams.items())))
            anlys = MCDSAnalysis(engine=self._engine, sampleDataSet=sds, name=sAnSpec[self.abbrevCol],
                                 customData=sAnSpec[customCols].copy(), logData=self.logData, **dAnlysParams)

            # Start running pre-analysis in parallel, but don't wait for it's finished, go on
            anlysFut = anlys.submit()
            
            # Store pre-analysis object and associated "future" for later use (should be running soon or later).
            dAnlyses[anlysFut] = anlys
            
            # Next analysis (loop).

        logger.info('All analyses started ; now waiting for their end, and results ...')

        # Wait for and gather results of all analyses.
        self.results = self._getResults(dAnlyses)

        # Set results specs for traceability.
        self.results.updateSpecs(analyses=dfExplParamSpecs, analyser=self.flatSpecs(),
                                 runtime=pd.Series(autods.runtime, name='Version'))
        
        # Done.
        logger.info(f'Analyses completed ({len(self.results)} results).')

        return self.results


class MCDSPreAnalysisResultsSet(MCDSAnalysisResultsSet):

    """A specialized results set for MCDS pre-analyses
    (simpler post-computations that base class MCDSAnalysisResultsSet)"""
    
    # Computed columns specs (name translation + position).
    Super = MCDSAnalysisResultsSet
    _firstResColInd = len(MCDSEngine.statSampCols()) + len(MCDSAnalysis.MIRunColumns)
    DComputedCols = {Super.CLSightRate: _firstResColInd + 10, # After Encounter Rate / Left|Right Trunc. Dist.
                     Super.CLDeltaAic: _firstResColInd + 12, # Before AIC
                     Super.CLChi2: _firstResColInd + 14, # Before all Chi2 tests 
                     Super.CLDeltaDCv: _firstResColInd + 72, # Before Density of animals / Cv 
                     # And, at the end ...
                     **{cl: -1 for cl in [Super.CLCmbQuaBal1, Super.CLCmbQuaBal2, Super.CLCmbQuaBal3,
                                          Super.CLCmbQuaChi2, Super.CLCmbQuaKS, Super.CLCmbQuaDCv]}}

    DfComputedColTrans = \
        pd.DataFrame(index=DComputedCols.keys(),
                     data=dict(en=['Obs Rate', 'Delta AIC', 'Chi2 P', 'Delta CoefVar Density',
                                   'Qual Bal 1', 'Qual Bal 2', 'Qual Bal 3',
                                   'Qual Chi2+', 'Qual KS+', 'Qual DCv+'],
                               fr=['Taux Obs', 'Delta AIC', 'Chi2 P', 'Delta CoefVar Densité',
                                   'Qual Equi 1', 'Qual Equi 2', 'Qual Equi 3',
                                   'Qual Chi2+', 'Qual KS+', 'Qual DCv+']))

    # Needed presence in base class, but use inhibited.
    CLFinalSelection = None

    def __init__(self, miCustomCols=None, dfCustomColTrans=None, miSampleCols=None, sampleIndCol=None,
                       sortCols=[], sortAscend=[], distanceUnit='Meter', areaUnit='Hectare',
                       surveyType='Point', distanceType='Radial', clustering=False):
        
        """
        Parameters:
        :param miSampleCols: columns to use for grouping by sample ; defaults to miCustomCols if None
        :param sampleIndCol: multi-column index for the sample Id column ; no default, must be there !
        """

        # Initialise base.
        super().__init__(miCustomCols=miCustomCols, dfCustomColTrans=dfCustomColTrans,
                         miSampleCols=miSampleCols, sampleIndCol=sampleIndCol,
                         sortCols=sortCols, sortAscend=sortAscend,
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         surveyType=surveyType, distanceType=distanceType, clustering=clustering)

    # Post-computations.
    def postComputeColumns(self):
        
        self._postComputeChi2()
        self._postComputeDeltaAicDcv()
        self._postComputeQualityIndicators()


# Default strategy for pre-analyses model choice sequence (if one fails, take next in order, and so on)
ModelEstimCritDef = 'AIC'
ModelCVIntervalDef = 95
ModelStrategyDef = [dict(keyFn=kf, adjSr='COSINE', estCrit=ModelEstimCritDef, cvInt=ModelCVIntervalDef) \
                     for kf in['HNORMAL', 'HAZARD', 'UNIFORM', 'NEXPON']]

# MCDSPreAnalyser: Run a bunch of MCDS pre-analyses
class MCDSPreAnalyser(MCDSAnalyser):

    def __init__(self, dfMonoCatObs, dfTransects=None, effortConstVal=1, dSurveyArea=dict(), 
                 transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                 sampleSelCols=['Species', 'Pass', 'Adult', 'Duration'], sampleDecCols=['Effort', 'Distance'],
                 sampleSpecCustCols=[], abbrevCol='SampAbbrev', abbrevBuilder=None, sampleIndCol='SampleNum',
                 distanceUnit='Meter', areaUnit='Hectare',
                 surveyType='Point', distanceType='Radial', clustering=False,
                 resultsHeadCols=dict(before=['SampleNum'], after=['SampleAbbrev'], 
                                      sample=['Species', 'Pass', 'Adult', 'Duration']),
                 workDir='.', runMethod='subprocess.run', runTimeOut=300, logProgressEvery=5):

        """Ctor
        
        Parameters:
        :param dfExplParamSpecs: Explicit sample specs, as a DataFrame
          (generated through explicitVariantSpecs, as an example),
        :param implParamSpecs: Implicit sample specs, suitable for explicitation
          through explicitVariantSpecs
        :param dModelStrategy: Sequence of fallback models to use when analyses fails.
        :param threads: Number of parallel threads to use (default None: no parallelism, no asynchronism)
        :param runMethod: for calling MCDS engine executable : 'os.system' or 'subprocess.run'
        :param timeOut: engine call time limit (s) ; None => no limit ;
                WARNING: Not implemented (no way) for 'os.system' run method (no solution found)
        """

        super().__init__(dfMonoCatObs=dfMonoCatObs, dfTransects=dfTransects, 
                         effortConstVal=effortConstVal, dSurveyArea=dSurveyArea, 
                         transectPlaceCols=transectPlaceCols, passIdCol=passIdCol, effortCol=effortCol,
                         sampleSelCols=sampleSelCols, sampleDecCols=sampleDecCols,
                         anlysSpecCustCols=sampleSpecCustCols,
                         abbrevCol=abbrevCol, abbrevBuilder=abbrevBuilder, sampleIndCol=sampleIndCol,
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         surveyType=surveyType, distanceType=distanceType, clustering=clustering,
                         resultsHeadCols=resultsHeadCols, anlysIndCol=None, 
                         workDir=workDir, runMethod=runMethod, runTimeOut=runTimeOut,
                         logProgressEvery=logProgressEvery)

        assert runTimeOut is None or runMethod != 'os.system', \
               f"Can't care about {runTimeOut}s execution time limit with os.system run method (not implemented)"

    def setupResults(self):
    
        """Build an empty results objects.
        """
    
        miCustCols, dfCustColTrans, miSampCols, sampIndMCol, sortCols, sortAscend = \
            self.prepareResultsColumns()
        
        return MCDSPreAnalysisResultsSet(miCustomCols=miCustCols, dfCustomColTrans=dfCustColTrans,
                                         miSampleCols=miSampCols, sampleIndCol=sampIndMCol,
                                         sortCols=sortCols, sortAscend=sortAscend,
                                         distanceUnit=self.distanceUnit, areaUnit=self.areaUnit,
                                         surveyType=self.surveyType, distanceType=self.distanceType,
                                         clustering=self.clustering)
    
    def run(self, dfExplSampleSpecs=None, implSampleSpecs=None, dModelStrategy=ModelStrategyDef, threads=None):
    
        """Run specified analyses
        
        Call explicitParamSpecs(..., check=True) before this to make sure user specs are OK

        Parameters:
        :param dfExplParamSpecs: Explicit sample specs, as a DataFrame
          (generated through explicitVariantSpecs, as an example),
        :param implParamSpecs: Implicit sample specs, suitable for explicitation
          through explicitVariantSpecs
        :param dModelStrategy: Sequence of fallback models to use when analyses fails.
        :param threads: Number of parallel threads to use (default None: no parallelism, no asynchronism)
        """
    
        # Executor (parallel or sequential).
        self._executor = Executor(threads=threads)

        # MCDS analysis engine (a sequential one: 'cause MCDSPreAnalysis does the parallel stuff itself).

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
        
        # Explicitate and complete analysis specs, and check for usability
        # (should be also done before calling run, to avoid failure).
        dfExplSampleSpecs, _, _, _, checkVerdict, checkErrors = \
            self.explicitParamSpecs(implSampleSpecs, dfExplSampleSpecs, dropDupes=True, check=True)
        assert checkVerdict, 'Pre-analysis params check failed: {}'.format('; '.join(checkErrors))
        
        # For each sample to analyse :
        runHow = 'in sequence' if threads <= 1 else f'{threads} parallel threads'
        logger.info('Running {} MCDS pre-analyses ({}) ...'.format(len(dfExplSampleSpecs), runHow))
        
        dAnlyses = dict()
        for sampInd, sSampSpec in dfExplSampleSpecs.iterrows():
            
            logger.info(f'#{sampInd+1}/{len(dfExplSampleSpecs)} : {sSampSpec[self.abbrevCol]}')

            # Select data sample to process
            sds = self._mcDataSet.sampleDataSet(sSampSpec[self.sampleSelCols])
            if not sds:
                continue # No data => no analysis.

            # Pre-analysis object
            anlys = MCDSPreAnalysis(engine=self._engine, executor=self._executor,
                                    sampleDataSet=sds, name=sSampSpec[self.abbrevCol],
                                    customData=sSampSpec[customCols].copy(),
                                    logData=False, modelStrategy=dModelStrategy)

            # Start running pre-analysis in parallel, but don't wait for it's finished, go on
            anlysFut = anlys.submit()
            
            # Store pre-analysis object and associated "future" for later use (should be running soon or later).
            dAnlyses[anlysFut] = anlys
            
            # Next analysis (loop).

        logger.info('All analyses started ; now waiting for their end, and results ...')

        # Wait for and gather results of all analyses.
        self.results = self._getResults(dAnlyses)

        # Set results specs for traceability.
        self.results.updateSpecs(samples=dfExplSampleSpecs, models=pd.DataFrame(dModelStrategy),
                                 analyser=self.flatSpecs(), runtime=pd.Series(autods.runtime, name='Version'))
        
        # Done.
        logger.info('Analyses completed.')

        return self.results

    def exportDSInputData(self, dfExplSampleSpecs=None, implSampleSpecs=None, format='Distance'):
    
        """Export specified data samples to the specified DS input format, for "manual" DS analyses
        
        Parameters:
        :param dfExplParamSpecs: Explicit sample specs, as a DataFrame
          (generated through explicitVariantSpecs, as an example),
        :param implParamSpecs: Implicit sample specs, suitable for explicitation
          through explicitVariantSpecs
        :param format: output files format, only 'Distance' supported for the moment.
        """
    
        assert format == 'Distance', 'Only Distance format supported for the moment'
    
        # MCDS analysis engine
        self._engine = MCDSEngine(workDir=self.workDir, runMethod=self.runMethod, timeOut=self.runTimeOut,
                                  distanceUnit=self.distanceUnit, areaUnit=self.areaUnit,
                                  surveyType=self.surveyType, distanceType=self.distanceType,
                                  clustering=self.clustering)

        # Explicitate and complete analysis specs, and check for usability
        # (should be also done before calling run, to avoid failure).
        dfExplSampleSpecs, _, _, _, checkVerdict, checkErrors = \
            self.explicitParamSpecs(implSampleSpecs, dfExplSampleSpecs, dropDupes=True, check=True)
        assert checkVerdict, 'Sample specs check failed: {}'.format('; '.join(checkErrors))
        
        # For each sample to export:
        logger.info('Exporting {} samples to {} format ...'.format(len(dfExplSampleSpecs), format))
        logger.debug(dfExplSampleSpecs)

        for sampInd, sSampSpec in dfExplSampleSpecs.iterrows():
            
            # Selection des données
            sds = self._mcDataSet.sampleDataSet(sSampSpec[self.sampleSelCols])
            if not sds:
                logger.warning('#{}/{} => No data in {} sample, no file exported'
                               .format(sampInd+1, len(dfExplSampleSpecs), sSampSpec[self.abbrevCol]))
                continue

            # Export to Distance
            fpn = pl.Path(self.workDir) / '{}-dist.txt'.format(sSampSpec[self.abbrevCol])
            fpn = self._engine.buildDistanceDataFile(sds, tgtFilePathName=fpn)

            logger.info('#{}/{} => {}'.format(sampInd+1, len(dfExplSampleSpecs), fpn.name))

        # Done.
        self._engine.shutdown()

        logger.info(f'Done exporting.')

if __name__ == '__main__':

    import sys

    sys.exit(0)
