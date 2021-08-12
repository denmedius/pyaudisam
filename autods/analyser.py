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

import autods.log as log

logger = log.logger('ads.anr')

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
        return dFlatSpecs

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

                dfExplPartSpecs = pd.DataFrame(data=np.tile(dfExplPartSpecs, [nInitSpecs, 1]), columns=dfExplPartSpecs.columns)

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
    
    # Usefull parameter columns labels
    CLParTruncLeft  = ('parameters', 'left truncation distance', 'Value')
    CLParTruncRight = ('parameters', 'right truncation distance', 'Value')

    # Computed column labels
    CLDeltaAic = ('detection probability', 'Delta AIC', 'Value')
    CLDeltaDCv = ('density/abundance', 'density of animals', 'Delta Cv')
    CLChi2     = ('detection probability', 'chi-square test probability determined', 'Value')

    CLCmbQuaBal1 = ('combined quality', 'balanced 1', 'Value')
    CLCmbQuaBal2 = ('combined quality', 'balanced 2', 'Value')
    CLCmbQuaBal3 = ('combined quality', 'balanced 3', 'Value')
    CLCmbQuaChi2 = ('combined quality', 'more Chi2', 'Value')
    CLCmbQuaKS   = ('combined quality', 'more KS', 'Value')
    CLCmbQuaDCV  = ('combined quality', 'more DCV', 'Value')

    CLCAFilSor = 'auto filter sort'
    CLTTruncGroup = 'Group'
    CLAFilSorGrpTruncLeft  = (CLCAFilSor, 'left truncation distance', CLTTruncGroup)
    CLAFilSorGrpTruncRight = (CLCAFilSor, 'right truncation distance', CLTTruncGroup)

    #CL = ('', '', 'Value')

    def __init__(self, miCustomCols=None, dfCustomColTrans=None, miSampleCols=None, sampleIndCol=None,
                       sortCols=[], sortAscend=[], distanceUnit='Meter', areaUnit='Hectare',
                       surveyType='Point', distanceType='Radial', clustering=False):
        
        """
        Parameters:
        :param miSampleCols: columns to use for grouping by sample ; defaults to miCustomCols if None
        :param sampleIndCol: multi-column index for the sample Id column ; no default, must be there !
        """

        assert sampleIndCol is not None

        # Computed columns specs (name translation + position).
        firstResColInd = len(MCDSEngine.statSampCols()) + len(MCDSAnalysis.MIRunColumns)
        cls = self
        DComputedCols = {cls.CLDeltaAic: firstResColInd + 11, # Right before AIC
                         cls.CLChi2: firstResColInd + 16, # Right before all Chi2 tests 
                         cls.CLDeltaDCv: firstResColInd + 67, # Right before Density of animals / CV 
                         cls.CLCmbQuaBal1: -1, # At the end ...
                         cls.CLCmbQuaBal2: -1,
                         cls.CLCmbQuaBal3: -1,
                         cls.CLCmbQuaChi2: -1,
                         cls.CLCmbQuaKS: -1,
                         cls.CLCmbQuaDCV: -1,
                         #cls.CL: ,
                         #cls.CL: ,
                        }    
        DfComputedColTrans = \
            pd.DataFrame(index=DComputedCols.keys(),
                         data=dict(en=['Delta AIC', 'Chi2 P', 'Delta CoefVar Density',
                                       'Qual Bal1', 'Qual Bal2', 'Qual Bal3',
                                       'Qual Chi2+', 'Qual KS+', 'Qual DCV+'],
                                   fr=['Delta AIC', 'Chi2 P', 'Delta CoefVar Densité',
                                       'Qual Equi1', 'Qual Equi2', 'Qual Equi3',
                                       'Qual Chi2+', 'Qual KS+', 'Qual DCV+']))


        # Initialise base.
        super().__init__(MCDSAnalysis, miCustomCols=miCustomCols, dfCustomColTrans=dfCustomColTrans,
                                       dComputedCols=DComputedCols, dfComputedColTrans=DfComputedColTrans,
                                       sortCols=sortCols, sortAscend=sortAscend)
        
        # Sample columns
        self.miSampleCols = miSampleCols if miSampleCols is not None else self.miCustomCols
        self.sampleIndCol = sampleIndCol
    
        # Descriptive parameters, not used in computations actually.
        self.distanceUnit = distanceUnit
        self.areaUnit = areaUnit
        self.surveyType = surveyType
        self.distanceType = distanceType
        self.clustering =clustering

    def copy(self, withData=True):
    
        """Clone function, with optional data copy"""
    
        # Create new instance with same ctor params.
        clone = MCDSAnalysisResultsSet(miCustomCols=self.miCustomCols, dfCustomColTrans=self.dfCustomColTrans,
                                       miSampleCols=self.miSampleCols, sampleIndCol=self.sampleIndCol,
                                       sortCols=self.sortCols, sortAscend=self.sortAscend,
                                       distanceUnit=self.distanceUnit, areaUnit=self.areaUnit,
                                       surveyType=self.surveyType, distanceType=self.distanceType,
                                       clustering=self.clustering)

        # Copy data if needed.
        if withData:
            clone._dfData = self._dfData.copy()
            clone.rightColOrder = self.rightColOrder
            clone.postComputed = self.postComputed

        return clone
    
    # Get translate names of custom columns
    def transSampleColumns(self, lang):
        
        return self.dfCustomColTrans.loc[self.miSampleCols, lang].to_list()

    # Post-computations : Actual Chi2 value, from multiple tests done.
    MaxChi2Tests = 3 # TODO: Really a constant, or actually depends on some analysis params ?
    CLsChi2All = [('detection probability', 'chi-square test probability (distance set {})'.format(i), 'Value') \
                  for i in range(MaxChi2Tests, 0, -1)]
    
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

    # Post-computations : Delta AIC/DCV per sampleCols + truncation param. cols group => AIC - min(group).
    CLAic = ('detection probability', 'AIC value', 'Value')
    CLDCv = ('density/abundance', 'density of animals', 'Cv')
    CLsTruncDist = [('encounter rate', 'left truncation distance', 'Value'),
                    ('encounter rate', 'right truncation distance (w)', 'Value')]

    def _postComputeDeltaAicDcv(self):
        
        logger.debug(f'Post-computing Delta AIC/DCV: {self.CLDeltaAic} / {self.CLDeltaDCv}')
        
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

    # Post computations : Quality indicators.
    CLNObs = ('encounter rate', 'number of observations (n)', 'Value')
    CLNTotObs = ('sample stats', 'total number of observations', 'Value')  # Must equal MCDSEngine._MIStatSampCols[0] !
    CLNTotPars = ('detection probability', 'total number of parameters (m)', 'Value')
    CLKS = ('detection probability', 'Kolmogorov-Smirnov test probability', 'Value')
    CLCvMUw = ('detection probability', 'Cramér-von Mises (uniform weighting) test probability', 'Value')
    CLCvMCw = ('detection probability', 'Cramér-von Mises (cosine weighting) test probability', 'Value')
    CLDCv = ('density/abundance', 'density of animals', 'Cv')

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
    def _combinedQualityMoreDCV(cls, sRes):
        return (sRes[[cls.CLChi2, cls.CLKS, cls.CLCvMUw, cls.CLCvMCw]].prod()
                * cls._normNObs(sRes) * cls._normNTotPars(sRes, a=0.2, b=0.6)
                * cls._normCVDens(sRes, a=12) ** 2) ** (1.0/8)

    def _postComputeQualityIndicators(self):
        
        logger.debug('Post-computing Quality Indicators')

        self._dfData[self.CLCmbQuaBal1] = self._dfData.apply(self._combinedQualityBalanced1, axis='columns')
        self._dfData[self.CLCmbQuaBal2] = self._dfData.apply(self._combinedQualityBalanced2, axis='columns')
        self._dfData[self.CLCmbQuaBal3] = self._dfData.apply(self._combinedQualityBalanced3, axis='columns')
        self._dfData[self.CLCmbQuaChi2] = self._dfData.apply(self._combinedQualityMoreChi2, axis='columns')
        self._dfData[self.CLCmbQuaKS]   = self._dfData.apply(self._combinedQualityMoreKS, axis='columns')
        self._dfData[self.CLCmbQuaDCV]  = self._dfData.apply(self._combinedQualityMoreDCV, axis='columns')
        
    # Post computations : Quality indicators.
    def _postComputeTruncationGroups(self):
        
        logger.debug('Post-computing Truncation Groups')

        # TODO: Simple solution = copy from MCDSTruncOptanalysisResultsSet and remove optimTrunc stuff
        # TODO: After that, a more simple groupby should be faster and simpler ... but need for new testing !
        
    # Post-computations : All of them.
    def postComputeColumns(self):
        
        self._postComputeChi2()
        self._postComputeDeltaAicDcv()
        self._postComputeQualityIndicators()
        self._postComputeTruncationGroups()


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

    def prepareResultsColumns(self):
        
        DAnlr2ResChapName = dict(before='header (head)', sample='header (sample)', after='header (tail)')

        # a. Sample multi-index columns
        sampleSelCols = self.resultsHeadCols['sample']
        sampMCols = [(DAnlr2ResChapName['sample'], col, 'Value') for col in sampleSelCols]
        miSampCols = pd.MultiIndex.from_tuples(sampMCols)

        # b. Full custom multi-index columns to append and prepend to raw analysis results
        beforeCols = self.resultsHeadCols['before']
        custMCols = [(DAnlr2ResChapName['before'], col, 'Value') for col in beforeCols]
        custMCols += sampMCols
        
        afterCols = self.resultsHeadCols['after']
        custMCols += [(DAnlr2ResChapName['after'], col, 'Value') for col in afterCols]

        customCols = beforeCols + sampleSelCols + afterCols
        miCustCols = pd.MultiIndex.from_tuples(custMCols)

        # c. Translation for it (well, no translation actually ... only one language forced for all !)
        dfCustColTrans = pd.DataFrame(index=miCustCols, data={lang: customCols for lang in ['fr', 'en']})

        # d. The 3-columns index for the sample index column
        sampIndMCol = (DAnlr2ResChapName[self.sampIndResHChap], self.sampleIndCol, 'Value')

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
                                      clustering=self.clustering)
    
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
        self.results.updateSpecs(analyser=self.flatSpecs(), analyses=dfExplParamSpecs)
        
        # Done.
        logger.info(f'Analyses completed ({len(self.results)} results).')

        return self.results


# Default strategy for model choice sequence (if one fails, take next in order, and so on)
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
        self.results.updateSpecs(analyser=self.flatSpecs(), samples=dfExplSampleSpecs,
                                 models=pd.DataFrame(dModelStrategy))
        
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
        logger.info(f'Done exporting.')

if __name__ == '__main__':

    import sys

    sys.exit(0)
