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

# Submodule "analyser": Run a bunch of DS analyses according to a user-friendly set of analysis specs

import copy
import re
import pathlib as pl
from packaging import version

import math
import numpy as np
import pandas as pd

from . import log, runtime
from .data import MonoCategoryDataSet, ResultsSet
from .executor import Executor
from .engine import MCDSEngine
from .analysis import DSAnalysis, MCDSAnalysis, MCDSPreAnalysis

logger = log.logger('ads.anr')


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

        nRows = max(len(crits) for crits in dVariants.values())

        return pd.DataFrame({colName: fixedLengthList(variants, nRows) for colName, variants in dVariants.items()})

    @staticmethod
    def _dropCommentColumns(dfSpecs):
    
        """Drop (in-place) comment columns from a spec dataframe (implicit or explicit)
        """
        
        cols2drop = [col for col in dfSpecs.columns if not col.strip() or col.startswith('Unnamed:')
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
        else:
            raise NotImplementedError(f'Unsupported file extension {ext} for input partial specs file')
            
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
        :param computedCols: Name and computing method for explicit columns to add (after appending :param varIndCol)
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
        dSameColsPsNames = dict()  # { cols: [table names] }
        
        for psName, dfPsValues in ddfPartSpecs.items():
            
            colSetId = ':'.join(sorted(dfPsValues.columns))
            if colSetId not in dSameColsPsNames:
                dSameColsPsNames[colSetId] = list()
                
            dSameColsPsNames[colSetId].append(psName)

        # For each group, concat. tables into one, after expliciting if needed
        ldfExplPartSpecs = list()

        for lPsNames in dSameColsPsNames.values():

            ldfSameColsPartSpecs = list()
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

            if commCols:  # Any column in common : left join each left row to each matching right row

                dfExplSpecs = dfExplSpecs.join(dfExplPartSpecs.set_index(commCols), on=commCols)

            else:  # No columns in common : combine each left row with all right rows

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
                parName = next(iter(parName for parName in int2UserSpecREs
                                    if any(re.search(pat, specName, flags=re.IGNORECASE)
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


class _FilterSortSteps(object):

    """Log = history of filter and sort steps for a given scheme"""

    def __init__(self, filSorSchId, resultsSet, lang):
        self.schemeId = filSorSchId
        self.lang = lang
        self.results = resultsSet  # only for resultsSet.transColumn(...)
        self.steps = list()  # of [stepName, propName, propValue]

    def copy(self):
        clone = _FilterSortSteps(filSorSchId=self.schemeId, resultsSet=self.results, lang=self.lang)
        clone.steps = copy.deepcopy(self.steps)
        return clone
    
    def transColumns(self, columns):
        if isinstance(columns, (list, dict)):
            return ', '.join(self.results.transColumn(col, self.lang) for col in columns)
        else:
            return self.results.transColumn(columns, self.lang)

    def append(self, stepName, propName, propValue, transColumns=False):
        """Append a step, property, value record in the log
        (translate value as columns label(s) if specified: transColumns=True)"""
        logger.debug2(f'* {stepName}: {propName} = {propValue}')
        if transColumns:
            propValue = self.transColumns(propValue)
        self.steps.append([stepName, propName, propValue])

    def toList(self):
        return [[self.schemeId] + step for step in self.steps]


class _FilterSortCache(object):

    def __init__(self):
        self.dResults = dict()

    def copy(self):
        clone = _FilterSortCache()
        clone.dResults = {schId: (iFilSor.copy(), filSorSteps.copy())
                          for schId, (iFilSor, filSorSteps) in self.dResults.items()}
        return clone

    def clear(self):
        self.dResults.clear()

    def update(self, schemeId, iFilSor, filSorSteps):
        # Add / update a detached copy
        self.dResults[schemeId] = iFilSor.copy(), filSorSteps.copy()

    def get(self, schemeId):
        if schemeId in self.dResults:
            # return a detached copy.
            iFilSor, filSorSteps = self.dResults[schemeId]
            logger.info1(f'Filter and sort scheme "{schemeId}" found in cache.')
            return iFilSor.copy(), filSorSteps.copy()
        return None, None


class MCDSAnalysisResultsSet(AnalysisResultsSet):

    """A specialized results set for MCDS analyses, with extra. post-computed columns : Delta AIC, Chi2 P"""
    
    # Shortcut to already existing and useful columns names.
    CLRunStatus = MCDSAnalysis.CLRunStatus
    CLParEstKeyFn = MCDSAnalysis.CLParEstKeyFn
    CLParEstAdjSer = MCDSAnalysis.CLParEstAdjSer
    CLParEstSelCrit = MCDSAnalysis.CLParEstSelCrit
    CLParEstCVInt = MCDSAnalysis.CLParEstCVInt
    CLParTruncLeft = MCDSAnalysis.CLParTruncLeft
    CLParTruncRight = MCDSAnalysis.CLParTruncRight
    CLParModFitDistCuts = MCDSAnalysis.CLParModFitDistCuts
    CLParModDiscrDistCuts = MCDSAnalysis.CLParModDiscrDistCuts

    CLEffort = ('encounter rate', 'effort (L or K or T)', 'Value')
    CLPDetec = ('detection probability', 'probability of detection (Pw)', 'Value')
    CLPDetecMin = ('detection probability', 'probability of detection (Pw)', 'Lcl')
    CLPDetecMax = ('detection probability', 'probability of detection (Pw)', 'Ucl')
    CLPDetecCv = ('detection probability', 'probability of detection (Pw)', 'Cv')
    CLPDetecDf = ('detection probability', 'probability of detection (Pw)', 'Df')
    CLEswEdr = ('detection probability', 'effective strip width (ESW) or effective detection radius (EDR)', 'Value')
    CLEswEdrMin = ('detection probability', 'effective strip width (ESW) or effective detection radius (EDR)', 'Lcl')
    CLEswEdrMax = ('detection probability', 'effective strip width (ESW) or effective detection radius (EDR)', 'Ucl')
    CLEswEdrCv = ('detection probability', 'effective strip width (ESW) or effective detection radius (EDR)', 'Cv')
    CLEswEdrDf = ('detection probability', 'effective strip width (ESW) or effective detection radius (EDR)', 'Df')
    CLDensity = ('density/abundance', 'density of animals', 'Value')
    CLDensityMin = ('density/abundance', 'density of animals', 'Lcl')
    CLDensityMax = ('density/abundance', 'density of animals', 'Ucl')
    CLDensityCv = ('density/abundance', 'density of animals', 'Cv')
    CLDensityDf = ('density/abundance', 'density of animals', 'Df')
    CLNumber = ('density/abundance', 'number of animals, if survey area is specified', 'Value')
    CLNumberMin = ('density/abundance', 'number of animals, if survey area is specified', 'Lcl')
    CLNumberMax = ('density/abundance', 'number of animals, if survey area is specified', 'Ucl')
    CLNumberCv = ('density/abundance', 'number of animals, if survey area is specified', 'Cv')
    CLNumberDf = ('density/abundance', 'number of animals, if survey area is specified', 'Df')

    DCLParTruncDist = dict(left=CLParTruncLeft, right=CLParTruncRight)

    # Computed Column Labels
    # a. Chi2 determined, Delta AIC, delta DCv
    CLChi2 = ('detection probability', 'chi-square test probability determined', 'Value')
    CLDeltaAic = ('detection probability', 'Delta AIC', 'Value')
    CLDeltaDCv = ('density/abundance', 'density of animals', 'Delta Cv')

    # b. Observation rate and combined quality indicators
    CLSightRate = ('encounter rate', 'observation rate', 'Value')
    CLCmbQuaBal1 = ('combined quality', 'balanced 1', 'Value')
    CLCmbQuaBal2 = ('combined quality', 'balanced 2', 'Value')
    CLCmbQuaBal3 = ('combined quality', 'balanced 3', 'Value')
    CLCmbQuaChi2 = ('combined quality', 'more Chi2', 'Value')
    CLCmbQuaKS = ('combined quality', 'more KS', 'Value')
    CLCmbQuaDCv = ('combined quality', 'more DCv', 'Value')

    # c. Automated filtering and grouping + sorting
    CLCAutoFilSor = 'auto filter sort'  # Label "Chapter" (1st level)
    CLTTruncGroup = 'Group'  # Label "Type" (3rd level)
    CLTSortOrder = 'Order'  # Label "Type" (3rd level)
    CLTPreSelection = 'Pre-selection'  # Label "Type" (3rd level)

    #   i. Close truncation group identification
    CLGroupTruncLeft = (CLCAutoFilSor, CLParTruncLeft[1], CLTTruncGroup)
    CLGroupTruncRight = (CLCAutoFilSor, CLParTruncRight[1], CLTTruncGroup)

    #   ii. Order inside groups with same = identical truncation parameters (distances and model cut points)
    CLGrpOrdSmTrAic = (CLCAutoFilSor, 'AIC (same trunc)', CLTSortOrder)

    #   iii. Order inside groups of close truncation distances
    CLGrpOrdClTrChi2KSDCv = (CLCAutoFilSor, 'Chi2 KS DCv (close trunc)', CLTSortOrder)
    # CLGrpOrdClTrChi2 = (CLCAutoFilSor, 'Chi2 (close trunc)', CLTSortOrder)
    CLGrpOrdClTrDCv = (CLCAutoFilSor, 'DCv (close trunc)', CLTSortOrder)
    
    CLGrpOrdClTrQuaBal1 = (CLCAutoFilSor, 'Bal. quality 1 (close trunc)', CLTSortOrder)
    CLGrpOrdClTrQuaBal2 = (CLCAutoFilSor, 'Bal. quality 2 (close trunc)', CLTSortOrder)
    CLGrpOrdClTrQuaBal3 = (CLCAutoFilSor, 'Bal. quality 3 (close trunc)', CLTSortOrder)
    CLGrpOrdClTrQuaChi2 = (CLCAutoFilSor, 'Bal. quality Chi2+ (close trunc)', CLTSortOrder)
    CLGrpOrdClTrQuaKS = (CLCAutoFilSor, 'Bal. quality KS+ (close trunc)', CLTSortOrder)
    CLGrpOrdClTrQuaDCv = (CLCAutoFilSor, 'Bal. quality DCv+ (close trunc)', CLTSortOrder)
    
    #   iv. Global order
    CLGblOrdChi2KSDCv = (CLCAutoFilSor, 'Chi2 KS DCv (global)', CLTSortOrder)

    CLGblOrdQuaBal1 = (CLCAutoFilSor, 'Bal. quality 1 (global)', CLTSortOrder)
    CLGblOrdQuaBal2 = (CLCAutoFilSor, 'Bal. quality 2 (global)', CLTSortOrder)
    CLGblOrdQuaBal3 = (CLCAutoFilSor, 'Bal. quality 3 (global)', CLTSortOrder)
    CLGblOrdQuaChi2 = (CLCAutoFilSor, 'Bal. quality Chi2+ (global)', CLTSortOrder)
    CLGblOrdQuaKS = (CLCAutoFilSor, 'Bal. quality KS+ (global)', CLTSortOrder)
    CLGblOrdQuaDCv = (CLCAutoFilSor, 'Bal. quality DCv+ (global)', CLTSortOrder)

    CLGblOrdDAicChi2KSDCv = (CLCAutoFilSor, 'DeltaAIC Chi2 KS DCv (global)', CLTSortOrder)

    # Computed columns specs (name translation + position).
    _firstResColInd = len(MCDSEngine.statSampCols()) + len(MCDSAnalysis.MIRunColumns)
    DComputedCols = {CLSightRate: _firstResColInd + 10,  # After Encounter Rate / Left|Right Trunc. Dist.
                     CLDeltaAic: _firstResColInd + 12,  # Before AIC
                     CLChi2: _firstResColInd + 14,  # Before all Chi2 tests
                     CLDeltaDCv: _firstResColInd + 72,  # Before Density of animals / Cv
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
                                   'Order Close Trunc Chi2 KS DCv',  # 'Order Close Trunc Chi2',
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
                                   'Ordre Tronc Proch Chi2 KS DCv',  # 'Ordre Tronc Proch Chi2',
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

        # Parameters for filter and sort key generation schemes
        self.ldFilSorKeySchemes = ldFilSorKeySchemes

        # Short (but unique) Ids for already seen "filter and sort" schemes,
        # based on the scheme name and an additional int suffix when needed ;
        # Definition: 2 equal schemes (dict ==) have the same Id
        self.dFilSorSchemes = dict()  # Unique Id => value

        # Cache for results of filter and sort schemes
        self.filSorCache = _FilterSortCache()

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
            clone.dfSamples = None if self.dfSamples is None else self.dfSamples.copy()
            clone.dFilSorSchemes = copy.deepcopy(self.dFilSorSchemes)
            clone.filSorCache = self.filSorCache.copy()

        return clone
    
    def onDataChanged(self):

        """React to results data (_dfData) changes that invalidate calculated data,
        but only when the calculus is coded in this class ; other calculi impacts taken care in base classes"""

        self.dfSamples = None
        self.dFilSorSchemes = dict()
        self.filSorCache.clear()

    def dropRows(self, sbSelRows):
    
        super().dropRows(sbSelRows)

        self.onDataChanged()
        
    def setData(self, dfData, postComputed=False, acceptNewCols=False):
        
        super().setData(dfData, postComputed=postComputed, acceptNewCols=acceptNewCols)

        self.onDataChanged()

    # Get translate names of custom columns
    def transSampleColumns(self, lang):
        
        return self.dfCustomColTrans.loc[self.miSampleCols, lang].to_list()

    # Post-computations : Actual Chi2 value, from multiple tests done.
    MaxChi2Tests = 3  # TODO: Really a constant, or actually depends on some analysis params ?
    CLsChi2All = [('detection probability', 'chi-square test probability (distance set {})'.format(i), 'Value')
                  for i in range(MaxChi2Tests, 0, -1)]
    
    @staticmethod
    def _determineChi2Value(sChi2AllDists):
        for chi2 in sChi2AllDists:
            if not np.isnan(chi2):
                return chi2
        return np.nan

    def _postComputeChi2(self):
        
        logger.debug(f'Post-computing actual Chi2: {self.CLChi2}')
        
        # Last value of all the tests done.
        chi2AllColLbls = [col for col in self.CLsChi2All if col in self._dfData.columns]
        if chi2AllColLbls:
            self._dfData[self.CLChi2] = self._dfData[chi2AllColLbls].apply(self._determineChi2Value, axis='columns')

    # Post-computations : Delta AIC/DCv per sampleCols + truncation param. cols group => AIC - min(group).
    CLAic = ('detection probability', 'AIC value', 'Value')
    CLDCv = CLDensityCv
    CLsTruncDist = [('encounter rate', 'left truncation distance', 'Value'),
                    ('encounter rate', 'right truncation distance (w)', 'Value')]

    def _postComputeDeltaAicDCv(self):
        
        logger.debug(f'Post-computing Delta AIC/DCv: {self.CLDeltaAic} / {self.CLDeltaDCv}')
        
        # a. Minimum AIC & DCv per group
        #    (drop all-NaN sample selection columns (sometimes, it happens) for a working groupby())
        groupColLbls = self.miSampleCols.append(pd.MultiIndex.from_tuples(self.CLsTruncDist))
        groupColLbls = [col for col in groupColLbls
                        if col in self._dfData.columns and not self._dfData[col].isna().all()]
        df2Join = self._dfData.groupby(groupColLbls, dropna=False)[[self.CLAic, self.CLDCv]].min()
        
        # b. Rename computed columns to target 'Delta XXX'
        df2Join.columns = pd.MultiIndex.from_tuples([self.CLDeltaAic, self.CLDeltaDCv])

        # c. Join the column to the target data-frame
        self._dfData = self._dfData.join(df2Join, on=groupColLbls)

        # d. Compute delta-AIC & DCv in-place
        self._dfData[self.CLDeltaAic] = self._dfData[self.CLAic] - self._dfData[self.CLDeltaAic]
        self._dfData[self.CLDeltaDCv] = self._dfData[self.CLDCv] - self._dfData[self.CLDeltaDCv]

    # Post computations : Useful columns for quality indicators.
    CLNObs = ('encounter rate', 'number of observations (n)', 'Value')
    CLNTotObs = MCDSEngine.MIStatSampCols[0]
    CLMinObsDist = MCDSEngine.MIStatSampCols[1]
    CLMaxObsDist = MCDSEngine.MIStatSampCols[2]
    CLKeyFn = ('detection probability', 'key function type', 'Value')
    CLNTotPars = ('detection probability', 'total number of parameters (m)', 'Value')
    CLNAdjPars = ('detection probability', 'number of adjustment term parameters (NAP)', 'Value')
    CLKS = ('detection probability', 'Kolmogorov-Smirnov test probability', 'Value')
    CLCvMUw = ('detection probability', 'Cramér-von Mises (uniform weighting) test probability', 'Value')
    CLCvMCw = ('detection probability', 'Cramér-von Mises (cosine weighting) test probability', 'Value')

    # Post computations : Quality indicators.
    CLsQuaIndicSources = [CLKeyFn, CLNAdjPars, CLNTotPars, CLNObs, CLNTotObs, CLChi2, CLKS, CLCvMUw, CLCvMCw, CLDCv]

    CIKeyFn = CLsQuaIndicSources.index(CLKeyFn)
    CINAdjPars = CLsQuaIndicSources.index(CLNAdjPars)
    CINTotPars = CLsQuaIndicSources.index(CLNTotPars)
    CINObs = CLsQuaIndicSources.index(CLNObs)
    CINTotObs = CLsQuaIndicSources.index(CLNTotObs)
    CIChi2 = CLsQuaIndicSources.index(CLChi2)
    CIKS = CLsQuaIndicSources.index(CLKS)
    CICvMUw = CLsQuaIndicSources.index(CLCvMUw)
    CICvMCw = CLsQuaIndicSources.index(CLCvMCw)
    CIDCv = CLsQuaIndicSources.index(CLDCv)

    @classmethod
    def _combinedQualityBalanced1(cls, aRes):

        """Historical QualBal1 indicator (March 2021), optimized through numpy

        Returns:
             np.array (shape: aRes rows, 1 column)
        """

        chi2KsCvMs = aRes[:, cls.CIChi2:cls.CICvMCw + 1].prod(axis=1)
        normNObs = aRes[:, cls.CINObs].astype(float) / aRes[:, cls.CINTotObs].astype(float)
        normNTotPars = 1 / (0.2 * np.maximum(2, aRes[:, cls.CINTotPars].astype(float)) + 0.6)
        normCVDens = np.exp(-12 * np.square(aRes[:, cls.CIDCv].astype(float)))

        return np.power(chi2KsCvMs * normNObs * normNTotPars * normCVDens, 1 / 7.0)  # shape: aRes rows, 1 column

    # Raw key function indicator and relevant universal numpy function.
    # DNormKeyFn = dict(HNORMAL=1.00, UNIFORM=0.75, HAZARD=0.5, NEXPON=0.1)  # Not better
    _ufnNormKeyFn = \
        np.frompyfunc(lambda keyFn: dict(HNORMAL=1.0, UNIFORM=0.9, HAZARD=0.6, NEXPON=0.1).get(keyFn, 0.0), 1, 1)

    CLsNewQuaIndics = [CLCmbQuaBal2, CLCmbQuaBal3, CLCmbQuaChi2, CLCmbQuaKS, CLCmbQuaDCv]

    @classmethod
    def _combinedQualityAll(cls, aRes):

        """New quality indicators (August, 2021 and later), optimized through numpy
        (QualBal2, QualBal3, QualMoreChi2, QualMoreKS, QualMoreDCv)

        Returns:
             tuple(np.array) (shape: aRes rows, 1 column each, order of cls.CLsNewQuaIndics)
        """

        # Common computations (all times)
        chi2 = aRes[:, cls.CIChi2].astype(float)
        ks = aRes[:, cls.CIKS].astype(float)
        dcv = aRes[:, cls.CIDCv].astype(float)
        chi2KsCvMs = aRes[:, cls.CIChi2:cls.CICvMCw + 1].astype(float).prod(axis=1)
        normNObs = aRes[:, cls.CINObs].astype(float) / aRes[:, cls.CINTotObs].astype(float)

        # Common computations (October 2021)
        nAdjPars2 = np.square(aRes[:, cls.CINAdjPars]).astype(float)
        normKeyFn = cls._ufnNormKeyFn(aRes[:, cls.CIKeyFn])
        normChi2KsCvMsNObsKFn = chi2KsCvMs * normNObs * normKeyFn

        # QualBal2 (August 2021)
        # normNTotPars2 = 1 / (0.2 * np.maximum(1, aRes[:, cls.CINTotPars].astype(float)) + 0.8)
        # normCVDens2 = np.exp(-16 * np.square(dcv))
        # normChi2KsCvMsNObsTotPDcv2 = chi2KsCvMs * normNObs * normNTotPars2 * normCVDens2
        # quaBal2 = np.power(normChi2KsCvMsNObsTotPDcv2, 1 / 7.0)

        # QualBal2 (October 2021): A more devaluating version for CVDens, using KeyFn, replacing NTotPars by NAdjPars
        normNAdjPars2 = np.exp(-0.15 * nAdjPars2)
        normCVDens2 = np.exp(-20 * np.square(dcv))
        quaBal2 = np.power(normChi2KsCvMsNObsKFn * normNAdjPars2 * normCVDens2, 1 / 8.0)

        # QualBal3 (August 2021)
        # normNTotPars3 = 1 / (0.3 * np.maximum(1, aRes[:, cls.CINTotPars].astype(float)) + 0.7)
        # normCVDens3 = np.exp(-20 * np.square(dcv))
        # normChi2KsCvMsNObsTotPDcv3 = chi2KsCvMs * normNObs * normNTotPars3 * normCVDens3
        # quaBal3 = np.power(normChi2KsCvMsNObsTotPDcv3, 1 / 7.0)

        # QualBal3 (October 2021): Same as QualBal2, but even more devaluating for CVDens
        normNAdjPars3 = np.exp(-0.17 * nAdjPars2)
        normCVDens3 = np.exp(-63 * np.power(dcv, 2.8))
        normChi2KsCvMsNObsKFnAdjPDcv3 = normChi2KsCvMsNObsKFn * normNAdjPars3 * normCVDens3
        quaBal3 = np.power(normChi2KsCvMsNObsKFnAdjPDcv3, 1 / 8.0)

        # QualMoreX (March 2021)
        # normNTotParsM = 1 / (0.2 * np.maximum(2, aRes[:, cls.CINTotPars].astype(float)) + 0.6)
        # normCVDensM = np.exp(-12 * np.square(dcv))
        # normChi2KsCvMsNObsTotPDCvM = chi2KsCvMs * normNObs * normNTotParsM * normCVDensM
        # moreChi2 = np.power(normChi2KsCvMsNObsTotPDCvM * chi2, 1 / 8.0)
        # moreKS = np.power(normChi2KsCvMsNObsTotPDCvM * ks, 1 / 8.0)
        # moreDCv = np.power(normChi2KsCvMsNObsTotPDCvM * normCVDensM, 1 / 8.0)

        # QualMoreX (October 2021): Follow QualBal3 update (were based on historical QualBal1)
        moreChi2 = np.power(normChi2KsCvMsNObsKFnAdjPDcv3 * chi2, 1 / 9.0)
        moreKS = np.power(normChi2KsCvMsNObsKFnAdjPDcv3 * ks, 1 / 9.0)
        moreDCv = np.power(normChi2KsCvMsNObsKFnAdjPDcv3 * normCVDens3, 1 / 9.0)

        return quaBal2, quaBal3, moreChi2, moreKS, moreDCv

    # Killer values for base MCDS indicators
    KilrNObs = 0
    KilrStaTest = 0
    KilrDensCv = 1e5
    KilrNPars = 1e3
    KilrNTotObs = 1e9
    KilrBalQua = 0

    def _postComputeQualityIndicators(self):

        cls = self

        logger.debug('Post-computing Quality Indicators')

        # Sighting rate (not 100% due to truncations).
        self._dfData[cls.CLSightRate] = 100 * self._dfData[cls.CLNObs] / self._dfData[cls.CLNTotObs]  # [0,1] => %

        # Prepare data for computations
        logger.debug1('* Pre-processing source data')

        # a. extract the useful columns, after adding them if not present
        #    (NaN value, except for CLKeyFn, that MUST be there anyway)
        for miCol in cls.CLsQuaIndicSources:
            if miCol not in self._dfData.columns and miCol != cls.CLKeyFn:
                self._dfData[miCol] = np.nan
        dfCompData = self._dfData[cls.CLsQuaIndicSources].copy()

        # b. historical bal quality  indicator 1
        logger.debug1('* Balanced quality 1')
        self._dfData[cls.CLCmbQuaBal1] = cls._combinedQualityBalanced1(dfCompData.values)

        # c. newer quality indicators
        #    (NaN value MUST kill down these indicators to compute => we have to enforce this)
        dfCompData.fillna({cls.CLNObs: cls.KilrNObs,
                           cls.CLChi2: cls.KilrStaTest, cls.CLKS: cls.KilrStaTest,
                           cls.CLCvMUw: cls.KilrStaTest, cls.CLCvMCw: cls.KilrStaTest,
                           cls.CLDCv: cls.KilrDensCv,  # Usually considered good under 0.3
                           cls.CLNTotObs: cls.KilrNTotObs,  # Should slap down _normObs whatever NObs
                           cls.CLNAdjPars: cls.KilrNPars,  # Should slap down _normNAdjPars whatever NObs
                           cls.CLNTotPars: cls.KilrNPars},
                          inplace=True)

        logger.debug1('* Balanced quality 2, 3, Chi2+, KS+, DCv+')
        for miCol, aIndic in zip(cls.CLsNewQuaIndics, cls._combinedQualityAll(dfCompData.values)):
            self._dfData[miCol] = aIndic

        # For some unknown reason, the theorically better code below raises some odd exceptions like (depends):
        # * index-join on non unique index not implemented
        # * KeyError: None of <items of cls.CLsNewQuaIndics> exists in index
        # whereas the same code works in devarchive2.ipynb/Development : Optimise _postComputeQualityIndicators).
        # self._dfData[cls.CLsNewQuaIndics] = np.stack(cls._combinedQualityAll(dfCompData.values), axis=1)

    # Post computations : Truncations groups.
    @staticmethod
    def _groupingIntervals(sValues, minDist, maxLen, epsilon=1e-6):

        """Build a list of value grouping intervals from a series of values
        taking care of min distance and max length constraints

        Parameters:
        :param sValues: pd.Series of values to group
        :param minDist: minimal distance between consecutive intervals (left.max - right.min > minIntrvDist)
        :param maxLen: max length of intervals (but ... see below)

        TODO: Fix current implementation that actually does not produce intervals of max length maxLen, but 1.5*maxLen !

        :return: pd.DataFrame of left-closed and right-open resulting intervals (columns = vmin, vsup)
        """

        # Cleanup and sort (ascending) distance series first
        # (+ for some reason, need for enforcing float dtype ... otherwise dtype='O' !?)
        sValues = sValues.dropna().astype(float).sort_values()

        # If not a single cleaned up distance to examine, stop here.
        if sValues.empty:
            return pd.DataFrame()

        # List non-null differences between consecutive sorted distances
        dfIntrvs = pd.DataFrame(dict(v=sValues.values))
        dfIntrvs['vdelta'] = dfIntrvs.v.diff()
        dfIntrvs.loc[dfIntrvs.v.idxmin(), 'vdelta'] = np.inf
        dfIntrvs.dropna(inplace=True)
        dfIntrvs = dfIntrvs[dfIntrvs.vdelta > 0].copy()

        # Deduce start (min) and end (sup) for each such interval (left-closed, right-open)
        dfIntrvs['vmin'] = dfIntrvs.loc[dfIntrvs.vdelta > minDist, 'v']
        dfIntrvs['vsup'] = dfIntrvs.loc[dfIntrvs.vdelta > minDist, 'v'].shift(-1).dropna()
        dfIntrvs.loc[dfIntrvs['vmin'].idxmax(), 'vsup'] = np.inf
        dfIntrvs.dropna(inplace=True)
        dfIntrvs['vsup'] = dfIntrvs['vsup'].apply(lambda vs: sValues[sValues < vs].max() + epsilon)
        dfIntrvs = dfIntrvs[['vmin', 'vsup']].reset_index(drop=True)

        # If these intervals are two wide, cut them up in equal sub-intervals and make them new intervals
        lsNewIntrvs = list()
        for _, sIntrv in dfIntrvs.iterrows():

            if sIntrv.vsup - sIntrv.vmin > maxLen:
                # TODO: Well, this actually does not produce intervals of max length maxLen, but 1.5*maxLen !
                nSubIntrvs = round((sIntrv.vsup - sIntrv.vmin) / maxLen)
                subIntrvLen = (sIntrv.vsup - sIntrv.vmin) / nSubIntrvs
                lsNewIntrvs += [pd.Series(dict(vmin=sIntrv.vmin + nInd * subIntrvLen,
                                               vsup=min(sIntrv.vmin + (nInd + 1) * subIntrvLen,
                                                        sIntrv.vsup)))
                                for nInd in range(nSubIntrvs)]
            else:
                lsNewIntrvs.append(sIntrv)

        dfIntrvs = pd.DataFrame(lsNewIntrvs).reset_index(drop=True)
        dfIntrvs.sort_values(by='vmin', inplace=True)

        return dfIntrvs

    @staticmethod
    def _intervalIndex(value, dfIntervals):
        """Compute the index of the interval to which a value belongs, if any
        :return: 0 for NaN values, -1 for values that belong to no interval, from-1 interval index otherwise
        """
        if pd.isnull(value):
            return 0
        dfWhere = dfIntervals[(dfIntervals.vmin <= value) & (dfIntervals.vsup > value)]
        if dfWhere.empty:
            return -1
        return 1 + dfWhere.index[0]

    @classmethod
    def _sampleDistTruncGroups(cls, dfSampRes, ldIntrvSpecs, intrvEpsilon=1e-6):

        """Compute distance truncation groups for 1 sample, for each target distance truncation column"""

        # For each truncation "method" (left or right)
        dTruncGroups = dict()
        for dIntrvSpecs in ldIntrvSpecs:

            truncCol = cls.DCLParTruncDist[dIntrvSpecs['col']]
            logger.debug4(f'  - {truncCol[1]}')

            # Compute distance grouping intervals
            dfIntrvs = cls._groupingIntervals(sValues=dfSampRes[truncCol], minDist=dIntrvSpecs['minDist'],
                                              maxLen=dIntrvSpecs['maxLen'], epsilon=intrvEpsilon)

            # Deduce index of belonging interval for each distance
            # (special case when no truncation: num=0 if NaN truncation distance)
            dTruncGroups[dIntrvSpecs['col']] = dfSampRes[truncCol].apply(cls._intervalIndex, dfIntervals=dfIntrvs)

        return dTruncGroups

    def listSamples(self, rebuild=False):

        """List result samples (if really needed of specified)"""

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

    def _distTruncGroups(self):

        """Compute distance truncation groups for all samples, for each target distance truncation column"""

        # For each sample,
        dTruncGroups = dict()  # key=ldIntrvSpecs[*]['col'], value=list(Series of group nums)
        for lblSamp, sSamp in self.listSamples().iterrows():

            # Select sample rows
            dfSampRes = self._dfData.loc[self._dfData[self.sampleIndCol] == lblSamp]
            logger.debug2('#{} {} : {} rows'
                          .format(lblSamp, ', '.join([f'{k[1]}={v}' for k, v in sSamp.items()]), len(dfSampRes)))

            # Compute truncation groups for this sample
            dSampTruncGroups = self._sampleDistTruncGroups(dfSampRes, ldIntrvSpecs=self.ldTruncIntrvSpecs,
                                                           intrvEpsilon=self.truncIntrvEpsilon)

            # Store them for later concatenation
            for colAlias, sGrpNums in dSampTruncGroups.items():
                if colAlias not in dTruncGroups:
                    dTruncGroups[colAlias] = list()
                dTruncGroups[colAlias].append(sGrpNums)

        # Concat series of computed group nums (opt or not) for each target distance column to group
        return {colAlias: pd.concat(lsGrpNums) for colAlias, lsGrpNums in dTruncGroups.items()}

    DCLGroupTruncDist = dict(left=CLGroupTruncLeft, right=CLGroupTruncRight)

    def _postComputeTruncationGroups(self):

        """Compute and add truncation group columns for later filtering and sorting"""

        logger.debug('Post-computing Truncation Groups')

        # Compute distance truncation groups for all samples, for each target distance truncation column
        # and update result table.
        for colAlias, sGrpNums in self._distTruncGroups().items():
            self._dfData[self.DCLGroupTruncDist[colAlias]] = sGrpNums

    # Post computations : filtering and sorting.
    # a. Column Labels for computed group and sort orders
    #    N.B. "close truncations" means "identical = same truncations" here (but see MCDSOptanalyserResultsSet)
    # See above

    # b. Schemes for computing filtering and sorting keys (see inherited _postComputeFilterSortKeys).
    AutoFilSorKeySchemes = \
        [  # Orders inside groups with identical truncation params.
         dict(key=CLGrpOrdSmTrAic,  # Best AIC, for same left and right truncations (but variable nb of cut points)
              sort=[CLParTruncLeft, CLParTruncRight, CLDeltaAic, CLChi2, CLKS, CLDCv, CLNObs, CLRunStatus],
              ascend=[True, True, True, False, False, True, False, True],
              group=[CLParTruncLeft, CLParTruncRight, CLParModFitDistCuts]),

         # Orders inside groups of close truncation params.
         # dict(key=CLGrpOrdClTrChi2,  # Best Chi2 inside groups of close truncation params
         #      sort=[CLGroupTruncLeft, CLGroupTruncRight,
         #            CLChi2],
         #      ascend=[True, True, False],
         #      group=[CLGroupTruncLeft, CLGroupTruncRight]),
         # dict(key=CLGrpOrdClTrKS,  # Best KS inside groups of close truncation params
         #      sort=[CLGroupTruncLeft, CLGroupTruncRight,
         #            CLKS],
         #      ascend=[True, True, False],
         #      group=[CLGroupTruncLeft, CLGroupTruncRight]),
         dict(key=CLGrpOrdClTrDCv,  # Best DCv inside groups of close truncation params
              sort=[CLGroupTruncLeft, CLGroupTruncRight,
                    CLDCv],
              ascend=[True, True, True],
              group=[CLGroupTruncLeft, CLGroupTruncRight]),
         dict(key=CLGrpOrdClTrChi2KSDCv,  # Best Chi2 & KS & DCv inside groups of close truncation params
              sort=[CLGroupTruncLeft, CLGroupTruncRight, CLChi2, CLKS, CLDCv, CLNObs, CLRunStatus],
              ascend=[True, True, False, False, True, False, True],
              group=[CLGroupTruncLeft, CLGroupTruncRight]),

         dict(key=CLGrpOrdClTrQuaBal1,  # Best Combined Quality 1 inside groups of close truncation params
              sort=[CLGroupTruncLeft, CLGroupTruncRight,
                    CLCmbQuaBal1],
              ascend=[True, True, False],
              group=[CLGroupTruncLeft, CLGroupTruncRight]),
         dict(key=CLGrpOrdClTrQuaBal2,  # Best Combined Quality 2 inside groups of close truncation params
              sort=[CLGroupTruncLeft, CLGroupTruncRight,
                    CLCmbQuaBal2],
              ascend=[True, True, False],
              group=[CLGroupTruncLeft, CLGroupTruncRight]),
         dict(key=CLGrpOrdClTrQuaBal3,  # Best Combined Quality 3 inside groups of close truncation params
              sort=[CLGroupTruncLeft, CLGroupTruncRight,
                    CLCmbQuaBal3],
              ascend=[True, True, False],
              group=[CLGroupTruncLeft, CLGroupTruncRight]),
         dict(key=CLGrpOrdClTrQuaChi2,  # Best Qualité combinée Chi2+ inside groups of close truncation params
              sort=[CLGroupTruncLeft, CLGroupTruncRight,
                    CLCmbQuaChi2],
              ascend=[True, True, False],
              group=[CLGroupTruncLeft, CLGroupTruncRight]),
         dict(key=CLGrpOrdClTrQuaKS,  # Best Combined Quality KS+ inside groups of close truncation params
              sort=[CLGroupTruncLeft, CLGroupTruncRight,
                    CLCmbQuaKS],
              ascend=[True, True, False],
              group=[CLGroupTruncLeft, CLGroupTruncRight]),
         dict(key=CLGrpOrdClTrQuaDCv,  # Best Combined Quality DCv+ inside groups of close truncation params
              sort=[CLGroupTruncLeft, CLGroupTruncRight,
                    CLCmbQuaDCv],
              ascend=[True, True, False],
              group=[CLGroupTruncLeft, CLGroupTruncRight]),

         # Global orders (no grouping by close or identical truncations)
         dict(key=CLGblOrdChi2KSDCv,
              sort=[CLChi2, CLKS, CLDCv, CLNObs, CLRunStatus],
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
              sort=[CLParTruncLeft, CLParTruncRight, CLParModFitDistCuts,
                    CLDeltaAic, CLChi2, CLKS, CLDCv, CLNObs, CLRunStatus],
              ascend=[True, True, True, True, False, False, True, False, True], napos='first')]

    # Enforce uniqueness of keys in filter and sort key specs.
    assert len(AutoFilSorKeySchemes) == len(set(scheme['key'] for scheme in AutoFilSorKeySchemes)), \
           'Duplicated scheme key in MCDSAnalysisResultsSet.AutoFilSorKeySchemes'

    # Enforce uniqueness of sort and group column in filter and sort key specs.
    assert all(len(scheme['sort']) == len(set(scheme['sort'])) for scheme in AutoFilSorKeySchemes), \
           'Duplicated sort column spec in some scheme of MCDSAnalysisResultsSet.AutoFilSorKeySchemes'
    assert all(len(scheme.get('group', [])) == len(set(scheme.get('group', []))) for scheme in AutoFilSorKeySchemes), \
           'Duplicated group column spec in some scheme of MCDSAnalysisResultsSet.AutoFilSorKeySchemes'

    # Check sort vs ascend list lengths in filter and sort key specs.
    assert all(isinstance(scheme['ascend'], bool) or len(scheme['ascend']) == len(scheme['sort'])
               for scheme in AutoFilSorKeySchemes), \
           'Inconsistent ascend vs sort specs in some scheme of MCDSAnalysisResultsSet.AutoFilSorKeySchemes'

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
        {CLParModFitDistCuts: (CLParModFitDistCuts[0], CLParModFitDistCuts[1], 'Sortable'),
         CLParModDiscrDistCuts: (CLParModDiscrDistCuts[0], CLParModDiscrDistCuts[1], 'Sortable')}

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
        {CLParModFitDistCuts: (CLParModFitDistCuts[0], CLParModFitDistCuts[1], 'Hashable'),
         CLParModDiscrDistCuts: (CLParModDiscrDistCuts[0], CLParModDiscrDistCuts[1], 'Hashable')}

    @classmethod
    def _sampleFilterSortKeys(cls, dfSampRes, ldFilSorKeySchemes):

        """Compute filter and sort keys for 1 sample, for each pre-defined scheme

        Parameters:
        :param ldFilSorKeySchemes: Filter-sort key generation schemes to use (ex. cls.AutoFilSorKeySchemes).
        """

        # Make a copy to avoid modifying source data.
        dfSampRes = dfSampRes.copy()

        # For each filter and sort scheme
        dFilSorKeys = dict()
        for scheme in ldFilSorKeySchemes:

            logger.debug3('* scheme {}'.format(scheme))

            # Workaround to-be-sorted problematic columns.
            sortCols = list()
            for col in scheme['sort']:
                if col in cls.DCLUnsortableCols:
                    wkrndSortCol = cls.DCLUnsortableCols[col]
                    logger.debug3('{} => {}'.format(col, wkrndSortCol))
                    dfSampRes[wkrndSortCol] = dfSampRes[col].apply(cls._toSortable)
                    col = wkrndSortCol  # Will rather sort with this one !
                sortCols.append(col)

            # Sort results
            dfSampRes.sort_values(by=sortCols, ascending=scheme['ascend'],
                                  na_position=scheme.get('napos', 'last'), inplace=True)

            # Compute order (target series is indexed like dfSampRes).
            if 'group' in scheme:  # Partial = 'group' order.

                # Workaround for to-be-grouped problematic columns.
                groupCols = list()
                for col in scheme['group']:
                    if col in cls.DCLUnhashableCols:
                        wkrndGroupCol = cls.DCLUnhashableCols[col]
                        logger.debug3('{} => {}'.format(col, wkrndGroupCol))
                        dfSampRes[wkrndGroupCol] = dfSampRes[col].apply(cls._toHashable)
                        col = wkrndGroupCol  # Will rather group with this one !
                    groupCols.append(col)

                sSampOrder = dfSampRes.groupby(groupCols, dropna=False).cumcount()

            else:  # Global order.
                sSampOrder = pd.Series(data=range(len(dfSampRes)), index=dfSampRes.index)

            # Done: next scheme.
            dFilSorKeys[scheme['key']] = sSampOrder

        return dFilSorKeys

    def _filterSortKeySchemes(self):
        """Select filter and sort keys schemes to apply (predefined if not overriden at ctor time)"""
        return self.ldFilSorKeySchemes or self.AutoFilSorKeySchemes

    def _filterSortKeys(self):

        """Compute filter and sort keys for all samples, for each scheme target key column"""

        # Retrieve filter and sort keys schemes to apply.
        ldFilSorKeySchemes = self._filterSortKeySchemes()

        # For each sample,
        dFilSorKeys = dict()  # key=<target column for key>, value=<Series of key values>)
        for lblSamp, sSamp in self.listSamples().iterrows():

            # Select sample rows
            dfSampRes = self._dfData.loc[self._dfData[self.sampleIndCol] == lblSamp]
            logger.debug2('#{} {} : {} rows'
                          .format(lblSamp, ', '.join([f'{k[1]}={v}' for k, v in sSamp.items()]), len(dfSampRes)))

            # Compute key values for all schemes for this sample
            dSampKeys = self._sampleFilterSortKeys(dfSampRes, ldFilSorKeySchemes)

            # Store them for later concatenation
            for colLbl, sFSKeys in dSampKeys.items():
                if colLbl not in dFilSorKeys:
                    dFilSorKeys[colLbl] = list()  # list(Series)
                dFilSorKeys[colLbl].append(sFSKeys)

        # Concat series of computed group nums (opt or not) for each target distance column to group
        return {colLbl: pd.concat(lsFSKeys) for colLbl, lsFSKeys in dFilSorKeys.items()}

    def _postComputeFilterSortKeys(self):

        """Compute and add partial or global order columns for later filtering and sorting"""

        logger.debug('Post-computing Filter and Sort keys')

        # Compute keys for each sample and add relevant columns to results.
        for colLbl, sFSKeys in self._filterSortKeys().items():
            self._dfData[colLbl] = sFSKeys

    # Post-computations : All of them.
    def postComputeColumns(self):
        
        self._postComputeChi2()
        self._postComputeDeltaAicDCv()
        self._postComputeQualityIndicators()
        self._postComputeTruncationGroups()
        self._postComputeFilterSortKeys()

    # Tools for actually filtering results
    @classmethod
    def _indexOfDuplicates(cls, dfRes, keep='first', subset=list(), round2decs=dict()):

        """Compute the indices of duplicates to remove in a data-frame,
        keep=first means that the first item of an "equality" set is the one kept at the end
        Warning: No sorting done here => do it before to get the right first !
        """
        if round2decs:
            # dfRes = dfRes.round(round2decs)  # Buggy (pandas 1.0.x up to 1.1.2): forgets columns !?!?!?
            if len(subset) > 0:
                pass  # TODO: Optimise = only copy subset cols
            dfRes = dfRes.copy()
            for col, dec in round2decs.items():
                if len(subset) == 0 or col in subset:
                    dfRes[col] = dfRes[col].apply(lambda x: x if pd.isnull(x) else round(x, ndigits=dec))

        return dfRes[dfRes.duplicated(keep=keep, subset=subset)].index

    @classmethod
    def _indexOfWorstOneCriterion(cls, dfRes, sampleIds, sampleIdCol, critCol, ascendCrit=True, nTgtRes=10):

        """Filtering function enforcing a target number of best output results based on 1 given criterion.
        ascendCrit=True means that the best criterion values are the smallest ones
        """

        i2Drop = pd.Index([], dtype=dfRes.index.dtype)

        # For each sample to filter ...
        for sampId in sampleIds:
            # Extract sample results, and sort them  based on the criterion.
            dfSampRes = dfRes[dfRes[sampleIdCol] == sampId].sort_values(by=critCol, ascending=ascendCrit)

            # Done for this sample.
            i2Drop = i2Drop.append(dfSampRes.index[nTgtRes:])  # Will drop indices after the N best's ones

        return i2Drop

    @classmethod
    def _indexOfWorstMultiOrderCriteria(cls, dfRes, critCols=list(), supCrit=1):

        """Filtering function enforcing a max(sup) order for output results based on multiple given criterion
        (results will be dropped if not in the supCrit best ones for at least 1 of the order criteria)
        """

        sb2keep = pd.Series(data=False, index=dfRes.index)
        for critCol in critCols:
            sb2keep |= (dfRes[critCol] < supCrit)

        return dfRes[~sb2keep].index

    # Constants for filSorSchemeId
    MainSchSpecNames = ['method', 'deduplicate', 'filterSort',
                        'preselCols', 'preselAscs', 'preselThrhs', 'preselNum']
    FinalQuaShortNames = {CLChi2: 'k2', CLKS: 'ks', CLCvMUw: 'cu', CLCvMCw: 'cw',
                          CLDCv: 'dc', CLSightRate: 'sr',
                          CLCmbQuaBal1: 'q1', CLCmbQuaBal2: 'q2', CLCmbQuaBal3: 'q3',
                          CLCmbQuaChi2: 'qk2', CLCmbQuaKS: 'qks', CLCmbQuaDCv: 'qdc'}

    def filSorSchemeId(self, scheme):

        """Human readable but unique identification of a filter and sort scheme
        
        Built on the scheme name format and an additional int suffix when needed.
        
        Definition: 2 equal schemes (dict ==) have the same Id

        Parameters:
        :param scheme: the scheme to identify
                       as a dict(method= filterSortOnXXX method to use,
                                 deduplicate= dict(dupSubset=, dDupRounds=) of deduplication params
                                     (if not or partially given, see filterSortOnXXX defaults)
                                 filterSort= dict of other <method> params (see filterSortOnXXX methods),
                                 preselCols= target columns for generating auto-preselection ones,
                                             containing [1, preselNum] ranks ; default: []
                                 preselAscs= Rank direction to use for each column (list),
                                             or all (single bool) ; default: True
                                             (True means that lower values are "better" ones)
                                 preselThrhs= Eliminating threshold for each column (list),
                                              or all (single number) ; default: 0.2
                                              (eliminated above if preselAscs True, below otherwise)
                                 preselNum= number of (best) pre-selections to keep for each sample) ;
                                            default: 5

        :return: the unique Id.
        """

        cls = self

        # Check scheme specification (1st level properties: presence of mandatory ones, authorised list, ...)
        props = scheme.keys()
        assert all(prop in self.MainSchSpecNames for prop in props), \
               'Unknown filter and sort scheme property/ies: {}' \
               .format(', '.join(prop for prop in props if prop not in self.MainSchSpecNames))
        mandProps = ['method']
        assert all(prop in props for prop in mandProps), \
               'Missing filter and sort scheme mandatory property/ies: {}' \
               .format(', '.join(prop for prop in mandProps if prop not in props))
        method = scheme['method']
        assert callable(method), 'Filter and sort scheme method must be callable'
        assert method is MCDSAnalysisResultsSet.filterSortOnExecCode \
               or method is MCDSAnalysisResultsSet.filterSortOnExCAicMulQua, \
               'Unsupported filter and sort scheme method: ' + str(method)

        # Compute the heading "name" part of the Id
        schemeId = 'ExCode' if method is MCDSAnalysisResultsSet.filterSortOnExecCode else 'ExAicMQua'
        methArgs = scheme.get('filterSort', {})
        if 'sightRate' in methArgs:
            schemeId += '-r{:.1f}'.format(methArgs['sightRate']).replace('.', '')
        if 'whichBestQua' in methArgs:
            schemeId += 'm{}'.format(len(methArgs['whichBestQua']))
        if 'whichFinalQua' in methArgs and method is not MCDSAnalysisResultsSet.filterSortOnExecCode:
            assert methArgs['whichFinalQua'] in cls.FinalQuaShortNames, \
                   'Unsupported quality indicator for filtering: ' + str(methArgs['whichFinalQua'])
            schemeId += cls.FinalQuaShortNames[methArgs['whichFinalQua']]
        if 'nFinalRes' in methArgs:
            schemeId += 'd{}'.format(methArgs['nFinalRes'])

        # The Id must be unique: enforce it through a uniqueness suffix only if needed
        if schemeId in self.dFilSorSchemes:  # Seems there's a possible collision ...
            closeSchemes = {schId: schVal for schId, schVal in self.dFilSorSchemes.items()
                            if schId.startswith(schemeId)}
            for schId, schVal in closeSchemes.items():  # Check if not simply an exact match
                if scheme == schVal:
                    return schId   # Bingo !
            schemeId += '-' + str(len(closeSchemes) + 1)  # No exact match => new unused Id !

        # Register new scheme and Id.
        self.dFilSorSchemes[schemeId] = scheme
        
        return schemeId

    @classmethod
    def _filterOnExecCode(cls, dfFilSorRes, filSorSteps, sampleIndCol, dupSubset, dDupRounds):

        """Inplace filter out results based on exec code and truncation params duplicates

        Details:
        1. drop results with error ExecCode >= 3
        2. drop results with identical truncation distances (keep best exec codes)

        Note: This doesn't actually modifies a single bit of the results set, but returns the resulting
              filtered and sorted index, suitable for indexing on self.dfData / dfTransData ...

        Parameters:
        :param dfFilSorRes: results table to update
        :param filSorSteps: filter and sort step list to update
        :param sampleIndCol: sample index = identification column
        :param dupSubset: Subset of (3-level multi-index) columns for detecting duplicates (as a list of tuples)
                          Warning: self.sampleIndCol is automatically prepended to this list if not already inside
        :param dDupRounds: {col: nb decimals} => number of decimals to keep (after rounding)
                           for a sub-set or all of dupSubset columns
        """

        # 1. Filter-out results obtained with some computation error (whatever sample).
        dfFilSorRes.drop(dfFilSorRes[dfFilSorRes[cls.CLRunStatus] > MCDSEngine.RCWarnings].index, inplace=True)
        stepId = 'run status'
        filSorSteps.append(stepId, 'column', cls.CLRunStatus, transColumns=True)
        filSorSteps.append(stepId, 'max value', MCDSEngine.RCWarnings)
        filSorSteps.append(stepId, 'results', len(dfFilSorRes))

        # 2. Filter-out results which are duplicates with respect to sample and truncation distances,
        # keeping best run status code first.
        dfFilSorRes.sort_values(by=[sampleIndCol, cls.CLParTruncLeft, cls.CLParTruncRight, cls.CLRunStatus],
                                ascending=True, na_position='first', inplace=True)
        if sampleIndCol not in dupSubset:
            dupSubset = [sampleIndCol] + dupSubset
        dfFilSorRes.drop(cls._indexOfDuplicates(dfFilSorRes, keep='first', subset=dupSubset, round2decs=dDupRounds),
                         inplace=True)
        stepId = 'duplicates on params'
        filSorSteps.append(stepId, 'param. names', dupSubset, transColumns=True)
        filSorSteps.append(stepId, 'param. precisions', ', '.join(filSorSteps.transColumns(miCol) + f': {nDec}'
                                                                  for miCol, nDec in dDupRounds.items()))
        filSorSteps.append(stepId, 'results', len(dfFilSorRes))

    LDupSubsetDef = [CLNObs, CLEffort, CLDeltaAic, CLChi2, CLKS, CLCvMUw, CLCvMCw, CLDCv,
                     CLPDetec, CLPDetecMin, CLPDetecMax, CLDensity, CLDensityMin, CLDensityMax]
    DDupRoundsDef = {CLDeltaAic: 1, CLChi2: 2, CLKS: 2, CLCvMUw: 2, CLCvMCw: 2, CLDCv: 2,
                     CLPDetec: 3, CLPDetecMin: 3, CLPDetecMax: 3, CLDensity: 2, CLDensityMin: 2, CLDensityMax: 2}

    @classmethod
    def _sortOnQua(cls, dfFilSorRes, filSorSteps, sampleIndCol, whichQua=CLCmbQuaBal3, ascendQua=False):

        """Inplace sort results per sample based on truncation distances and a customisable quality indicator

        Parameters:
        :param dfFilSorRes: results table to update
        :param filSorSteps: filter and sort step list to update
        :param sampleIndCol: sample index = identification column
        :param whichQua: Quality indicator (not order of) column to use
        :param ascendQua: if True, lower whichQua indicator values are better
        """

        sortCols = [cls.CLParTruncLeft, cls.CLParTruncRight, whichQua]
        dfFilSorRes.sort_values(by=[sampleIndCol] + sortCols, ascending=[True, True, True, ascendQua],
                                na_position='first', inplace=True)
        stepId = 'final sorting'
        filSorSteps.append(stepId, 'columns', sortCols, transColumns=True)
        filSorSteps.append(stepId, 'lower last column is better ?', ascendQua)

    def filterSortOnExecCode(self, schemeId, lang, whichFinalQua=CLCmbQuaBal3, ascFinalQua=False,
                             dupSubset=LDupSubsetDef, dDupRounds=DDupRoundsDef):

        """Minimal filter and sort scheme

        Details:
        1. drop results with error ExecCode >= 3
        2. drop results with identical truncation distances (keep best exec codes)
        3. sort on truncation distances and specified indicator

        Note: This doesn't actually modifies a single bit of the results set, but returns the resulting
              filtered and sorted index, suitable for indexing on self.dfData / dfTransData ...

        Parameters:
        :param schemeId: Scheme identification, for traceability
        :param lang: Translation language, for traceability
        :param whichFinalQua: Quality indicator (not order of) column to use for final sorting
        :param ascFinalQua: if True, lower whichFinalQua indicator values are better
        :param dupSubset: Subset of (3-level multi-index) columns for detecting duplicates (as a list of tuples)
                          Warning: self.sampleIndCol is automatically prepended to this list if not already inside
        :param dDupRounds: {col: nb decimals} => number of decimals to keep (after rounding)
                           for a sub-set or all of dupSubset columns

        :return: tuple(index of selected and sorted results, log of filter & sort steps accomplished)
        """

        cls = self

        logger.debug(f'Filter and sort scheme "{schemeId}": Applying.')

        filSorSteps = _FilterSortSteps(schemeId, resultsSet=self, lang=lang)

        # 0. Retrieve results to filter and sort.
        dfFilSorRes = self.getData(copy=True)
        filSorSteps.append('before', 'datetime', pd.Timestamp.now())
        filSorSteps.append('before', 'results', len(dfFilSorRes))

        # 1&2. Filter-out results with some computation error, and also duplicates based on same truncation params.
        self._filterOnExecCode(dfFilSorRes, filSorSteps, sampleIndCol=self.sampleIndCol,
                               dupSubset=dupSubset, dDupRounds=dDupRounds)

        # 3. Final sorting : increasing order for truncation distances (and final quality indicator if needed).
        self._sortOnQua(dfFilSorRes, filSorSteps, sampleIndCol=self.sampleIndCol,
                        whichQua=whichFinalQua, ascendQua=ascFinalQua)

        # Done.
        filSorSteps.append('after', 'results', len(dfFilSorRes))

        return dfFilSorRes.index, filSorSteps

    @classmethod
    def _filterOnAicMultiQua(cls, dfFilSorRes, filSorSteps, sampleIndCol,
                             minSightRate=90, nBestAicOrd=2,
                             nBestMQuaOrd=1, whichBestMQuaOrd=[CLGrpOrdClTrChi2KSDCv, CLGrpOrdClTrQuaBal3],
                             nFinalQua=10, whichFinalQua=CLCmbQuaBal3, ascFinalQua=False):

        """Inplace filter out results based on a mostly customisable selection of quality indicators

        Details:
        1. Per sample and group of IDENTICAL left and right truncation distances,
           keep only the results with CLGrpOrdSmTrAic < nBestAicOrd ;
           Note: this is generally close to: the nBestAicOrd best AIC values
            (and if equal, the best Chi2, KS, DCv, NObs, CodEx ... etc)
           but may actually give more than nBestAicOrd rows per sample and group of ...
        2. Per sample and group of close truncation distances (see _postComputeTruncationGroups),
           keep only the results with a least one of the whichBestMQuaOrd indicator orders < nBestMQuaOrd ;
           Note: this will probably give more than nBestMQuaOrd rows par sample and group of ...
        3. Eliminate sighting rates < minSightRate,
        4. Keep only the nFinalQua best results, with respect to whichFinalQua indicator
           (ascendFinalQua=True meaning that lower values are better)

        Parameters:
        :param dfFilSorRes: results table to update
        :param filSorSteps: filter and sort step list to update
        :param sampleIndCol: sample index = identification column
        :param minSightRate: Minimal observation rate (ratio of NTot Obs / NObs, seldom=1 because of dist. truncations)
        :param nBestAicOrd: Upper limit (excluded) for CLGrpOrdSmTrAic quality indicator order
                            (per sample and IDENTICAL truncation parameters)
        :param nBestMQuaOrd: Upper limit (excluded) for whichBestMQuaOrd for quality indicator orders
                             (keep results with at least 1 of them under the limit)
        :param whichBestMQua: Quality indicator order columns to use for filtering best results per sample
                              and close truncation distances
        :param nFinalQua: Final nb of best whichFinalQua indicator (not order of) results to keep per sample
        :param whichFinalQua: Quality indicator (not order of) column to use
                              for final "best results per sample" selection
        :param ascFinalQua: if True, lower whichFinalQua indicator values are better
        """

        # 1. Filter-out results with poorest AIC, per groups of same sample and IDENTICAL truncation distances.
        stepId = 'best AIC'
        filSorSteps.append(stepId, 'column', cls.CLGrpOrdSmTrAic, transColumns=True)
        filSorSteps.append(stepId, 'upper limit (excluded)', nBestAicOrd)
        dfFilSorRes.drop(dfFilSorRes[dfFilSorRes[cls.CLGrpOrdSmTrAic] >= nBestAicOrd].index,
                         inplace=True)
        filSorSteps.append(stepId, 'results', len(dfFilSorRes))

        # 2. Filter-out results not in N best ones for at least 1 in a specified set of quality indicator,
        # per groups of same sample and CLOSE truncation distances.
        stepId = 'best results for >= 1 indicator'
        filSorSteps.append(stepId, 'selected indicator orders', whichBestMQuaOrd, transColumns=True)
        filSorSteps.append(stepId, 'order limit (excluded) / indicator', nBestMQuaOrd)
        i2Drop = cls._indexOfWorstMultiOrderCriteria(dfFilSorRes, critCols=whichBestMQuaOrd, supCrit=nBestMQuaOrd)
        dfFilSorRes.drop(labels=i2Drop, inplace=True)
        filSorSteps.append(stepId, 'results', len(dfFilSorRes))

        # 3. Filter-out results with insufficient considered sightings rate
        #    (due to a small sample or truncations params).
        stepId = 'non-outlier sightings'
        filSorSteps.append(stepId, 'min sighting rate', minSightRate)
        dfFilSorRes.drop(dfFilSorRes[dfFilSorRes[cls.CLSightRate] < minSightRate].index, inplace=True)
        filSorSteps.append(stepId, 'results', len(dfFilSorRes))

        # 4. Filter-out eventually too numerous results, keeping only the N best ones
        #    with respect to the specified quality indicator (real indicator, not relevant close trunc order).
        stepId = 'final best results'
        filSorSteps.append(stepId, 'column', whichFinalQua, transColumns=True)
        filSorSteps.append(stepId, 'lower is better ?', ascFinalQua)
        filSorSteps.append(stepId, 'max results', nFinalQua)
        i2Drop = cls._indexOfWorstOneCriterion(dfFilSorRes, sampleIds=dfFilSorRes[sampleIndCol].unique(),
                                               sampleIdCol=sampleIndCol, nTgtRes=nFinalQua,
                                               critCol=whichFinalQua, ascendCrit=ascFinalQua)
        dfFilSorRes.drop(labels=i2Drop, inplace=True)
        filSorSteps.append(stepId, 'results', len(dfFilSorRes))

    def filterSortOnExCAicMulQua(self, schemeId, lang, sightRate=95, nBestAIC=2, nBestQua=1,
                                 whichBestQua=[CLGrpOrdClTrChi2KSDCv, CLGrpOrdClTrDCv, CLGrpOrdClTrQuaBal1,
                                               CLGrpOrdClTrQuaChi2, CLGrpOrdClTrQuaKS, CLGrpOrdClTrQuaDCv],
                                 nFinalRes=10, whichFinalQua=CLCmbQuaBal3, ascFinalQua=False,
                                 dupSubset=LDupSubsetDef, dDupRounds=DDupRoundsDef):

        """Filter and sort scheme for selecting best results with respect to a set of quality indicators,
        all taken at the same priority, except for one used to limit results at the end,
        all of this after same filtering on exec code and truncation param duplicates as in filterSortOnExecCode

        Details:
        1. Same filtering as filterSortOnExecCode
        2. Per sample and group of IDENTICAL left and right truncation distances,
           keep only the results with CLGrpOrdSmTrAic < nBestAIC ;
           Note: this is generally close to: the nBestAIC best AIC values
            (and if equal, the best Chi2, KS, DCv, NObs, CodEx ... etc)
           but may actually give more than nBestAIC rows per sample and group of ...
        3. Per sample and group of close truncation distances (see _postComputeTruncationGroups),
           keep only the results with a least one of the whichBestQua indicator orders < nBestQua ;
           Note: this will probably give more than nBestQua rows par sample and group of ...
        4. Eliminate sighting rates < sightRate,
        5. Keep only the nFinalRes best results, with respect to whichFinalQua indicator
           (ascendFinalQua=True meaning that lower values are better)
        6. Finally, sort by truncation distances (no truncation first, shorter distances first, ... simpler first)
           and by best whichFinalQua indicator values

        Note: This doesn't actually modifies a single bit of the results set, but returns the resulting
              filtered and sorted index, suitable for indexing on self.dfData / dfTransData ...

        Parameters:
        :param schemeId: Scheme identification, for traceability
        :param lang: Translation language, for traceability
        :param sightRate: Minimal observation rate (ratio of NTot Obs / NObs, not 1 because of dist. truncations)
        :param nBestAIC: Nb of best AIC results to keep per sample and IDENTICAL truncation parameters
        :param nBestQua: Nb of best results to keep per sample with respect to each quality indicator specified
                         through its related order column name in whichBestQua
        :param whichBestQua: Quality indicator order columns to use for filtering best results per sample
                             (at most nBestQua best results are kept for each related indicator ;
                              to be retained, a result MUST be among the nBestQua best ones for ALL
                              the specified indicators)
        :param nFinalRes: Final nb of best whichFinalQua results to keep per sample
        :param whichFinalQua: Quality indicator order column to use for final "best results per sample" selection
        :param ascFinalQua: if True, lower whichFinalQua indicator values are better
        :param dupSubset: Subset of (3-level multi-index) columns for detecting duplicates (as a list of tuples)
                          Warning: self.sampleIndCol is automatically prepended to this list if not already inside
        :param dDupRounds: {col: nb decimals} => number of decimals to keep (after rounding)
                           for a sub-set or all of dupSubset columns

        :return: tuple(index of selected and sorted results, log of filter & sort steps accomplished)
        """

        logger.debug(f'Filter and sort scheme "{schemeId}": Applying.')

        filSorSteps = _FilterSortSteps(schemeId, resultsSet=self, lang=lang)

        # 0. Retrieve results to filter and sort.
        dfFilSorRes = self.getData(copy=True)
        filSorSteps.append('before', 'datetime', pd.Timestamp.now())
        filSorSteps.append('before', 'results', len(dfFilSorRes))

        # 1. Filter-out results with some computation error, and also duplicates based on same truncation params.
        self._filterOnExecCode(dfFilSorRes, filSorSteps, sampleIndCol=self.sampleIndCol,
                               dupSubset=dupSubset, dDupRounds=dDupRounds)

        # 2. Filter-out results with poorest AIC, per groups of same sample and IDENTICAL truncation distances.
        # 3. Filter-out results with poorest values of specified quality indicators,
        #    per groups of same sample and CLOSE truncation distances.
        # 4. Filter-out results with insufficient considered sightings rate
        #    (due to a small sample or truncations params).
        # 5. Filter-out eventually too numerous results, keeping only the N best ones
        #    with respect to the specified quality indicator.
        self._filterOnAicMultiQua(dfFilSorRes, filSorSteps, sampleIndCol=self.sampleIndCol,
                                  minSightRate=sightRate, nBestAicOrd=nBestAIC,
                                  nBestMQuaOrd=nBestQua, whichBestMQuaOrd=whichBestQua,
                                  nFinalQua=nFinalRes, whichFinalQua=whichFinalQua, ascFinalQua=ascFinalQua)

        # 6. Final sorting : increasing order for truncation distances (and final quality indicator if needed).
        self._sortOnQua(dfFilSorRes, filSorSteps, sampleIndCol=self.sampleIndCol,
                        whichQua=whichFinalQua, ascendQua=ascFinalQua)

        # Done.
        filSorSteps.append('after', 'results', len(dfFilSorRes))

        return dfFilSorRes.index, filSorSteps

    def _preselColumn(self, srcCol):
        """Pre-selection column label (from source column) and translation"""
        return ((srcCol[0], srcCol[1], self.CLTPreSelection),
                dict(fr='Pré-sélection ' + self.transColumn(srcCol, 'fr'),
                     en='Pre-selection ' + self.transColumn(srcCol, 'en')))

    def _addPreselColumns(self, dfFilSorRes, filSorSteps,
                          preselCols=[CLCmbQuaBal1], preselAscend=True, preselThresh=[0.2], nSamplePreSels=5):

        """Add (in-place) a pre-selection column to a filtered and sorted translated results table

        Parameters:
        :param dfFilSorRes: the filtered and sorted table to update
        :param filSorSteps: the filtered and sorted step log to update
        :param nSamplePreSels: Max number of generated pre-selections per sample
        :param preselCols: Results columns to use for generating auto-preselection indices (in [1, nSamplePreSels])
        :param preselAscend: Order to use for each column (list[bool]), or all (single bool) ;
                             True means that lower values are "better" ones, and the other way round
        :param preselThresh: Value above (ascend=True) or below () which no preselection is proposed
                             (=> nan in target column), for each column (list[number]), or all (single number)

        """

        cls = self

        if isinstance(preselAscend, bool):
            preselAscend = [preselAscend] * len(preselCols)
        assert len(preselCols) == len(preselAscend), \
               'preselAscend must be a single bool or a list(bool) with len(preselCols)'

        if isinstance(preselThresh, (int, float)):
            preselThresh = [preselThresh] * len(preselCols)
        assert len(preselCols) == len(preselThresh), \
               'preselAscend must be a single number or a list(number) with len(preselCols)'

        filSorSteps.append('auto-preselection', 'Nb of pre-selections', nSamplePreSels)

        # Create each pre-selection column: rank per sample in preselCol/ascending (or not) order
        # up to nSamplePreSels (but no preselection under / over threshold).
        for srcCol, srcColAscend, srcColThresh in zip(preselCols, preselAscend, preselThresh):

            # Determine label and translation.
            tgtPreSelCol, dTgtPreSelColTrans = self._preselColumn(srcCol)
            self.addColumnsTrans({tgtPreSelCol: dTgtPreSelColTrans})

            filSorSteps.append('auto-preselection', 'pre-selection column', srcCol, transColumns=True)
            filSorSteps.append('auto-preselection', 'lower is better ?', srcColAscend)
            filSorSteps.append('auto-preselection', 'eliminating threshold', srcColThresh)

            # Compute contents and add to table
            # a. Rank all results
            dfFilSorRes.insert(dfFilSorRes.columns.get_loc(srcCol), tgtPreSelCol,
                               dfFilSorRes.groupby(self.miSampleCols.to_list())[[srcCol]]
                                          .transform(lambda s: s.rank(ascending=srcColAscend,
                                                                      method='dense', na_option='keep'))[srcCol])

            # b. Nullify too big ranks and "under" threshold values
            sbKillOnThresh = dfFilSorRes[srcCol] > srcColThresh if srcColAscend else dfFilSorRes[srcCol] < srcColThresh
            sbKillOnNumber = dfFilSorRes[tgtPreSelCol] > nSamplePreSels
            dfFilSorRes.loc[sbKillOnThresh | sbKillOnNumber, tgtPreSelCol] = np.nan

        # Create final empty selection column (for the user to self-decide at the end)
        # (right before the first added pre-selection column, no choice)
        if len(preselCols) > 0:
            dfFilSorRes.insert(dfFilSorRes.columns.get_loc(preselCols[0]) - 1, cls.CLFinalSelection, np.nan)

        return dfFilSorRes

    def dfFilSorData(self, scheme=dict(nameFmt='ExecCode', method=filterSortOnExecCode,
                                       filterSort=dict(whichFinalQua=CLCmbQuaBal3, ascFinalQua=False),
                                       preselCols=[CLCmbQuaBal3], preselAscs=False, preselThrhs=[0.2],
                                       preselNum=5),
                     columns=None, lang=None, rebuild=False):

        """Extract filtered and sorted data following the given scheme

        Note: Let R be MCDSAnalysisResultsSet, or a subclass (needed below).
        
        Parameters:
        :param scheme: filter and sort scheme to apply
                 as a dict(nameFmt= format string for generating the Id of the report
                           method= ResClass.filterSortOnXXX method to use,
                           deduplicate= dict(dupSubset=, dDupRounds=) of deduplication params
                               (if not or partially given, see RCLS.filterSortOnXXX defaults)
                           filterSort= dict of other <method> params,
                           preselCols= target columns for generating auto-preselection ones,
                                       containing [1, preselNum] ranks ; default: []
                           preselAscs= Rank direction to use for each column (list),
                                       or all (single bool) ; default: True
                                       (True means that lower values are "better" ones)
                           preselThrhs= Eliminating threshold for each column (list),
                                        or all (single number) ; default: 0.2
                                        (eliminated above if preselAscs True, below otherwise)
                           preselNum= number of (best) pre-selections to keep for each sample) ;
                                      default: 5)
                 examples: dict(nameFmt='ExecCode', => format string to generate the name of the report
                                method=R.filterSortOnExecCode,
                                filterSort=dict(whichFinalQua=CLCmbQuaBal3, ascFinalQua=False),
                                preselCols=[R.CLCmbQuaBal1, R.CLCmbQuaBal2], preselAscs=False,
                                preselThrhs=0.2, preselNum=5),
                           dict(nameFmt='AicCKCvQua-r{sightRate:.1f}d{nFinalRes}', 
                                method=R.filterSortOnExCAicMulQua,
                                deduplicate=dict(dupSubset=[R.CLNObs, R.CLEffort, R.CLDeltaAic, R.CLChi2,
                                                            R.CLKS, R.CLCvMUw, R.CLCvMCw, R.CLDCv]),
                                                 dDupRounds={R.CLDeltaAic: 1, R.CLChi2: 2, R.CLKS: 2,
                                                             R.CLCvMUw: 2, R.CLCvMCw: 2, R.CLDCv: 2})
                                filterSort=dict(sightRate=92.5, nBestAIC=3, nBestQua=1, 
                                                whichBestQua=[R.CLGrpOrdClTrChi2KSDCv, R.CLGrpOrdClTrDCv,
                                                              R.CLGrpOrdClTrQuaBal1, R.CLGrpOrdClTrQuaChi2,
                                                              R.CLGrpOrdClTrQuaKS, R.CLGrpOrdClTrQuaDCv],
                                                nFinalRes=12, whichFinalQua=R.CLCmbQuaBal1, ascFinalQua=False),
                                preselCols=[R.CLCmbQuaBal1, R.CLDCv], preselAscs=[False, True],
                                preselThrhs=[0.2, 0.5], preselNum=3)
        :param columns: Subset and order of columns to keep at the end (before translation) (None = [] = all)
                        Warning: No need to specify here pre-selection and final selection columns,
                                 as they'll be added automatically, and relocated at a non-customisable place.
        :param lang: Target language for column name translation (if None, no translation => keep original names)
        :param rebuild: If True, rebuild filtered and sorted table ; otherwise, simply reuse cached data
               if the results set didn't change enough meanwhile.

        :return: tuple(scheme id, result data-frame, log of completed filter and sort steps
                       as a list of [schemeId, step name, property name, property value])
        """

        # Check (1/2) if need for applying scheme : needed if rebuild requested or post-computation needed.
        isApplySchemeNeeded = rebuild or not self.postComputed
        filSorSchId = self.filSorSchemeId(scheme)

        # If not needed, check (2/2) also if same scheme have been applied yet
        if not isApplySchemeNeeded:
            iFilSor, filSorSteps = self.filSorCache.get(filSorSchId)
            isApplySchemeNeeded = iFilSor is None  # Not already applied => to apply now !

        # Apply scheme if needed.
        if isApplySchemeNeeded:

            # Do it => index of filtered and sorted results + log of steps
            iFilSor, filSorSteps = \
                scheme['method'](self, schemeId=filSorSchId, lang=lang or 'en',
                                 **scheme.get('filterSort', {}), **scheme.get('deduplicate', {}))

            # Update cache
            self.filSorCache.update(filSorSchId, iFilSor, filSorSteps)

        # Actually extract filtered and sorted rows and selected columns.
        dfFilSorRes = self.dfSubData(index=iFilSor, columns=columns, copy=True)

        # Add the preselection column (and update filter and sort log)
        if not dfFilSorRes.empty:
            dfFilSorRes = self._addPreselColumns(dfFilSorRes, filSorSteps,
                                                 nSamplePreSels=scheme.get('preselNum', 5),
                                                 preselCols=scheme.get('preselCols', []),
                                                 preselAscend=scheme.get('preselAscs', True),
                                                 preselThresh=scheme.get('preselThrhs', 0.2))

        # Final translation if specified.
        if lang:
            dfFilSorRes.columns = self.transColumns(dfFilSorRes.columns, lang)

        # Done.
        return filSorSchId, dfFilSorRes, filSorSteps.toList()


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
    IntSpecMinDist = 'MinDist'  # Left truncation distance
    IntSpecMaxDist = 'MaxDist'  # Right truncation distance
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
        sortAscend = [True] * len(sortCols)

        return miCustCols, dfCustColTrans, miSampCols, sampIndMCol, sortCols, sortAscend

    def setupResults(self, ldFilSorKeySchemes=None):
    
        """Build an empty results objects.

        Parameters:
        :param ldFilSorKeySchemes: Replacement for MCDSAnalysisResultsSet predefined filter-sort key schemes
                                   None => use predefined ones MCDSAnalysisResultsSet.AutoFilSorKeySchemes.
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
                                      truncIntrvEpsilon=self.truncIntrvEpsilon,
                                      ldFilSorKeySchemes=ldFilSorKeySchemes)
    
    def _getResults(self, dAnlyses):
    
        """Wait for and gather dAnalyses (MCDSAnalysis futures) results into a MCDSAnalysisResultsSet 
        """
    
        # Start of elapsed time measurement (yes, starting the analyses may take some time, but it is
        # negligible when compared to analysis time ; and better here for evaluating mean per analysis).
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

            # Report elapsed time and number of analyses completed until now
            # (once per self.logProgressEvery analyses though).
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
        customCols = self.resultsHeadCols['before'] + self.resultsHeadCols['sample'] + self.resultsHeadCols['after']
        
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
            logger.debug('Anlys params: {}'.format(', '.join(f'{k}:{v}' for k, v in dAnlysParams.items())))
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
                                 runtime=pd.Series(runtime, name='Version'))
        
        # Done.
        logger.info(f'Analyses completed ({len(self.results)} results).')

        return self.results


class MCDSPreAnalysisResultsSet(MCDSAnalysisResultsSet):

    """A specialized results set for MCDS pre-analyses
    (simpler post-computations that base class MCDSAnalysisResultsSet)"""
    
    # Computed columns specs (name translation + position).
    Super = MCDSAnalysisResultsSet
    _firstResColInd = len(MCDSEngine.statSampCols()) + len(MCDSAnalysis.MIRunColumns)
    DComputedCols = {Super.CLSightRate: _firstResColInd + 10,  # After Encounter Rate / Left|Right Trunc. Dist.
                     Super.CLDeltaAic: _firstResColInd + 12,  # Before AIC
                     Super.CLChi2: _firstResColInd + 14,  # Before all Chi2 tests
                     Super.CLDeltaDCv: _firstResColInd + 72,  # Before Density of animals / Cv
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
        self._postComputeDeltaAicDCv()
        self._postComputeQualityIndicators()


# Default strategy for pre-analyses model choice sequence (if one fails, take next in order, and so on)
ModelEstimCritDef = 'AIC'
ModelCVIntervalDef = 95
ModelStrategyDef = [dict(keyFn=kf, adjSr='COSINE', estCrit=ModelEstimCritDef, cvInt=ModelCVIntervalDef)
                    for kf in ['HNORMAL', 'HAZARD', 'UNIFORM', 'NEXPON']]


class MCDSPreAnalyser(MCDSAnalyser):

    """MCDSPreAnalyser: Run a bunch of MCDS pre-analyses
    """

    def __init__(self, dfMonoCatObs, dfTransects=None, effortConstVal=1, dSurveyArea=dict(),
                 transectPlaceCols=['Transect'], passIdCol='Pass', effortCol='Effort',
                 sampleSelCols=['Species', 'Pass', 'Adult', 'Duration'], sampleDecCols=['Effort', 'Distance'],
                 sampleSpecCustCols=[], abbrevCol='SampAbbrev', abbrevBuilder=None, sampleIndCol='SampleNum',
                 distanceUnit='Meter', areaUnit='Hectare',
                 surveyType='Point', distanceType='Radial', clustering=False,
                 resultsHeadCols=dict(before=['SampleNum'], after=['SampleAbbrev'], 
                                      sample=['Species', 'Pass', 'Adult', 'Duration']),
                 workDir='.', runMethod='subprocess.run', runTimeOut=300, logProgressEvery=5):

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
        :param dfExplSampleSpecs: Explicit sample specs, as a DataFrame
          (generated through explicitVariantSpecs, as an example),
        :param implSampleSpecs: Implicit sample specs, suitable for explicitation
          through explicitVariantSpecs
        :param dModelStrategy: Sequence of fallback models to use when analyses fails.
        :param threads: Number of parallel threads to use (default None: no parallelism, no asynchronism)
        """
    
        # Executor (parallel or sequential).
        self._executor = Executor(threads=threads)

        # MCDS analysis engine (a sequential one: 'cause MCDSPreAnalysis does the parallel stuff itself).

        # Failed try: Seems we can't stack ThreadPoolExecutors, as optimisations get run sequentially
        #             when using an Executor(threads=1) (means async) for self._engine ... 
        # engineExor = None if self.runMethod != 'os.system' or self.runTimeOut is None else Executor(threads=1)
        self._engine = MCDSEngine(workDir=self.workDir,  # executor=engineExor,
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
                continue  # No data => no analysis.

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
                                 analyser=self.flatSpecs(), runtime=pd.Series(runtime, name='Version'))
        
        # Done.
        logger.info('Analyses completed.')

        return self.results

    def exportDSInputData(self, dfExplSampleSpecs=None, implSampleSpecs=None, format='Distance'):
    
        """Export specified data samples to the specified DS input format, for "manual" DS analyses
        
        Parameters:
        :param dfExplSampleSpecs: Explicit sample specs, as a DataFrame
          (generated through explicitVariantSpecs, as an example),
        :param implSampleSpecs: Implicit sample specs, suitable for explicitation
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
