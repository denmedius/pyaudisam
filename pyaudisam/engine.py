# coding: utf-8
import shutil
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

# Submodule "engine": Interface to (external) DS engines

import sys
import os
import re
import pathlib as pl
import copy
import tempfile

from collections import namedtuple as ntuple

import subprocess as sproc

import numpy as np
import pandas as pd

from . import log, runtime

logger = log.logger('ads.eng', level=log.INFO)  # Initial config (can be changed later)

from .executor import Executor

# Keep Ruin'dows from opening a GPF dialog box every time a launched executable (like MCDS.exe) crashes !
# BUT: This does NOT work :-(
if sys.platform.startswith('win'):
    import ctypes
    import msvcrt
    ctypes.windll.kernel32.SetErrorMode(msvcrt.SEM_NOGPFAULTERRORBOX)

# Actual package install dir.
KInstDirPath = pl.Path(__file__).parent.resolve()


# DSEngine (abstract) classes.
# An engine for running multiple DS analyses with same options (engine ctor params),
# but various parameters (submitAnalysis() parameters), possibly as parallel threads / processes.
# Warning: No option change allowed while started analyses are running / all their getResults() have returned.
class DSEngine:
    
    # Possible values for options.
    DistUnits = ['Meter', 'Kilometer', 'Mile', 'Inch', 'Feet', 'Yard', 'Nautical mile']
    AreaUnits = ['Hectare', 'Acre'] + ['Sq. ' + distUnit for distUnit in DistUnits]
    
    # Forbidden chars in workDir path name (Distance DS engines are real red necks)
    # TODO: Stronger protection (more special chars ? more generic method, through re ?)
    ForbidPathChars = [' ', '(', ')', ',']
    
    # Find executable installation dir.
    @staticmethod
    def findExecutable(exeFileName, exeFileSearchPaths):

        exeFilePathName = None
        logger.info(f'Looking for {exeFileName} ...')
        for path in exeFileSearchPaths:
            exeFN = path / exeFileName
            if not exeFN.exists():
                logger.info(f'* checking {exeFN.as_posix()}: no,')
                continue

            logger.info(f'Found {exeFileName} here: {exeFN.as_posix()}.')
            exeFilePathName = exeFN
            break

        if not exeFilePathName:
            raise OSError(f'Could not find {exeFileName} ; please install Distance software (V6 or later)')

        return exeFilePathName
    
    # Find executable version.
    @staticmethod
    def findVersion():
        raise NotImplementedError('Abstract DSEngine.findVersion method must no be called: see derived classes')

    # Specifications of output stats.
    DfStatRowSpecs, DfStatModSpecs, DfStatModNotes, MIStatModCols, DfStatModColTrans = None, None, None, None, None
    
    def __init__(self, workDir='.', executor=None, runMethod='subprocess.run', timeOut=None,
                 distanceUnit='Meter', areaUnit='Hectare', **options):

        """Ctor
        :param workDir: As a simple str, or a pl.Path
        :param executor: Executor object to use (None => a sequential one will be auto-generated)
        :param runMethod: for calling engine executable : 'os.system' or 'subprocess.run'
        :param timeOut: engine call time limit (s) ; None => no limit ;
                WARNING: Not implemented (no way) for 'os.system' run method (see MCDSAnalysis for this)
        """

        # Check base options
        assert distanceUnit in self.DistUnits, \
               'Invalid distance unit {}: should be in {}'.format(distanceUnit, self.DistUnits)
        assert areaUnit in self.AreaUnits, \
               'Invalid area unit {}: should be in {}'.format(areaUnit, self.AreaUnits)
        
        # Save specific options (as a named tuple for easier use through dot operator).
        options = copy.deepcopy(options)
        options.update(distanceUnit=distanceUnit, areaUnit=areaUnit)
        self.OptionsClass = ntuple('Options', options.keys())
        self.options = self.OptionsClass(**options) 
        
        # Set executor for runAnalysis().
        self.executor = executor if executor is not None else Executor()
        assert timeOut is None or runMethod != 'os.system' or self.executor.isAsync(), \
               f"Can't care about {timeOut}s execution time limit" \
               " with a non-asynchronous executor and os.system run method"

        # Parameters for engine subprocess creation.
        self.runMethod = runMethod
        self.timeOut = timeOut
        
        # Check and prepare workdir if needed, and save.
        assert all(c not in str(workDir) for c in self.ForbidPathChars), \
               'Invalid character from "{}" in workDir folder "{}"' \
               .format(''.join(self.ForbidPathChars), workDir)
        self.workDir = pl.Path(workDir)
        self.workDir.mkdir(exist_ok=True)
        logger.info('DSEngine work folder: {}'.format(self.workDir.absolute().as_posix()))
    
    # Possible regexps for auto-detection of columns to import (into Distance / MCDS) from data sets / exports
    # (regexps are re.search'ed : any match _anywhere_inside_ the column name is OK).
    ImportFieldAliasREs = \
        {'STR_LABEL':  ['region', 'zone', 'secteur', 'strate', 'stratum'],
         'STR_AREA':   ['surface', 'area', 'ha', 'km2'],
         'SMP_LABEL':  ['point', 'lieu', 'location', 'transect'],
         'SMP_EFFORT': ['effort', 'passes', 'surveys', 'samplings', 'longueur', 'length'],
         'DISTANCE':   ['dist'],
         'ANGLE':      ['angl', 'azim', 'direct'],
         'SIZE':       ['nombre', 'nb', 'indiv', 'obj', 'tail', 'num', 'siz']}  # Cluster size
    
    # Data fields of decimal type.
    # TODO: Complete for non-'Point transect' modes
    DecimalFields = ['SMP_EFFORT', 'DISTANCE', 'ANGLE']
    
    # Match srcFields with tgtAliasREs ones ; keep remaining ones ; sort decimal fields.
    @classmethod
    def matchDataFields(cls, srcFields, tgtAliasREs={}):
        
        logger.debug('Matching required data columns:')
        
        # Try and match required data columns.
        matFields = list()
        matDecFields = list()
        for tgtField in tgtAliasREs:
            logger.debug1(' * ' + tgtField + ':')
            foundTgtField = False
            for srcField in srcFields:
                for pat in tgtAliasREs[tgtField]:
                    if re.search(pat, srcField, flags=re.IGNORECASE):
                        logger.debug2('  . ' + srcField)
                        matFields.append(srcField)
                        if tgtField in cls.DecimalFields:
                            matDecFields.append(srcField)
                        foundTgtField = True
                        break
                if foundTgtField:
                    break
            if not foundTgtField:
                raise KeyError('Could not find a match for expected field {} in sample data set columns [{}]'
                               .format(tgtField, ', '.join(srcFields)))
        
        # Extra fields.
        extFields = [field for field in srcFields if field not in matFields]

        logger.debug('... success.')
        
        return matFields, matDecFields, extFields

    # Set up a thread & process safe run folder for an analysis
    # * runPrefix : user-friendly prefix for the generated folder-name (maybe None)
    # Note: Not a class method because it uses self.workDir
    def setupRunFolder(self, runPrefix=None):

        # MCDS does not support folder and file names with spaces inside ...
        # And one never knows ... remove any other special chars also.
        if runPrefix is None:
            runPrefix = ''
        else:
            runPrefix = runPrefix.translate(str.maketrans({c: '' for c in ' ,.:;()/'})) + '-'

        return pl.Path(tempfile.mkdtemp(dir=self.workDir, prefix=runPrefix))

    # Engine in/out file names.
    CmdFileName = 'cmd.txt'
    DataFileName = 'data.txt'
    OutputFileName = 'output.txt'
    LogFileName = 'log.txt'
    StatsFileName = 'stats.txt'
    PlotsFileName = 'plots.txt'
        
    # Shutdown : release any used resource.
    # Post-condition: Instance can no more run analyses.
    def shutdown(self, executor=True):
    
        if executor and self.executor is not None:
            self.executor.shutdown()
        self.executor = None
   
    def __del__(self):
    
        self.shutdown()


# MCDS engine (Conventional Distance Sampling)
class MCDSEngine(DSEngine):
    
    # Possible survey types and distance types.
    SurveyTypes = ['Point', 'Line']
    DistTypes = ['Radial', 'Perpendicular', 'Radial & Angle']
    
    # First data fields in exports for distance / MCDS importing
    # (N/A ones may get removed before use, according to distance type and clustering options).
    FirstDataFields = {'Point': ['STR_LABEL', 'STR_AREA', 'SMP_LABEL', 'SMP_EFFORT', 'DISTANCE', 'SIZE'],
                       'Line': ['STR_LABEL', 'STR_AREA', 'SMP_LABEL', 'SMP_EFFORT', 'DISTANCE', 'ANGLE', 'SIZE']}

    # Estimator key functions (Order: Distance .chm doc, "MCDS Engine Stats File", note 2 below second table).
    EstKeyFns = ['UNIFORM', 'HNORMAL', 'NEXPON', 'HAZARD']
    EstKeyFnDef = EstKeyFns[0]

    # Estimator adjustment series (Order: Distance .chm doc, "MCDS Engine Stats File", note 3 below second table).
    EstAdjustFns = ['POLY', 'HERMITE', 'COSINE']
    EstAdjustFnDef = EstAdjustFns[0]

    # Estimator adjustment terms selection criterion.
    EstCriteria = ['AIC', 'AICC', 'BIC', 'LR']
    EstCriterionDef = EstCriteria[0]
    
    # Estimator confidence value for output interval.
    EstCVIntervalDef = 95  # %
    
    # Distance truncation / cut points parameters.
    DistMinDef = None  # No left truncation (min = 0)
    DistMaxDef = None  # No right truncation (max = max. distance observed)
    DistFitCutsDef = None  # Model fitting : Automatic engine distance cuts determination.
    DistDiscrCutsDef = None  # Distance values discretisation : None (keep exact values).

    # MCDS Executable
    # * MCDS.exe is an autonomous executable: you can put a copy anywhere, or nearly ... see below
    # * it'll be searched first in the folders (separated by a ;) referenced in the MCDS_PATH env. variable,
    # * then in a "Distance 7" sub-folder of ., then C:/PortableApps, C:/Program files (x86), C:/Program files,
    # * then in a "Distance 6" sub-folder of ., then C:/PortableApps, C:/Program files (x86), C:/Program files.
    # (if you install Distance 7 (or 6, or newer) the normal way, it will then of course be found :-)
    _ExeFileSearchPaths = [pl.Path(path) for path in os.environ.get('MCDS_PATH', '').split(';') if path] \
                           + [pl.Path(rootFolder) / f'Distance {majorVersion}'
                              for majorVersion in [7, 6]  # Latest first.
                              for rootFolder in ['.', 'C:/PortableApps', 'C:/Program files (x86)', 'C:/Program files']]
    ExeFilePathName = DSEngine.findExecutable('MCDS.exe', _ExeFileSearchPaths)

    # Output stats specs : load from external files (extracts from Distance doc).
    @classmethod
    def loadStatSpecs(cls, nMaxAdjParams=10):
        
        if MCDSEngine.DfStatRowSpecs is not None:
            return
        
        logger.debug('MCDS : Loading output stats specs ...')
        
        # Output stats row specifications
        fileName = KInstDirPath / 'mcds/stat-row-specs.txt'
        logger.debug1('* {}'.format(fileName))
        with open(fileName, mode='r', encoding='utf8') as fStatRowSpecs:
            statRowSpecLines = [line.rstrip('\n') for line in fStatRowSpecs.readlines() if not line.startswith('#')]
            statRowSpecs = [(statRowSpecLines[i].strip(), statRowSpecLines[i+1].strip())
                            for i in range(0, len(statRowSpecLines)-2, 3)]
            cls.DfStatRowSpecs = pd.DataFrame(columns=['Name', 'Description'],
                                              data=statRowSpecs).set_index('Name')
            assert not cls.DfStatRowSpecs.empty, 'Empty MCDS stats row specs'
        
        # Module and stats number to description table
        fileName = KInstDirPath / 'mcds/stat-mod-specs.txt'
        logger.debug1('* ' + fileName.as_posix())
        with open(fileName, mode='r', encoding='utf8') as fStatModSpecs:
            statModSpecLines = [line.rstrip('\n') for line in fStatModSpecs.readlines() if not line.startswith('#')]
            reModSpecNumName = re.compile('(.+) – (.+)')
            statModSpecs = list()
            moModule = None
            for line in statModSpecLines:
                # logger.debug2(line)
                if not line:
                    continue
                if moModule is None:
                    moModule = reModSpecNumName.match(line.strip())
                    continue
                if line == ' ':
                    moModule = None
                    continue
                moStatistic = reModSpecNumName.match(line.strip())
                modNum, modDesc, statNum, statDescNotes = \
                    moModule.group(1), moModule.group(2), moStatistic.group(1), moStatistic.group(2)
                for i in range(len(statDescNotes) - 1, -1, -1):
                    if not re.match(r'[\d ,]', statDescNotes[i]):
                        statDesc = statDescNotes[:i + 1]
                        statNotes = statDescNotes[i + 1:].replace(' ', '')
                        break
                modNum = int(modNum)
                if statNum.startswith('101 '):
                    for num in range(nMaxAdjParams):  # Assume no more than that ... a bit hacky !
                        statModSpecs.append((modNum, modDesc, 101+num,  # Make statDesc unique for later indexing
                                             statDesc.replace('each', f'A({num+1})'), statNotes))
                else:
                    statNum = int(statNum)
                    if modNum == 2 and statNum in [3, 20, 21, 22]:
                        if statNum == 3:
                            # MCDS 6.2: Actually, there are 0 or 3 of these, but anyway ...
                            for num in range(3):
                                statModSpecs.append((modNum, modDesc, num + 1 + 200,
                                                     # Change statNum & Make statDesc unique for later indexing
                                                     statDesc + f' (distance set {num + 1})', statNotes))
                        else:
                            # MCDS 7.4: 20, 21 & 22 are new and undocumented, but assume behaviour same as for 3.
                            for num in range(3):
                                statModSpecs.append((modNum, modDesc, num + 1 + 10 * (statNum + 1),
                                                     # Change statNum & Make statDesc unique for later indexing
                                                     statDesc + f' (distance set {num + 1})', statNotes))
                    else:
                        statModSpecs.append((modNum, modDesc, statNum, statDesc, statNotes))
            cls.DfStatModSpecs = pd.DataFrame(columns=['modNum', 'modDesc', 'statNum', 'statDesc', 'statNotes'],
                                              data=statModSpecs).set_index(['modNum', 'statNum'])
            assert not cls.DfStatModSpecs.empty, 'Empty MCDS stats module specs'
        
        # Produce a MultiIndex for output stats as columns in the desired order.
        indexItems = list()
        for lbl, modDesc, statDesc, statNotes in cls.DfStatModSpecs[['modDesc', 'statDesc', 'statNotes']].itertuples():
            indexItems.append((modDesc, statDesc, 'Value'))
            if '1' in statNotes:
                indexItems += [(modDesc, statDesc, valName) for valName in ['Cv', 'Lcl', 'Ucl', 'Df']]
        cls.MIStatModCols = pd.MultiIndex.from_tuples(indexItems)
    
        # Notes about stats.
        fileName = KInstDirPath / 'mcds/stat-mod-notes.txt'
        logger.debug1('* {}'.format(fileName))
        with open(fileName, mode='r', encoding='utf8') as fStatModNotes:
            statModNoteLines = [line.rstrip('\n') for line in fStatModNotes.readlines() if not line.startswith('#')]
            statModNotes = [(int(line[:2]), line[2:].strip()) for line in statModNoteLines if line]
            cls.DfStatModNotes = pd.DataFrame(data=statModNotes, columns=['Note', 'Text']).set_index('Note')
            assert not cls.DfStatModNotes.empty, 'Empty MCDS stats module notes'
            
        # DataFrame for translating 3-level multi-index columns to 1 level lang-translated columns
        fileName = KInstDirPath / 'mcds/stat-mod-trans.txt'
        logger.debug1('* {}'.format(fileName))
        cls.DfStatModColsTrans = pd.read_csv(fileName, sep='\t')
        cls.DfStatModColsTrans.set_index(['Module', 'Statistic', 'Figure'], inplace=True)
        
        logger.debug('MCDS : Loaded output stats specs.')
        
    # Accessors to class variables.
    @classmethod
    def statRowSpecs(cls):
        
        if cls.DfStatRowSpecs is None:
            cls.loadStatSpecs()    

        return cls.DfStatRowSpecs
        
    @classmethod
    def statModSpecs(cls):
        
        if cls.DfStatModSpecs is None:
            cls.loadStatSpecs()    

        return cls.DfStatModSpecs
        
    @classmethod
    def statModCols(cls):
        
        if cls.MIStatModCols is None:
            cls.loadStatSpecs()    

        return cls.MIStatModCols
        
    @classmethod
    def statModNotes(cls):
        
        if cls.DfStatModNotes is None:
            cls.loadStatSpecs()    

        return cls.DfStatModNotes
        
    @classmethod
    def statModColTrans(cls):
        
        if cls.DfStatModColsTrans is None:
            cls.loadStatSpecs()    

        return cls.DfStatModColsTrans

    # Sample stats columns
    MIStatSampCols = pd.MultiIndex.from_tuples([('sample stats', 'total number of observations', 'Value'),
                                                ('sample stats', 'minimal observation distance', 'Value'),
                                                ('sample stats', 'maximal observation distance', 'Value')])

    @classmethod
    def statSampCols(cls):
        
        return cls.MIStatSampCols
    
    # Sample stats columns translation
    _DfStatSampColsTrans = \
        pd.DataFrame(index=MIStatSampCols,
                     data=dict(en=['NTot Obs', 'Min Dist', 'Max Dist'], fr=['NTot Obs', 'Min Dist', 'Max Dist']))

    @classmethod
    def statSampColTrans(cls):

        return cls._DfStatSampColsTrans

    def __init__(self, workDir='.', executor=None, runMethod='subprocess.run', timeOut=None,
                 distanceUnit='Meter', areaUnit='Hectare',
                 surveyType='Point', distanceType='Radial', clustering=False):
        
        """Ctor.
        :param workDir: As a simple str, or a pl.Path
        :param executor: Executor object to use (None => a sequential one will be auto-generated)
        :param runMethod: Method used to run the MCDS executable : os.system or subprocess.run
        :param timeOut: Time-out (s) for analysis execution (None => no limit);
                        WARNING: NOT implemented here when 'os.system' runMethod ... see MCDSAnalysis
        """

        # Initialize dynamic class variables.
        MCDSEngine.loadStatSpecs()    

        # Check options
        assert surveyType in self.SurveyTypes, \
               'Invalid survey type {} : should be in {}'.format(surveyType, self.SurveyTypes)
        assert distanceType in self.DistTypes, \
               'Invalid distance type {} : should be in {}'.format(distanceType, self.DistTypes)
        
        # Specialise class level regexps for matching import fields,
        # according to distance type and clustering options
        self.importFieldAliasREs = self.ImportFieldAliasREs.copy()
        if not clustering:
            del self.importFieldAliasREs['SIZE']
        if distanceType != 'Radial & Angle':
            del self.importFieldAliasREs['ANGLE']            
    
        # Initialise base.
        firstDataFields = [fld for fld in self.FirstDataFields[surveyType] if fld in self.importFieldAliasREs]
        super().__init__(workDir=workDir, executor=executor, runMethod=runMethod, timeOut=timeOut,
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         surveyType=surveyType, distanceType=distanceType, clustering=clustering,
                         firstDataFields=firstDataFields)
                         
    # Command file template (for str.format()ing).
    CmdTxt = \
        '\n'.join(map(str.strip,
                  """{output}
                     {log}
                     {stats}
                     {plots}
                     {bootstrap}
                     {bootpgrss}
                     Options;
                     Type={survType};
                     Distance={distType} /Measure='{distUnit}';
                     Area /Units='{areaUnit}';
                     Object=Single;
                     SF=1;
                     Selection=Sequential;
                     Lookahead=1;
                     Maxterms=5;
                     Confidence={cvInterv};
                     print=Selection;
                     End;
                     Data /Structure=Flat;
                     Fields={dataFields};
                     Infile={dataFileName} /{echoData};
                     End;
                     Estimate;
                     Distance{distDiscrSpecs};
                     Density=All;
                     Encounter=All;
                     Detection=All;
                     Size=All;
                     Estimator /Key={estKeyFn} /Adjust={estAdjustFn} /Criterion={estCriterion};
                     Monotone=Strict;
                     Pick=AIC;
                     GOF{gOFitSpecs};
                     Cluster /Bias=GXLOG;
                     VarN=Empirical;
                     End;
                  """.split('\n'))) + '\n'
    
    # Build command file from options and params
    # * runDir : pl.Path where to create cmd file.
    # Note: Not a classmethod because it uses self.options
    def buildCmdFile(self, runDir, **params):

        # Default params values
        if 'logData' not in params:
            params['logData'] = False
        if 'estimKeyFn' not in params:
            params['estimKeyFn'] = self.EstKeyFnDef
        if 'estimAdjustFn' not in params:
            params['estimAdjustFn'] = self.EstAdjustFnDef
        if 'estimCriterion' not in params:
            params['estimCriterion'] = self.EstCriterionDef
        if 'cvInterval' not in params:
            params['cvInterval'] = self.EstCVIntervalDef
        if 'maxDist' not in params:
            params['maxDist'] = self.DistMaxDef
        if 'minDist' not in params:
            params['minDist'] = self.DistMinDef
        if 'fitDistCuts' not in params:
            params['fitDistCuts'] = self.DistFitCutsDef
        if 'discrDistCuts' not in params:
            params['discrDistCuts'] = self.DistDiscrCutsDef

        # Generate file contents
        # a. Compute non-trivial data fields
        distDiscrSpecs = ''
        gOFitSpecs = ''
        
        minDist = params['minDist']
        maxDist = params['maxDist']
        fitDistCuts = params['fitDistCuts']
        discrDistCuts = params['discrDistCuts']
        if discrDistCuts is not None:
            
            if isinstance(discrDistCuts, list):
                assert not (minDist is None or maxDist is None)
                distDiscrSpecs += ' /Int=' + ','.join(format(d, 'g') for d in [minDist] + discrDistCuts + [maxDist])
            elif isinstance(discrDistCuts, (int, float)):
                distDiscrSpecs += ' /NClass=' + format(discrDistCuts, 'g')
            # Other cases not supported, should be asserted by the caller.
        
        elif fitDistCuts is not None:  # Can't fit model on other distance intervals than used for discretisation.
            
            if isinstance(fitDistCuts, list):
                assert not (minDist is None or maxDist is None)
                gOFitSpecs += ' /Int=' + ','.join(format(d, 'g') for d in [minDist] + fitDistCuts + [maxDist])
            elif isinstance(fitDistCuts, (int, float)):
                gOFitSpecs += ' /NClass=' + format(fitDistCuts, 'g')
            # Other cases not supported, should be asserted by the caller.
                
        if minDist is not None:
            distDiscrSpecs += ' /Left=' + format(minDist, 'g')

        if maxDist is not None:
            distDiscrSpecs += ' /Width=' + format(maxDist, 'g')
            
        # b. Format contents string
        cmdTxt = self.CmdTxt.format(output=runDir/self.OutputFileName, log=runDir/self.LogFileName,
                                    stats=runDir/self.StatsFileName, plots=runDir/self.PlotsFileName,
                                    bootstrap='None', bootpgrss='None',  # No support for the moment.
                                    survType=self.options.surveyType, distType=self.options.distanceType,
                                    distUnit=self.options.distanceUnit, areaUnit=self.options.areaUnit,
                                    dataFields=','.join(self.options.firstDataFields),
                                    dataFileName=runDir/self.DataFileName,
                                    echoData=('' if params['logData'] else 'No') + 'Echo',
                                    estKeyFn=params['estimKeyFn'], estAdjustFn=params['estimAdjustFn'],
                                    estCriterion=params['estimCriterion'], cvInterv=params['cvInterval'],
                                    distDiscrSpecs=distDiscrSpecs, gOFitSpecs=gOFitSpecs)

        # Write file.
        cmdFileName = runDir / self.CmdFileName
        with open(cmdFileName, mode='w', encoding='utf-8') as cmdFile:
            cmdFile.write(cmdTxt)

        logger.debug('Commands written to {}'.format(cmdFileName))

        # Done.
        return cmdFileName
    
    # Workaround pd.DataFrame.to_csv(float_format='%.xf') not working when NaNs in serie
    @staticmethod
    def safeFloat2Str(val, prec=None, decPt='.'):
        strVal = '' if pd.isnull(val) else str(val) if prec is None \
                    else '{:.{prec}f}'.format(val, prec=prec)
        if decPt != '.':
            strVal = strVal.replace('.', decPt)
        return strVal

    # Build input data table from a sample data set (check and match mandatory columns, enforce order).
    # TODO: Add support for covariate columns (through extraFields)
    def buildExportTable(self, sampleDataSet, withExtraFields=True, decPoint='.'):
        
        # Match sampleDataSet table columns to MCDS expected fields from possible aliases
        matchFields, matchDecFields, extraFields = \
            self.matchDataFields(sampleDataSet.dfData.columns, self.importFieldAliasREs)
        exportFields = matchFields
        if withExtraFields:
            exportFields += extraFields
        else:
            extraFields.clear()
        
        logger.debug2('Final data columns export order: ' + str(exportFields))
        
        # Put columns in the right order (first data fields ... first, in the same order)
        dfExport = sampleDataSet.dfData[exportFields].copy()

        # Prepare safe export of decimal data with maybe some NaNs
        allDecFields = set(matchDecFields + sampleDataSet.decimalFields).intersection(exportFields)
        logger.debug2('Decimal columns: ' + str(allDecFields))
        for field in allDecFields:
            dfExport[field] = dfExport[field].apply(self.safeFloat2Str, decPt=decPoint)
                
        return dfExport, extraFields

    # Build MCDS input data file from a sample data set.
    # Note: Data = sighting order not changed, same as in input sampleDataSet !
    # * runDir : pl.Path where to create data file.
    # TODO: Add support for covariate columns (through extraFields)
    def buildDataFile(self, runDir, sampleDataSet):
        
        # Build data to export (check and match mandatory columns, enforce order, ignore extra cols).
        dfExport, extraFields = \
            self.buildExportTable(sampleDataSet, withExtraFields=False, decPoint='.')
        
        # Export.
        dataFileName = runDir / self.DataFileName
        dfExport.to_csv(dataFileName, index=False, sep='\t', encoding='utf-8', header=None)
        
        logger.debug('Data MCDS-exported to {}'.format(dataFileName))
        
        return dataFileName
    
    # Run status codes (from MCDS documentation)
    RCNotRun = 0
    RCOK = 1
    RCWarnings = 2
    RCErrors = 3
    RCFileErrors = 4
    RCOtherErrors = 5  # and above, straight from MCDS.exe.
    RCTimedOut = 555  # as named (through subprocess or concurrent.futures modules)
    
    @classmethod
    def wasRun(cls, runCode):
        return runCode != cls.RCNotRun
    
    @classmethod
    def success(cls, runCode):
        return runCode == cls.RCOK
    
    @classmethod
    def warnings(cls, runCode):
        return runCode == cls.RCWarnings
    
    @classmethod
    def errors(cls, runCode):
        return runCode >= cls.RCErrors
    
    @classmethod
    def _runThroughOSSystem(cls, execFileName, cmdFileName, forReal=True):

        """Run MCDS command through os.system

        Under Ruin'dows, this means an intermediate "cmd.exe" subprocess.
        """

        # Call executable (no " around cmdFile, don't forget the space after ',', ...)
        cmd = '"{}" 0, {}'.format(execFileName, cmdFileName)
        if forReal:
            logger.info1(f'Running MCDS through os.system({cmd}) ...')
            startTime = pd.Timestamp.now()
            status = os.system(cmd)
            elapsedTime = (pd.Timestamp.now() - startTime).total_seconds()
            logger.info2(f'... MCDS done : status={status}, elapsed={elapsedTime:.2f}s')
            
        # ... unless specified not to (input files generated, but no execution).
        else:
            logger.info1(f'NOT running MCDS through os.system({cmd}).')
            startTime = pd.NaT
            status = cls.RCNotRun
            elapsedTime = 0

        return status, startTime, elapsedTime

    # MCDS.exe subprocess creation flags under ruin'dows: no window please !
    # BUT: Does not help with fucking Ruin'dows crash window ... alas !
    ExeCrFlags = sproc.CREATE_NO_WINDOW if sys.platform.startswith('win') else 0

    @classmethod
    def _runThroughSubProcessRun(cls, execFileName, cmdFileName, forReal=True, timeOut=None):

        """Run MCDS command through subprocess.run

        Under Ruin'dows, no "cmd.exe" intermediate subprocess, but only a conhost.exe.
        """

        # Call executable (no " around cmdFile, don't forget the space after ',', ...)
        cmd = [str(execFileName), '0,', str(cmdFileName)]
        if forReal:
            logger.info1(f'Running MCDS through subprocess.run({cmd}, ) ...')
            startTime = pd.Timestamp.now()
            try:
                proc = sproc.run(cmd, text=True, stdout=sproc.PIPE, stderr=sproc.STDOUT,
                                 timeout=timeOut, creationflags=cls.ExeCrFlags)
                status = proc.returncode
                stdouterr = proc.stdout
            except sproc.TimeoutExpired as toExc:
                logger.error(f'MCDS timed out after {toExc.timeout:.2f}s')
                status = cls.RCTimedOut
                stdouterr = toExc.stdout
            elapsedTime = (pd.Timestamp.now() - startTime).total_seconds()
            logger.info3('MCDS stdout&err:')
            logger.info3(stdouterr)
            logger.info2(f'... MCDS done : status={status}, elapsed={elapsedTime:.2f}s')

        # ... unless specified not to (input files generated, but no execution).
        else:
            logger.info1(f'NOT running MCDS through subprocess.run({cmd}).')
            startTime = pd.NaT
            status = cls.RCNotRun
            elapsedTime = 0

        return status, startTime, elapsedTime

    @classmethod
    def _run(cls, execFileName, cmdFileName, forReal=True, method='subprocess.run', timeOut=None):

        """Run MCDS command through the given method
        """

        if method == 'os.system':
            return cls._runThroughOSSystem(execFileName, cmdFileName, forReal=forReal)
        elif method == 'subprocess.run':
            return cls._runThroughSubProcessRun(execFileName, cmdFileName, forReal=forReal, timeOut=timeOut)

        raise NotImplementedError(f'Unknown MCDSEngine run method "{method}"')

    # Run 1 MCDS analysis from the beginning to the end (blocking for the calling thread)
    # * runPrefix : user-friendly prefix for the generated folder-name (maybe None)
    def _runAnalysis(self, sampleDataSet, runPrefix='mcds', realRun=True, **analysisParms):
        
        # Create a new exclusive thread and process-safe run folder
        anlysStartTime = pd.Timestamp.now()
        runDir = self.setupRunFolder(runPrefix)
        logger.debug('Will run in {}'.format(runDir))
        
        # Generate data and command files into this folder
        logger.info2('MCDS analysis params: ' + str(analysisParms))
        self.buildDataFile(runDir, sampleDataSet)
        cmdFileName = self.buildCmdFile(runDir, **analysisParms)
        
        anlysElapsedTime = (pd.Timestamp.now() - anlysStartTime).total_seconds()

        # Run executable as an OS sub-process.
        runStatus, engStartTime, engElapsedTime = \
            self._run(self.ExeFilePathName, cmdFileName, forReal=realRun,
                      method=self.runMethod, timeOut=self.timeOut)
        anlysElapsedTime += engElapsedTime

        # Extract and decode results.
        startTime = pd.Timestamp.now()

        if self.success(runStatus) or self.warnings(runStatus):
            sResults = self.decodeStats(runDir)
        else:
            sResults = None

        anlysElapsedTime += (pd.Timestamp.now() - startTime).total_seconds()

        return runStatus, anlysStartTime, anlysElapsedTime + engElapsedTime, runDir, sResults
    
    # Start running an MCDS analysis, using the executor (possibly asynchronously if it is not a sequential one)
    def submitAnalysis(self, sampleDataSet, runPrefix='mcds', realRun=True, **analysisParms):
        
        # Check really implemented options
        assert self.options.surveyType == 'Point', \
               'Not yet implemented survey type {}'.format(self.options.surveyType)
        assert self.options.distanceType == 'Radial', \
               'Not yet implemented distance type {}'.format(self.options.distanceType)
        
        # Submit analysis work and return a Future object to ask from and wait for its results.
        return self.executor.submit(self._runAnalysis, sampleDataSet, runPrefix, realRun, **analysisParms)
    
    # Decode output stats file to a value series
    # Precondition: self.runAnalysis(...) was called and took place in :param:runDir
    # * runDir : string or pl.Path where to find stats file.
    # Warning: No support for more than 1 stratum, 1 sample, 1 estimator.
    @classmethod
    def decodeStats(cls, runDir):

        statsFileName = pl.Path(runDir) / cls.StatsFileName
        logger.debug('Decoding stats from {} ...'.format(statsFileName))
        
        # 1. Load table (text format, with space separated and fixed width columns,
        #    columns headers from cls.DfStatRowSpecs)
        dfStats = pd.read_csv(statsFileName, sep=' +', engine='python', names=cls.DfStatRowSpecs.index)
        dfInitStats = dfStats.copy()
        
        # 2. Remove Stratum, Sample and Estimator columns (no support for multiple ones for the moment)
        dfStats.drop(columns=['Stratum', 'Sample', 'Estimator'], inplace=True)
        
        # 3. Stack figure columns to rows, to get more comfortable
        dfStats.set_index(['Module', 'Statistic'], append=True, inplace=True)
        dfStats = dfStats.stack().reset_index()
        dfStats.rename(columns={'level_0': 'id', 'level_3': 'Figure', 0: 'Value'}, inplace=True)

        # 4. Fix multiple Module=2 & Statistic=3, 20, 21, 22 rows (before joining with cls.DfStatModSpecs)
        #    => change Statistic column for unicity = instance unambiguous identification
        newStatNum = 200
        for lbl, sRow in dfStats[(dfStats.Module == 2) & (dfStats.Statistic == 3)].iterrows():
            if dfStats.at[lbl, 'Figure'] == 'Value':
                newStatNum += 1
            dfStats.at[lbl, 'Statistic'] = newStatNum
        for oldStatNum in [20, 21, 22]:
            newStatNum = 10 * (oldStatNum + 1)
            for lbl, sRow in dfStats[(dfStats.Module == 2) & (dfStats.Statistic == oldStatNum)].iterrows():
                if dfStats.at[lbl, 'Figure'] == 'Value':
                    newStatNum += 1
                dfStats.at[lbl, 'Statistic'] = newStatNum

        # 5. Add descriptive / naming columns for modules and statistics,
        #    from cls.DfStatModSpecs (more user-friendly than numeric ids + help for detecting N/A figures)
        dfStats = dfStats.join(cls.DfStatModSpecs, on=['Module', 'Statistic'])
        
        # 6. Check that supposed N/A figures (as told by cls.DfStatModSpecs.statNotes) are really such
        #    Warning: There seems to be a bug in MCDS with Module=2 & Statistic=10x : some Cv values not always 0 ...
        sKeepOnlyValueFig = ~dfStats.statNotes.apply(lambda s: pd.notnull(s) and '1' in s)
        sFigs2Drop = (dfStats.Figure != 'Value') & sKeepOnlyValueFig
        assert ~dfStats[sFigs2Drop & ((dfStats.Module != 2) | (dfStats.Statistic < 100))].Value.any(), \
               'Error: Some so-called "N/A" stat figures are not zeroes !'
        
        # 7. Remove so-called N/A figures
        dfStats.drop(dfStats[sFigs2Drop].index, inplace=True)
        
        # 8. Make some values more readable (1/2): Get and convert the values.
        # Note: Since after pandas 2.2, we can't update the values in-place, as it would imply writing
        #       strings in a float column, and pandas 2 does no more accept it ; hence the 2-step process.
        lblKeyFn = (dfStats.modDesc == 'detection probability') & (dfStats.statDesc == 'key function type')
        keyFnName = cls.EstKeyFns[int(dfStats.loc[lblKeyFn, 'Value'].iloc[0]) - 1] if lblKeyFn.any() else None
        lblAdjFn = (dfStats.modDesc == 'detection probability') & (dfStats.statDesc == 'adjustment series type')
        adjFnName = cls.EstAdjustFns[int(dfStats.loc[lblAdjFn, 'Value'].iloc[0]) - 1] if lblAdjFn.any() else None

        # 9. Check if any unexpected module / statistic present, and if so, warn and fix
        #    (like after MCDS.exe changed version ... as an example)
        dfUnexptdStats = dfStats[dfStats[['modDesc', 'statDesc']].isnull().any(axis='columns')]
        if not dfUnexptdStats.empty:
            logger.warning('Ignoring unexpected modules / stats found in results'
                           ' from a likely unsupported yet MCDS version'
                           f'\n{dfUnexptdStats.to_string(min_rows=30, max_rows=30)}')
            dfStats.drop(dfUnexptdStats.index, inplace=True)

        # 10. Final indexing and transposition
        dfStats = dfStats.reindex(columns=['modDesc', 'statDesc', 'Figure', 'Value'])
        dfStats.set_index(['modDesc', 'statDesc', 'Figure'], inplace=True)
        dfStats = dfStats.T

        # 11. Make some values more readable (2/2): Update them in the final frame, if found in stats.
        if keyFnName:
            dfStats[('detection probability', 'key function type', 'Value')] = keyFnName
        if adjFnName:
            dfStats[('detection probability', 'adjustment series type', 'Value')] = adjFnName

        # That's all folks !
        logger.debug('Done decoding from {}.'.format(statsFileName))
        
        return dfStats.iloc[0]

    # Decode output log file to a string
    # Precondition: self.runAnalysis(...) was called and took place in :param:runDir
    # * runDir : string or pl.Path folder path-name where the analysis was run.
    @classmethod
    def decodeLog(cls, runDir):
        
        return dict(text=open(pl.Path(runDir) / cls.LogFileName).read().strip())
    
    # Decode output ... output file to a dict of chapters
    # Precondition: self.runAnalysis(...) was called and took place in :param:runDir
    # * runDir : string or pl.Path folder path-name where the analysis was run.
    @classmethod
    def decodeOutput(cls, runDir):
        
        outLst = open(pl.Path(runDir) / cls.OutputFileName).read().strip().split('\t')
        
        return [dict(id=title.translate(str.maketrans({c: '' for c in ' ,.-:()/'})),
                     title=title.strip(), text=text.strip('\n'))
                for title, text in [outLst[i:i+2] for i in range(0, len(outLst), 2)]]
            
    # Decode output plots file as a dict of plot dicts (key = output chapter title)
    # Precondition: self.runAnalysis(...) was called and took place in :param:runDir
    # * runDir : string or pl.Path folder path-name where the analysis was run.
    @classmethod
    def decodePlots(cls, runDir):
        
        dPlots = dict()
        lines = (line.strip() for line in open(pl.Path(runDir) / cls.PlotsFileName, 'r').readlines())
        for title in lines:
            
            title = title.strip()
            subTitle = next(lines).strip()
            xLabel = next(lines).strip()
            yLabel = next(lines).strip()
            xMin, xMax, yMin, yMax = [float(s) for s in next(lines).split()]
            nDataRows = int(next(lines))
            dataRows = list()
            for _ in range(nDataRows):
                dataRows.append([np.nan if '*' in s else float(s) for s in next(lines).split()])
                
            dPlots[title] = dict(title=title, subTitle=subTitle, dataRows=dataRows,  # nDataRows=nDataRows,
                                 xLabel=xLabel, yLabel=yLabel, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax)

        return dPlots

    @classmethod
    def loadDataFile(cls, runDir):

        runDir = pl.Path(runDir)
        with open(runDir / cls.CmdFileName, 'r') as cmdFile:
            fieldsLine = next(line for line in cmdFile.readlines() if line.startswith('Fields='))

        dataCols = fieldsLine.strip('\n;')[len('Fields='):].split(',')

        dataFilePathName = runDir / cls.DataFileName
        dfData = pd.read_csv(dataFilePathName, sep='\t', names=dataCols)

        logger.debug('Loaded {} rows with columns {} from MCDS data file {}.'
                     .format(len(dfData), ','.join(dataCols), dataFilePathName.as_posix()))

        return dfData

    # Columns names for exporting to Distance import format with explicit columns headers
    # (or for explicitating their contents in a standard way).
    FirstDistanceExportFields = \
        {'Point': dict(STR_LABEL='Region*Label', STR_AREA='Region*Area',
                       SMP_LABEL='Point transect*Label', SMP_EFFORT='Point transect*Survey effort',
                       DISTANCE='Observation*Radial distance',
                       SIZE='Observation*Cluster size'),
         'Line':  dict(STR_LABEL='Region*Label', STR_AREA='Region*Area',
                       SMP_LABEL='Line transect*Label', SMP_EFFORT='Line transect*Line length',
                       DISTANCE='Observation*Perp distance', ANGLE='Observation*Angle',
                       SIZE='Observation*Cluster size')}
 
    def distanceFields(self, dsFields):
        return [self.FirstDistanceExportFields[self.options.surveyType][name] for name in dsFields]
    
    # Compute sample stats
    def computeSampleStats(self, sampleDataSet):

        # Match sampleDataSet table columns to MCDS expected fields from possible aliases
        matchFields, _, _ = \
            self.matchDataFields(sampleDataSet.dfData.columns, self.importFieldAliasREs)
        
        # Extract useful columns and rename them for standard semantics.
        dfSample = sampleDataSet.dfData[matchFields]
        dfSample.columns = self.distanceFields(self.options.firstDataFields)

        # Compute sample stats.
        distCol = self.FirstDistanceExportFields[self.options.surveyType]['DISTANCE']
        stats = [sum(dfSample[distCol].notnull()), dfSample[distCol].min(), dfSample[distCol].max()]

        # Done
        return pd.Series(data=stats, index=self.MIStatSampCols)

    # Build Distance/MCDS input data file from a sample data set to given target folder and file name.
    def buildDistanceDataFile(self, sampleDataSet, tgtFilePathName, decimalPoint=',', withExtraFields=False):
                
        # Build data to export (check and match mandatory columns, enforce order, keep other cols).
        dfExport, extraFields = \
            self.buildExportTable(sampleDataSet, withExtraFields=withExtraFields, decPoint=decimalPoint)
                
        # Export.
        dfExport.to_csv(tgtFilePathName, index=False, sep='\t', encoding='utf-8',
                        header=self.distanceFields(self.options.firstDataFields) + extraFields)

        logger.debug('Data Distance-exported to {}.'.format(tgtFilePathName))
        
        return tgtFilePathName

    # Retrieve MCDS.exe version
    @classmethod
    def findVersion(cls):

        logger.debug(f'Running {cls.ExeFilePathName} to discover its version ...')

        versionStr = None
        lastExc = None
        for tmpDirRoot in [None, '.']:
            try:
                # Create an auto-cleanup temporary folder for running MCDS.
                logger.debug(f'* trying {tmpDirRoot or "default"} root dir for temporary run folder ...')
                with tempfile.TemporaryDirectory(dir=tmpDirRoot) as tmpDir:

                    if ' ' in tmpDir:
                        logger.debug(f'  => skipped {tmpDir}: some space(s) in path')
                        continue  # No space in folder path, MCDS does not support it

                    # Create an engine instance
                    engine = cls(workDir=tmpDir)

                    # Setup run folder
                    runPath = engine.setupRunFolder()

                    # Generate command file into this folder
                    cmdFileName = engine.buildCmdFile(runPath)

                    # Run executable as an OS sub-process.
                    engine._run(engine.ExeFilePathName, cmdFileName)

                    # Extract and decode results.
                    with open(runPath / cls.LogFileName) as logFile:
                        firstLine = logFile.readline().strip()
                        versionStr = firstLine[len('This is mcds.exe version'):].strip()

                    # Done, with success.
                    logger.debug(f'Successfully determined MCDS version: {versionStr}')
                    break

            except OSError as exc:
                lastExc = str(exc)
                logger.debug(f'  => exception: {lastExc}')

        if lastExc is not None:
            logger.warning(f'Failed to determine MCDS version: {lastExc}')
            logger.warning('Hints: Current folder must be writable and its path must not contain any space !')
        if versionStr is None:
            logger.warning('Failed to determine MCDS version: unsupported (new ?) log file format ?')

        return versionStr


# Discover and save MCDS engine version (can't do it in the class itself, contrary to ExePathName).
MCDSEngine.ExeVersion = MCDSEngine.findVersion()
