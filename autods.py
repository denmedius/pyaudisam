# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment


import sys
import os, shutil
import tempfile
import argparse
from packaging import version

import re
import datetime as dt
import codecs

from collections import OrderedDict as odict

import numpy as np
import pandas as pd

import jinja2


# DSEngine (abstract) classes.
class DSEngine(object):
    
    # Options possible values.
    DistUnits = ['Meter']
    AreaUnits = ['Hectare']
    
    # Forbidden chars in workDir path name (Distance DS engines are real red necks)
    # TODO: stronger protection (more special chars ? more generic method, through re ?)
    ForbidPathChars = [' ', '(', ')', ',', ':', ]
    
    # Distance software detection params.
    DistanceSuppVers = [7, 6] # Lastest first.
    DistancePossInstPaths = [sys.path[0], 
                             os.path.join('C:\\', 'Program files (x86)'),
                             os.path.join('C:\\', 'Program files')]

    # Find given executable installation dir.
    @staticmethod
    def findExecutable(exeFileName):

        exeFilePathName = None
        print('Looking for {} ...'.format(exeFileName))
        for path in DSEngine.DistancePossInstPaths:
            for ver in DSEngine.DistanceSuppVers:
                exeFileDir = os.path.join(path, 'Distance ' + str(ver))
                print(' - checking {} : '.format(exeFileDir), end='')
                if not os.path.exists(os.path.join(exeFileDir, exeFileName)):
                    print('no.')
                else:
                    print('yes !')
                    exeFilePathName = os.path.join(exeFileDir, exeFileName)
                    break
            if exeFilePathName:
                break

        if exeFilePathName:
            print('{} found in {}'.format(exeFileName, exeFileDir))
        else:
            raise Exception('Could not find {} ; please install Distance software (V6 or later)'.format(exeFileName))
            
        return exeFilePathName
    
    # Specifications of output stats.
    DfStatRowSpecs, DfStatModSpecs, DfStatModNotes, MIStatModCols, DfStatModColTrans = None, None, None, None, None
    
    def __init__(self, workDir='.',
                 distanceUnit='Meter', areaUnit='Hectare', **options):

        # Check base options
        assert distanceUnit in self.DistUnits, \
               'Invalid distance unit {}: should be in {}'.format(distanceUnit, self.DistUnits)
        assert areaUnit in self.AreaUnits, \
               'Invalid area unit {}: should be in {}'.format(areaUnit, self.AreaUnits)
        
        # Save workdir, base and specific options.
        self.options = options.copy()
        self.options.update(distanceUnit=distanceUnit, areaUnit=areaUnit)
        
        # Check and prepare workdir if needed.
        assert all(c not in workDir for c in self.ForbidPathChars), \
               'Invalid character from "{}" in workDir folder "{}"' \
               .format(''.join(self.ForbidPathChars), workDir)
        self.workDir = workDir
        os.makedirs(workDir, exist_ok=True)
            
    # Possible regexps for auto-detection of columns to import from data sets / export
    # TODO: Complete for non 'Point transect' modes
    ImportFieldAliasREs = \
        odict([('STR_LABEL', ['region', 'zone', 'strate', 'stratum']),
               ('STR_AREA', ['surface', 'area', 'ha', 'km2']),
               ('SMP_LABEL', ['point', 'lieu', 'location']),
               ('SMP_EFFORT', ['effort', 'passages', 'surveys', 'samplings']),
               ('DISTANCE', ['distance'])])
    
    # Associated Distance import fields.
    DistanceFields = \
        dict(STR_LABEL='Region*Label', STR_AREA='Region*Area', SMP_LABEL='Point transect*Label',
             SMP_EFFORT='Point transect*Survey effort', DISTANCE='Observation*Radial distance')
    
    def distanceFields(self, dsFields):
        return [self.DistanceFields[name] for name in dsFields]
    
    # Data fields of decimal type.
    # TODO: Complete for non 'Point transect' modes
    DecimalFields = ['SMP_EFFORT', 'DISTANCE']
    
    # Match srcFields with tgtAliasREs ones ; keep remaining ones ; sort decimal fields.
    @classmethod
    def matchDataFields(cls, srcFields, tgtAliasREs=odict()):
        
        print('Matching required data columns:', end=' ')
        
        # Try and match required data columns.
        matFields = list()
        matDecFields = list()
        for tgtField in tgtAliasREs:
            print(tgtField, end='=')
            foundTgtField = False
            for srcField in srcFields:
                for pat in tgtAliasREs[tgtField]:
                    if re.search(pat, srcField, flags=re.IGNORECASE):
                        print(srcField, end=', ')
                        matFields.append(srcField)
                        if tgtField in cls.DecimalFields:
                            matDecFields.append(srcField)
                        foundTgtField = True
                        break
                if foundTgtField:
                    break
            if not foundTgtField:
                raise Exception('Error: Failed to find a match for expected {} in dataset columns {}' \
                                .format(tgtField, srcFields))
        
        # Extra fields.
        extFields = [field for field in srcFields if field not in matFields]

        print('... success.')
        
        return matFields, matDecFields, extFields

    # Engine in/out file names.
    CmdFileName = 'cmd.txt'
    DataFileName = 'data.txt'
    OutputFileName = 'output.txt'
    LogFileName = 'log.txt'
    StatsFileName = 'stats.txt'
    PlotsFileName = 'plots.txt'
        
    # Setup run folder (all in and out files will go there)
    def setupRunFolder(self, runPrefix='ds', forceSubFolder=None):
        
        if not forceSubFolder:
            
            # MCDS does not support folder and file names with spaces inside ...
            # And one never knows ... replace other special chars also.
            runPrefix = runPrefix.translate(str.maketrans({c:'-' for c in ' ,.:;()/'}))
        
            self.runDir = tempfile.mkdtemp(dir=self.workDir, prefix=runPrefix+'-')
            
        else:
            
            self.runDir = os.path.join(self.workDir, forceSubFolder)
            
        print('Will run in', self.runDir)
        
        # Define input and output file pathnames
        def pathName(fileName):
            return os.path.join(self.runDir, fileName)
        
        self.cmdFileName   = pathName(self.CmdFileName)
        self.dataFileName  = pathName(self.DataFileName)
        self.outFileName   = pathName(self.OutputFileName)
        self.logFileName   = pathName(self.LogFileName)
        self.statsFileName = pathName(self.StatsFileName)
        self.plotsFileName  = pathName(self.PlotsFileName)
        
        return self.runDir

# MCDS engine (Conventional Distance Sampling)
class MCDSEngine(DSEngine):
    
    DistUnits = ['Meter']
    AreaUnit = ['Hectare']
    SurveyTypes = ['Point'] #, 'Line'] #TODO : Add Line support
    DistTypes = ['Radial'] #? 'Perpendicular', 'Radial & Angle']
    FirstDataFields = { 'Point' : ['STR_LABEL', 'STR_AREA', 'SMP_LABEL', 'SMP_EFFORT', 'DISTANCE'],
                      } #TODO : Add Line support
    FirstDistanceExportFields = { 'Point' : ['Region*Label', 'Region*Area', 'Point transect*Label',
                                             'Point transect*Survey effort', 'Observation*Radial distance'],
                                } #TODO : Add Line support

    # Estimator key functions (Order: Distance .chm doc, "MCDS Engine Stats File", note 2 below second table).
    EstKeyFns = ['UNIFORM', 'HNORMAL', 'NEXPON', 'HAZARD']
    EstKeyFnDef = EstKeyFns[0]

    # Estimator adjustment series (Order: Distance .chm doc, "MCDS Engine Stats File", note 3 below second table).
    EstAdjustFns = ['POLY', 'HERMITE', 'COSINE']
    EstAdjustFnDef = EstAdjustFns[0]

    # Estimator key functions (Order: Distance .chm doc, "MCDS Engine Stats File", note 2 below second table).
    EstCriterions = ['AIC', 'AICC', 'BIC', 'LR']
    EstCriterionDef = EstCriterions[0]
    
    # Estimator confidence value for output interval.
    EstCVIntervalDef = 95 # %

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
                     Print=Selection;
                     End;
                     Data /Structure=Flat;
                     Fields={dataFields};
                     Infile={dataFileName} /NoEcho;
                     End;
                     Estimate;
                     Distance;
                     Density=All;
                     Encounter=All;
                     Detection=All;
                     Size=All;
                     Estimator /Key={estKeyFn} /Adjust={estAdjustFn} /Criterion={estCriterion};
                     Monotone=Strict;
                     Pick=AIC;
                     GOF;
                     Cluster /Bias=GXLOG;
                     VarN=Empirical;
                     End;
                  """.split('\n'))) + '\n'
    
    # Executable 
    ExeFilePathName = DSEngine.findExecutable(exeFileName='MCDS.exe')

    # Output stats specs : load from external files (extracts from Distance doc).
    @classmethod
    def loadStatSpecs(cls, nMaxAdjParams=10):
        
        print('MCDS : Loading output stats specs ...')
        
        # Output stats row specifications
        fileName = 'mcds-stat-row-specs.txt'
        print('*', fileName)
        with open(fileName, mode='r', encoding='utf8') as fStatRowSpecs:
            statRowSpecLines = [line.rstrip('\n') for line in fStatRowSpecs.readlines() if not line.startswith('#')]
            statRowSpecs =  [(statRowSpecLines[i].strip(), statRowSpecLines[i+1].strip()) \
                             for i in range(0, len(statRowSpecLines)-2, 3)]
            cls.DfStatRowSpecs = pd.DataFrame(columns=['Name', 'Description'],
                                              data=statRowSpecs).set_index('Name')
            assert not cls.DfStatRowSpecs.empty, 'Empty MCDS stats row specs'
        
        # Module and stats number to description table
        fileName = 'mcds-stat-mod-specs.txt'
        print('*', fileName)
        with open(fileName, mode='r', encoding='utf8') as fStatModSpecs:
            statModSpecLines = [line.rstrip('\n') for line in fStatModSpecs.readlines() if not line.startswith('#')]
            reModSpecNumName = re.compile('(.+) – (.+)')
            statModSpecs = list()
            moModule = None
            for line in statModSpecLines:
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
                for i in range(len(statDescNotes)-1, -1, -1):
                    if not re.match('[\d ,]', statDescNotes[i]):
                        statDesc = statDescNotes[:i+1]
                        statNotes = statDescNotes[i+1:].replace(' ', '')
                        break
                modNum = int(modNum)
                if statNum.startswith('101 '):
                    for num in range(nMaxAdjParams): # Assume no more than that ... a bit hacky !
                        statModSpecs.append((modNum, modDesc, 101+num, # Make statDesc unique for later indexing
                                             statDesc.replace('each', 'A({})'.format(num+1)), statNotes))
                else:
                    statNum = int(statNum)
                    if modNum == 2 and statNum == 3: # Actually, there are 0 or 3 of these ...
                        for num in range(3):
                            statModSpecs.append((modNum, modDesc, num+201,
                                                 # Change statNum & Make statDesc unique for later indexing
                                                 statDesc+' (distance set {})'.format(num+1), statNotes))
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
        fileName = 'mcds-stat-mod-notes.txt'
        print('*', fileName)
        with open(fileName, mode='r', encoding='utf8') as fStatModNotes:
            statModNoteLines = [line.rstrip('\n') for line in fStatModNotes.readlines() if not line.startswith('#')]
            statModNotes =  [(int(line[:2]), line[2:].strip()) for line in statModNoteLines if line]
            cls.DfStatModNotes = pd.DataFrame(data=statModNotes, columns=['Note', 'Text']).set_index('Note')
            assert not cls.DfStatModNotes.empty, 'Empty MCDS stats module notes'
            
        # DataFrame for translating 3-level multi-index columns to 1 level lang-translated columns
        fileName = 'mcds-stat-mod-trans.txt'
        print('*', fileName)
        cls.DfStatModColsTrans = dfStatModTransExt = pd.read_csv(fileName, sep='\t')
        cls.DfStatModColsTrans.set_index(['Module', 'Statistic', 'Figure'], inplace=True)
        
        print('MCDS : Loaded output stats specs.')
        print()
        
    # Accessors to dynamic class variables.
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
        
    # Ctor
    def __init__(self, workDir='.',
                 distanceUnit='Meter', areaUnit='Hectare',
                 surveyType='Point', distanceType='Radial'):
        
        # Initialize some class variables.
        if MCDSEngine.DfStatRowSpecs is None:
            MCDSEngine.loadStatSpecs()    

        # Check options
        assert surveyType in self.SurveyTypes, \
               'Invalid survey type {} : should be in {}'.format(surveyType, self.SurveyTypes)
        assert distanceType in self.DistTypes, \
               'Invalid area unit{} : should be in {}'.format(distanceType, self.DistTypes)
        
        # Initialise base.
        super().__init__(workDir=workDir, 
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         surveyType=surveyType, distanceType=distanceType,
                         firstDataFields=self.FirstDataFields[surveyType])        
    
    # Build command file from options and params
    def buildCmdFile(self, **params):

        cmdTxt = self.CmdTxt.format(output=self.outFileName, log=self.logFileName,
                                    stats=self.statsFileName, plots=self.plotsFileName,
                                    bootstrap='None', bootpgrss='None', # No support for the moment.
                                    survType=self.options['surveyType'], distType=self.options['distanceType'],
                                    distUnit=self.options['distanceUnit'], areaUnit=self.options['areaUnit'],
                                    dataFields=', '.join(self.dataFields), dataFileName=self.dataFileName,
                                    estKeyFn=params['estimKeyFn'], estAdjustFn=params['estimAdjustFn'],
                                    estCriterion=params['estimCriterion'], cvInterv=params['cvInterval'])

        with open(self.cmdFileName, mode='w', encoding='utf-8') as cmdFile:
            cmdFile.write(cmdTxt)

        print('Commands written to', self.cmdFileName)

        return self.cmdFileName
    
    # Workaround pd.DataFrame.to_csv(float_format='%.xf') not working when NaNs in serie
    @staticmethod
    def safeFloat2Str(val, prec=None, decPt='.'):
        strVal = '' if pd.isnull(val) else str(val) if prec is None \
                    else '{:.{prec}f}'.format(val, prec=prec)
        if decPt != '.':
            strVal = strVal.replace('.', decPt)
        return strVal

    # Build input data table from data set (check and match mandatory columns, enforce order).
    # TODO: Add support for covariate columns (through extraFields)
    def buildExportTable(self, dataSet, withExtraFields=True, decPoint='.'):
        
        # Match dataSet table columns to MCDS expected fields from possible aliases
        matchFields, matchDecFields, extraFields = \
            self.matchDataFields(dataSet.dfData.columns, self.ImportFieldAliasREs)
        exportFields = matchFields
        if withExtraFields:
            exportFields += extraFields
        else:
            extraFields.clear()
        
        print('Final data columns in order:', exportFields)
        
        # Put columns in the right order (first data fields ... first, in the same order)
        dfExport = dataSet.dfData[exportFields].copy()

        # Prepare safe export of decimal data with may be some NaNs
        allDecFields = set(matchDecFields + dataSet.decimalFields).intersection(exportFields)
        print('Decimal columns:', allDecFields)
        for field in allDecFields:
            dfExport[field] = dfExport[field].apply(self.safeFloat2Str, decPt=decPoint)
                
        return dfExport, extraFields

    # Build MCDS input data file from data set.
    # TODO: Add support for covariate columns (through extraFields)
    def buildDataFile(self, dataSet):
        
        # Build data to export (check and match mandatory columns, enforce order, ignore extra cols).
        dfExport, extraFields = \
            self.buildExportTable(dataSet, withExtraFields=False, decPoint='.')
        
        # Save data fields for the engine : mandatory ones only
        self.dataFields = self.options['firstDataFields']
        
        # Export.
        dfExport.to_csv(self.dataFileName, index=False, sep='\t', encoding='utf-8', header=None)
        
        print('Data MCDS-exported to', self.dataFileName)
        
        return self.dataFileName
    
    # Run MCDS
    def run(self, dataSet, runPrefix='mcds', realRun=True, **analysisParms):
        
        # Create a new exclusive run folder
        self.setupRunFolder(runPrefix)
        
        # Generate data and command files into this folder
        _ = self.buildDataFile(dataSet)
        cmdFileName = self.buildCmdFile(**analysisParms)
        
        # Call executable (no " around cmdFile, don't forget the space after ',', ...)
        cmd = '"{}" 0, {}'.format(self.ExeFilePathName, cmdFileName)
        if realRun:
            print('Running MCDS :', cmd)
            self.runTime = pd.Timestamp.now()
            self.runStatus = os.system(cmd)
            print('Done : status =', self.runStatus)
            
        # ... unless specified not to (input files generated, but no execution).
        else:
            print('Not running MCDS :', cmd)
            self.runTime = pd.NaT
            self.runStatus = 0

        return self.runStatus, self.runTime, self.runDir
    
    # Decode output stats file to a value series
    # Precondition: self.run(...)
    # Warning: No support for more than 1 stratum, 1 sample, 1 estimator.
    def decodeStats(self):

        print('Decoding', self.statsFileName, end=' ... ')
        
        # 1. Load table (text format, with space separated and fixed width columns,
        #    columns headers from self.DfStatRowSpecs)
        dfStats = pd.read_csv(self.statsFileName, sep=' +', engine='python', names=self.DfStatRowSpecs.index)
        
        # 2. Remove Stratum, Sample and Estimator columns (no support for multiple ones for the moment)
        dfStats.drop(columns=['Stratum', 'Sample', 'Estimator'], inplace=True)
        
        # 3. Stack figure columns to rows, to get more confortable
        dfStats.set_index(['Module', 'Statistic'], append=True, inplace=True)
        dfStats = dfStats.stack().reset_index()
        dfStats.rename(columns={'level_0': 'id', 'level_3': 'Figure', 0: 'Value'}, inplace=True)

        # 4. Fix multiple Module=2 & Statistic=3 rows (before joining with self.DfStatModSpecs)
        newStatNum = 200
        for lbl, sRow in dfStats[(dfStats.Module == 2) & (dfStats.Statistic == 3)].iterrows():
            if dfStats.loc[lbl, 'Figure'] == 'Value':
                newStatNum += 1
            dfStats.loc[lbl, 'Statistic'] = newStatNum
        newStatNum = 201
        
        # 5. Add descriptive / naming columns for modules and statistics,
        #    from self.DfStatModSpecs (more user friendly than numeric ids + help for detecting N/A figures)
        dfStats = dfStats.join(self.DfStatModSpecs, on=['Module', 'Statistic'])
        
        # 6. Check that supposed N/A figures (as told by self.dfStatModuleSpecs.statNotes) are really such
        #    Warning: There must be a bug in MCDS with Module=2 & Statistic=10x : some Cv values not always 0 ...
        sKeepOnlyValueFig = ~dfStats.statNotes.apply(lambda s: pd.notnull(s) and '1' in s)
        sFigs2Drop = (dfStats.Figure != 'Value') & sKeepOnlyValueFig
        assert ~dfStats[sFigs2Drop & ((dfStats.Module != 2) | (dfStats.Statistic < 100))].Value.any(), \
               'Warning: Somme so-called "N/A" figures are not zeroes !'
        
        # 7. Remove so-called N/A figures
        dfStats.drop(dfStats[sFigs2Drop].index, inplace=True)
        
        # 8. Make some values more readable.
        lblKeyFn = (dfStats.Module == 2) & (dfStats.Statistic == 13)
        dfStats.loc[lblKeyFn, 'Value'] = dfStats.loc[lblKeyFn, 'Value'].astype(int).apply(lambda n: self.EstKeyFns[n-1])
        lblAdjFn = (dfStats.Module == 2) & (dfStats.Statistic == 14)
        dfStats.loc[lblAdjFn, 'Value'] = dfStats.loc[lblAdjFn, 'Value'].astype(int).apply(lambda n: self.EstAdjustFns[n-1])
        
        # 9. Final indexing
        dfStats = dfStats.reindex(columns=['modDesc', 'statDesc', 'Figure', 'Value'])
        dfStats.set_index(['modDesc', 'statDesc', 'Figure'], inplace=True)

        # That's all folks !
        print('done.')
        
        return dfStats.T.iloc[0]

    # Build Distance/MCDS input data file from data set.
    def buildDistanceDataFile(self, dataSet, tgtFilePathName, withExtraFields=False):
                
        # Build data to export (check and match mandatory columns, enforce order, keep other cols).
        dfExport, extraFields = self.buildExportTable(dataSet, withExtraFields=withExtraFields, decPoint=',')
        
        # Export.
        dfExport.to_csv(tgtFilePathName, index=False, sep='\t', encoding='utf-8',
                        header=self.distanceFields(self.options['firstDataFields']) + extraFields)

        print('Data Distance-exported to', tgtFilePathName)
        
        return tgtFilePathName

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
    
    def __init__(self, engine, dataSet, namePrefix='mcds',
                 estimKeyFn=EngineClass.EstKeyFnDef, estimAdjustFn=EngineClass.EstAdjustFnDef, 
                 estimCriterion=EngineClass.EstCriterionDef, cvInterval=EngineClass.EstCVIntervalDef):
        
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
        assert cvInterval > 0 and cvInterval < 100,\
               'Invalid cvInterval {}% : should be in {}'.format(cvInterval, ']0%, 100%[')

        # Build name from main params
        name = '-'.join([namePrefix] + [p[:3].lower() for p in [estimKeyFn, estimAdjustFn]])
        if estimCriterion != self.EngineClass.EstCriterionDef:
            name += '-' + estimCriterion.lower()
        if cvInterval != self.EngineClass.EstCVIntervalDef:
            name += '-' + str(cvInterval)

        # Initialise base.
        super().__init__(engine, dataSet, name)
        
        # Save params.
        self.estimKeyFn = estimKeyFn
        self.estimAdjustFn = estimAdjustFn
        self.estimCriterion = estimCriterion
        self.cvInterval = cvInterval
    
    # Run columns for output : analysis params + root engine output (3-level multi-index)
    MIRunColumns = pd.MultiIndex.from_tuples([('parameters', 'estimator key function', 'Value'),
                                              ('parameters', 'estimator adjustment series', 'Value'),
                                              ('parameters', 'estimator selection criterion', 'Value'),
                                              ('parameters', 'CV interval', 'Value')] + DSAnalysis.RunRunColumns)
    
    # DataFrame for translating 3-level multi-index columns to 1 level lang-translated columns
    DfRunColumnTrans = \
        pd.DataFrame(index=MIRunColumns,
                     data=dict(en=['ModKeyFn', 'ModAdjSer', 'ModChcCrit', 'ConfInter'] + DSAnalysis.DRunRunColumnTrans['en'],
                               fr=['FnCléMod', 'SérAjustMod', 'CritChxMod', 'InterConf']+ DSAnalysis.DRunRunColumnTrans['fr']))
    
    def run(self, realRun=True):
        
        self.runStatus, self.runTime, self.runDir = \
            self.engine.run(dataSet=self.dataSet, runPrefix=self.name, realRun=realRun,
                            estimKeyFn=self.estimKeyFn, estimAdjustFn=self.estimAdjustFn,
                            estimCriterion=self.estimCriterion, cvInterval=self.cvInterval)
        
        # Load and decode output stats.
        sResults = pd.Series(data=[self.estimKeyFn, self.estimAdjustFn, self.estimCriterion,
                                   self.cvInterval, self.runStatus, self.runTime, self.runDir],
                             index=self.MIRunColumns)
        
        if self.runStatus in [1, 2]:
            sResults = sResults.append(self.engine.decodeStats())
            
        # TODO: output (text and curves), log
        
        print()
        
        return sResults
    
# A tabular input data set for multiple analyses, with 1 or 0 individual per row
# Warning:
# * Only Point transect supported as for now
# * No change made afterwards on decimal precision : provide what you need !
# * No change made afterwards on the order of rows : provide what you need !
# * Support provided for pandas.DataFrame, Excel .xlsx file and tab-separated .csv/.txt files,
#   and OpenDoc .ods file when pandas >= 0.25
class DataSet(object):
    
    SupportedFileExts = ['.xlsx', '.csv', '.txt'] \
                        + (['.ods'] if version.parse(pd.__version__).release >= (0, 25) else [])
    
    # Wrapper around pd.read_csv for smart ./, decimal character management (pandas is not smart on this)
    # TODO: Make this more efficient
    @staticmethod
    def csv2df(fileName, decCols, sep='\t'):
        df = pd.read_csv(fileName, sep=sep)
        allRight = True
        for col in decCols:
            if df[col].dropna().apply(lambda v: isinstance(v, str)).any():
                allRight = False
                break
        if not allRight:
            df = pd.read_csv(fileName, sep=sep, decimal=',')
        return df
    
    def __init__(self, source, decimalFields=list()):
        
        assert (isinstance(source, str) and os.path.isfile(source)) \
               or isinstance(source, pd.DataFrame), \
               'source must be a pandas.DataFrame or an existing filename'
        
        if isinstance(source, str):
            ext = os.path.splitext(source)[1].lower()
            assert ext in self.SupportedFileExts, \
                   'Unsupported source file type {}: not from {{{}}}' \
                   .format(ext, ','.join(self.SupportedFileExts))
            if ext in ['.xlsx', '.ods']:
                dfData = pd.read_excel(source)
            elif ext in ['.csv', '.txt']:
                dfData = self.csv2df(source, decCols=decimalFields, sep='\t')
        else:
            dfData = source.copy()
        
        assert not dfData.empty, 'No data in set'
        assert len(dfData.columns) >= 5, 'Not enough columns (should be at leat 5)'
        
        assert all(field in dfData.columns for field in decimalFields), \
               'Some declared decimal field(s) are not in source columns {}' \
               .format(','.join(dfData.columns))
        
        self._dfData = dfData
        self.decimalFields = decimalFields
        
    @property
    def dfData(self):
        
        return self._dfData

    @dfData.setter
    def dfData(self, dfData):
        
        raise NotImplementedError('No changed allowed to data ; create a new dataset !')
        
# A result set for multiple analyses from the same engine.
# With ability to prepend custom heading columns to the engine output stat ones.
# And to get a 3-level multi-index columned or a mono-indexed translated columned version of the data table.
class ResultsSet(object):
    
    def __init__(self, analysisClass, miCustomCols=None, dfCustomColTrans=None):
        
        assert issubclass(analysisClass, DSAnalysis), 'analysisClass must derive from DSAnalysis'
        assert miCustomCols is None or isinstance(miCustomCols, pd.MultiIndex), \
               'customCols must be None or a pd.MultiIndex'
        assert miCustomCols is None or len(miCustomCols.levels) == 3, \
               'customCols must have 3 levels if not None'
        
        self.analysisClass = analysisClass
    
        # 3-level multi-index columns (module, statistic, figure)
        self.miAnalysisCols = analysisClass.MIRunColumns.append(analysisClass.EngineClass.statModCols())
        self.miCustomCols = miCustomCols
            
        # DataFrame for translating 3-level multi-index columns to 1 level lang-translated columns
        self.dfAnalysisColTrans = analysisClass.DfRunColumnTrans.append(analysisClass.EngineClass.statModColTrans())
        self.dfCustomColTrans = dfCustomColTrans
        
        self._dfData = pd.DataFrame()
        self.rightColOrder = False
    
    # sResult : Series for result cols values
    # sCustomHead : Series for custom cols values
    def append(self, sResult, sCustomHead=None):
        
        assert isinstance(sResult, pd.Series), 'sResult : Can only append a pd.Series'
        assert sCustomHead is None or isinstance(sCustomHead, pd.Series), 'sCustom : Can only append a pd.Series'
        
        if sCustomHead is not None:
            sResult = sCustomHead.append(sResult)
        
        self._dfData = self._dfData.append(sResult, ignore_index=True)
        
        # Appending (or concat'ing) often changes columns order
        self.rightColOrder = False
        
    @property
    def dfData(self):
        
        if not self.rightColOrder:
            
            # Enforce right columns order, but remove columns with no relevant data.
            miTgtColumns = self.miAnalysisCols
            if self.miCustomCols is not None:
                miTgtColumns = self.miCustomCols.append(miTgtColumns)
            self._dfData = self._dfData.reindex(columns=miTgtColumns)
            self._dfData.dropna(how='all', axis='columns', inplace=True)
            
            # No need to do it again, until next append() !
            self.rightColOrder = True
        
        return self._dfData

    @dfData.setter
    def dfData(self, dfData):
        
        assert isinstance(dfData, pd.DataFrame), 'dfData must be a pd.DataFrame'
        
        self._dfData = dfData.copy()
        
        # Let's assume that columns order is dirty.
        self.rightColOrder = False
        
    # Access a mono-indexed translated columns version of the data table
    def dfTransData(self, lang='en', subset=None):
        
        assert lang in ['en', 'fr'], 'No support for "{}" language'.format(lang)
        assert subset is None or isinstance(subset, list) or isinstance(subset, pd.MultiIndex), \
               'subset columns must be specified as None (all), or as a list of tuples, or as a pandas.MultiIndex'
        
        # Build translation table for lang from custom one and analysis one
        dfColTrans = self.dfAnalysisColTrans
        if self.dfCustomColTrans is not None:
            dfColTrans = self.dfCustomColTrans.append(dfColTrans)
        dTr = dfColTrans[lang].to_dict()
        
        # Make a copy of / extract selected columns of dfData, and translate column names.
        if subset is None:
            dfTrData = self.dfData.copy()
        else:
            if isinstance(subset, list):
                miSubset = pd.MultiIndex.from_tuples(subset)
            else:
                miSubset = subset
            dfTrData = self.dfData.reindex(columns=miSubset)
        dfTrData.columns = [dTr.get(col, col) for col in dfTrData.columns]
        
        return dfTrData

    # Save data to Excel.
    def toExcel(self, fileName, sheetName=None):
        
        self.dfData.to_excel(fileName, sheet_name=sheetName or 'AllResults')

    # Load data from Excel (assuming ctor params match with Excel sheet column names and list,
    #  which can well be ensured by using the same ctor params as used for saving !).
    def fromExcel(self, fileName, sheetName=None):
        
        self.dfData = pd.read_excel(fileName, sheet_name=sheetName or 'AllResults', 
                                    header=[0, 1, 2], skiprows=[3], index_col=0)

# Results reports class (Excel and HTML)
class ResultsReport(object):

    def __init__(self, resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                       synthCols=None, lang='en', attachedDir='.', tgtFolder='.', tgtPrefix='results'):
    
        assert os.path.isdir(tgtFolder), 'Target folder {} doesn\'t seem to exist ...'.format(tgtFolder)
        assert synthCols is None or isinstance(synthCols, list) or isinstance(synthCols, pd.MultiIndex), \
               'synthesis columns must be specified as None (all), or as a list of tuples, or as a pandas.MultiIndex'
        
        self.resultsSet = resultsSet
        self.synthCols = synthCols
        
        self.runFolderCol = resultsSet.dfAnalysisColTrans.loc[DSAnalysis.RunFolderColumn, lang]

        self.lang = lang
        self.title = title
        self.subTitle = subTitle
        self.anlysSubTitle = anlysSubTitle
        self.description = description
        self.keywords = keywords
        
        self.attachedDir = attachedDir
        
        self.tgtPrefix = tgtPrefix
        self.tgtFolder = tgtFolder
        
    # Translation table for HTML report.
    DTrans = dict(en={ 'RunFolder': 'Analysis', 'Synthesis': 'Synthesis', 'Details': 'Details',
                       'Synthesis table': 'Synthesis table', 'Detailed results': 'Detailed results',
                       'Download Excel': 'Download as Excel(TM) file',
                       'Summary computation log': 'Summary computation log', 'Detailed computation log': 'Detailed computation log',
                       'Click on analysis # for details': 'Click on analysis number for detailed report' },
                  fr={ 'DossierExec': 'Analyse', 'Synthesis': 'Synthèse', 'Details': 'Détails',
                       'Synthesis table': 'Tableau de synthèse', 'Detailed results': 'Résultats en détails',
                       'Download Excel': 'Télécharger le classeur Excel (TM)',
                       'Summary computation log': 'Résumé des calculs', 'Detailed computation log': 'Détail des calculs',
                       'Click on analysis # for details': 'Cliquer sur le numéro de l\'analyse pour le rapport détaillé' })

    # Output file pathname generation.
    def targetFilePathName(self, suffix, prefix=None, tgtFolder=None):
        
        return os.path.join(tgtFolder or self.tgtFolder, (prefix or self.tgtPrefix) + suffix)
    
    # HTML report generation.
    def toHtml(self):
        
        # Build and configure jinja2 environnement.
        env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'),
                                 trim_blocks=True, lstrip_blocks=True)
        #env.filters.update(trace=_jcfPrint2StdOut) # Template debugging ...
        
        # Install needed attached files.
        attSrcDir = os.path.join('AutoDS', 'report')
        for fn in ['autods.css', 'fa-feather-alt.svg', 'fa-angle-up.svg',
                   'fa-file-excel-rgr.svg', 'fa-file-excel-hover-rgr.svg']:
            shutil.copy(os.path.join(attSrcDir, fn), self.tgtFolder)
            
        # Postprocess synthesis table :
        # a. Add run folder columns, and as the 1st one (will serve as the analysis id and link to associated detailled report)
        synCols = [DSAnalysis.RunFolderColumn] + [col for col in self.synthCols if col != DSAnalysis.RunFolderColumn]
        dfSyn = self.resultsSet.dfTransData(self.lang, subset=synCols)
        
        # b. Links to each analysis detailled report.
        #dfSyn[self.runFolderCol] = \
        #    dfSyn[self.runFolderCol].apply(lambda p: '<a href="./{p}/index.html">{p}</a>' \
        #                                             .format(p=os.path.relpath(p, self.tgtFolder).replace(os.sep, '/')))
        dfSyn[self.runFolderCol] = \
            dfSyn.apply(lambda an: '<a href="./{p}/index.html">{n:03d}</a>' \
                                   .format(p=os.path.relpath(an[self.runFolderCol], self.tgtFolder).replace(os.sep, '/'),
                                           n=an.name), axis='columns')
        dfSyn.rename(columns={ self.runFolderCol: self.DTrans[self.lang][self.runFolderCol] }, inplace=True)
       
        # Postprocess synthesis table.
        dfDet = self.resultsSet.dfTransData(self.lang)

        # a. Add run folder columns, and as the 1st one (will serve as the analysis id and link to associated detailled report)
        detTrCols = [self.runFolderCol] + [col for col in dfDet if col != self.runFolderCol]
        dfDet = dfDet.reindex(columns=detTrCols)
       
        # b. Links to each analysis detailled report.
        #dfDet[self.runFolderCol] = \
        #    dfDet[self.runFolderCol].apply(lambda p: '<a href="{p}/index.html">{p}</a>' \
        #                                             .format(p=os.path.relpath(p, self.tgtFolder).replace(os.sep, '/')))
        dfDet[self.runFolderCol] = \
            dfDet.apply(lambda an: '<a href="./{p}/index.html">{n:03d}</a>' \
                                   .format(p=os.path.relpath(an[self.runFolderCol], self.tgtFolder).replace(os.sep, '/'),
                                           n=an.name), axis='columns')
        dfDet.rename(columns={ self.runFolderCol: self.DTrans[self.lang][self.runFolderCol] }, inplace=True)

        # Generate top report page.
        genDateTime = dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        tmpl = env.get_template('autods-top-tmpl.html')
        topHtmlPathName = self.targetFilePathName(suffix='.html')
        xlFileUrl = os.path.basename(self.targetFilePathName(suffix='.xlsx')).replace(os.sep, '/')
        html = tmpl.render(synthesis=dfSyn.to_html(escape=False, index=False),
                           details=dfDet.to_html(escape=False, index=False),
                           title=self.title, subtitle=self.subTitle, keywords=self.keywords,
                           xlUrl=xlFileUrl, tr=self.DTrans[self.lang], genDateTime=genDateTime)
        html = '\n'.join(line.rstrip() for line in html.split('\n') if line.rstrip())

        # Write top HTML to file.
        with codecs.open(topHtmlPathName, mode='w', encoding='utf-8-sig') as tgtFile:
            tgtFile.write(html)

        # Generate detailled report page for each analysis
        tmpl = env.get_template('autods-anlys-tmpl.html')
        
        dfSynthRes = self.resultsSet.dfTransData(self.lang, subset=self.synthCols)
        dfDetRes = self.resultsSet.dfTransData(self.lang)
        for lblAnlys in dfDetRes.index:
        
            anlysFolder = dfDetRes.at[lblAnlys, self.runFolderCol]

            # Postprocess synthesis table :
            dfSyn = dfSynthRes.loc[lblAnlys].to_frame().T
            dfSyn.index = dfSyn.index.map(lambda n: '{:03d}'.format(n))
            
            # Postprocess detailed table :
            dfDet = dfDetRes.loc[lblAnlys].to_frame().T
            dfDet.index = dfDet.index.map(lambda n: '{:03d}'.format(n))
            
            # Generate analysis report page.
            subTitle = 'Analyse {:03d} : {}'.format(lblAnlys, self.anlysSubTitle)
            log = dict(text=open(os.path.join(anlysFolder, DSEngine.LogFileName)).read().strip())
            outLst = open(os.path.join(anlysFolder, DSEngine.OutputFileName)).read().strip().split('\t')
            output = [dict(id=title.translate(str.maketrans({c:'' for c in ' ,.-:()/'})), 
                           title=title.strip(), text=text.strip('\n')) \
                      for title, text in [outLst[i:i+2] for i in range(0, len(outLst), 2)]]
            html = tmpl.render(synthesis=dfSyn.to_html(escape=False, index=True),
                               details=dfDet.to_html(escape=False, index=True),
                               log=log, output=output,
                               title=self.title, subtitle=subTitle, keywords=self.keywords,
                               tr=self.DTrans[self.lang], genDateTime=genDateTime)
            html = '\n'.join(line.rstrip() for line in html.split('\n') if line.rstrip())

            # Write analysis HTML to file.
            htmlPathName = self.targetFilePathName(tgtFolder=anlysFolder, prefix='index', suffix='.html')
            with codecs.open(htmlPathName, mode='w', encoding='utf-8-sig') as tgtFile:
                tgtFile.write(html)

        return topHtmlPathName

    # Génération du rapport Excel.
    def toExcel(self):
        
        xlsxPathName = os.path.join(self.tgtFolder, self.tgtPrefix + '.xlsx')
        
        with pd.ExcelWriter(xlsxPathName) as xlsxWriter:

            self.resultsSet.dfTransData(self.lang, subset=self.synthCols) \
              .to_excel(xlsxWriter, sheet_name=self.DTrans[self.lang]['Synthesis'], index=True)
            self.resultsSet.dfTransData(self.lang) \
              .to_excel(xlsxWriter, sheet_name=self.DTrans[self.lang]['Details'], index=True)

        return xlsxPathName

if __name__ == '__main__':

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
    dataSet = DataSet(source=args.dataFile)

    # Create DS engine
    engine = MCDSEngine(workDir=args.workDir,
                        distanceUnit='Meter', areaUnit='Hectare',
                        surveyType='Point', distanceType='Radial')

    # Create and run analysis
    analysis = MCDSAnalysis(engine=engine, dataSet=dataSet, namePrefix=args.engineType,
                            estimKeyFn=args.keyFn, estimAdjustFn=args.adjustFn,
                            estimCriterion=args.criterion, cvInterval=args.cvInterval)

    dResults = analysis.run(realRun=not args.debug)
    
    # Print results
    print('Results:\n' + '\n'.join('* {}: {}'.format(k, v) for k, v in dResults.items()))

    sys.exit(analysis.runStatus)
