# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment


import sys
import os, shutil
import copy
import tempfile
import argparse
from packaging import version

import re
import datetime as dt
import codecs

from collections import OrderedDict as odict, namedtuple as ntuple

import numpy as np
import pandas as pd

import jinja2
import matplotlib.pyplot as plt


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
        
        # Save specific options (as a named tuple for easier use).
        options = copy.deepcopy(options)
        options.update(distanceUnit=distanceUnit, areaUnit=areaUnit)
        self.OptionsClass = ntuple('Options', options.keys())
        self.options = self.OptionsClass(**options) 
        
        # Check and prepare workdir if needed, and save.
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
    
    # Distance truncation / cut points parameters.
    DistMinDef = None # No left truncation (min = 0)
    DistMaxDef = None # No right truncation (max = max. distance observed)
    DistFitCutsDef = None # Model fitting : Automatic engine distance cuts determination.
    DistDiscrCutsDef = None # Distance values discretisation : None (keep exact values).

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
        cls.DfStatModColsTrans = pd.read_csv(fileName, sep='\t')
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
    def buildCmdFile(self, **params):

        # Default params values
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
        # a. Compute non trivial data fields
        distDiscrSpecs = ''
        gOFitSpecs = ''
        
        minDist = params['minDist']
        maxDist = params['maxDist']
        fitDistCuts = params['fitDistCuts']
        discrDistCuts = params['discrDistCuts']
        if discrDistCuts is not None:
            
            if isinstance(discrDistCuts, list):
                assert not (minDist is None or maxDist is None)
                distDiscrSpecs += ' /Int=' + ','.join(str(d) for d in [minDist] + discrDistCuts + [maxDist])
            elif isinstance(discrDistCuts, int):
                distDiscrSpecs += ' /NClass=' + str(discrDistCuts)
            # Other cases not supported, dans should be asserted by the caller.
        
        elif fitDistCuts is not None: # Can't fit model on other distance intervals than used for discretisation.
            
            if isinstance(fitDistCuts, list):
                assert not (minDist is None or maxDist is None)
                gOFitSpecs += ' /Int=' + ','.join(str(d) for d in [minDist] + fitDistCuts + [maxDist])
            elif isinstance(fitDistCuts, int):
                gOFitSpecs += ' /NClass=' + str(fitDistCuts)
            # Other cases not supported, dans should be asserted by the caller.
                
        if minDist is not None:
            distDiscrSpecs += ' /Left=' + str(minDist)

        if maxDist is not None:
            distDiscrSpecs += ' /Width=' + str(maxDist)
                
        # b. Format contents string
        cmdTxt = self.CmdTxt.format(output=self.outFileName, log=self.logFileName,
                                    stats=self.statsFileName, plots=self.plotsFileName,
                                    bootstrap='None', bootpgrss='None', # No support for the moment.
                                    survType=self.options.surveyType, distType=self.options.distanceType,
                                    distUnit=self.options.distanceUnit, areaUnit=self.options.areaUnit,
                                    dataFields=', '.join(self.dataFields), dataFileName=self.dataFileName,
                                    estKeyFn=params['estimKeyFn'], estAdjustFn=params['estimAdjustFn'],
                                    estCriterion=params['estimCriterion'], cvInterv=params['cvInterval'],
                                    distDiscrSpecs=distDiscrSpecs, gOFitSpecs=gOFitSpecs)

        # Write file.
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
    # Note: Data order not changed, same as in input dataSet !
    # TODO: Add support for covariate columns (through extraFields)
    def buildDataFile(self, dataSet):
        
        # Build data to export (check and match mandatory columns, enforce order, ignore extra cols).
        dfExport, extraFields = \
            self.buildExportTable(dataSet, withExtraFields=False, decPoint='.')
        
        # Save data fields for the engine : mandatory ones only
        self.dataFields = self.options.firstDataFields
        
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
        #    Warning: There seems to be a bug in MCDS with Module=2 & Statistic=10x : some Cv values not always 0 ...
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

    # Decode output log file to a string
    # Precondition: self.run(...)
    @classmethod
    def decodeLog(cls, anlysFolder):
        
        return dict(text=open(os.path.join(anlysFolder, cls.LogFileName)).read().strip())
    
    # Decode output ... output file to a dict of chapters
    # Precondition: self.run(...)
    @classmethod
    def decodeOutput(cls, anlysFolder):
        
        outLst = open(os.path.join(anlysFolder, cls.OutputFileName)).read().strip().split('\t')
        
        return [dict(id=title.translate(str.maketrans({c:'' for c in ' ,.-:()/'})), 
                     title=title.strip(), text=text.strip('\n')) \
                for title, text in [outLst[i:i+2] for i in range(0, len(outLst), 2)]]
            
    # Decode output plots file as a dict of plot dicts (key = output chapter title)
    # Precondition: self.run(...)
    @classmethod
    def decodePlots(cls, anlysFolder):
        
        dPlots = dict()
        lines = (line.strip() for line in open(os.path.join(anlysFolder, cls.PlotsFileName), 'r').readlines())
        for title in lines:
            
            title = title.strip()
            subTitle = next(lines).strip()
            xLabel = next(lines).strip()
            yLabel = next(lines).strip()
            xMin, xMax, yMin, yMax = [float(s) for s in next(lines).split()]
            nDataRows = int(next(lines))
            dataRows = list()
            for l in range(nDataRows):
                dataRows.append([np.nan if '*' in s else float(s) for s in next(lines).split()])
                
            dPlots[title] = dict(title=title, subTitle=subTitle, dataRows=dataRows, #nDataRows=nDataRows,,
                                 xLabel=xLabel, yLabel=yLabel, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax)

        return dPlots

    # Build Distance/MCDS input data file from data set.
    def buildDistanceDataFile(self, dataSet, tgtFilePathName, withExtraFields=False):
                
        # Build data to export (check and match mandatory columns, enforce order, keep other cols).
        dfExport, extraFields = self.buildExportTable(dataSet, withExtraFields=withExtraFields, decPoint=',')
        
        # Export.
        dfExport.to_csv(tgtFilePathName, index=False, sep='\t', encoding='utf-8',
                        header=self.distanceFields(self.options.firstDataFields) + extraFields)

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
    
    @staticmethod
    def checkDistCuts(distCuts, minDist, maxDist):
        if distCuts is None:
            return # OK !
        if isinstance(distCuts, int):
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
    
    def __init__(self, engine, dataSet, name=None,
                 estimKeyFn=EngineClass.EstKeyFnDef, estimAdjustFn=EngineClass.EstAdjustFnDef, 
                 estimCriterion=EngineClass.EstCriterionDef, cvInterval=EngineClass.EstCVIntervalDef,
                 minDist=EngineClass.DistMinDef, maxDist=EngineClass.DistMaxDef, 
                 fitDistCuts=EngineClass.DistFitCutsDef, discrDistCuts=EngineClass.DistDiscrCutsDef):
        
        """
            :param minDist: None or >=0 
            :param maxDist: None or > :param minDist:
            :param fitDistCuts: None or int = number of equal sub-intervals of [:param minDist:, :param maxDist:] 
                                or list of distance values inside [:param minDist:, :param maxDist:]
                                ... for model fitting
            :param discrDistCuts: None or int = number of equal sub-intervals of [:param minDist:, :param maxDist:] 
                                or list of distance values inside [:param minDist:, :param maxDist:]
                                ... for distance values discretisation
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
        assert minDist is None or minDist >= 0, \
               'Invalid left truncation distance {}: should be None or >= 0'.format(minDist)
        assert maxDist is None or minDist is None or minDist <= maxDist, \
               'Invalid right truncation distance {}:' \
               ' should be None or >= left truncation distance if specified, or >= 0'.format(maxDist)
        self.checkDistCuts(fitDistCuts, minDist, maxDist)
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
            self.engine.run(dataSet=self.dataSet, runPrefix=self.name, realRun=realRun,
                            estimKeyFn=self.estimKeyFn, estimAdjustFn=self.estimAdjustFn,
                            estimCriterion=self.estimCriterion, cvInterval=self.cvInterval,
                            minDist=self.minDist, maxDist=self.maxDist,
                            fitDistCuts=self.fitDistCuts, discrDistCuts=self.discrDistCuts)
        
        # Load and decode output stats.
        sResults = pd.Series(data=[self.estimKeyFn, self.estimAdjustFn, self.estimCriterion,
                                   self.cvInterval, self.minDist, self.maxDist, self.fitDistCuts, self.discrDistCuts,
                                   self.runStatus, self.runTime, self.runDir],
                             index=self.MIRunColumns)
        
        if self.runStatus in [1, 2]:
            sResults = sResults.append(self.engine.decodeStats())
            
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
    
    def __init__(self, analysisClass, miCustomCols=None, dfCustomColTrans=None,
                       dComputedCols=None, dfComputedColTrans=None):
        
        assert issubclass(analysisClass, DSAnalysis), 'analysisClass must derive from DSAnalysis'
        assert miCustomCols is None or isinstance(miCustomCols, pd.MultiIndex), \
               'customCols must be None or a pd.MultiIndex'
        assert miCustomCols is None or len(miCustomCols.levels) == 3, \
               'customCols must have 3 levels if not None'
        
        self.analysisClass = analysisClass
        self.engineClass = analysisClass.EngineClass
    
        # 3-level multi-index columns (module, statistic, figure)
        self.miAnalysisCols = analysisClass.MIRunColumns.append(self.engineClass.statModCols())
        for col, ind in dComputedCols.items(): # Add post-computed columns at the right place
            self.miAnalysisCols = self.miAnalysisCols.insert(ind, col)
        self.computedCols = list(dComputedCols.keys())
        self.miCustomCols = miCustomCols
        
        # DataFrames for translating 3-level multi-index columns to 1 level lang-translated columns
        self.dfAnalysisColTrans = analysisClass.DfRunColumnTrans.append(self.engineClass.statModColTrans())
        self.dfAnalysisColTrans = self.dfAnalysisColTrans.append(dfComputedColTrans)
        self.dfCustomColTrans = dfCustomColTrans
        
        self._dfData = pd.DataFrame() # The real data (frame).
        self.rightColOrder = False # self._dfData columns are assumed to be in a wrong order.
        self.postComputed = False # Post-computation not yet done.
    
    def __len__(self):
        
        return len(self._dfData)
    
    # sResult : Series for result cols values
    # sCustomHead : Series for custom cols values
    def append(self, sResult, sCustomHead=None):
        
        assert not self.postComputed, 'Can\'t append after columns post-computation'
        
        assert isinstance(sResult, pd.Series), 'sResult : Can only append a pd.Series'
        assert sCustomHead is None or isinstance(sCustomHead, pd.Series), 'sCustom : Can only append a pd.Series'
        
        if sCustomHead is not None:
            sResult = sCustomHead.append(sResult)
        
        self._dfData = self._dfData.append(sResult, ignore_index=True)
        
        # Appending (or concat'ing) often changes columns order
        self.rightColOrder = False
        
    # Post-computations.
    def postComputeColumns(self):
        
        # Derive class to really compute things (=> work directly on self._dfData),
        pass
        
    @property
    def dfData(self):
        
        # Do post-computation if not already done.
        if not(self._dfData.empty or self.postComputed):
            
            # Make sure we jave a MultiIndex for columns (append breaks this, not fromExcel)
            if not isinstance(self._dfData.columns, pd.MultiIndex):
                self._dfData.columns = pd.MultiIndex.from_tuples(self._dfData.columns)
            
            # Post-compute as specified (or not).
            self.postComputeColumns()
            self.postComputed = True # No need to do it again !
        
        # Enforce right columns order.
        if not(self._dfData.empty or self.rightColOrder):
            
            #print('dfData: reordering')
            miTgtColumns = self.miAnalysisCols
            if self.miCustomCols is not None:
                miTgtColumns = self.miCustomCols.append(miTgtColumns)
            self._dfData = self._dfData.reindex(columns=miTgtColumns)
            self.rightColOrder = True # No need to do it again, until next append() !
        
        # This is also documentation line !
        assert isinstance(self._dfData.columns, pd.MultiIndex)
        
        # Don't return columns with no relevant data.
        return self._dfData.dropna(how='all', axis='columns')

    @dfData.setter
    def dfData(self, dfData):
        
        assert isinstance(dfData, pd.DataFrame), 'dfData must be a pd.DataFrame'
        
        self._dfData = dfData.copy()
        
        # Let's assume that columns order is dirty.
        self.rightColOrder = False
        
        # Remove any computed column (will be recomputed later on).
        self._dfData.drop(columns=self.computedCols, inplace=True)
        
        # Post-computation not yet done.
        self.postComputed = False
    
    # Get translate names of custom columns
    def transCustomColumns(self, lang):
        
        return self.dfCustomColTrans[lang].to_list()
    
    # Build translation table for lang from custom one and analysis one
    def transTable(self):
        
        dfColTrans = self.dfAnalysisColTrans
        if self.dfCustomColTrans is not None:
            dfColTrans = self.dfCustomColTrans.append(dfColTrans)
            
        return dfColTrans
        
    # Access a mono-indexed translated columns version of the data table
    def dfTransData(self, lang='en', subset=None):
        
        assert lang in ['en', 'fr'], 'No support for "{}" language'.format(lang)
        assert subset is None or isinstance(subset, list) or isinstance(subset, pd.MultiIndex), \
               'subset columns must be specified as None (all), or as a list of tuples, or as a pandas.MultiIndex'
        
        # Make a copy of / extract selected columns of dfData.
        if subset is None:
            dfTrData = self.dfData
        else:
            if isinstance(subset, list):
                miSubset = pd.MultiIndex.from_tuples(subset)
            else: # pd.MultiIndex
                miSubset = subset
            dfTrData = self.dfData.reindex(columns=miSubset)
        dfTrData = dfTrData.copy()
        
        # Translate column names.
        dfTrData.columns = [self.transTable()[lang].get(col, str(col)) for col in dfTrData.columns]
        
        return dfTrData

    # Save data to Excel.
    def toExcel(self, fileName, sheetName=None):
        
        self.dfData.to_excel(fileName, sheet_name=sheetName or 'AllResults')

    # Load (overwrite) data from Excel (assuming ctor params match with Excel sheet column names and list,
    # which can well be ensured by using the same ctor params as used for saving !).
    def fromExcel(self, fileName, sheetName=None):
        
        self.dfData = pd.read_excel(fileName, sheet_name=sheetName or 'AllResults', 
                                    header=[0, 1, 2], skiprows=[3], index_col=0)

        
# A specialized results set for MCDS analyses,
# with extra. post-computed columns : Delta AIC, Chi2 P
class MCDSResultsSet(ResultsSet):
    
    DeltaAicColInd = ('detection probability', 'Delta AIC', 'Value')
    Chi2ColInd = ('detection probability', 'chi-square test probability determined', 'Value')

    def __init__(self, miCustomCols=None, dfCustomColTrans=None):
        
        # Setup computed columns specs.
        dCompCols = odict([(self.DeltaAicColInd, 22), # Right before AIC
                           (self.Chi2ColInd, 24)]) # Right before all Chi2 tests 
        dfCompColTrans = \
            pd.DataFrame(index=dCompCols.keys(),
                         data=dict(en=['Delta AIC', 'Chi2 P'], fr=['Delta AIC', 'Chi2 P']))

        # Initialise base.
        super().__init__(MCDSAnalysis, miCustomCols=miCustomCols, dfCustomColTrans=dfCustomColTrans,
                                       dComputedCols=dCompCols, dfComputedColTrans=dfCompColTrans)
        
    # Post-computations.
    def postComputeColumns(self):
        
        #print('postComputeColumns: ...')
        
        # Compute Delta AIC (AIC - min(group)) per { species, periods, precision, duration } group.
        # a. Minimum AIC per group
        aicColInd = ('detection probability', 'AIC value', 'Value')
        aicGroupColInds = [('sample', 'Sample', 'Value')]
        df2Join = self._dfData.groupby(aicGroupColInds)[[aicColInd]].min()
        
        # b. Rename computed columns to target
        df2Join.columns = pd.MultiIndex.from_tuples([self.DeltaAicColInd])
        
        # c. Join the column to the target data-frame
        #print(self._dfData.columns, df2Join.columns)
        self._dfData = self._dfData.join(df2Join, on=aicGroupColInds)
        
        # d. Compute delta-AIC in-place
        self._dfData[self.DeltaAicColInd] = self._dfData[aicColInd] - self._dfData[self.DeltaAicColInd]

        # Compute determined Chi2 test probability (last value of all the tests done).
        def determineChi2(sChi2All):
            for chi2 in sChi2All:
                if not np.isnan(chi2):
                    return chi2
            return np.nan
        chi2AllColInds = [('detection probability', 'chi-square test probability (distance set {})'.format(i), 'Value') \
                          for i in range(3, 0, -1)]
        self._dfData[self.Chi2ColInd] = self._dfData[chi2AllColInds].apply(determineChi2, axis='columns')


# Base for results reports classes (abstract)
class ResultsReport(object):

    # Translation table for output documents.
    DTrans = dict(en={ }, fr={ })

    def __init__(self, resultsSet, title, subTitle, description, keywords, dCustomTrans=dict(),
                       lang='en', attachedDir='.', tgtFolder='.', tgtPrefix='results'):
    
        assert len(resultsSet) > 0, 'Can\'t build reports with nothing inside'
        assert os.path.isdir(tgtFolder), 'Target folder {} doesn\'t seem to exist ...'.format(tgtFolder)
        
        self.resultsSet = resultsSet
        
        self.trRunFolderCol = resultsSet.dfAnalysisColTrans.loc[DSAnalysis.RunFolderColumn, lang]
        self.dfEnColTrans = None # EN to other languages column name translation

        self.lang = lang
        self.title = title
        self.subTitle = subTitle
        self.description = description
        self.keywords = keywords
        
        self.dTrans = copy.deepcopy(dCustomTrans)
        for lang in self.DTrans.keys():
            if lang not in self.dTrans:
                self.dTrans[lang] = dict()
            self.dTrans[lang].update(self.DTrans[lang])
        
        self.attachedDir = attachedDir
        
        self.tgtPrefix = tgtPrefix
        self.tgtFolder = tgtFolder
        
        self.tmplEnv = None
        
    # Translate string
    def tr(self, s):
        return self.dTrans[self.lang].get(s, s)
    
    # Translate EN-translated column(s) name(s) to self.lang one
    # * colNames : str (1), list(N) or dict (N keys) of column names
    def trEnColNames(self, colNames):
        
        if self.lang == 'en':
            return colNames
        
        if self.dfEnColTrans is None:
            self.dfEnColTrans = self.resultsSet.transTable()
            self.dfEnColTrans.set_index('en', inplace=True, drop=True)
            
        def trEnColName(cn): # Assuming there's only 1 match !
            return self.dfEnColTrans.at[cn, self.lang]
        
        if isinstance(colNames, str):
            trColNames = trEnColName(colNames)
        elif isinstance(colNames, list):
            trColNames = [trEnColName(k) for k in colNames]
        elif isinstance(colNames, dict):
            trColNames = { trEnColName(k) : v for k, v in colNames.items() }
        # Otherwise ... boom: trColNames undefined !
        
        return trColNames
        
    # Output file pathname generation.
    def targetFilePathName(self, suffix, prefix=None, tgtFolder=None):
        
        return os.path.join(tgtFolder or self.tgtFolder, (prefix or self.tgtPrefix) + suffix)
    
    def relativeRunFolderUrl(self, runFolderPath):
        return os.path.relpath(runFolderPath, self.tgtFolder).replace(os.sep, '/')
    
    # Install needed attached files for HTML report.
    def installAttFiles(self, attFiles):
        
        attSrcDir = os.path.join('AutoDS', 'report')
        for fn in attFiles:
            shutil.copy(os.path.join(attSrcDir, fn), self.tgtFolder)        
    
    # Get Jinja2 template environment for HTML reports.
    def getTemplateEnv(self):
        
        # Build and configure jinja2 environnement if not already done.
        if self.tmplEnv is None:
            self.tmplEnv = jinja2.Environment(loader=jinja2.FileSystemLoader('.'),
                                              trim_blocks=True, lstrip_blocks=True)
            #self.tmplEnv.filters.update(trace=_jcfPrint2StdOut) # Template debugging ...

        return self.tmplEnv
    
    # Final formatting of translated data tables, for HTML or SpreadSheet rendering
    # (sort, convert units, round values, and style).
    # To be specialized in derived classes (here, we do nothing) !
    # Note: Use trEnColNames method to pass from EN-translated columns names to self.lang-ones
    # Return a pd.DataFrame.Styler
    def finalFormatData(self, dfTrData, sort=True, convert=True, round=True, style=True):
        
        return dfTrData.style # Nothing done here, specialize in derived class if needed.

    
# Results full reports class (Excel and HTML, targeting similar result as in Distance 6+)
class ResultsFullReport(ResultsReport):

    # Translation table.
    DTrans = dict(en={ 'RunFolder': 'Analysis', 'Synthesis': 'Synthesis', 'Details': 'Details',
                       'Synthesis table': 'Synthesis table',
                       'Click on analysis # for details': 'Click on analysis number to get to detailed report',
                       'Detailed results': 'Detailed results',
                       'Download Excel': 'Download as Excel(TM) file',
                       'Summary computation log': 'Summary computation log',
                       'Detailed computation log': 'Detailed computation log',
                       'Previous analysis': 'Previous analysis', 'Next analysis': 'Next analysis',
                       'Back to top': 'Back to global report',
                       'Page generated with': 'Page générée via', 'with icons from': 'avec les pictogrammes de',
                       'and': 'et', 'in': 'dans', 'sources': 'sources', 'on': 'le' },
                  fr={ 'DossierExec': 'Analyse', 'Synthesis': 'Synthèse', 'Details': 'Détails',
                       'Synthesis table': 'Tableau de synthèse',
                       'Click on analysis # for details': 'Cliquer sur le numéro de l\'analyse pour accéder au rapport détaillé',
                       'Detailed results': 'Résultats en détails',
                       'Download Excel': 'Télécharger le classeur Excel (TM)',
                       'Summary computation log': 'Résumé des calculs', 'Detailed computation log': 'Détail des calculs',
                       'Previous analysis': 'Analyse précédente', 'Next analysis': 'Analyse suivante',
                       'Back to top': 'Retour au rapport global',
                       'Page generated with': 'Page generated with', 'with icons from': 'with icons from',
                       'and': 'and', 'in': 'in', 'sources': 'sources', 'on': 'on' })

    def __init__(self, resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                       synthCols=None, dCustomTrans=dict(), lang='en',
                       attachedDir='.', tgtFolder='.', tgtPrefix='results'):
    
        assert synthCols is None or isinstance(synthCols, list) or isinstance(synthCols, pd.MultiIndex), \
               'synthesis columns must be specified as None (all), or as a list of tuples, or as a pandas.MultiIndex'
        
        super().__init__(resultsSet, title, subTitle, description, keywords,
                         dCustomTrans=dCustomTrans, lang=lang,
                         attachedDir=attachedDir, tgtFolder=tgtFolder, tgtPrefix=tgtPrefix)
        
        self.synthCols = synthCols

        self.anlysSubTitle = anlysSubTitle
        
    # Attached files for HTML report.
    AttachedFiles = ['autods.css', 'fa-feather-alt.svg', 'fa-angle-up.svg', 'fa-file-excel.svg', 'fa-file-excel-hover.svg',
                     'fa-arrow-left-hover.svg', 'fa-arrow-left.svg', 'fa-arrow-right-hover.svg', 'fa-arrow-right.svg',
                     'fa-arrow-up-hover.svg', 'fa-arrow-up.svg']
    
    # Plot ... data to be plot, and draw resulting figure to image files.
    PlotImgPrfxQqPlot = 'qqplot'
    PlotImgPrfxDetProb = 'detprob'
    PlotImgPrfxProbDens = 'probdens'
    
    @classmethod
    def generatePlots(cls, plotsData, tgtFolder, imgFormat='png', figSize=(12, 6),
                      grid=True, bgColor='#f9fbf3', transparent=False, trColors=['blue', 'red']):
        
        imgFormat = imgFormat.lower()
        
        # For each plot, 
        dPlots = dict()
        for title, pld in plotsData.items():
            
            # Plot a figure from the plot data (3 possible types, from title).
            if 'Qq-plot' in title:
                
                tgtFileName = cls.PlotImgPrfxQqPlot
                
                n = len(pld['dataRows'])
                df2Plot = pd.DataFrame(data=pld['dataRows'],
                                       columns=['If the fit was perfect ...', 'Real observations'],
                                       index=np.linspace(0.5/n, 1.0-0.5/n, n))
                
                axes = df2Plot.plot(figsize=figSize, color=trColors, grid=grid,
                                    xlim=(pld['xMin'], pld['xMax']),
                                    ylim=(pld['yMin'], pld['yMax']))

            elif 'Detection Probability' in title:
                
                tgtFileName = cls.PlotImgPrfxDetProb + title.split(' ')[-1] # Assume last "word" is the hist. number
                
                df2Plot = pd.DataFrame(data=pld['dataRows'], 
                                       columns=[pld['xLabel'], pld['yLabel'] + ' (sampled)',
                                                pld['yLabel'] + ' (fitted)'])
                df2Plot.set_index(pld['xLabel'], inplace=True)
                
                axes = df2Plot.plot(figsize=figSize, color=trColors, grid=grid,
                                    xlim=(pld['xMin'], pld['xMax']), 
                                    ylim=(pld['yMin'], pld['yMax']))
        
            elif 'Pdf' in title:
                
                tgtFileName = cls.PlotImgPrfxProbDens + title.split(' ')[-1] # Assume last "word" is the Pdf number
                
                df2Plot = pd.DataFrame(data=pld['dataRows'], 
                                       columns=[pld['xLabel'], pld['yLabel'] + ' (sampled)',
                                                pld['yLabel'] + ' (fitted)'])
                df2Plot.set_index(pld['xLabel'], inplace=True)
                
                axes = df2Plot.plot(figsize=figSize, color=trColors, grid=grid,
                                    xlim=(pld['xMin'], pld['xMax']), 
                                    ylim=(pld['yMin'], pld['yMax']))
        
            # Finish plotting.
            axes.legend(df2Plot.columns, fontsize=12)
            axes.set_title(label=pld['title'] + ' : ' + pld['subTitle'],
                           fontdict=dict(fontsize=16), pad=20)
            axes.set_xlabel(pld['xLabel'], fontsize=12)
            axes.set_ylabel(pld['yLabel'], fontsize=12)
            if not transparent:
                axes.set_facecolor(bgColor)
                axes.figure.patch.set_facecolor(bgColor)
                
            # Generate an image file for the plot figure (forcing the specified patch background color).
            tgtFileName = tgtFileName + '.' + imgFormat
            axes.figure.savefig(os.path.join(tgtFolder, tgtFileName),
                                box_inches='tight', transparent=transparent,
                                facecolor=axes.figure.get_facecolor(), edgecolor='none')
            plt.close(axes.figure)

            # Save image URL.
            dPlots[title] = tgtFileName
                
        return dPlots
    
    # Top page
    def toHtmlTop(self):
        
        print('Top page ...')
        
        # Generate post-processed and translated synthesis table.
        # a. Add run folder column if not selected (will serve to generate the link to the analysis detailed report)
        synCols = self.synthCols
        if DSAnalysis.RunFolderColumn not in synCols:
            synCols += [DSAnalysis.RunFolderColumn]
        dfSyn = self.resultsSet.dfTransData(self.lang, subset=synCols)
        dfSyn[self.trRunFolderCol] = dfSyn[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
        
        # b. Links to each analysis detailled report.
        dfSyn.index = \
            dfSyn.apply(lambda an: '<a href="./{p}/index.html">{n:03d}</a>' \
                                   .format(p=an[self.trRunFolderCol], n=an.name), axis='columns')
       
        # c. Post-format as specified in actual class.
        dfsSyn = self.finalFormatData(dfSyn)

        # Generate post-processed and translated detailed table.
        dfDet = self.resultsSet.dfTransData(self.lang)

        # a. Add run folder column if not there (will serve to generate the link to the analysis detailed report)
        detTrCols = list(dfDet.columns)
        if self.trRunFolderCol not in detTrCols:
            detTrCols += [self.trRunFolderCol]
        dfDet[self.trRunFolderCol] = dfDet[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
        dfDet = dfDet.reindex(columns=detTrCols)
       
        # b. Links to each analysis detailed report.
        dfDet.index = \
            dfDet.apply(lambda an: '<a href="./{p}/index.html">{n:03d}</a>' \
                                   .format(p=an[self.trRunFolderCol], n=an.name), axis='columns')
        
        # c. Post-format as specified in actual class.
        dfsDet = self.finalFormatData(dfDet, convert=False, round=False)

        # Generate top report page.
        genDateTime = dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        tmpl = self.getTemplateEnv().get_template('autods-top.htpl')
        xlFileUrl = os.path.basename(self.targetFilePathName(suffix='.xlsx')).replace(os.sep, '/')
        html = tmpl.render(synthesis=dfsSyn.render(), #escape=False, index=False),
                           details=dfsDet.render(), #escape=False, index=False),
                           title=self.title, subtitle=self.subTitle, keywords=self.keywords,
                           xlUrl=xlFileUrl, tr=self.dTrans[self.lang], genDateTime=genDateTime)
        html = re.sub('(?:[ \t]*\\\n){2,}', '\n'*2, html) # Cleanup blank line series to one only.

        # Write top HTML to file.
        htmlPathName = self.targetFilePathName(suffix='.html')
        with codecs.open(topHtmlPathName, mode='w', encoding='utf-8-sig') as tgtFile:
            tgtFile.write(html)

        return htmlPathName
    
    # Analyses pages.
    PlotImgFormat = 'png'
    def toHtmlAnalyses(self):
        
        # Generate translated synthesis and detailed tables.
        dfSynthRes = self.resultsSet.dfTransData(self.lang, subset=self.synthCols)
        dfDetRes = self.resultsSet.dfTransData(self.lang)

        print('Analyses pages ({}) ...'.format(len(dfSynthRes)))

        # 1. 1st pass : Generate previous / next list (for navigation buttons)
        sCurrUrl = dfSynthRes[self.trRunFolderCol]
        sCurrUrl = sCurrUrl.apply(lambda path: self.targetFilePathName(tgtFolder=path, prefix='index', suffix='.html'))
        sCurrUrl = sCurrUrl.apply(lambda path: os.path.relpath(path, self.tgtFolder).replace(os.sep, '/'))
        dfAnlysUrls = pd.DataFrame(dict(current=sCurrUrl, previous=np.roll(sCurrUrl, 1), next=np.roll(sCurrUrl, -1)))

        # 2. 2nd pass : Generate
        topHtmlPathName = self.targetFilePathName(suffix='.html')
        tmpl = self.getTemplateEnv().get_template('autods-anlys.htpl')
        engineClass = self.resultsSet.engineClass
        trCustCols = self.resultsSet.transCustomColumns(self.lang)
        
        for lblAnlys in dfDetRes.index:
            
            sAnlysCustCols = dfDetRes.loc[lblAnlys, trCustCols]
            print('  #{}'.format(lblAnlys), ' '.join(['{}={}'.format(k, v) for k, v in sAnlysCustCols.iteritems()]))
        
            anlysFolder = dfDetRes.at[lblAnlys, self.trRunFolderCol]

            # Postprocess synthesis table :
            dfSyn = dfSynthRes.loc[lblAnlys].to_frame().T
            dfSyn.index = dfSyn.index.map(lambda n: '{:03d}'.format(n))
            dfsSyn = self.finalFormatData(dfSyn)
            
            # Postprocess detailed table :
            dfDet = dfDetRes.loc[lblAnlys].to_frame().T
            dfDet.index = dfDet.index.map(lambda n: '{:03d}'.format(n))
            dfsDet = self.finalFormatData(dfDet, convert=False, round=False)
            
            # Generate analysis report page.
            genDateTime = dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            subTitle = 'Analyse {:03d} : {}'.format(lblAnlys, self.anlysSubTitle)
            sAnlysUrls = dfAnlysUrls.loc[lblAnlys]
            html = tmpl.render(synthesis=dfsSyn.render(), #escape=False, index=True),
                               details=dfsDet.render(), #(escape=False, index=True),
                               log=engineClass.decodeLog(anlysFolder),
                               output=engineClass.decodeOutput(anlysFolder),
                               plots=self.generatePlots(engineClass.decodePlots(anlysFolder),
                                                        anlysFolder, imgFormat=self.PlotImgFormat),
                               title=self.title, subtitle=subTitle, keywords=self.keywords,
                               navUrls=dict(prevAnlys='../'+sAnlysUrls.previous,
                                            nextAnlys='../'+sAnlysUrls.next,
                                            back2Top='../'+os.path.basename(topHtmlPathName)),
                               tr=self.dTrans[self.lang], genDateTime=genDateTime)
            html = re.sub('(?:[ \t]*\\\n){2,}', '\n'*2, html) # Cleanup blank line series to one only.

            # Write analysis HTML to file.
            htmlPathName = self.targetFilePathName(tgtFolder=anlysFolder, prefix='index', suffix='.html')
            with codecs.open(htmlPathName, mode='w', encoding='utf-8-sig') as tgtFile:
                tgtFile.write(html)
        
    # HTML report generation (based on results.dfTransData).
    def toHtml(self):
        
        # Install needed attached files.
        self.installAttFiles(self.AttachedFiles)
            
        # Generate full report page for each analysis
        topHtmlPathName = self.toHtmlTop()

        # Generate report page for each analysis
        self.toHtmlAnalyses()

        print('... done.')
                
        return topHtmlPathName

    # Génération du rapport Excel.
    def toExcel(self):
        
        xlsxPathName = os.path.join(self.tgtFolder, self.tgtPrefix + '.xlsx')
        
        with pd.ExcelWriter(xlsxPathName) as xlsxWriter:
            
            # Synthesis
            dfSyn = self.resultsSet.dfTransData(self.lang, subset=self.synthCols)
            dfSyn[self.trRunFolderCol] = dfSyn[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
            
            # ... Convert run folder columns to hyperlink if present
            def toHyperlink(path):
                return '=HYPERLINK("file:///{path}", "{path}")'.format(path=path)
            if DSAnalysis.RunFolderColumn in self.synthCols:                
                dfSyn[self.trRunFolderCol] = dfSyn[self.trRunFolderCol].apply(toHyperlink)
            
            dfsSyn = self.finalFormatData(dfSyn)
            
            dfsSyn.to_excel(xlsxWriter, sheet_name=self.tr('Synthesis'), index=True)
            
            # Details
            dfDet = self.resultsSet.dfTransData(self.lang)
            
            dfSyn[self.trRunFolderCol] = dfSyn[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
            dfDet[self.trRunFolderCol] = dfDet[self.trRunFolderCol].apply(toHyperlink)
            
            dfsDet = self.finalFormatData(dfDet, convert=False, round=False)
            
            dfsDet.to_excel(xlsxWriter, sheet_name=self.tr('Details'), index=True)

        return xlsxPathName

# Results pre-reports class 
# (HTML only, targeting very simple mono-table synthesis for fully automatic pre-analyses,
#  in order to give the user hints about what analyses are to be done, and with what parameter values).
class ResultsPreReport(ResultsFullReport):

    # Translation table.
    DTrans = dict(en={ 'RunFolder': 'Analysis', 'Synthesis': 'Synthesis', 'Details': 'Details',
                       'Synthesis table': 'Synthesis table',
                       'Click on analysis # for details': 'Click on analysis number to get to detailed report',
                       'Sample': 'Sample', 'Parameters': 'Parameters', 'Results': 'Results',
                       'PDF': 'Detection probability density',
                       'Detailed results': 'Detailed results',
                       'Download Excel': 'Download as Excel(TM) file',
                       'Summary computation log': 'Summary computation log',
                       'Detailed computation log': 'Detailed computation log',
                       'Previous analysis': 'Previous analysis', 'Next analysis': 'Next analysis',
                       'Back to top': 'Back to global report',
                       'Page generated with': 'Page générée via', 'with icons from': 'avec les pictogrammes de',
                       'and': 'et', 'in': 'dans', 'sources': 'sources', 'on': 'le' },
                  fr={ 'DossierExec': 'Analyse', 'Synthesis': 'Synthèse', 'Details': 'Détails',
                       'Synthesis table': 'Tableau de synthèse',
                       'Click on analysis # for details': 'Cliquer sur le numéro de l\'analyse pour accéder au rapport détaillé',
                       'Sample': 'Echantillon', 'Parameters': 'Paramètres', 'Results': 'Résultats',
                       'PDF': 'Densité de probabilité de détection',
                       'Detailed results': 'Résultats en détails',
                       'Download Excel': 'Télécharger le classeur Excel (TM)',
                       'Summary computation log': 'Résumé des calculs', 'Detailed computation log': 'Détail des calculs',
                       'Previous analysis': 'Analyse précédente', 'Next analysis': 'Analyse suivante',
                       'Back to top': 'Retour au rapport global',
                       'Page generated with': 'Page generated with', 'with icons from': 'with icons from',
                       'and': 'and', 'in': 'in', 'sources': 'sources', 'on': 'on' })

    def __init__(self, resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                       sampleCols, paramCols, resultCols, anlysSynthCols=None, 
                       dCustomTrans=dict(), pdfPlotHeight=256, lang='en', attachedDir='.', tgtFolder='.', tgtPrefix='results'):
    
        super().__init__(resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                         synthCols=anlysSynthCols, dCustomTrans=dCustomTrans, lang=lang,
                         attachedDir=attachedDir, tgtFolder=tgtFolder, tgtPrefix=tgtPrefix)
    
        self.sampleCols = sampleCols
        self.paramCols = paramCols
        self.resultCols = resultCols
        self.pdfPlotHeight = pdfPlotHeight

    @staticmethod
    def series2VertTable(ser):
        
        return re.sub('\\\n *', '', ser.to_frame().to_html(header=False))
    
        #return ''.join('<p>{}: {}</p>'.format(k, v) for k, v in dictOrSeries.items())
        
    def pdfImageHtmlElement(self, runFolder):
        
        for pdfInd in range(3, 0, -1):
            pdfFileName = '{}{}.{}'.format(self.PlotImgPrfxDetProb, pdfInd, self.PlotImgFormat)
            if os.path.isfile(os.path.join(runFolder, pdfFileName)):
                return '<img src="./{}/{}" style="height: {}px" />' \
                       .format(self.relativeRunFolderUrl(runFolder), pdfFileName, self.pdfPlotHeight)
        
        return 'PDF image file not found'
        
    # Top page
    def toHtmlTop(self):
        
        print('Top page ...')
        
        # Generate the table to display from raw results (index + 4 columns : sample, params, results, PDF plot)
        # 1. Get translated and post-formated detailed results
        dfDet = self.resultsSet.dfTransData(self.lang)
        dfsDet = self.finalFormatData(dfDet)
        dfDet = dfsDet.data

        # 2. Translate sample, parameter and result columns
        dTransResCol = self.resultsSet.transTable()
        sampleTrCols = [dTransResCol[self.lang].get(col, str(col)) for col in self.sampleCols]
        paramTrCols = [dTransResCol[self.lang].get(col, str(col)) for col in self.paramCols]
        resultTrCols = [dTransResCol[self.lang].get(col, str(col)) for col in self.resultCols]
        
        # 3. Fill target table index and columns
        dfSyn = pd.DataFrame(dict(Sample=dfDet[sampleTrCols].apply(self.series2VertTable, axis='columns'),
                                  Parameters=dfDet[paramTrCols].apply(self.series2VertTable, axis='columns'),
                                  Results=dfDet[resultTrCols].apply(self.series2VertTable, axis='columns'),
                                  PDF=dfDet[self.trRunFolderCol].apply(self.pdfImageHtmlElement)))
        
        dfSyn.index = \
            dfDet.apply(lambda an: '<a href="./{p}/index.html">{n:03d}</a>' \
                                   .format(p=self.relativeRunFolderUrl(an[self.trRunFolderCol]), n=an.name),
                        axis='columns')
        
        # 4. Translate table columns.
        dfSyn.columns = [self.tr(col) for col in dfSyn.columns]

        self.dfSyn = dfSyn
        
        # Generate top report page.
        genDateTime = dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        tmpl = self.getTemplateEnv().get_template('autods-pretop.htpl')
        xlFileUrl = os.path.basename(self.targetFilePathName(suffix='.xlsx')).replace(os.sep, '/')
        html = tmpl.render(synthesis=dfSyn.to_html(escape=False),
                           title=self.title, subtitle=self.subTitle, keywords=self.keywords,
                           xlUrl=xlFileUrl, tr=self.dTrans[self.lang], genDateTime=genDateTime)
        html = re.sub('(?:[ \t]*\\\n){2,}', '\n'*2, html) # Cleanup blank lines series to one only

        # Write top HTML to file.
        htmlPathName = self.targetFilePathName(suffix='.html')
        with codecs.open(htmlPathName, mode='w', encoding='utf-8-sig') as tgtFile:
            tgtFile.write(html)

        return htmlPathName
    
    # HTML report generation (based on results.dfTransData).
    def toHtml(self):
        
        # Install needed attached files.
        self.installAttFiles(self.AttachedFiles)
            
        # Generate full report page for each analysis
        # (done first to have PDF image files generated for top report page generation just below).
        self.toHtmlAnalyses()
        
        # Generate top report page.
        topHtmlPathName = self.toHtmlTop()

        print('... done.')
                
        return topHtmlPathName


# A specialized full report for MCDS analyses, with actual output formating
class MCDSResultsFullReport(ResultsFullReport):

    DCustTrans = \
        dict(en={ 'Note: Some figures rounded or converted': 
                     "<strong>Note</strong>: Densities are expressed per square km,"
                     " and most figures have been rounded for readability",
                  'Note: All figures untouched, as output by MCDS': 
                     "<strong>Note</strong>: All values have been left untouched,"
                     " as outuput by MCDS (no rounding, no conversion)" },
             fr={ 'Note: Some figures rounded or converted':
                      "<strong>N.B.</strong> Les densités sont exprimées par km carré, et presque toutes les valeurs"
                      " ont été arrondies pour la lisibilité",
                  'Note: All figures untouched, as output by MCDS':
                      "<strong>N.B.</strong> Aucune valeur n'a été convertie ou arrondie,"
                      " elles sont toutes telles que produites par MCDS" })
    
    def __init__(self, resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                       synthCols=None, lang='en', attachedDir='.', tgtFolder='.', tgtPrefix='results'):
    
        super().__init__(resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                         synthCols, self.DCustTrans, lang, attachedDir, tgtFolder, tgtPrefix)
        
    # Styling colors
    cChrGray = '#869074'
    cBckGreen, cBckGray = '#e0ef8c', '#dae3cb'
    cSclGreen, cSclOrange, cSclRed = '#cbef8c', '#f9da56', '#fe835a'
    cChrInvis = '#e8efd1' # body background
    scaledColors = [cSclGreen, cSclOrange, cSclRed]
    scaledColorsRvd = list(reversed(scaledColors))
    
    dExCodeColors = dict(zip([1, 2, 3], scaledColors))
    
    @classmethod
    def colorExecCodes(cls, sCodes):
        return ['background-color: ' + cls.dExCodeColors.get(c, cls.dExCodeColors[3]) for c in sCodes]
    
    @classmethod
    def scaledColorV(cls, v, thresholds, colors): # len(thresholds) == len(colors) - 1
        if pd.isnull(v):
            return cls.cBckGray
        for ind, thresh in enumerate(thresholds):
            if v > thresh:
                return colors[ind]
        return colors[-1]
    
    @classmethod
    def scaledColorS(cls, sValues, thresholds, colors):
        return ['background-color: ' + cls.scaledColorV(v, thresholds, colors) for v in sValues]
    
    @staticmethod
    def isNull(o):
        return not isinstance(o, list) and pd.isnull(o)
    
    @staticmethod
    def shortenDistCuts(distCuts):
        short = str(distCuts)
        if short in ['None', 'nan']:
            return None
        else:
            return short.translate(str.maketrans({c:'' for c in '[] '})).replace('.0,', ',')

    # Final formatting of translated data tables, for HTML or SpreadSheet rendering
    # (sort, convert units, round values, and style).
    # Note: Use trEnColNames method to pass from EN-translated columns names to self.lang-ones
    # Return a pd.DataFrame.Styler
    def finalFormatData(self, dfTrData, sort=True, convert=True, round=True, style=True):
        
        # Sorting
        df = dfTrData
        if sort:
            
            df.sort_values(by=[col for col in self.trEnColNames(['Sample', 'Delta AIC']) if col in df.columns], 
                           ascending=[True, True], inplace=True)
        
        # Converting to other units, or so.
        kVarDens = 1.0
        if convert:
            
            for col in self.trEnColNames(['Density', 'Min Density', 'Max Density']): # 'CoefVar Density', 
                if col in df.columns:
                    df[col] *= 1000000 / 10000 # ha => km2
            
            col = self.trEnColNames('CoefVar Density')
            if col in df.columns:
                kVarDens = 100.0
                df[col] *= kVarDens # [0, 1] => %
            
            for col in self.trEnColNames(['Fit Dist Cuts', 'Discr Dist Cuts']):
                if col in df.columns:
                    df[col] = df[col].apply(self.shortenDistCuts)
            
        # Reducing float precision
        if round:
            
            dColDecimals = { **{ col: 2 for col in ['Delta AIC', 'Chi2 P', 'KS P',
                                                    'PDetec', 'Min PDetec', 'Max PDetec'] },
                             **{ col: 1 for col in ['AIC', 'EDR/ESW', 'Min EDR/ESW', 'Max EDR/ESW',
                                                    'Density', 'Min Density', 'Max Density', 'CoefVar Density'] } }
            df = df.round(decimals={ col: dec for col, dec in self.trEnColNames(dColDecimals).items() if col in df.columns })
        
        # Styling
        dfs = df.style
        if style:
            
            col = self.trEnColNames('Delta AIC')
            if col in df.columns:
                dfs.set_properties(subset=pd.IndexSlice[df[df[col] == 0].index, :],
                                   **{'background-color': self.cBckGreen})
               
            col = self.trEnColNames('ExCod')
            if col in df.columns:
                dfs.apply(self.colorExecCodes, subset=[col], axis='columns')
            
            col = self.trEnColNames('CoefVar Density')
            if col in df.columns:
                dfs.apply(self.scaledColorS, subset=[col], axis='columns',
                          thresholds=[v * kVarDens for v in [0.3, 0.2]], colors=self.scaledColorsRvd)
            
            col = self.trEnColNames('KS P')
            if col in df.columns:
                dfs.apply(self.scaledColorS, subset=[col], axis='columns',
                          thresholds=[0.7, 0.2], colors=self.scaledColors)
            
            col = self.trEnColNames('Chi2 P')
            if col in df.columns:
                dfs.apply(self.scaledColorS, subset=[col], axis='columns',
                          thresholds=[0.7, 0.2], colors=self.scaledColors)
            
            col = self.trEnColNames('ExCod')
            if col in df.columns:
                dfs.set_properties(subset=pd.IndexSlice[df[~df[col].isin([1, 2])].index, :],
                                   **{'color': self.cChrGray})
            
            dfs.where(self.isNull, 'color: transparent').where(self.isNull, 'text-shadow: none')
        
        return dfs


# A specialized full report for MCDS analyses, with actual output formating
class MCDSResultsPreReport(ResultsPreReport):

    def __init__(self, resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                       sampleCols, paramCols, resultCols, anlysSynthCols=None, 
                       dCustomTrans=dict(), pdfPlotHeight=256, lang='en', attachedDir='.', tgtFolder='.', tgtPrefix='results'):

        super().__init__(resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                         sampleCols, paramCols, resultCols, anlysSynthCols,
                         dCustomTrans, pdfPlotHeight, lang, attachedDir, tgtFolder, tgtPrefix)
        
    # Final formatting of translated data tables, for HTML or SpreadSheet rendering
    # (sort, convert units, round values, and style).
    # Note: Use trEnColNames method to pass from EN-translated columns names to self.lang-ones
    # Return a pd.DataFrame.Styler
    def finalFormatData(self, dfTrData, sort=True, convert=True, round=True, style=True):
        
        # Sorting
        df = dfTrData
        if sort:
            
            pass # No sort to be done here
        
        # Converting to other units, or so.
        kVarDens = 1.0
        if convert:
            
            for col in self.trEnColNames(['Density', 'Min Density', 'Max Density']):
                if col in df.columns:
                    df[col] *= 1000000 / 10000 # ha => km2
            
            col = self.trEnColNames('CoefVar Density')
            if col in df.columns:
                kVarDens = 100.0
                df[col] *= kVarDens # [0, 1] => %
            
        # Reducing float precision
        if round:
            
            dColDecimals = { **{ col: 2 for col in ['Delta AIC', 'Chi2 P', 'KS P',
                                                    'PDetec', 'Min PDetec', 'Max PDetec'] },
                             **{ col: 1 for col in ['AIC', 'EDR/ESW', 'Min EDR/ESW', 'Max EDR/ESW',
                                                    'Density', 'Min Density', 'Max Density', 'CoefVar Density'] } }
            df = df.round(decimals={ col: dec for col, dec in self.trEnColNames(dColDecimals).items() if col in df.columns })
        
        # Styling
        dfs = df.style
        if style:
            
            pass # No styling to be done here.
        
        return dfs


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
    analysis = MCDSAnalysis(engine=engine, dataSet=dataSet, name=args.engineType,
                            estimKeyFn=args.keyFn, estimAdjustFn=args.adjustFn,
                            estimCriterion=args.criterion, cvInterval=args.cvInterval)

    dResults = analysis.run(realRun=not args.debug)
    
    # Print results
    print('Results:\n' + '\n'.join('* {}: {}'.format(k, v) for k, v in dResults.items()))

    sys.exit(analysis.runStatus)
