# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Engines : Interface to (external) DS engines
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment


import sys
import os
import re
import pathlib as pl
import copy
import tempfile

from collections import OrderedDict as odict, namedtuple as ntuple

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger('autods')

# Actual package install dir.
KInstDirPath = pl.Path(__file__).parent.resolve()


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
    DistancePossInstPaths = [pl.Path('C:/Program files (x86)'), pl.Path('C:/Program files'), KInstDirPath]

    # Find given executable installation dir.
    # Note: MCDS.exe is an autonomous executable : simply put it in a "Distance 7" subfolder
    #       of this package's one, and it'll work ! Or else install Distance 7 (or later) the normal way :-)
    @staticmethod
    def findExecutable(exeFileName):

        exeFilePathName = None
        logger.debug('Looking for {} ...'.format(exeFileName))
        for ver in DSEngine.DistanceSuppVers:
            for path in DSEngine.DistancePossInstPaths:
                exeFN = path / 'Distance {}'.format(ver) / exeFileName
                if not exeFN.exists():
                    logger.debug('  Checking {} : No,'.format(exeFN))
                else:
                    logger.info('Found {} here: {}.'.format(exeFileName, exeFN))
                    exeFilePathName = exeFN
                    break
            if exeFilePathName:
                break

        if not exeFilePathName:
            raise Exception('Could not find {} ; please install Distance software (V6 or later)'.format(exeFileName))
            
        return exeFilePathName
    
    # Specifications of output stats.
    DfStatRowSpecs, DfStatModSpecs, DfStatModNotes, MIStatModCols, DfStatModColTrans = None, None, None, None, None
    
    def __init__(self, workDir='.', # As a simple str, or a pl.Path
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
        assert all(c not in str(workDir) for c in self.ForbidPathChars), \
               'Invalid character from "{}" in workDir folder "{}"' \
               .format(''.join(self.ForbidPathChars), workDir)
        self.workDir = pl.Path(workDir)
        self.workDir.mkdir(exist_ok=True)
            
    # Possible regexps for auto-detection of columns to import from data sets / export
    # TODO: Complete for non 'Point transect' modes
    ImportFieldAliasREs = \
        odict([('STR_LABEL', ['region', 'zone', 'secteur', 'strate', 'stratum']),
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
        
        logger.debug('Matching required data columns:')
        
        # Try and match required data columns.
        matFields = list()
        matDecFields = list()
        for tgtField in tgtAliasREs:
            logger.debug(' * ' + tgtField + ':')
            foundTgtField = False
            for srcField in srcFields:
                for pat in tgtAliasREs[tgtField]:
                    if re.search(pat, srcField, flags=re.IGNORECASE):
                        logger.debug('  . ' + srcField)
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

        logger.debug('... success.')
        
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
            
        logger.debug('Will run in ' + str(self.runDir))
        
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

    # Estimator adjustment terms selection criterion.
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
    ExeFilePathName = DSEngine.findExecutable('MCDS.exe')

    # Output stats specs : load from external files (extracts from Distance doc).
    @classmethod
    def loadStatSpecs(cls, nMaxAdjParams=10):
        
        if MCDSEngine.DfStatRowSpecs is not None:
            return
        
        logger.debug('MCDS : Loading output stats specs ...')
        
        # Output stats row specifications
        fileName = KInstDirPath / 'mcds/stat-row-specs.txt'
        logger.debug('* ' + str(fileName))
        with open(fileName, mode='r', encoding='utf8') as fStatRowSpecs:
            statRowSpecLines = [line.rstrip('\n') for line in fStatRowSpecs.readlines() if not line.startswith('#')]
            statRowSpecs =  [(statRowSpecLines[i].strip(), statRowSpecLines[i+1].strip()) \
                             for i in range(0, len(statRowSpecLines)-2, 3)]
            cls.DfStatRowSpecs = pd.DataFrame(columns=['Name', 'Description'],
                                              data=statRowSpecs).set_index('Name')
            assert not cls.DfStatRowSpecs.empty, 'Empty MCDS stats row specs'
        
        # Module and stats number to description table
        fileName = KInstDirPath / 'mcds/stat-mod-specs.txt'
        logger.debug('* ' + str(fileName))
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
        fileName = KInstDirPath / 'mcds/stat-mod-notes.txt'
        logger.debug('* ' + str(fileName))
        with open(fileName, mode='r', encoding='utf8') as fStatModNotes:
            statModNoteLines = [line.rstrip('\n') for line in fStatModNotes.readlines() if not line.startswith('#')]
            statModNotes =  [(int(line[:2]), line[2:].strip()) for line in statModNoteLines if line]
            cls.DfStatModNotes = pd.DataFrame(data=statModNotes, columns=['Note', 'Text']).set_index('Note')
            assert not cls.DfStatModNotes.empty, 'Empty MCDS stats module notes'
            
        # DataFrame for translating 3-level multi-index columns to 1 level lang-translated columns
        fileName = KInstDirPath / 'mcds/stat-mod-trans.txt'
        logger.debug('* ' + str(fileName))
        cls.DfStatModColsTrans = pd.read_csv(fileName, sep='\t')
        cls.DfStatModColsTrans.set_index(['Module', 'Statistic', 'Figure'], inplace=True)
        
        logger.debug('MCDS : Loaded output stats specs.')
        logger.debug('')
        
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
    def buildCmdFile(self, **params):

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
                                    echoData=('' if params['logData'] else 'No') + 'Echo',
                                    estKeyFn=params['estimKeyFn'], estAdjustFn=params['estimAdjustFn'],
                                    estCriterion=params['estimCriterion'], cvInterv=params['cvInterval'],
                                    distDiscrSpecs=distDiscrSpecs, gOFitSpecs=gOFitSpecs)

        # Write file.
        with open(self.cmdFileName, mode='w', encoding='utf-8') as cmdFile:
            cmdFile.write(cmdTxt)

        logger.debug('Commands written to ' + str(self.cmdFileName))

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
        
        logger.debug('Final data columns export order: ' + str(exportFields))
        
        # Put columns in the right order (first data fields ... first, in the same order)
        dfExport = dataSet.dfData[exportFields].copy()

        # Prepare safe export of decimal data with may be some NaNs
        allDecFields = set(matchDecFields + dataSet.decimalFields).intersection(exportFields)
        logger.debug('Decimal columns: ' + str(allDecFields))
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
        
        logger.debug('Data MCDS-exported to ' + str(self.dataFileName))
        
        return self.dataFileName
    
    # Run status codes (from MCDS documentation)
    RCNotRun      = 0
    RCOK          = 1
    RCWarnings    = 2
    RCErrors      = 3
    RCFileErrors  = 4
    RCOtherErrors = 5 # and above.
    
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
            logger.info('Running MCDS : ' + cmd)
            self.runTime = pd.Timestamp.now()
            self.runStatus = os.system(cmd)
            logger.info('Run MCDS : status = ' + str(self.runStatus))
            
        # ... unless specified not to (input files generated, but no execution).
        else:
            logger.info('Not running MCDS : ' + cmd)
            self.runTime = pd.NaT
            self.runStatus = self.RCNotRun

        return self.runStatus, self.runTime, self.runDir
    
    # Decode output stats file to a value series
    # Precondition: self.run(...)
    # Warning: No support for more than 1 stratum, 1 sample, 1 estimator.
    def decodeStats(self):

        logger.debug('Decoding ' + str(self.statsFileName) + ' ... ')
        
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
        logger.debug('Done decoding stats.')
        
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

        logger.debug('Data Distance-exported to ' + tgtFilePathName)
        
        return tgtFilePathName
   

if __name__ == '__main__':

    sys.exit(0)