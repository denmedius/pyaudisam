# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment


import sys
import os
import tempfile

import re

from collections import OrderedDict as odict

import numpy as np
import pandas as pd


# Workaround df.to_csv(float_format='%.xf') not working when NaNs in serie
def safeFloat2Str(val, prec=None, decPt='.'):
    strVal = '' if np.isnan(val) else str(val) if prec is None \
                else '{:.{prec}f}'.format(val, prec=prec)
    if decPt != '.':
        strVal = strVal.replace('.', decPt)
    return strVal


# DSEngine classes.
class DSEngine(object):
    
    # Distance software detection params.
    DistanceSuppVers = [7, 6] # Lastest first.
    DistancePossInstPaths = [os.path.join('C:/', 'Program files (x86)'),
                             os.path.join('C:/', 'Program files')]

    # Find given executable installation dir.
    def findDistExecutable(self, exeFileName):

        self.exeFilePathName = None
        print('Looking for {} ...'.format(exeFileName))
        for path in self.DistancePossInstPaths:
            for ver in self.DistanceSuppVers:
                exeFileDir = os.path.join(path, 'Distance ' + str(ver))
                print(' - checking {} : '.format(exeFileDir), end='')
                if not os.path.exists(os.path.join(exeFileDir, exeFileName)):
                    print('no.')
                else:
                    print('yes !')
                    self.exeFilePathName = os.path.join(exeFileDir, exeFileName)
                    break
            if self.exeFilePathName:
                break

        if self.exeFilePathName:
            print('{} found in {}'.format(exeFileName, exeFileDir))
        else:
            print('Error : Could not find {} ; please install Distance software (V6 or later)'.format(exeFileName))
    
    # Options possible values.
    DistUnits = ['Meter']
    AreaUnits = ['Hectare']
    
    def __init__(self, exeFileName, workDir='.',
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
        self.workDir = workDir
        if not os.path.isdir(workDir):
            os.makedirs(workDir)
        
        # Detect engine executable installation folder
        self.findDistExecutable(exeFileName)
            
    def __bool__(self):
        
        return self.exeFilePathName is not None

    # Possible regexps for auto-detection of columns to import from data sets / export
    # TODO: Complete for non 'Point transect' modes
    ImportFieldAliasREs = \
        odict([('STR_LABEL', ['region', 'zone', 'strate', 'stratum']),
               ('STR_AREA', ['surface', 'area', 'ha', 'km2']),
               ('SMP_LABEL', ['point', 'lieu', 'location']),
               ('SMP_EFFORT', ['effort', 'passages', 'surveys', 'samplings']),
               ('DISTANCE', ['distance'])])
    
    # Data fields of decimal type.
    # TODO: Complete for non 'Point transect' modes
    DecimalFields = ['SMP_EFFORT', 'DISTANCE']
    
    # Match srcFields with tgtAliasREs ones ; keep remaining ones ; sort decimal fields.
    def matchDataFields(self, srcFields, tgtAliasREs=odict()):
        
        print('Matching required data columns:', end=' ')
        
        matFields = list()
        matDecFields = list()
        for tgtField in tgtAliasREs:
            print(tgtField, end='=')
            foundTgtField = False
            for srcField in srcFields:
                for pat in tgtAliasREs[tgtField]:
                    if re.match(pat, srcField, flags=re.IGNORECASE):
                        print(srcField, end=', ')
                        matFields.append(srcField)
                        if tgtField in self.DecimalFields:
                            matDecFields.append(srcField)
                        foundTgtField = True
                        break
                if foundTgtField:
                    break
            if not foundTgtField:
                raise Exception('Error: Failed to find a match for expected {} in dataset columns {}' \
                                .format(tgtField, srcFields))
                
        remFields = [field for field in srcFields if field not in matFields]

        print('... success.')
        
        return matFields, remFields, matDecFields

    # Setup run folder (all in and out files will go there)
    def setupRunFolder(self, runName='ds'):
        
        self.runDir = tempfile.mkdtemp(dir=self.workDir, prefix=runName+'-')
        
        print('Will run in', self.runDir)
        
        return self.runDir
    
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

    # Command file template (for str.format()ing).
    CmdTxt = \
        '\n'.join(map(str.strip,
                  """{output}
                     {log}
                     {stats}
                     {bootstrap}
                     None
                     None
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
                  """.split())) + '\n'

    cmdFileName   = 'cmd.txt'
    dataFileName  = 'data.txt'
    outFileName   = 'output.txt'
    logFileName   = 'log.txt'
    statsFileName = 'stats.txt'
    bootFileName  = 'bootstrap.txt'
        
    # Ctor
    def __init__(self, workDir='.',
                 distanceUnit='Meter', areaUnit='Hectare',
                 surveyType='Point', distanceType='Radial'):
        
        # Check options
        assert surveyType in self.SurveyTypes, \
               'Invalid survey type {} : should be in {}'.format(surveyType, self.SurveyTypes)
        assert distanceType in self.DistTypes, \
               'Invalid area unit{} : should be in {}'.format(distanceType, self.DistTypes)
        
        # Initialise base.
        super().__init__(exeFileName='MCDS.exe', workDir=workDir, 
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         surveyType=surveyType, distanceType=distanceType,
                         firstDataFields=self.FirstDataFields[surveyType])        
    
    # Build command file from options and params
    def buildCmdFile(self, **params):

        cmdTxt = self.CmdTxt.format(output=self.outFileName, log=self.logFileName,
                                    stats=self.statsFileName, bootstrap=self.bootFileName,
                                    survType=self.options['surveyType'], distType=self.options['distanceType'],
                                    distUnit=self.options['distanceUnit'], areaUnit=self.options['areaUnit'],
                                    dataFields=self.dataFields, dataFileName=self.dataFileName,
                                    estKeyFn=params['estimKeyFn'], estAdjustFn=params['estimAdjustFn'],
                                    estCriterion=params['estimCriterion'], cvInterv=params['cvInterval'])

        tgtPathName = os.path.join(self.runDir, self.cmdFileName)
        with open(tgtPathName, 'w') as cmdFile:
            cmdFile.write(cmdTxt)

        print('Commands written to', tgtPathName)

        return tgtPathName
    
    # Build input data table from data set (check and match mandatory columns, enforce order).
    def buildExportTable(self, dfData, decimalFields=list(), decPoint='.'):
        
        # Match dataSet table columns to MCDS expected fields from possible aliases
        matchFields, otherFields, matchDecFields = \
            self.matchDataFields(dfData.columns, self.ImportFieldAliasREs)
        exportFields = matchFields + otherFields
        
        print('Final data columns in order:', exportFields)
        
        # Put columns in the right order (first data fields ... first, in the same order)
        dfExport = dfData[exportFields].copy()

        # Prepare safe export of decimal data with may be some NaNs
        allDecFields = set(matchDecFields + decimalFields)
        print('Decimal columns:', allDecFields)
        for field in allDecFields:
            dfExport[field] = dfExport[field].apply(safeFloat2Str, decPt=decPoint)
                
        return dfExport, otherFields

    # Build MCDS input data file from data set.
    def buildDataFile(self, dataSet):
        
        # Build data to export (check and match mandatory columns, enforce order, keep other cols).
        dfExport, otherFields = self.buildExportTable(dataSet.dfData, dataSet.decimalFields, decPoint='.')
        
        # Save data fields for the engine : mandatory ones + remaining ones
        self.dataFields = self.options['firstDataFields'] + otherFields
        
        # Export.
        tgtPathName = os.path.join(self.runDir, self.dataFileName)
        dfExport.to_csv(tgtPathName, index=False, sep='\t', encoding='utf-8', header=None)
        
        print('Data MCDS-exported to', tgtPathName)
        
        return tgtPathName
    
    # Run MCDS
    def __call__(self, dataSet, runName='mcds', realRun=True, **analysisParms):
        
        # Create a new exclusive run folder
        self.setupRunFolder(runName)
        
        # Generate data and command file into this folder
        _ = self.buildDataFile(dataSet)
        cmdFileName = self.buildCmdFile(**analysisParms)
        
        # Call executable ...
        cmd = '"{}" 0, {}'.format(self.exeFilePathName, cmdFileName)
        if realRun:
            print('Running MCDS :', cmd)
            self.runStatus = os.system(cmd)
            print('Done : status =', self.runStatus)
            
        # ... unless specified not to (input files generated, but no execution).
        else:
            print('Not running MCDS :', cmd)
            self.runStatus = 2

        return self.runStatus, self.runDir

    # Build Distance/MCDS input data file from data set.
    def buildDistanceDataFile(self, dfData, tgtFilePathName, decimalFields=list()):
                
        # Build data to export (check and match mandatory columns, enforce order, keep other cols).
        dfExport, otherFields = self.buildExportTable(dfData, decimalFields, decPoint=',')
        
        # Export.
        dfExport.to_csv(tgtFilePathName, index=False, sep='\t', encoding='utf-8',
                        header=self.FirstDistanceExportFields[self.options['surveyType']] + otherFields)

        print('Data Distance-exported to', tgtFilePathName)
        
        return tgtFilePathName

# Analysis : Gather input params, data set, results, debug and log files
class DSAnalysis(object):
    
    def __init__(self, engine, dataSet, name):
        
        self.engine = engine
        self.dataSet = dataSet
        self.name = name
        
class MCDSAnalysis(DSAnalysis):
    
    EstKeyFns = ['UNIFORM', 'HNORMAL', 'HAZARD'] #, 'NEXPON']
    EstAdjustFns = ['COSINE', 'POLY', 'HERMITE']
    EstCriterions = ['AIC']

    def __init__(self, engine, dataSet, namePrefix='mcds',
                 estimKeyFn='HNORMAL', estimAdjustFn='COSINE', estimCriterion='AIC', cvInterval=95):
        
        # Check analysis params
        assert estimKeyFn in self.EstKeyFns, \
               'Invalid estimate key function {}: should be in {}'.format(estimKeyFn, self.EstKeyFns)
        assert estimAdjustFn in self.EstAdjustFns, \
               'Invalid estimate adjust function {}: should be in {}'.format(estimAdjustFn, self.EstAdjustFns)
        assert estimCriterion in self.EstCriterions, \
               'Invalid estimate criterion {}: should be in {}'.format(estimCriterion, self.EstCriterions)
        assert cvInterval > 0 and cvInterval < 100,\
               'Invalid cvInterval {}% : should be in {}'.format(cvInterval, ']0%, 100%[')

        # Build name from main params
        name = '-'.join([namePrefix] + [p[:4] for p in [estimKeyFn, estimAdjustFn, estimCriterion]] \
                        + [str(cvInterval)])

        # Initialise base.
        super().__init__(engine, dataSet, name)
        
        # Save params.
        self.estimKeyFn = estimKeyFn
        self.estimAdjustFn = estimAdjustFn
        self.estimCriterion = estimCriterion
        self.cvInterval = cvInterval
    
    def run(self, realRun=True):
        
        self.rc, self.filesDir = self.engine(dataSet=self.dataSet, runName=self.name,
                                             estimKeyFn=self.estimKeyFn, estimAdjustFn=self.estimAdjustFn,
                                             estimCriterion=self.estimCriterion, cvInterval=self.cvInterval)
        
        # Load and decode results and log.
        dResults = odict([('estimKeyFn', self.estimKeyFn), ('estimAdjustFn', self.estimAdjustFn),
                          ('estimCriterion', self.estimCriterion), ('cvInterval', self.cvInterval),
                          ('filesDir', self.filesDir), ('rc', self.rc),
                          #TODO
                         ])
        
        return dResults
    
# A data set for multiple analyses, with 1 or 0 individual per line
# Warning:
# * Only Point transect supported as for now
# * No further change in decimal precision : provide what you need !
class DataSet(object):
    
    def __init__(self, dfData, decimalFields=list()):
        
        assert not dfData.empty, 'No data in set'
        assert all(field in dfData.columns for field in decimalFields), \
               'Some declared decimal field(s) are not in dfData columns {}' \
               .format(','.join(dfData.columns))
        
        self.dfData = dfData
        self.decimalFields = decimalFields


if __name__ == '__main__':

    print('A coder ...')
    
    sys.exit(0)
