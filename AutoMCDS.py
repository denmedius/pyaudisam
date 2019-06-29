# coding: utf-8

import sys
import os
import numpy as np
import pandas as pd


class MCDSEngine(object):
    
    DistUnits = ['Meter']
    AreaUnits = ['Hectare']
    DataFields = ['STR_LABEL', 'STR_AREA', 'SMP_LABEL', 'SMP_EFFORT', 'DISTANCE']
    SurveyTypes = ['Point', 'Line']
    DistTypes = ['Radial'] #? 'Perpendicular', 'Radial & Angle']

    def __init__(self, distanceUnit='Meter', areaUnit='Hectare',
                 surveyType='Point', distanceType='Radial'
                 dataFields=['STR_LABEL', 'STR_AREA', 'SMP_LABEL', 'SMP_EFFORT', 'DISTANCE']):
        
        self.cmdFileName = 'mcds-cmd.txt',
        self.outFileName = 'mcds-output.txt',
        self.logFileName = 'mcds.log',
        self.statsFileName = 'mcds-stats.txt',
        self.bootFileName = 'mcds-bootstrap.txt')
        
        # Contrôle des options
        assert distUnit in self.DistUnits, \
               'Invalid distance unit {}: should be in {}'.format(distUnit, self.DistUnits)
        assert areaUnit in self.AreaUnits, \
               'Invalid area unit {}: should be in {}'.format(areaUnit, self.AreaUnits)
        assert all(field in self.DataFields for field in dataFields),\
               'Invalid data field {}: should be in {}'.format(dataFields, self.DataFields)
        assert surveyType in self.SurveyTypes, \
               'Invalid survey type {} : should be in {}'.format(surveyType, self.SurveyTypes)
        assert distanceType in self.DistTypes, \
               'Invalid area unit{} : should be in {}'.format(distanceType, self.DistTypes)
        
        self.options.update(distanceUnit=distanceUnit, areaUnit=areaUnit,
                            surveyType=surveyType, distanceType=distanceType,
                            dataFields=dataFields)
        
        # Détection du dossier d'install. de Distance
        self.mcdsExe = 'MCDS.exe'
        possVers = [7, 6]
        possPaths = [os.path.join('C:\\', 'Program files (x86)'),
                     os.path.join('C:', 'Program files')]
        print('Recherche {} ...'.format(mcdsExe))
        for path in possPaths:
            for ver in possVers:
                self.mcdsInstPath = os.path.join(path, 'Distance ' + str(ver))
                print(' - essai dans {} : '.format(mcdsInstPath), end='')
                if not os.path.exists(os.path.join(self.mcdsInstPath, self.mcdsExe)):
                    self.mcdsInstPath = None
                    print('non.')
                else:
                    print('eureka !')
                    break
            if self.mcdsInstPath:
                break

        if self.mcdsInstPath:
            print('{} trouvé dans {}'.format(mcdsExe, mcdsInstPath))
        else:
            print('Erreur : Impossible de trouver {}'.format(mcdsExe))
            
    def __bool__(self):
        
        return self.mcdsInstPath is not None

    def _buildCmdFile(self, dataFileName='mcds-data.txt', **params):

        cmdTxt = """{output}
        {log}
        {stats}
        {bootstrap}
        None
        None
        Options;
        Type={optSurvType};
        Distance={optDistType} /Measure='{optDistUnit}';
        Area /Units='{optAreaUnit}';
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

        """.format(output=self.outFileName, log=self.logFileName,
                   stats=self.statsFileName, bootstrap=self.bootFileName,
                   optSurvType=self.options['surveyType'], optDistType=self.options['distanceType'],
                   optDistUnit=self.options['distanceUnit'], optAreaUnit=self.options['areaUnit'],
                   dataFields=self.options['dataFields'],
                   dataFileName=dataFileName,
                   estKeyFn=params['estimKeyFn'], estAdjustFn=params['estimAdjustFn'],
                   estCriterion=params['estimCriterion'], cvInterv=params['cvInterval'])

        with open(self.cmdFileName, 'w') as cmdFile:
            cmdFile.write(cmdTxt)

        return self.cmdFileName

    # Run MCDS
    def __call__(self, dataFileName='mcds-data.txt', **params):

        cmdFileName = _buildCmdFile(dataFileName, **params)
        
        print('Running MCDS :')
        cmd = '"{}" 0, {}'.format(os.path.join(self.mcdsInstPath, self.mcdsExe), cmdFileName)
        print(cmd)
        self.rc = os.system(cmd)
        print('RC=', self.rc)

        return rc

# Génération du fichier de commande
class MCDSAnalysis(object):
    
    EstKeyFns = ['UNIFORM', 'HNORMAL', 'HAZARD'] #, 'NEXPON']
    EstAdjustFns = ['COSINE', 'POLY', 'HERMITE']
    EstCriterions = ['AIC']

    def __init__(self, engine):
        
        self.engine = engine
    
    def __call__(self, estimKeyFn='HNORMAL', estimAdjustFn='COSINE', estimCriterion='AIC', cvInterval=95):
        
        assert estimKeyFn in self.EstKeyFns, \
               'Invalid estimate key function {}: should be in {}'.format(estimKeyFn, self.EstKeyFns)
        assert estimAdjustFn in self.EstAdjustFns, \
               'Invalid estimate adjust function {}: should be in {}'.format(estimAdjustFn, self.EstAdjustFns)
        assert estimCriterion in self.EstCriterions, \
               'Invalid estimate criterion {}: should be in {}'.format(estimCriterion, self.EstCriterions)
        assert cvInterval > 0 and cvInterval < 100,\
               'Invalid cvInterval {}% : should be in {}'.format(cvInterval, ']0%, 100%[')
        
        
        
if __name__ == '__main__':

    print('A coder ...')

    sys.exit(0)
