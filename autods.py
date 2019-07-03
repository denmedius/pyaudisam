# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment


import sys
import os
import numpy as np
import pandas as pd


class DSEngine(object):
    
    DistUnits = ['Meter']
    AreaUnits = ['Hectare']
    
    def __init__(self, exeFileName, workDir='.',
                 distanceUnit='Meter', areaUnit='Hectare', **options):

        # Check base options
        assert distUnit in self.DistUnits, \
               'Invalid distance unit {}: should be in {}'.format(distUnit, self.DistUnits)
        assert areaUnit in self.AreaUnits, \
               'Invalid area unit {}: should be in {}'.format(areaUnit, self.AreaUnits)
        
        # Save base and specific options.
        self.options = dict(distanceUnit=distanceUnit, areaUnit=areaUnit)
        
        # Check and prepare workdir if needed.
        self.workDir = workDir
        if not os.path.isdir(workDir):
            os.makedirs(workDir)
        
        # Detect engine executable installation folder
        self.exeFilePathName = None
        possVers = [7, 6]
        possPaths = [os.path.join('C:/', 'Program files (x86)'),
                     os.path.join('C:/', 'Program files')]
        print('Looking for {} ...'.format(exeFileName))
        for path in possPaths:
            for ver in possVers:
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
            
    def __bool__(self):
        
        return self.exeFilePathName is not None

class MCDSEngine(DSEngine):
    
    DataFields = ['STR_LABEL', 'STR_AREA', 'SMP_LABEL', 'SMP_EFFORT', 'DISTANCE']
    SurveyTypes = ['Point'] #, 'Line'] #TODO : Add Line support
    DistTypes = ['Radial'] #? 'Perpendicular', 'Radial & Angle']
    DataFields = { 'Point' : ['STR_LABEL', 'STR_AREA', 'SMP_LABEL', 'SMP_EFFORT', 'DISTANCE'],
                 } #TODO : Add Line support

    def __init__(self, workDir='.',
                 distanceUnit='Meter', areaUnit='Hectare',
                 surveyType='Point', distanceType='Radial'):
        
        self.cmdFileName   = 'cmd.txt',
        self.outFileName   = 'output.txt',
        self.logFileName   = 'log.txt',
        self.statsFileName = 'stats.txt',
        self.bootFileName  = 'bootstrap.txt')
        
        # Check options
        assert all(field in self.DataFields for field in dataFields),\
               'Invalid data field {}: should be in {}'.format(dataFields, self.DataFields)
        assert surveyType in self.SurveyTypes, \
               'Invalid survey type {} : should be in {}'.format(surveyType, self.SurveyTypes)
        assert distanceType in self.DistTypes, \
               'Invalid area unit{} : should be in {}'.format(distanceType, self.DistTypes)
        
        # Initialise base.
        super().__init__(exeFileName='MCDS.exe', workDir=workDir, 
                         distanceUnit=distanceUnit, areaUnit=areaUnit,
                         **dict(surveyType=surveyType, distanceType=distanceType,
                                dataFields=DataFields[surveyType]))
        
    def _buildCmdFile(self, dataFileName='data.txt', **params):

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
    def __call__(self, dataSet, runName='mcds', **analysisParms):
        
        self.lastDataDir = tempfile.mkdtemp(dir=self.workdir, prefix=runName+'-')
        
        dataFileName = os.path.join(self.lastDataDir, 'data.txt')
        
        dataSet.export2MCDS(dataFileName)

        cmdFileName = _buildCmdFile(dataFileName, **analysisParms)
        
        cmd = '"{}" 0, {}'.format(self.exeFilePathName, cmdFileName)
        print('Running MCDS :', cmd)
        self.lastExeRC = os.system(cmd)
        print('Done : RC=', self.lastExeRC)

        return self.lastExeRC, self.lastDataDir

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

    def __init__(self, engine, dataSet, namePrefix='mcsd',
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
    
    def run(self):
        
        rc, dataDir = engine(self.dataSet, runName=self.name,
                             estimKeyFn=self.estimKeyFn, estimAdjustFn=self.estimAdjustFn,
                             estimCriterion=self.estimCriterion, cvInterval=self.cvInterval)
        
        # analyser log et fic resultats
        TODO
        
        return sResults (dont id unique run (dataDir ?))
    
# A data set for multiple analyses, with 1 or 0 individual per line
# Warning: Only Point transect supported as for now
class DataSet(object):
    
    ExportCols = ['Region', 'Surface', 'Point', 'Effort', 'Distance']
    
    def __init__(self, dfData):
        
        assert all(col in self.ExportCols for col in dfData.columns), \
               'Some columns from \{{}\} are missing'.format(','.join(self.ExportCols))
        assert not dfData.empty, 'Not data in set'
        
        self.dfData = dfData
        
    def export2MCDS(self, tgtFileName):
        
        #TODO
        pass
        
    def export2Distance(self, tgtFileName, distPrec=1):
        
        dfExport = self.dfData[self.ExportCols].copy()

        # Workaround df.to_csv(float_format='%.xf') not working when NaNs in serie
        def safeFloat2Str(val, prec=1, decPt='.'):
            strVal = '' if np.isnan(val) else '{:.{prec}f}'.format(val, prec=prec)
            if decPt != '.':
                strVal = strVal.replace('.', decPt)
            return strVal
        
        dfExport.Distance = dfExport.Distance.apply(safeFloat2Str, prec=distPrec, decPt=',')
        dfExport.Effort = dfExport.Effort.apply(safeFloat2Str, prec=0)

        # Mise en forme finale
        dfExport.sort_values(by=['Point'], inplace=True)

        # Export final.
        dfExport.to_csv(tgtFileName, index=False, sep='\t', encoding='utf-8',
                        header=['Region*Label', 'Region*Area', 'Point transect*Label',
                                'Point transect*Survey effort', 'Observation*Radial distance'])
        
        
# Exemple avec ADCD 2019 Papier
def extraireJeuDonnees(dfTout, espece, passages=['A', 'B'], duree='10mn'): # Suppose uniquement des mâles
    
    assert all(p in ['A', 'B'] for p in passages)
    assert duree in ['5mn', '10mn']
    assert espece in dfTout.ESPECE.unique()
    
    # Passages
    dfJeu = dfTout[(dfTout.ESPECE == espece) & (dfTout.PASSAGE.isin(passages))].copy()
    
    # Durée
    if duree == '10mn':
        dfJeu['NOMBRE'] = dfJeu[['PER5MN', 'PER10MN']].sum(axis='columns')
    else:
        dfJeu['NOMBRE'] = dfJeu['PER5MN']
    dfJeu.drop(dfJeu[dfJeu.NOMBRE.isnull()].index, inplace=True)
    assert all(dfJeu.NOMBRE == 1)
        
    # Effort
    dfJeu['EFFORT'] = len(passages)
        
    # Nettoyage
    dfJeu.drop(['PER5MN', 'PER10MN'], axis='columns', inplace=True)
    
    return dfJeu
            
def ajouterAbsences(dfJeu, effort, pointsPapier):
    
    assert not dfJeu.empty, 'Erreur : Il n\'y aurait que des absences !'

    zone, surface, espece = dfJeu.iloc[0][['ZONE', 'HA', 'ESPECE']]
    dAbsence = { 'ZONE': zone, 'HA': surface, 'POINT': None, 'ESPECE': espece,
                 'DISTANCE': np.nan, 'EFFORT': effort, 'MALE': None,
                 'NOMBRE': np.nan, 'DATE': pd.NaT, 'OBSERVATEUR': None, 'PASSAGE': None }

    pointsManquants = [p for p in pointsPapier if p not in dfJeu.POINT.unique()]
    for p in pointsManquants:
        dAbsence.update(POINT=p)
        dfJeu = dfJeu.append(dAbsence, ignore_index=True)
    
    return dfJeu, len(pointsManquants)

if __name__ == '__main__':

    print('A finir de coder ...')
    
    # Paramètres
    workDir = 'ACDC-Auto'
            
    # Exemple avec ADCD 2019 Papier
    pointsPapier = \
        list(map(int, """23,39,40,41,42,55,56,57,58,59,60,72,73,74,75,76,88,89,90,91,
                         105,106,109,110,112,113,122,123,125,126,127,128,129,130,141,142,143,144,145,146,
                         147,148,157,158,159,160,161,162,163,164,165,166,174,175,176,177,178,179,180,181,
                         182,183,184,185,192,193,194,195,196,197,198,199,200,201,202,210,211,212,213,214,
                         215,216,218,219,228,229,232,233,245,246,247,250,262,263,265,266,280,281,282,283,
                         284,299,300,301""".split(',')))

    dfMales = pd.read_excel('ACDC/ACDC2019-Papyrus-DonneesBrutes.xlsx', sheet_name='ResultIndivMales')
    dfMales.rename(columns={ 'ha': 'HA', 'Distance en m': 'DISTANCE', 'Mâle\xa0?': 'MALE', 'Date': 'DATE',
                             'Période': 'PASSAGE', '0-5mn': 'PER5MN', '5-10 mn': 'PER10MN' }, inplace=True)

    assert all(dfMales.MALE.str.lower() == 'oui')

    print('Nb mâles   :', len(dfMales))
    print('Nb espèces :', len(dfMales.ESPECE.unique()))

    dfToDo = pd.read_excel('ACDC/ACDC2019-Papyrus-DonneesBrutes.xlsx', sheet_name='AFaire')
    toDoCols = ['ESPECE', 'MALES', 'PERIODE']
    assert all(col in toDoCols for col in dfToDo.columns)
    dfToDo = dfToDo.reindex(toDoCols, axis='columns')
    dfToDo.sort_values(by='MALES', ascending=False, inplace=True)
            
    dfParams = pd.read_excel('ACDC/ACDC2019-Papyrus-DonneesBrutes.xlsx', sheet_name='ParamsAnalyses')
    paramCols = ['KeyFn', 'AdjustFn', 'Criterion', 'CVInterval']
    assert all(col in paramCols for col in dfParams.columns)
    dfParams = dfParams.reindex(paramCols, axis='columns')

    mcds = MCDSEngine(workDir=workDir,
                      distanceUnit='Meter', areaUnit='Hectare',
                      surveyType='Point', distanceType='Radial')
            
    dfAnalyses = pd.DataFrame() # Analyses : nom, params et resultats
    for index, sToDo in dfToDo.iterrows():

        espece, nbIndivs, passage = sToDo
        passages = [p for p in passage]

        print(espece, ':', passage)

        for duree in ['5mn', '10mn']:

            print('-', duree, end=' : ')

            dfJeu = extraireJeuDonnees(dfMales, espece, passages, duree)

            print(len(dfJeu), 'mâles', end=', ')

            dfJeu, nAbsences = ajouterAbsences(dfJeu, effort=len(passages), pointsPapier=pointsPapier)

            print(nAbsences, 'absences', end=' ')

            jeu = DataSet(dfJeu)
            
            for index, sParams in dfParams.iterrows():
                analyse = MCDSAnalysis(engine=mcds, dataSet=jeu, 
                                       namePrefix='{}-{}-{}'.format(espece, duree, passage),
                                       **dict(estimKeyFn=sParams['KeyFn'], estimAdjustFn=sParams['AdjustFn'],
                                              estimCriterion=sParams['Criterion'], cvInterval=sParams['CVInterval'])
                sAnalyse = analyse.run()
                sAnalyse['id'] = (espece, passage, duree, index)
                for name, value in sParams.iteritems():
                    sAnalyse[name] = value
                dfAnalyse = dfAnalyses.append(sAnalyse, ignoreIndex=True)
    
    dfAnalyses.to_excel('ACDC/ACDC2019-Papyrus-ResultatsAnalyses.xlsx')
    
    sys.exit(0)
