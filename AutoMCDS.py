# coding: utf-8

import sys
import os
import numpy as np
import pandas as pd


# Détection du dossier d'install. de Distance
mcdsExe = 'MCDS.exe'
possVers = [7, 6]
possPaths = [os.path.join('C:\\', 'Program files (x86)'),
             os.path.join('C:', 'Program files')]
print('Recherche {} ...'.format(mcdsExe))
for path in possPaths:
    for ver in possVers:
        mcdsInstPath = os.path.join(path, 'Distance ' + str(ver))
        print(' - essai dans {} : '.format(mcdsInstPath), end='')
        if not os.path.exists(os.path.join(mcdsInstPath, mcdsExe)):
            mcdsInstPath = None
            print('non.')
        else:
            print('eureka !')
            break
    if mcdsInstPath:
        break

if mcdsInstPath:
    print('{} trouvé dans {}'.format(mcdsExe, mcdsInstPath))
else:
    print('Erreur : Impossible de trouver {}'.format(mcdsExe))


# Run MCDS
def runMCDSEngine(cmdFileName):

    print('Running MCDS :')
    cmd = '"{}" 0, {}'.format(os.path.join(mcdsInstPath, mcdsExe), cmdFileName)
    print(cmd)
    rc = os.system(cmd)
    print('RC=', rc)

    return rc

# Génération du fichier de commande
def buildInputFile(dataFileName='mcds-data.txt'):

    cmdFileName = 'mcds-cmd.txt'

    outFileName = 'mcds-output.txt'
    logFileName = 'mcds.log'
    statsFileName = 'mcds-stats.txt'
    bootFileName = 'mcds-bootstrap.txt'

    KeyFns = ['UNIFORM', 'HNORMAL', 'HAZARD'] #, 'NEXPON']
    AdjustFns = ['COSINE', 'POLY', 'HERMITE']

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
    Confidence={optCVInterv};
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

    """.format(output=outFileName, log=logFileName,
               stats=statsFileName, bootstrap=bootFileName,
               optSurvType='Point', optDistType='Radial',
               optDistUnit='Meter', optAreaUnit='Hectare', optCVInterv=95,
               dataFields='STR_LABEL, STR_AREA, SMP_LABEL, SMP_EFFORT, DISTANCE',
               dataFileName=dataFileName,
               estKeyFn='HNORMAL', estAdjustFn='COSINE', estCriterion='AIC')

    with open(cmdFileName, 'w') as cmdFile:
        cmdFile.write(cmdTxt)

    return cmdFileName

if __name__ == '__main__':

    print('A coder ...')

    sys.exit(0)
