# coding: utf-8

# ACDC : l'Autre Carré D'à Côté (Cournols - Olloix - Montaigut-le-Blanc - Ludesse)
#        alias le plateau de JPD
#
#  Outils de manipulation des données pour le Distance Sampling (basé sur autods)
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

import numpy as np
import pandas as pd

#import autods as ads

# Suppose uniquement des mâles
def extraireJeuDonnees(dfTout, espece, passages=['A', 'B'], duree='10mn'):
    
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

#
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