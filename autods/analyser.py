# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Analyser: Run a bunch of DS analyses according to a user-friendly set of analysis specs
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment


import sys

from collections import OrderedDict as odict

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger('autods')

# An explicit tabular set of analysis specs (1 analysis per row, all explicited parameters)
class AnalysisSpecSet(object):

    pass # Soon available, through functions just below :-)
    
# Generation of a table of implicit "partial variant" specification,
# from a list of possible data selection criteria for each variable.
# "Partial variant" because its only about a sub-set of variants
# * dVariants : { target columns name: list of possibles criteria for data selection }
def implicitPartialVariantSpecs(dVariants):
    
    def fixedLengthList(toComplete, length):
        return toComplete + [np.nan]*(length - len(toComplete))

    nRows = max(len(l) for l in dVariants.values())

    return pd.DataFrame({ colName : fixedLengthList(variants, nRows) for colName, variants in dVariants.items() })

# Generation of a table of explicit "partial variant" specifications, from an implicit one
# (= generate all combinations of variants)
def explicitPartialVariantSpecs(dfImplSpecs):
    
    dfExplSpecs = dfImplSpecs[dfImplSpecs.columns[:1]].dropna()
    
    # For each implicit specs column (but the first)
    for col in dfImplSpecs.columns[1:]:
        
        # Get variants
        sVariants = dfImplSpecs[col].dropna()
        
        # Duplicate current explicit table as much as variants are many
        dfExplSpecs = dfExplSpecs.loc[np.repeat(dfExplSpecs.index.to_numpy(), len(sVariants))]
        
        # Add the new columns by tiling the variants along the whole index range
        dfExplSpecs[col] = np.tile(sVariants.to_numpy(), len(dfExplSpecs) // len(sVariants))
        
        # Reset index for easy next duplication
        dfExplSpecs.reset_index(inplace=True, drop=True)

    # Done.
    return dfExplSpecs

# Generation of a table of explicit variant specifications,
# from a set of implicit and explicit partial variant specs tables
# * oddfPartSpecs : the odict of name => partial specs table
#   Warning: implicit tables are only found by their name containing "_impl"
def explicitVariantSpecs(oddfPartSpecs):
    
    assert len(oddfPartSpecs.keys()) > 0, "Error: Can't explicit variants with no partial variant"
    
    # Group partial specs tables with same column sets (according to column names)
    odSameColsPsNames = odict() # { sorted(cols): [table names] }
    
    for psName, dfPsValues in oddfPartSpecs.items():
        
        colSetId = ':'.join(sorted(dfPsValues.columns))
        if colSetId not in odSameColsPsNames:
            odSameColsPsNames[colSetId] = list()
            
        odSameColsPsNames[colSetId].append(psName)

    # For each group, concat. tables into one, after expliciting if needed
    ldfExplPartSpecs = list()

    for lPsNames in odSameColsPsNames.values():

        ldfSameColsPartSpecs= list()
        for psName in lPsNames:

            dfPartSpecs = oddfPartSpecs[psName]

            # Implicit specs case:
            if '_impl' in psName:

                dfPartSpecs = explicitPartialVariantSpecs(dfPartSpecs)

            # Now, specs are explicit.
            ldfSameColsPartSpecs.append(dfPartSpecs)

        # Concat groups of same columns set explicit specs
        ldfExplPartSpecs.append(pd.concat(ldfSameColsPartSpecs, ignore_index=True))
    
    # Combinaison des specs explicites (dans l'ordre)
    dfExplSpecs = ldfExplPartSpecs[0]

    for dfExplPartSpecs in ldfExplPartSpecs[1:]:

        commCols = [col for col in dfExplSpecs.columns if col in dfExplPartSpecs.columns]

        if commCols: # Any column in common : left join each left row to each matching right row

            dfExplSpecs = dfExplSpecs.join(dfExplPartSpecs.set_index(commCols), on=commCols)

        else: # No columns in common : combine each left row with all right rows

            nInitSpecs = len(dfExplSpecs)

            dfExplSpecs = dfExplSpecs.loc[np.repeat(dfExplSpecs.index, len(dfExplPartSpecs))]
            dfExplSpecs.reset_index(drop=True, inplace=True)

            dfExplPartSpecs = pd.DataFrame(data=np.tile(dfExplPartSpecs, [nInitSpecs, 1]), columns=dfExplPartSpecs.columns)

            dfExplSpecs = pd.concat([dfExplSpecs, dfExplPartSpecs], axis='columns')

        dfExplSpecs.reset_index(drop=True, inplace=True)
    
    # Done.
    return dfExplSpecs

# Analyser: Run a bunch of DS analyses on samples extracted from a sightings data set,
#           according to a user-friendly set of analysis specs
class Analyser(object):

    def __init__(self, rawDataSet, dfSampleSet, rawAnlysSpecs):

        pass # Soon available ...
    
    def __call_(self, ):
    
        pass # Soon available ...


if __name__ == '__main__':

    sys.exit(0)
