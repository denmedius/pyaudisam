# coding: utf-8

"""ACDC 2019 Study: Parameters for Distance Sampling analyses with pyaudisam
("Naturalist" field data subset)

See howto.md for what it is and how to use it.

Jean-Philippe MEURET <fefeqe22.vucuqu82 (at) murena.io>
"""

import sys

import pathlib as pl
import datetime as dt

import pandas as pd

from pyaudisam.optanalyser import MCDSTruncOptanalysisResultsSet as rs


# Field data ####################################################################

# Root folder for linked files below
_dataDir = pl.Path('.')

# Base columns.
speciesCol = 'Espèce'
passIdCol = 'Passage'
distanceCol = 'Distance'

effortCol = 'Effort'

# Geographical Id columns for transects (whatever on-field visit, aka pass)
transectPlaceCols = ['Point']

# Field sighting and point transect description file.
studyName = 'ACDC2019'
subStudyName = '-Nat'

surveyDataFile = _dataDir / f'{studyName}{subStudyName}-ObsIndivDist.xlsx'
indivDistDataSheet = 'Donnees'
transectsDataSheet = 'Inventaires'


# General Distance Sampling parameters ##################################################
distanceUnit = 'Meter'
areaUnit = 'Sq. Kilometer'
surveyType = 'Point'
distanceType = 'Radial'

clustering = False

passEffort = 1  # Constant effort = 1 per pass on each point transect.

studyAreaSpecs = dict(Zone='ACDC', Surface=24)  # km2


# Parameters for pre-analyses ##########################################################
# a. Selection / identification of samples.
sampleSelCols = [speciesCol, passIdCol, 'Adulte', 'Durée']
sampleIndCol = 'Echant'  # Unique Id integer
sampleAbbrevCol = 'Abrev. Echant'  # Sample abbreviation (human-readable)
sampleSpecCustCols = []

# b. Sample abbreviation (human-readable) generation, from sample attributes 
def sampleAbbrev(sSamp):
    abbrvs = [''.join(word[:4].title() for word in sSamp[speciesCol].split(' ')[:2])]
    if passIdCol in sSamp.index and not pd.isnull(sSamp[passIdCol]) and sSamp[passIdCol]:
        abbrvs.append(sSamp[passIdCol].replace('+', ''))
    if 'Durée' in sSamp.index:
        abbrvs.append(sSamp['Durée'].replace('+', ''))
    if 'Adulte' in sSamp.index:
        abbrvs.append(sSamp.Adulte.replace('+', ''))
    return '-'.join(abbrvs)

# c. Samples to be analysed
sampleSpecFile = _dataDir / f'ACDC2019-Samples.xlsx'

# d. Pre-analysis specific parameters
preResultsHeadCols = dict(before=[sampleIndCol],
                          sample=sampleSelCols,
                          after=[sampleAbbrevCol] + sampleSpecCustCols)

modelPreStrategy = \
    [dict(keyFn=kf, adjSr=js, estCrit='AIC', cvInt=95)
     for js in ['COSINE', 'POLY']  # , 'HERMITE'] # Hermite : not worth the computation cost
     for kf in ['HNORMAL', 'UNIFORM', 'HAZARD']]  # , 'NEXPON']] # Avoid, because g'(0) << 0 !!!

# e. Parameters for computations run and follow-up
runPreAnalysisMethod = 'subprocess.run'
runPreAnalysisTimeOut = 300
logPreAnalysisData = False
logPreAnalysisProgressEvery = 5


# Parameters for analyse ###############################################################
# a. Analysis identification
analysisIndCol = 'Analyse'  # Unique integer
analysisAbbrevCol = 'Abrev. Analyse' # Analysis abbreviation (human-readable)

# b. Analysis abbreviation (human-readable) generation, from main analysis attributes 
def analysisAbbrev(sAnlys):
    abbrevs = [sampleAbbrev(sAnlys)]
    abbrevs += [sAnlys['FonctionClé'][:3].lower(), sAnlys['SérieAjust'][:3].lower()]
    dTroncAbbrv = {'l': 'TrGche' if 'TrGche' in sAnlys.index else 'TroncGche',
                   'r': 'TrDrte' if 'TrDrte' in sAnlys.index else 'TroncDrte',
                   'm': 'NbTrModel' if 'NbTrModel' in sAnlys.index else 'NbTrchMod',
                   'd': 'NbTrDiscr'}
    for abbrev, name in dTroncAbbrv.items():
        if name in sAnlys.index and not pd.isnull(sAnlys[name]):
            abbrevs.append('{}{}'.format(abbrev, sAnlys[name][0].lower() if isinstance(sAnlys[name], str)
                                                                         else int(sAnlys[name])))
    return '-'.join(abbrevs)

# c. Analysis-specific parameters
analysisSpecCustCols = []

analysisParamCols = ['FonctionClé', 'SérieAjust', 'TrGche', 'TrDrte', 'NbTrchMod']

resultsHeadCols = dict(before=[analysisIndCol, sampleIndCol],
                       sample=sampleSelCols,
                       after=analysisParamCols + [analysisAbbrevCol] + analysisSpecCustCols)

defEstimKeyFn = 'HNORMAL'
defEstimAdjustFn = 'COSINE'
defEstimCriterion = 'AIC'
defCVInterval = 95

defMinDist = None
defMaxDist = None,
defFitDistCuts = None
defDiscrDistCuts = None

ldTruncIntrvSpecs = [dict(col='left', minDist=5.0, maxLen=5.0),
                     dict(col='right', minDist=25.0, maxLen=25.0)]
truncIntrvEpsilon = 1e-6

# d. Analyses to run : auto-extraction from opt-analyses spec. file ...
#    (by simply removing 'auto' truncation parameters)
analysisSpecFile = _dataDir / 'ACDC2019-AnalysesToDo.autogen.xlsx'

_ddfAnlysSpecs = pd.read_excel(_dataDir / f'ACDC2019-OptAnalysesToDo.xlsx', sheet_name=None)
with pd.ExcelWriter(analysisSpecFile) as xlWrtr:
    for sn in _ddfAnlysSpecs:
        if sn != 'TroncaturesAuto_impl':
            _ddfAnlysSpecs[sn].to_excel(xlWrtr, sheet_name=sn, index=False)

# e. Parameters for computations run and follow-up
runAnalysisMethod = 'subprocess.run'
runAnalysisTimeOut = 300
logAnalysisData = False
logAnalysisProgressEvery = 5


# Parameters for opt-analyses ###########################################################
# a. Specific parameters
optAnalysisSpecCustCols = []

optAnalysisParamCols = ['FonctionClé', 'SérieAjust', 'TrGche', 'TrDrte', 'NbTrchMod']

optResultsHeadCols = dict(before=[analysisIndCol, sampleIndCol],
                          sample=sampleSelCols,
                          after=optAnalysisParamCols + [analysisAbbrevCol] + optAnalysisSpecCustCols)

defExpr2Optimise = 'balq3'
defMinimiseExpr = False
defOutliersMethod = 'tucquant'
defOutliersQuantCutPct = 5
defFitDistCutsFctr = dict(min=2/3, max=3/2)
defDiscrDistCutsFctr = dict(min=1/3, max=1)

defSubmitTimes = 1
defSubmitOnlyBest = 1

dDefSubmitOtherParams = dict()

defCoreEngine = 'zoopt'
defCoreMaxIters = 150
defCoreTermExprValue = None
defCoreAlgorithm = 'racos'
defCoreMaxRetries = 0

# b. Opt-analyses to run
optAnalysisSpecFile = _dataDir / f'ACDC2019-OptAnalysesToDo.xlsx'

# c. Parameters for computations run and follow-up
runOptAnalysisMethod = 'subprocess.run'
runOptAnalysisTimeOut = 300
logOptAnalysisData = False
logOptAnalysisProgressEvery = 50
logOptimisationProgressEvery = 5
backupOptimisationsEvery = 50


# Report parameters (all types of analysis) #############################################
studyLang = 'en'

reportStudyTitle = f'ACDC 2019 Naturalist'
reportStudyDescr = "Estimation of common breeding bird populations on the Cournols - Olloix plateau -" \
                   " in 2019 though ~100 distance sampling point transects of 5 and 10mn (2 seasonal passes)" \
                   " ; using the Naturalist smartphone app. for field surveys"
reportStudyKeywords = 'ACDC, Cournols, Olloix, 2019, Distance Sampling, Naturalist, smartphone'

# Pre-analysis report ###############################################################
preReportStudyTitle = reportStudyTitle
preReportStudyDescr = reportStudyDescr
preReportStudyKeywords = reportStudyKeywords
preReportStudySubTitle = 'Pre-analysis report (30 most numerous species)'
preReportAnlysSubTitle = 'Détails of pre-analyses'

preReportPlotParams = dict(plotImgSize=(640, 400), superSynthPlotsHeight=288,
                           plotImgFormat='png', plotImgQuality=90,
                           plotLineWidth=1, plotDotWidth=4,
                           plotFontSizes=dict(title=11, axes=10, ticks=9, legend=10))

# Select columns for the various report tables
# a. Main HTML page (super-synthesis) : Column #1 (top) = sample description
preReportSampleCols = [('header (head)', sampleIndCol, 'Value')] \
                      + [('header (sample)', col, 'Value') for col in sampleSelCols]

# b. Main HTML page (super-synthesis) : Column #1 (bottom) = analysis model
preReportParamCols = [rs.CLParEstKeyFn, rs.CLParEstAdjSer]

# c. Main HTML page /super-synthesis :
#    Columns #2 & #3 = results (right before columns #4, #5, and #6 with DS plots)
preReportResultCols = [rs.CLRunStatus,
                       rs.CLNObs, rs.CLMaxObsDist, rs.CLEffort,
                       rs.CLAic, rs.CLChi2, rs.CLKS, rs.CLDCv,
                       rs.CLPDetec, rs.CLEswEdr,
                       rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax,
                       rs.CLNumber, rs.CLNumberMin, rs.CLNumberMax]

# d. Detail HTML pages and Excel synthesis sheet : synthesis table.
preReportSynthCols = preReportSampleCols + preReportParamCols \
                     + [rs.CLRunStatus,
                        rs.CLNObs, rs.CLMaxObsDist, rs.CLEffort,
                        rs.CLAic, rs.CLChi2, rs.CLKS, rs.CLDCv,
                        rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax,
                        rs.CLNumber, rs.CLNumberMin, rs.CLNumberMax,
                        rs.CLEswEdr, rs.CLEswEdrMin, rs.CLEswEdrMax,
                        rs.CLPDetec, rs.CLPDetecMin, rs.CLPDetecMax,
                        rs.CLRunFolder]

# e. Super-synthesis table, synthesis and details, for HTML or Excel : sorting parameters.
preReportSortCols = [('header (head)', sampleIndCol, 'Value')]
preReportSortAscend = True

preReportRebuild = False


# (Opt-)analysis 'full' report #######################################################
# 1. Analysis report specifics
anlysFullReportStudyTitle = reportStudyTitle
anlysFullReportStudySubTitle = 'Rapport d\'analyse (30 espèces les plus nombreuses)'
anlysFullReportAnlysSubTitle = 'Détail des analyses'
anlysFullReportStudyDescr = reportStudyDescr
anlysFullReportStudyKeywords = reportStudyKeywords + ', full'

# 2. Opt-analysis report specifics
optAnlysFullReportStudyTitle = anlysFullReportStudyTitle
optAnlysFullReportStudySubTitle = anlysFullReportStudySubTitle
optAnlysFullReportAnlysSubTitle = anlysFullReportAnlysSubTitle
optAnlysFullReportStudyDescr = anlysFullReportStudyDescr
optAnlysFullReportStudyKeywords = anlysFullReportStudyKeywords

# 3. Common to analysis and opt-analysis reports : plotting parameters
fullReportPlotParams = dict(plotImgSize=(640, 400), superSynthPlotsHeight=288,
                            plotImgFormat='png', plotImgQuality=90,
                            plotLineWidth=1, plotDotWidth=4,
                            plotFontSizes=dict(title=11, axes=10, ticks=9, legend=10))

# Select columns for the various report tables
# a. Main HTML page (super-synthesis) : Column #1 (top) = sample description
fullReportSampleCols = [('header (head)', sampleIndCol, 'Value')] \
                       + [('header (sample)', col, 'Value') for col in sampleSelCols] \
                       + [rs.CLNTotObs, rs.CLMaxObsDist]

# b. Main HTML page (super-synthesis) : Column #1 (bottom) = analysis model and truncation parameters
fullReportParamCols = [rs.CLParEstKeyFn, rs.CLParEstAdjSer,
                       rs.CLParTruncLeft, rs.CLParTruncRight, rs.CLParModFitDistCuts]

# c. Main HTML page / super-synthesis :
#    Columns #2 & #3 = results (right before columns #4, #5, and #6 with DS plots)
fullReportResultCols = [
    ('header (head)', analysisIndCol, 'Value'),
    rs.CLRunStatus,
    rs.CLEffort, rs.CLNObs, rs.CLSightRate, rs.CLNAdjPars,
    rs.CLAic, rs.CLChi2, rs.CLKS, rs.CLDCv,
    rs.CLCmbQuaBal3, rs.CLCmbQuaBal2, rs.CLCmbQuaBal1,
    rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax,
    rs.CLNumber, rs.CLNumberMin, rs.CLNumberMax,
    rs.CLEswEdr, rs.CLPDetec
]

# d. Main and detail HTML pages and Excel synthesis sheet : synthesis table.
fullReportSynthCols = \
    fullReportSampleCols + [('header (head)', analysisIndCol, 'Value')] + fullReportParamCols \
    + [rs.CLRunStatus,
       rs.CLEffort, rs.CLNObs, rs.CLSightRate, rs.CLNAdjPars,
       rs.CLDeltaAic, rs.CLChi2, rs.CLKS, rs.CLCvMUw, rs.CLCvMCw, rs.CLDCv,
       rs.CLCmbQuaBal3, rs.CLCmbQuaBal2, rs.CLCmbQuaBal1,
       rs.CLCmbQuaChi2, rs.CLCmbQuaKS, rs.CLCmbQuaDCv,
       rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax,
       rs.CLNumber, rs.CLNumberMin, rs.CLNumberMax,
       rs.CLEswEdr, rs.CLEswEdrMin, rs.CLEswEdrMax,
       rs.CLPDetec, rs.CLPDetecMin, rs.CLPDetecMax,
       rs.CLRunFolder]

# e. Super-synthesis table, synthesis and details, for HTML or Excel : sorting parameters.
fullReportSortCols = \
    [('header (head)', sampleIndCol, 'Value')] \
    + [rs.CLParTruncLeft, rs.CLParTruncRight,
       rs.CLDeltaAic, rs.CLCmbQuaBal3]
fullReportSortAscend = [True, True, True, True, False]

fullReportRebuild = False


# (Opt-)Analysis 'auto-filtered' reports d'(opt-)analyse 'auto-filtrés' ##########################
# 1. Analysis report specifics
anlysFilsorReportStudyTitle = reportStudyTitle
anlysFilsorReportStudySubTitle = 'Auto-filtered report: "{fsId}" method (30 most numerous species)'
anlysFilsorReportAnlysSubTitle = 'Analysis details'
anlysFilsorReportStudyDescr = reportStudyDescr
anlysFilsorReportStudyKeywords = reportStudyKeywords + ', auto-filter'

# 2. Opt-analysis report specifics
optAnlysFilsorReportStudyTitle = anlysFilsorReportStudyTitle
optAnlysFilsorReportStudySubTitle = anlysFilsorReportStudySubTitle
optAnlysFilsorReportAnlysSubTitle = anlysFilsorReportAnlysSubTitle
optAnlysFilsorReportStudyDescr = anlysFilsorReportStudyDescr
optAnlysFilsorReportStudyKeywords = anlysFilsorReportStudyKeywords

# 3. Common to analysis and opt-analysis reports
# a. Plotting parameters
filsorReportPlotParams = dict(plotImgSize=(640, 400), superSynthPlotsHeight=288,
                              plotImgFormat='png', plotImgQuality=90,
                              plotLineWidth=1, plotDotWidth=4,
                              plotFontSizes=dict(title=11, axes=10, ticks=9, legend=10))

# b. Available filter & sort schemes
_whichBestQua = [rs.CLGrpOrdClTrChi2KSDCv, rs.CLGrpOrdClTrDCv, rs.CLCmbQuaBal3, rs.CLCmbQuaBal2, rs.CLCmbQuaBal1,
                 rs.CLGrpOrdClTrQuaChi2, rs.CLGrpOrdClTrQuaKS, rs.CLGrpOrdClTrQuaDCv]

_dupSubset = [rs.CLNObs, rs.CLEffort, rs.CLDeltaAic, rs.CLChi2, rs.CLKS, rs.CLCvMUw, rs.CLCvMCw, rs.CLDCv,
              rs.CLPDetec, rs.CLPDetecMin, rs.CLPDetecMax, rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax]
_dDupRounds = {rs.CLDeltaAic: 1, rs.CLChi2: 2, rs.CLKS: 2, rs.CLCvMUw: 2, rs.CLCvMCw: 2, rs.CLDCv: 2,
               rs.CLPDetec: 3, rs.CLPDetecMin: 3, rs.CLPDetecMax: 3, rs.CLDensity: 2, rs.CLDensityMin: 2,
               rs.CLDensityMax: 2}

_whichFinalQua = rs.CLCmbQuaBal3
_ascFinalQua = False

filsorReportSchemes = [dict(method=rs.filterSortOnExCAicMulQua,
                            deduplicate=dict(dupSubset=_dupSubset, dDupRounds=_dDupRounds),
                            filterSort=dict(sightRate=97.5, nBestAIC=2, nBestQua=1, whichBestQua=_whichBestQua,
                                            nFinalRes=8, whichFinalQua=_whichFinalQua, ascFinalQua=_ascFinalQua),
                            preselCols=[rs.CLCmbQuaBal3, rs.CLCmbQuaBal2, rs.CLCmbQuaBal1],
                            preselAscs=False, preselThrhs=0.4, preselNum=3),
                       dict(method=rs.filterSortOnExCAicMulQua,
                            deduplicate=dict(dupSubset=_dupSubset, dDupRounds=_dDupRounds),
                            filterSort=dict(sightRate=95, nBestAIC=2, nBestQua=1, whichBestQua=_whichBestQua,
                                            nFinalRes=10, whichFinalQua=_whichFinalQua, ascFinalQua=_ascFinalQua),
                            preselCols=[rs.CLCmbQuaBal3, rs.CLCmbQuaBal2, rs.CLCmbQuaBal1],
                            preselAscs=False, preselThrhs=0.3, preselNum=4),
                       dict(method=rs.filterSortOnExCAicMulQua,
                            deduplicate=dict(dupSubset=_dupSubset, dDupRounds=_dDupRounds),
                            filterSort=dict(sightRate=92.5, nBestAIC=3, nBestQua=1, whichBestQua=_whichBestQua,
                                            nFinalRes=12, whichFinalQua=_whichFinalQua, ascFinalQua=_ascFinalQua),
                            preselCols=[rs.CLCmbQuaBal3, rs.CLCmbQuaBal2, rs.CLCmbQuaBal1],
                            preselAscs=False, preselThrhs=0.2, preselNum=5),
                       dict(method=rs.filterSortOnExecCode,
                            deduplicate=dict(dupSubset=_dupSubset, dDupRounds=_dDupRounds),
                            filterSort=dict(whichFinalQua=_whichFinalQua, ascFinalQua=_ascFinalQua),
                            preselCols=[rs.CLCmbQuaBal3, rs.CLCmbQuaBal2, rs.CLCmbQuaBal1],
                            preselAscs=False, preselThrhs=0.1, preselNum=10)]

# Select columns for the various report tables
# a. Main HTML page (super-synthesis) : Column #1 (top) = sample description
filsorReportSampleCols = [('header (head)', sampleIndCol, 'Value')] \
                         + [('header (sample)', col, 'Value') for col in sampleSelCols] \
                         + [rs.CLNTotObs, rs.CLMaxObsDist]

# b. Main HTML page (super-synthesis) : Column #1 (bottom) = analysis model and truncation parameters
filsorReportParamCols = [rs.CLParEstKeyFn, rs.CLParEstAdjSer,
                         rs.CLParTruncLeft, rs.CLParTruncRight, rs.CLParModFitDistCuts]

# c. Main HTML page / super-synthesis :
#    Columns #2 & #3 = results (right before columns #4, #5, and #6 with DS plots)
filsorReportResultCols = [('header (head)', analysisIndCol, 'Value'),
                          rs.CLRunStatus,
                          rs.CLEffort, rs.CLNObs, rs.CLSightRate, rs.CLNAdjPars,
                          rs.CLAic, rs.CLChi2, rs.CLKS, rs.CLDCv,
                          rs.CLCmbQuaBal3, rs.CLCmbQuaBal2, rs.CLCmbQuaBal1,
                          rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax,
                          rs.CLNumber, rs.CLNumberMin, rs.CLNumberMax,
                          rs.CLEswEdr, rs.CLPDetec]

# d. Main and detail HTML pages and Excel synthesis sheet : auto-filtered and synthesis tables.
filsorReportSynthCols = \
    filsorReportSampleCols + [('header (head)', analysisIndCol, 'Value')] + filsorReportParamCols \
    + [rs.CLRunStatus,
       rs.CLEffort, rs.CLNObs, rs.CLSightRate, rs.CLNAdjPars,
       rs.CLDeltaAic, rs.CLChi2, rs.CLKS, rs.CLCvMUw, rs.CLCvMCw, rs.CLDCv,
       rs.CLCmbQuaBal3, rs.CLCmbQuaBal2, rs.CLCmbQuaBal1,
       rs.CLCmbQuaChi2, rs.CLCmbQuaKS, rs.CLCmbQuaDCv,
       rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax,
       rs.CLNumber, rs.CLNumberMin, rs.CLNumberMax,
       rs.CLEswEdr, rs.CLEswEdrMin, rs.CLEswEdrMax,
       rs.CLPDetec, rs.CLPDetecMin, rs.CLPDetecMax,
       # rs.CLGroupTruncLeft, rs.CLGroupTruncRight,
       # rs.CLGrpOrdSmTrAic,
       # rs.CLGrpOrdClTrChi2KSDCv,  # rs.CLGrpOrdClTrChi2,
       # rs.CLGrpOrdClTrDCv,
       # rs.CLGrpOrdClTrQuaBal1, rs.CLGrpOrdClTrQuaBal2, rs.CLGrpOrdClTrQuaBal3, rs.CLGrpOrdClTrQuaChi2,
       # rs.CLGrpOrdClTrQuaKS, rs.CLGrpOrdClTrQuaDCv,
       # rs.CLGblOrdChi2KSDCv, rs.CLGblOrdQuaBal1, rs.CLGblOrdQuaBal2, rs.CLGblOrdQuaBal3,
       # rs.CLGblOrdQuaChi2, rs.CLGblOrdQuaKS, rs.CLGblOrdQuaDCv,
       # rs.CLGblOrdDAicChi2KSDCv,
       rs.CLRunFolder]

# e. Super-synthesis table, synthesis and details, for HTML or Excel : sorting parameters.
filsorReportSortCols = [('header (head)', sampleIndCol, 'Value'), _whichFinalQua]
filsorReportSortAscend = [True, False]

filsorReportRebuild = False
