# coding: utf-8

"""Parameter module for command line validation tests of pyaudisam (for -p option)

(as far as possible same data and parameters than in valtests.ipynb)

Parameters (pass them through -k key1=value1,key2=value2,... options to pyaudisam main command-line script):
:param lang: report language, en or fr

Usage (after cloning https://github.com/denmedius/pyaudisam to <pyaudisam dir>):
* cd <pyaudisam dir>
* python -m pyaudisam -p tests/valtests-ds-params -n -w tests/tmp/mcds-preanlr ...
  * -x/--distexport : export input files for manual analyses in Distance 6+ software
  * -e/--preanalyses : run pre-analyses for samples specified in this param file (see below)
  * -t/--prereports excel,html : generate english HTML and Excel reports of pre-analyses results
  * -s/--speparams lang=fr -t/--prereports html : generate french HTML report of pre-analyses results
* python -m pyaudisam -p tests/valtests-ds-params -n -w tests/tmp/mcds-anlr ...
  * -a/--analyses : run analyses specified in this param file (see below)
  * -r/--reports excel,html:mqua-r92 : generate auto-filtered Excel report (all filters)
     and HTML report (ExAicMQua-r920m6q3d12 filter)... of analyses results
* python -m pyaudisam -p tests/valtests-ds-params -n -w tests/tmp/mcds-optanlr ...
  * -o/--optanalyses : run opt-analyses specified in this param file (see below)
  * -f/--optreports excel:full,html:r92,html:r96 : generate full Excel report and 2 HTML reports
     (ExAicMQua-r920m6q3d12 and ExAicMQua-r960m6q3d8 filters) ... of opt-analyses results
* Note: if -u option not present, nothing actually run, no file written, only checks done and informations
  listed about what should happen if ... useful before jumping, isn't it ?
"""

import pathlib as pl

import pandas as pd

# from pyaudisam import log
from pyaudisam.optimisation import Interval
from pyaudisam.optanalyser import MCDSTruncOptanalysisResultsSet as rs

# Pas d'autres fichiers de paramètres inclus ici via pyaudisam.utils.loadPythonData.
parameterFiles = []


# Input data ##############################################

instDir = pl.Path(__file__).parent

studyName = 'valtests'
subStudyName = ''

# Input data
speciesCol = 'Espèce'
passIdCol = 'Passage'
distanceCol = 'Distance'
effortCol = 'Effort'
transectPlaceCols = ['Point']  # Columns for identifying places of transects, whatever pass number

# Survey field data: Individualised sightings + transects definition
surveyDataFile = instDir / 'refin/ACDC2019-Naturalist-ExtraitObsIndiv.ods'
indivDistDataSheet = 'DonnéesIndiv'
transectsDataSheet = 'Inventaires'

# DS analysis common parameters ##############################################
# a. Input / output data

# b. DS and run params
distanceUnit = 'Meter'
areaUnit = 'Hectare'
surveyType = 'Point'
distanceType = 'Radial'

clustering = False

passEffort = 1  # Constant effort value: 1 per pass over each transect = point.

studyAreaSpecs = dict(Zone='ACDC', Surface='2400')  # ha

# DS pre-analysis parameters ######################################################
# a. Input / output data
sampleSelCols = [speciesCol, passIdCol, 'Adulte', 'Durée']  # Columns for selecting and also identifying samples
sampleIndCol = 'NumEchant'  # Unique sample Id (number)
sampleAbbrevCol = 'AbrevEchant'  # Sample abbreviation (computed through analysisAbbrev() below)

def sampleAbbrev(sSample):
    """Short string for sample 'identification'"""
    abrvSpe = ''.join(word[:4].title() for word in sSample['Espèce'].split(' ')[:2])
    sampAbbrev = '{}-{}-{}-{}'.format(abrvSpe, sSample.Passage.replace('+', ''),
                                      sSample.Adulte.replace('+', ''), sSample['Durée'])
    return sampAbbrev

sampleSpecCustCols = []

# b. Samples to analyse
sampleSpecFile = instDir / 'refin/ACDC2019-Naturalist-ExtraitSpecsEchants.ods'

# c. DS and run params
preResultsHeadCols = dict(before=[sampleIndCol],
                          sample=sampleSelCols,
                          after=[sampleAbbrevCol] + sampleSpecCustCols)

modelPreStrategy = [dict(keyFn=kf, adjSr=js, estCrit='AIC', cvInt=95)
                    for js in ['COSINE', 'POLY', 'HERMITE']  # HERMITE: longer computation, for nothing better (?)
                    for kf in ['HNORMAL', 'HAZARD', 'UNIFORM', 'NEXPON']]  # NEXPON: problem: g'(0) << 0 !!!

runPreAnalysisMethod = 'subprocess.run'
runPreAnalysisTimeOut = 300
logPreAnalysisData = False
logPreAnalysisProgressEvery = 5

# DS analysis parameters ########################################################
# a. Input / output data
def analysisAbbrev(sAnlys):
    """Short string for analysis 'identification'"""
    abbrevs = [sampleAbbrev(sAnlys)]
    abbrevs += [sAnlys['FonctionClé'][:3].lower(), sAnlys['SérieAjust'][:3].lower()]
    dTroncAbrv = {'l': 'TrGche' if 'TrGche' in sAnlys.index else 'TroncGche',
                  'r': 'TrDrte' if 'TrDrte' in sAnlys.index else 'TroncDrte',
                  'm': 'NbTrches' if 'NbTrches' in sAnlys.index else 'NbTrModel'
                  if 'NbTrModel' in sAnlys.index else 'NbTrchMod',
                  'd': 'NbTrDiscr'}
    for abrv, name in dTroncAbrv.items():
        if name in sAnlys.index and not pd.isnull(sAnlys[name]):
            abbrevs.append('{}{}'.format(abrv, sAnlys[name][0].lower() if isinstance(sAnlys[name], str)
                                                                       else int(sAnlys[name])))
    return '-'.join(abbrevs)

analysisIndCol = 'NumAnlys'  # Analysis abbreviation (computed through sampleAbbrev() below)
analysisAbbrevCol = 'AbrevAnlys'  # Unique analysis Id (number)

# b. DS and run params
analysisSpecCustCols = []

resultsHeadCols = dict(before=[analysisIndCol, sampleIndCol],
                       sample=sampleSelCols,
                       after=[analysisAbbrevCol] + analysisSpecCustCols)

analysisParamCols = ['FonctionClé', 'SérieAjust', 'TrGche', 'TrDrte', 'NbTrchMod']

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

runAnalysisMethod = 'subprocess.run'
runAnalysisTimeOut = 300
logAnalysisData = False
logAnalysisProgressEvery = 5

# c. Analyses to run (reuse valtests notebook spec. file)
_ddfAnlysSpecs = pd.read_excel(instDir / 'refin/ACDC2019-Naturalist-ExtraitSpecsAnalyses.xlsx', sheet_name=None)
analysisSpecFile = instDir / 'tmp/ACDC2019-Naturalist-ExtraitSpecsAnalyses.xlsx'
with pd.ExcelWriter(analysisSpecFile) as xlWrtr:
    for sn in ['Echant1_impl', 'Echant2_impl', 'Modl_impl', 'Params1_expl', 'Params2_expl']:
        _ddfAnlysSpecs[sn].to_excel(xlWrtr, sheet_name=sn, index=False)

# DS opt-analysis parameters ########################################################
# a. Input / output data

# b. DS and run params
optAnalysisParamCols = ['FonctionClé', 'SérieAjust', 'TrGche', 'TrDrte', 'NbTrchMod', 'MultiOpt']
optAnalysisSpecCustCols = []

optResultsHeadCols = dict(before=[analysisIndCol, sampleIndCol],
                          sample=sampleSelCols,
                          after=['FonctionClé', 'SérieAjust', 'TrGche', 'TrDrte', 'NbTrchMod']
                                + [analysisAbbrevCol] + optAnalysisSpecCustCols)

runOptAnalysisMethod = 'subprocess.run'
runOptAnalysisTimeOut = 300
logOptAnalysisData = False
logOptAnalysisProgressEvery = 5
logOptimisationProgressEvery = 3
backupOptimisationsEvery = 5

defExpr2Optimise = 'chi2'
defMinimiseExpr = False
defOutliersMethod = 'tucquant'
defOutliersQuantCutPct = 7
defFitDistCutsFctr = Interval(min=0.6, max=1.4)
defDiscrDistCutsFctr = Interval(min=0.5, max=1.2)

defSubmitTimes = 1
defSubmitOnlyBest = None

dDefSubmitOtherParams = dict()

defCoreEngine = 'zoopt'
defCoreMaxIters = 100
defCoreTermExprValue = None
defCoreAlgorithm = 'racos'
defCoreMaxRetries = 0

# c. Opt-analyses to run (reuse valtests notebook spec. file)
optAnalysisSpecFile = instDir / 'refin/ACDC2019-Naturalist-ExtraitSpecsOptanalyses.xlsx'


# Reports (all types) ##############################################################
# Use pre-parameter 'lang' if available (you can pass it through -k lang=fr)
studyLang = 'en' if 'lang' not in dir() else lang
assert studyLang in ['en', 'fr']

# Pre-reports ######################################################################
preReportStudyTitle = 'PyAuDiSam Validation: Pre-analyses'
preReportStudySubTitle = 'Pre-analysis results report'
preReportAnlysSubTitle = 'Pre-analysis results details'
preReportStudyDescr = "Easy and parallel run through MCDSPreAnalyser"
preReportStudyKeywords = 'pyaudisam, validation, pre-analysis'

preReportPlotParams = dict(plotImgSize=(640, 400), superSynthPlotsHeight=288,
                           plotImgFormat='png', plotImgQuality=90,
                           plotLineWidth=1, plotDotWidth=4,
                           plotFontSizes=dict(title=11, axes=10, ticks=9, legend=10))

# Column selection for the various report tables
# a. Main HTML page (super-synthesis): Column 1 (top) for sample description
preReportSampleCols = [('header (head)', sampleIndCol, 'Value')] \
                      + [('header (sample)', col, 'Value') for col in sampleSelCols] \
                      + [rs.CLNTotObs, rs.CLMinObsDist, rs.CLMaxObsDist]

# b. Main HTML page (super-synthesis): Column 1 (bottom) for analysis model parameters
preReportParamCols = [rs.CLParEstKeyFn, rs.CLParEstAdjSer]  # rs.CLParEstCVInt, rs.CLParEstSelCrit]

# c. Main HTML page (super-synthesis): Columns 2 & 3 for analysis results (columns 4, 5, & 6 for plots)
preReportResultCols = [rs.CLRunStatus,
                       rs.CLNObs, rs.CLEffort,
                       rs.CLAic, rs.CLChi2, rs.CLKS, rs.CLDCv,
                       rs.CLCmbQuaBal1, rs.CLCmbQuaBal2, rs.CLCmbQuaBal3,
                       rs.CLPDetec, rs.CLEswEdr,
                       rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax,
                       rs.CLNumber, rs.CLNumberMin, rs.CLNumberMax]

# d. Detail HTML pages & Excel reports : Synthesis table
preReportSynthCols = [('header (head)', sampleIndCol, 'Value')] \
                     + [('header (sample)', col, 'Value') for col in sampleSelCols] \
                     + [rs.CLParEstKeyFn, rs.CLParEstAdjSer,
                        rs.CLNTotObs, rs.CLNObs, rs.CLNTotPars, rs.CLEffort, rs.CLDeltaAic,
                        rs.CLChi2, rs.CLKS, rs.CLCvMUw, rs.CLCvMCw, rs.CLDCv,
                        rs.CLSightRate,
                        rs.CLCmbQuaBal1, rs.CLCmbQuaBal2, rs.CLCmbQuaBal3,
                        rs.CLCmbQuaChi2, rs.CLCmbQuaKS, rs.CLCmbQuaDCv,
                        rs.CLPDetec, rs.CLPDetecMin, rs.CLPDetecMax,
                        rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax,
                        rs.CLNumber, rs.CLNumberMin, rs.CLNumberMax]

# e. Excel & HTML super-synthesis, synthesis & details : Sorting parameters.
preReportSortCols = [('header (head)', sampleIndCol, 'Value')]
preReportSortAscend = True

preReportRebuild = False


# Full reports ######################################################################
# 1. Specific to analysis reports
anlysFullReportStudyTitle = 'PyAuDiSam Validation: Analyses'
anlysFullReportStudySubTitle = 'Global analysis full report'
anlysFullReportAnlysSubTitle = 'Detailed report'
anlysFullReportStudyDescr = 'Easy and parallel run through MCDSAnalyser'
anlysFullReportStudyKeywords = 'pyaudisam, validation, analysis, full, report'

# 2. Specific to opt-analysis reports
optAnlysFullReportStudyTitle = 'PyAuDiSam Validation: Opt-analyses'
optAnlysFullReportStudySubTitle = 'Global opt-analysis full report'
optAnlysFullReportAnlysSubTitle = 'Detailed report'
optAnlysFullReportStudyDescr = 'Easy and parallel run through MCDSTruncationOptAnalyser'
optAnlysFullReportStudyKeywords = 'pyaudisam, validation, opt-analysis, full, report'

# 3. Common to analysis reports and opt-analysis reports
# Plot parameters
fullReportPlotParams = \
    dict(plotImgSize=(640, 400), superSynthPlotsHeight=288,
         plotImgFormat='png', plotImgQuality=90,
         plotLineWidth=1, plotDotWidth=4,
         plotFontSizes=dict(title=11, axes=10, ticks=9, legend=10))

# Column selection for the various report tables
# a. Main HTML page (super-synthesis): Column 1 (top) for sample description
fullReportSampleCols = \
    [('header (head)', sampleIndCol, 'Value')] \
    + [('header (sample)', col, 'Value') for col in sampleSelCols] \
    + [rs.CLNTotObs, rs.CLMinObsDist, rs.CLMaxObsDist]

# b. Main HTML page (super-synthesis): Column 1 (bottom) for analysis model parameters
fullReportParamCols = \
    [rs.CLParEstKeyFn, rs.CLParEstAdjSer,
     # rs.CLParEstCVInt, rs.CLParEstSelCrit,
     rs.CLParTruncLeft, rs.CLParTruncRight, rs.CLParModFitDistCuts]

# c. Main HTML page (super-synthesis): Columns 2 & 3 for analysis results (columns 4, 5, & 6 for plots)
fullReportResultCols = \
    [('header (head)', analysisIndCol, 'Value'),
     rs.CLRunStatus,
     rs.CLNObs, rs.CLEffort, rs.CLSightRate, rs.CLNAdjPars,
     rs.CLAic, rs.CLChi2, rs.CLKS, rs.CLDCv,
     rs.CLCmbQuaBal3, rs.CLCmbQuaBal2, rs.CLCmbQuaBal1,
     rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax,
     rs.CLNumber, rs.CLNumberMin, rs.CLNumberMax,
     rs.CLEswEdr, rs.CLPDetec]

# d. Main & detail HTML pages & Excel reports : Synthesis table
fullReportSynthCols = \
    [('header (head)', sampleIndCol, 'Value')] \
    + [('header (sample)', col, 'Value') for col in sampleSelCols] \
    + [('header (head)', analysisIndCol, 'Value')] + fullReportParamCols \
    + [rs.CLNTotObs, rs.CLNObs, rs.CLNTotPars, rs.CLEffort,
       rs.CLDeltaAic, rs.CLChi2, rs.CLKS, rs.CLCvMUw, rs.CLCvMCw, rs.CLDCv,
       rs.CLPDetec, rs.CLPDetecMin, rs.CLPDetecMax,
       rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax,
       rs.CLSightRate,
       rs.CLCmbQuaBal1, rs.CLCmbQuaBal2, rs.CLCmbQuaBal3,
       rs.CLCmbQuaChi2, rs.CLCmbQuaKS, rs.CLCmbQuaDCv,
       rs.CLGrpOrdSmTrAic,
       rs.CLGrpOrdClTrChi2KSDCv,  # rs.CLGrpOrdClTrChi2,
       rs.CLGrpOrdClTrDCv,
       rs.CLGrpOrdClTrQuaBal1, rs.CLGrpOrdClTrQuaBal2, rs.CLGrpOrdClTrQuaBal3, rs.CLGrpOrdClTrQuaChi2,
       rs.CLGrpOrdClTrQuaKS, rs.CLGrpOrdClTrQuaDCv,
       rs.CLGblOrdChi2KSDCv, rs.CLGblOrdQuaBal1, rs.CLGblOrdQuaBal2, rs.CLGblOrdQuaBal3,
       rs.CLGblOrdQuaChi2, rs.CLGblOrdQuaKS, rs.CLGblOrdQuaDCv,
       rs.CLGblOrdDAicChi2KSDCv,
       rs.CLRunFolder]

# e. Excel & HTML super-synthesis, synthesis & details : Sorting parameters.
fullReportSortCols = \
    [('header (head)', sampleIndCol, 'Value')] \
    + [rs.CLParTruncLeft, rs.CLParTruncRight,
       rs.CLDeltaAic, rs.CLCmbQuaBal3]
fullReportSortAscend = [True] * (len(fullReportSortCols) - 1) + [False]

fullReportRebuild = False


# Auto-filtered reports ######################################################################
# a. Specific to analysis reports
anlysFilsorReportStudyTitle = 'PyAuDiSam Validation: Analyses'
anlysFilsorReportStudySubTitle = 'Global analysis auto-filtered report'
anlysFilsorReportAnlysSubTitle = 'Detailed report'
anlysFilsorReportStudyDescr = 'Automated filtering et sorting : method "{fsId}" ;' \
                              ' after easy and parallel run through MCDSAnalyser'
anlysFilsorReportStudyKeywords = 'pyaudisam, validation, analysis, auto-filter, report'

# c. Specific to opt-analysis reports
optAnlysFilsorReportStudyTitle = 'PyAuDiSam Validation: Opt-analyses'
optAnlysFilsorReportStudySubTitle = 'Global opt-analysis auto-filtered report'
optAnlysFilsorReportAnlysSubTitle = 'Detailed report'
optAnlysFilsorReportStudyDescr = 'Automated filtering et sorting : method "{fsId}" ;' \
                                 ' after easy and parallel run through MCDSTruncationOptAnalyser'
optAnlysFilsorReportStudyKeywords = 'pyaudisam, validation, opt-analysis, auto-filter, report'

# c. Common to analysis reports and opt-analysis reports
filsorReportPlotParams = dict(plotImgSize=(640, 400), superSynthPlotsHeight=288,
                              plotImgFormat='png', plotImgQuality=90,
                              plotLineWidth=1, plotDotWidth=4,
                              plotFontSizes=dict(title=11, axes=10, ticks=9, legend=10))

# Available filter & sort schemes
_whichFinalQua = rs.CLCmbQuaBal3
_ascFinalQua = False

_whichBestQua = [rs.CLGrpOrdClTrChi2KSDCv, rs.CLGrpOrdClTrDCv, _whichFinalQua,
                rs.CLGrpOrdClTrQuaChi2, rs.CLGrpOrdClTrQuaKS, rs.CLGrpOrdClTrQuaDCv]

_dupSubset = [rs.CLNObs, rs.CLEffort, rs.CLDeltaAic, rs.CLChi2, rs.CLKS, rs.CLCvMUw, rs.CLCvMCw, rs.CLDCv, 
              rs.CLPDetec, rs.CLPDetecMin, rs.CLPDetecMax, rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax]
_dDupRounds = {rs.CLDeltaAic: 1, rs.CLChi2: 2, rs.CLKS: 2, rs.CLCvMUw: 2, rs.CLCvMCw: 2, rs.CLDCv: 2, 
               rs.CLPDetec: 3, rs.CLPDetecMin: 3, rs.CLPDetecMax: 3,
               rs.CLDensity: 2, rs.CLDensityMin: 2, rs.CLDensityMax: 2}

filsorReportSchemes = \
    [dict(method=rs.filterSortOnExecCode,
          deduplicate=dict(dupSubset=_dupSubset, dDupRounds=_dDupRounds),
          filterSort=dict(whichFinalQua=_whichFinalQua, ascFinalQua=_ascFinalQua),
          preselCols=[rs.CLCmbQuaBal1, rs.CLCmbQuaBal2, rs.CLCmbQuaBal3],
          preselAscs=False, preselThrhs=0.2, preselNum=4),
     dict(method=rs.filterSortOnExCAicMulQua,
          deduplicate=dict(dupSubset=_dupSubset, dDupRounds=_dDupRounds),
          filterSort=dict(sightRate=90, nBestAIC=4, nBestQua=2, whichBestQua=_whichBestQua,
                          nFinalRes=15, whichFinalQua=_whichFinalQua, ascFinalQua=_ascFinalQua),
          preselCols=[rs.CLCmbQuaBal1, rs.CLCmbQuaBal2, rs.CLCmbQuaBal3],
          preselAscs=False, preselThrhs=0.2, preselNum=3),
     dict(method=rs.filterSortOnExCAicMulQua,
          deduplicate=dict(dupSubset=_dupSubset, dDupRounds=_dDupRounds),
          filterSort=dict(sightRate=92, nBestAIC=3, nBestQua=2, whichBestQua=_whichBestQua,
                          nFinalRes=12, whichFinalQua=_whichFinalQua, ascFinalQua=_ascFinalQua),
          preselCols=[rs.CLCmbQuaBal1, rs.CLCmbQuaBal2, rs.CLCmbQuaBal3],
          preselAscs=False, preselThrhs=0.2, preselNum=3),
     dict(method=rs.filterSortOnExCAicMulQua,
          deduplicate=dict(dupSubset=_dupSubset, dDupRounds=_dDupRounds),
          filterSort=dict(sightRate=94, nBestAIC=2, nBestQua=1, whichBestQua=_whichBestQua,
                          nFinalRes=10, whichFinalQua=_whichFinalQua, ascFinalQua=_ascFinalQua),
          preselCols=[rs.CLCmbQuaBal1, rs.CLCmbQuaBal2, rs.CLCmbQuaBal3],
          preselAscs=False, preselThrhs=0.2, preselNum=3),
     dict(method=rs.filterSortOnExCAicMulQua,
          deduplicate=dict(dupSubset=_dupSubset, dDupRounds=_dDupRounds),
          filterSort=dict(sightRate=96, nBestAIC=2, nBestQua=1, whichBestQua=_whichBestQua,
                          nFinalRes=8, whichFinalQua=_whichFinalQua, ascFinalQua=_ascFinalQua),
          preselCols=[rs.CLCmbQuaBal1, rs.CLCmbQuaBal2, rs.CLCmbQuaBal3],
          preselAscs=False, preselThrhs=0.2, preselNum=3)]

# Column selection for the various report tables
# a. Main HTML page (super-synthesis): Column 1 (top) for sample description
filsorReportSampleCols = \
    [('header (head)', sampleIndCol, 'Value')] \
    + [('header (sample)', col, 'Value') for col in sampleSelCols] \
    + [rs.CLNTotObs, rs.CLMinObsDist, rs.CLMaxObsDist]

# b. Main HTML page (super-synthesis): Column 1 (bottom) for analysis model parameters
filsorReportParamCols = \
    [('header (head)', analysisIndCol, 'Value'),
     rs.CLParEstKeyFn, rs.CLParEstAdjSer,
     # rs.CLParEstCVInt, rs.CLParEstSelCrit,
     rs.CLParTruncLeft, rs.CLParTruncRight, rs.CLParModFitDistCuts]

# c. Main HTML page (super-synthesis): Columns 2 & 3 for analysis results (columns 4, 5, & 6 for plots)
filsorReportResultCols = \
    [rs.CLRunStatus,
     rs.CLNObs, rs.CLEffort, rs.CLSightRate, rs.CLNAdjPars,
     rs.CLAic, rs.CLChi2, rs.CLKS, rs.CLDCv,
     rs.CLCmbQuaBal1, rs.CLCmbQuaBal2, rs.CLCmbQuaBal3,
     rs.CLEswEdr, rs.CLPDetec,
     rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax,
     rs.CLNumber, rs.CLNumberMin, rs.CLNumberMax]

# d. Main & detail HTML pages & Excel reports : Auto-filtered & synthesis tables
filsorReportSynthCols = \
    [('header (head)', sampleIndCol, 'Value')] \
    + [('header (sample)', col, 'Value') for col in sampleSelCols] \
    + filsorReportParamCols \
    + [rs.CLNTotObs, rs.CLNObs, rs.CLNTotPars, rs.CLEffort,
       rs.CLDeltaAic, rs.CLChi2, rs.CLKS, rs.CLCvMUw, rs.CLCvMCw, rs.CLDCv,
       rs.CLSightRate,
       rs.CLCmbQuaBal1, rs.CLCmbQuaBal2, rs.CLCmbQuaBal3,
       rs.CLCmbQuaChi2, rs.CLCmbQuaKS, rs.CLCmbQuaDCv,
       rs.CLPDetec, rs.CLPDetecMin, rs.CLPDetecMax,
       rs.CLDensity, rs.CLDensityMin, rs.CLDensityMax,
       rs.CLNumber, rs.CLNumberMin, rs.CLNumberMax,
       rs.CLGrpOrdSmTrAic,
       rs.CLGrpOrdClTrChi2KSDCv,  # rs.CLGrpOrdClTrChi2,
       rs.CLGrpOrdClTrDCv,
       rs.CLGrpOrdClTrQuaBal1, rs.CLGrpOrdClTrQuaBal2, rs.CLGrpOrdClTrQuaBal3, rs.CLGrpOrdClTrQuaChi2,
       rs.CLGrpOrdClTrQuaKS, rs.CLGrpOrdClTrQuaDCv,
       rs.CLGblOrdChi2KSDCv, rs.CLGblOrdQuaBal1, rs.CLGblOrdQuaBal2, rs.CLGblOrdQuaBal3,
       rs.CLGblOrdQuaChi2, rs.CLGblOrdQuaKS, rs.CLGblOrdQuaDCv,
       rs.CLGblOrdDAicChi2KSDCv]

# e. Excel & HTML super-synthesis, synthesis & details : Sorting parameters.
filsorReportSortCols = [('header (head)', sampleIndCol, 'Value'), _whichFinalQua]
filsorReportSortAscend = [True, False]

filsorReportRebuild = False
