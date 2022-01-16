# coding: utf-8

# PyAuDiSam: Automation of Distance Sampling analyses with Distance software (http://distancesampling.org/)

# Copyright (C) 2021 Jean-Philippe Meuret

# This program is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see https://www.gnu.org/licenses/.

# Package main script, for when pyaudisam is invoked through "python -m"
import re
import sys
import atexit
import tempfile
import pathlib as pl
import argparse

import pandas as pd

from . import log, runtime
from .utils import loadPythonData
from .analyser import MCDSAnalyser, MCDSPreAnalyser, FilterSortSchemeIdManager
from .optanalyser import MCDSTruncationOptanalyser
from .report import MCDSResultsPreReport, MCDSResultsFullReport, MCDSResultsFilterSortReport


class Logger(object):

    """Local logger, taking care of output log file at exit time"""

    DefLogNamePrefix = 'pyaudisam-main'

    def __init__(self, runTimestamp, level=log.INFO):

        self.runTimestamp = runTimestamp
        self.openOpr = 'Checking'
        self.dOprStart = dict()

        # Log to sys.stdout, and also to a temporary log file (unique folder).
        self.runLogFileName = f'{self.DefLogNamePrefix}.{runTimestamp}.log'
        self.runLogFileName = pl.Path(tempfile.mkdtemp(prefix='pyaudisam')) / self.runLogFileName
        log.configure(handlers=[sys.stdout, self.runLogFileName], reset=True,
                      loggers=[dict(name='matplotlib', level=log.WARNING),
                               dict(name='ads', level=log.INFO2),
                               dict(name='ads.eng', level=log.INFO),
                               dict(name='ads.exr', level=log.INFO),
                               # dict(name='ads.dat', level=log.DEBUG),
                               # dict(name='ads.rep', level=log.DEBUG),
                               dict(name='ads.anr', level=log.INFO1),
                               dict(name='main', level=level)])

        # Fallback final log file path-name as long as it is not specified : current folder, generic (timestamped) name.
        self.finalLogFileName = pl.Path('.') / self.runLogFileName.name

        # Setup an atexit handler to move and rename the session log file in a user-friendly place if possible.
        atexit.register(self.giveBackLogFile)

        # Add logging methods to this logger
        self.logger = log.logger(name='main')
        for meth in dir(self.logger):
            if any(meth.startswith(prefix) for prefix in ['critical', 'error', 'warning', 'info', 'debug']):
                setattr(self, meth, getattr(self.logger, meth))

        # Done.
        self.info1(f'Logging session to temporary ' + self.runLogFileName.as_posix())
        self.info('Current folder: ' + pl.Path().absolute().as_posix())
        self.info('Command line: {} {}'.format(pl.Path(sys.argv[0]).as_posix(), ' '.join(sys.argv[1:])))

    def setFinalLogPrefix(self, prefix=None):

        self.finalLogFileName = None if prefix is None else pl.Path(prefix + f'.{self.runTimestamp}.log')

    def giveBackLogFile(self):

        if any(arg in sys.argv for arg in ['-h', '--help']):
            self.finalLogFileName = None  # No need for a log file here !

        if self.finalLogFileName is not None:
            self.info1(f'Giving back session log to {self.finalLogFileName}')

        # Release the log file
        log.configure(handlers=[sys.stdout], reset=True, loggers=[dict(name='main', level=log.INFO)])

        # Actually move and rename the log file if it is needed, or delete it if not.
        if self.finalLogFileName is not None:
            self.finalLogFileName.parent.mkdir(parents=True, exist_ok=True)
            self.runLogFileName.rename(self.finalLogFileName)
        else:
            self.runLogFileName.unlink()

        # Remove initial parent folder, now empty (was specially created for).
        self.runLogFileName.parent.rmdir()

    def setRealRun(self, realRun=True):

        if realRun:
            self.openOpr = 'Running'
            self.info('This is a real run: requested operation will be actually run !')
        else:
            self.openOpr = 'Checking'
            self.warning('Not a real run, only checking requested operations ...')

    def openOperation(self, oprText):

        self.info(f'{self.openOpr} {oprText} ...')
        self.dOprStart[oprText] = pd.Timestamp.now()

    def closeOperation(self, oprText):

        elapsed = str(pd.Timestamp.now() - self.dOprStart[oprText]).replace('0 days ', '')
        self.info(f'Done {self.openOpr.lower()} {oprText} ({elapsed}).')
        del self.dOprStart[oprText]


def decodeReportArg(repArg, repName='report'):

    """Decode report argument value

    Mini-language examples: none ; excel ; excel,html ; html:full ; excel,html:mqua-r925,html:mqua-r950

    :returns: dict(format: list(type)) with format in {'none', 'excel', 'html'}, 'none' excluding any other,
              and list(type) empty for 'none', empty or ['full'] for 'excel' format,
              and containing at least 1 of {full, <filter-sort method regex search string>*} for 'html' format.
    """

    # Separate format[:type] items
    repItems = [item.lower() for item in repArg.split(',')]

    # List types for each format
    repSpecs = dict()
    for repItem in repItems:
        fmt, typ = repItem.split(':') + ([None] if ':' not in repItem else [])
        if fmt not in repSpecs:
            repSpecs[fmt] = list()
        if typ not in repSpecs[fmt] and typ is not None:
            repSpecs[fmt].append(typ)

    # 'none' format kept iff alone
    repSpecs = {fmt: typs for fmt, typs in repSpecs.items() if fmt != 'none' or len(repSpecs) == 1}
    if 'none' in repSpecs:
        repSpecs.clear()

    # Check formats
    unsupRepFmts = [fmt for fmt in repSpecs if fmt not in ['none', 'html', 'excel']]
    if unsupRepFmts:
        logger.error('Unsupported {} format(s) {}'.format(repName, ', '.join(unsupRepFmts)))
        sys.exit(2)

    logger.debug1(f'{repName}: {repSpecs}')

    return repSpecs


# 0. Configure logging.
runTimestamp = pd.Timestamp.now().strftime('%y%m%d-%H%M%S')  # Date+time of the run (for log file, ... etc).

logger = Logger(runTimestamp, level=log.DEBUG2 if '-v' in sys.argv or '--verbose' in sys.argv else log.INFO1)

logger.info('Computation platform:')
for k, v in runtime.items():
    logger.info(f'* {k}: {v}')

# 1. Parse, check and post-process command line args.
argser = argparse.ArgumentParser(prog='pyaudisam',  # usage='python -m pyaudisam',
                                 description='Prepare or run (and / or generate reports for) many distance'
                                             ' sampling (pre-)analyses using a DS engine from Distance software',
                                 epilog='Exit codes:'
                                        ' 0 if OK,'
                                        ' 2 if any command line argument issue,'
                                        ' 1 if any other (unexpected) issue.')

argser.add_argument('-u', '--run', dest='realRun', action='store_true', default=False,
                    help='Actually run specified operation (not only run diagnosis of)'
                         ' => as long as -u/--run is not there, you can try any option,'
                         ' it wont start or write anything ... fell free, you are safe :-)')
argser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,
                    help='Display more infos about the work to be done and export sample / (opt-)analysis'
                         ' spec. files when relevant in the current directory ;')
argser.add_argument('-p', '--params', dest='paramFile', type=str, required=True,
                    help='Path-name of python file (.py assumed if no extension / suffix given) specifying'
                         ' export / (opt)analysis / report parameters')
argser.add_argument('-s', '--speparams', dest='speParams', type=str, default='',
                    help='Comma-separated key=value items specifying "special" parameters'
                         'defined before the parameter file is loaded, just as overridable built-in variables'
                         r' (syntax and limitations: string-only values, with no space or ,;\'"$&! inside'
                         ' => use it only for few and simple args like switches and simple names')
argser.add_argument('-w', '--workdir', dest='workDir', type=str, default='.',
                    help='Work folder = where to store DS analyses sub-folders and output files'
                         ' (Note: a timestamp sub-folder YYMMDD-HHMMSS is auto-appended, if not already such)')
argser.add_argument('-n', '--notimestamp', dest='noTimestamp', action='store_true', default=False,
                    help='Inhibit auto-timestamped work sub-folder creation (under work folder)')
argser.add_argument('-x', '--distexport', dest='distExport', action='store_true', default=False,
                    help='Export one Distance input file for each specified sample of the survey data'
                         ' (Note: a sample spec. file and a survey data file must be also specified, through -p)')
argser.add_argument('-e', '--preanalyses', dest='preAnalyses', action='store_true', default=False,
                    help='Run pre-analyses for the specified samples of the survey data'
                         ' (Note: a sample spec. file and a survey data file must be also specified, through -p)')
argser.add_argument('-a', '--analyses', dest='analyses', action='store_true', default=False,
                    help='Run analyses for the specified samples of the survey data'
                         ' (Note: an analysis spec. file and a survey data file must be also specified, through -p)')
argser.add_argument('-o', '--optanalyses', dest='optAnalyses', action='store_true', default=False,
                    help='Run opt-analyses for the specified samples of the survey data'
                         ' (Note: an opt-analysis spec. file and a survey data file must be also specified,'
                         ' through -p)')
argser.add_argument('-c', '--recoveropts', dest='recoverOpts', action='store_true', default=False,
                    help='Restart optimisations at the point they were when interrupted (for any reason), from the'
                         ' last usable recovery file found in the work folder (use in conjonction with -w & -n)'
                         ' (Note: if no usable recovery file found, the restart will fail: no automatic restart from'
                         ' scratch ; remove -c to do so)')
argser.add_argument('-t', '--prereports', dest='preReports', type=str, default='none',
                    help='Which reports to generate from pre-analyses results, through comma-separated keywords'
                         ' among {excel, html, none} (case does not matter, none ignored if not alone)')
argser.add_argument('-r', '--reports', dest='reports', type=str, default='none',
                    help='Which reports to generate from analyses results, through comma-separated format:type items'
                         ' with format among {excel, html, none}, "none" ignored if not alone ; type ignored'
                         ' for format="excel", and among {full, <filter-sort method regex search string>*}'
                         ' (at least one of) for format=html (case does not matter) ;'
                         ' note: available filter-sort methods are automatically listed'
                         ' when auto-filtering parameters are specified, so run command without -u first !'
                         ' examples: none ; excel ; excel,html:full ; html:mqua92,html:full,excel,html:mqua950')
argser.add_argument('-f', '--optreports', dest='optReports', type=str, default='none',
                    help='Which reports to generate from opt-analyses results (same mini-language as for -r)')
argser.add_argument('-l', '--logprefix', dest='logPrefix', type=str, default=None,
                    help='Target log file path-name prefix (will be postfixed by .<YYMMDD-HHMMSS timestamp>.log)'
                         f" (Default: <work folder>/{logger.DefLogNamePrefix} if -u/--run, else 'none' ;"
                         "if special value 'none', no log saved)")
argser.add_argument('-m', '--threads', dest='threads', type=int, default=0,
                    help='Number of parallel threads to use for (pre/opt-)analyses / report generation'
                         ' (default: 0 => auto-determined actual number of parallel threads from CPU specs ;'
                         ' 1 for no parallelism ; for any other choice, first check your actual CPU specs)')
argser.add_argument('-g', '--engine', dest='engineType', type=str, default='MCDS',
                    choices=['MCDS'],
                    help='The Distance engine to use, among MCDS, ... and no other for the moment'
                         ' (insensitive to case)')

args = argser.parse_args()
logger.debug(f'Arguments: {args}')

logger.setRealRun(args.realRun)

if args.threads == 1:
    args.threads = None  # No need for asynchronism: enforce sequential run.

args.preReports = decodeReportArg(args.preReports, repName='pre-analysis report')
args.reports = decodeReportArg(args.reports, repName='analysis report')
args.optReports = decodeReportArg(args.optReports, repName='opt-analysis report')

logger.info1('Arguments:')
for k, v in vars(args).items():
    logger.info1(f'* {k}: {v}')

# 2. Load parameter python file, passing "special" parameters if any.
speParamItems = args.speParams.split(',') if args.speParams else list()
if any(item.count('=') != 1 for item in speParamItems):
    logger.error(f'Syntax error in pre-parameters: "{args.speParams}"'
                 ' (should be "name1=value1,name2=value2,...")')
    sys.exit(2)

speParams = dict([item.split('=') for item in speParamItems])
paramFile, pars = loadPythonData(path=args.paramFile, **speParams)
if not pars:
    logger.error(f'Failed to load parameter file {paramFile.as_posix()}')
    sys.exit(2)
logger.debug1('Parameters: ' + ', '.join(vars(pars)))

paramFiles = [paramFile.as_posix()]
if 'paramFiles' in vars(pars):
    paramFiles += pars.paramFiles

# 3. More checks on args and parameters.
# a. Check filter and sort report args
if 'filsorReportSchemes' in vars(pars):
    filsorSchemeIdMgr = FilterSortSchemeIdManager()
    filsorReportSchemes = {filsorSchemeIdMgr.schemeId(sch): sch for sch in pars.filsorReportSchemes}
    logger.info('Available filter & sort report schemes: ' + ', '.join(filsorReportSchemes.keys()))

if 'html' in args.reports and not args.reports['html']:
    logger.error('HTML analysis report: MUST specify type / filter & sort method')
    sys.exit(2)

if 'html' in args.reports and ('full' not in args.reports['html'] or len(args.reports['html']) > 1):

    filsorMatches = {reSchId: [schId for schId in filsorReportSchemes if re.search(reSchId, schId, flags=re.I)]
                     for reSchId in args.reports['html'] if reSchId != 'full'}
    if any(len(matches) != 1 for matches in filsorMatches.values()) \
       or len(filsorMatches) != len(set(match for matches in filsorMatches.values() for match in matches)):
        logger.error('HTML analysis report: Bad or ambiguous filter & sort method specification(s) {}'
                     .format(' ; '.join('{} => [{}]'.format(reSchId, ', '.join(matchSchemes))
                                        for reSchId, matchSchemes in filsorMatches.items())))
        sys.exit(2)

    filsorAnlysReportSchemes = {schId: schValue for schId, schValue in filsorReportSchemes.items()
                                if schId in set(match for matches in filsorMatches.values() for match in matches)}
    if not filsorAnlysReportSchemes:
        logger.error('HTML analysis report: No filter & sort method specified')
        sys.exit(2)

if 'html' in args.optReports and not args.optReports['html']:
    logger.error('HTML opt-analysis report: MUST specify type / filter & sort method')
    sys.exit(2)

if 'html' in args.optReports and ('full' not in args.optReports['html'] or len(args.optReports['html']) > 1):

    filsorMatches = {reSchId: [schId for schId in filsorReportSchemes if re.search(reSchId, schId, flags=re.I)]
                     for reSchId in args.optReports['html'] if reSchId != 'full'}
    if any(len(matches) != 1 for matches in filsorMatches.values()) \
       or len(filsorMatches) != len(set(match for matches in filsorMatches.values() for match in matches)):
        logger.error('HTML opt-analysis report: Bad or ambiguous filter & sort method specification(s) {}'
                     .format(' ; '.join('{} => [{}]'.format(reSchId, ', '.join(matchSchemes))
                                        for reSchId, matchSchemes in filsorMatches.items())))
        sys.exit(2)

    filsorOptAnlysReportSchemes = {schId: schValue for schId, schValue in filsorReportSchemes.items()
                                   if schId in set(match for matches in filsorMatches.values() for match in matches)}
    if not filsorOptAnlysReportSchemes:
        logger.error('HTML opt-analysis report: No filter & sort method specified')
        sys.exit(2)

# 4. Output folder for results, reports ... etc.
#    (post-fixed with the run timestamp, if not already specified)
workDir = args.workDir if 'workDir' in vars(args) else pars.workDir if 'workDir' in vars(pars) else '.'
workDir = pl.Path(workDir)
if not(args.noTimestamp or re.match('.*[0-9]{6}-[0-9]{4,6}$', workDir.name)):
    workDir = workDir / runTimestamp
logger.info(f'Work folder: {workDir.as_posix()}')

# 5. Now we can setup the final session log file path-name prefix !
if args.logPrefix is None:
    if args.realRun:  # Default
        args.logPrefix = workDir.as_posix() + f'/{pars.studyName}{pars.subStudyName}'
elif args.logPrefix.lower() == 'none':
    args.logPrefix = None
logger.setFinalLogPrefix(args.logPrefix)

# 6. Really something to do ?
emptyRun = not any([args.distExport, args.preAnalyses, args.preReports,
                    args.analyses, args.reports, args.optAnalyses, args.optReports])
if emptyRun:
    logger.warning('No operation specified: nothing to do actually !')

# 7. Load input data if needed:
# a. Survey data
# * individualised data with distance from observer to observed "object",
# * point transect definition
if not emptyRun:

    surveyDataFile = pars.surveyDataFile if 'surveyDataFile' in vars(pars) else None
    if surveyDataFile:
        surveyDataFile = pl.Path(surveyDataFile)
        if surveyDataFile.is_file():
            indivDistSheet = pars.indivDistDataSheet if 'indivDistDataSheet' in vars(pars) else 0
            transectSheet = pars.transectsDataSheet if 'transectsDataSheet' in vars(pars) else 1
            logger.info1(f'Loading survey data and transects infos from file {surveyDataFile.as_posix()} ...')
            with pd.ExcelFile(surveyDataFile) as xlInFile:
                dfMonoCatObs = pd.read_excel(xlInFile, sheet_name=indivDistSheet)
                dfTransects = pd.read_excel(xlInFile, sheet_name=transectSheet)
            logger.info1(f'... found {len(dfMonoCatObs)} mono-category sightings and {len(dfTransects)} transects')
        else:
            logger.error(f'Could not find survey data file {surveyDataFile.as_posix()}')
            sys.exit(2)
    else:
        logger.error('No survey data file specified, can\'t export Distance file or run any type of analysis')
        sys.exit(2)

# 5.b. Sample specs
if args.distExport or args.preAnalyses:

    sampleSpecFile = pars.sampleSpecFile if 'sampleSpecFile' in vars(pars) else None
    if sampleSpecFile:
        sampleSpecFile = pl.Path(sampleSpecFile)
        if not sampleSpecFile.is_file():
            logger.error(f'Could not find sample spec. file {sampleSpecFile.as_posix()}')
            sys.exit(2)
    else:
        logger.error('No sample spec. file specified, can\'t export Distance file or run pre-analyses')
        sys.exit(2)

# 8. Export input files for Distance software, for manual analyses (if specified to).
if args.distExport:

    oprText = 'export of input data files for Distance'
    logger.openOperation(oprText)

    # a. Create PreAnalyser object.
    preAnlysr = MCDSPreAnalyser(dfMonoCatObs, dfTransects=dfTransects, dSurveyArea=pars.studyAreaSpecs,
                                effortConstVal=pars.passEffort, effortCol=pars.effortCol,
                                transectPlaceCols=pars.transectPlaceCols, passIdCol=pars.passIdCol,
                                sampleSelCols=pars.sampleSelCols,
                                sampleDecCols=[pars.effortCol, pars.distanceCol],
                                sampleIndCol=pars.sampleIndCol, sampleSpecCustCols=pars.sampleSpecCustCols,
                                abbrevCol=pars.sampleAbbrevCol, abbrevBuilder=pars.sampleAbbrev,
                                distanceUnit=pars.distanceUnit, areaUnit=pars.areaUnit,
                                surveyType=pars.surveyType, distanceType=pars.distanceType,
                                clustering=pars.clustering,
                                workDir=workDir)

    # b. Check sample specs.
    dfExplSampleSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols, verdict, reasons = \
        preAnlysr.explicitParamSpecs(implParamSpecs=sampleSpecFile, dropDupes=True, check=True)

    assert userParamSpecCols == []  # No analysis params here (auto. generated by PreAnalyser)
    assert intParamSpecCols == []  # Idem
    assert verdict
    assert not reasons

    logger.info2(f'Explicit sample specs:\n{dfExplSampleSpecs.to_string()}')
    logger.info1(f'From sample specs, {len(dfExplSampleSpecs)} samples to export')
    sampSpecFileName = f'{pars.studyName}{pars.subStudyName}-samples-explispecs.xlsx'
    if args.verbose and not args.realRun:
        dfExplSampleSpecs.to_excel(sampSpecFileName, index=False)

    # b. Export 1 Distance input data file for each specified sample.
    # dfSamples = Analyser.explicitVariantSpecs(partSpecs=sampSpecFile, varIndCol=pars.sampleIndCol)
    if args.realRun:
        preAnlysr.exportDSInputData(implSampleSpecs=sampleSpecFile, format='Distance')
        sampSpecFileName = workDir / sampSpecFileName
        dfExplSampleSpecs.to_excel(sampSpecFileName, index=False)
    preAnlysr.shutdown()  # Not really needed actually.

    logger.closeOperation(oprText)

# 9. Run pre-analyses (if specified to).
PreAnalyser = MCDSPreAnalyser
resultsWord = 'resultats' if pars.studyLang == 'fr' else 'results'
presFileName = workDir / f'{pars.studyName}{pars.subStudyName}-preanalyses-{resultsWord}.xlsx'
if args.preAnalyses:

    oprText = 'pre-analyses'
    logger.openOperation(oprText)

    # a. Create PreAnalyser object.
    preAnlysr = PreAnalyser(dfMonoCatObs, dfTransects=dfTransects, dSurveyArea=pars.studyAreaSpecs,
                            effortConstVal=pars.passEffort, effortCol=pars.effortCol,
                            transectPlaceCols=pars.transectPlaceCols, passIdCol=pars.passIdCol,
                            sampleSelCols=pars.sampleSelCols,
                            sampleDecCols=[pars.effortCol, pars.distanceCol],
                            sampleIndCol=pars.sampleIndCol, sampleSpecCustCols=pars.sampleSpecCustCols,
                            abbrevCol=pars.sampleAbbrevCol, abbrevBuilder=pars.sampleAbbrev,
                            distanceUnit=pars.distanceUnit, areaUnit=pars.areaUnit,
                            surveyType=pars.surveyType, distanceType=pars.distanceType,
                            clustering=pars.clustering,
                            resultsHeadCols=pars.preResultsHeadCols,
                            workDir=workDir,
                            runMethod=pars.runPreAnalysisMethod, runTimeOut=pars.runPreAnalysisTimeOut,
                            logData=pars.logPreAnalysisData, logProgressEvery=pars.logPreAnalysisProgressEvery)

    # b. Check sample specs.
    dfExplSampleSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols, verdict, reasons = \
        preAnlysr.explicitParamSpecs(implParamSpecs=sampleSpecFile, dropDupes=True, check=True)

    assert userParamSpecCols == []  # No analysis params here (auto. generated by PreAnalyser)
    assert verdict
    assert not reasons

    logger.info2(f'Explicit sample specs:\n{dfExplSampleSpecs.to_string()}')
    logger.info2(f'Pre-analysis model fallback strategy:\n{pd.DataFrame(pars.modelPreStrategy).to_string()}')
    logger.info1(f'From sample specs, {len(dfExplSampleSpecs)} samples to pre-analyse')
    sampSpecFileName = f'{pars.studyName}{pars.subStudyName}-samples-explispecs.xlsx'
    if args.verbose and not args.realRun:
        dfExplSampleSpecs.to_excel(sampSpecFileName, index=False)

    if any(col not in dfExplSampleSpecs.columns for col in pars.sampleSpecCustCols):
        logger.error('Missing custom (pass-through) column(s) in sample specs: {}'
                     .format(', '.join(col for col in pars.sampleSpecCustCols
                                       if col not in dfExplSampleSpecs.columns)))
        sys.exit(2)

    # c. Run pre-analyses.
    if args.realRun:
        preResults = preAnlysr.run(implSampleSpecs=sampleSpecFile, dModelStrategy=pars.modelPreStrategy,
                                   threads=args.threads)
        sampSpecFileName = workDir / sampSpecFileName
        dfExplSampleSpecs.to_excel(sampSpecFileName, index=False)
    preAnlysr.shutdown()

    # d. Save results to disk.
    # if 'dfSampleStats' in dir(): # Qq specs supplémentaires
    #    preResults.updateSpecs(sampleStats=dfSampleStats)
    if args.realRun:
        preResults.toExcel(presFileName)

    logger.closeOperation(oprText)

# 10. Generate pre-analysis reports (if specified to).
PreReport = MCDSResultsPreReport
reportWord = 'rapport' if pars.studyLang == 'fr' else 'report'
if args.preReports:

    # a. Load pre-analyses results if not just computed
    if not args.preAnalyses:

        if not presFileName.is_file():
            logger.error(f'Cannot generate pre-analysis reports: results file not found {presFileName.as_posix()}')
            sys.exit(2)

        logger.info(f'Loading pre-analysis results from {presFileName.as_posix()}')

        preAnlysr = PreAnalyser(dfMonoCatObs, dfTransects=dfTransects, dSurveyArea=pars.studyAreaSpecs,
                                effortConstVal=pars.passEffort, effortCol=pars.effortCol,
                                transectPlaceCols=pars.transectPlaceCols, passIdCol=pars.passIdCol,
                                sampleSelCols=pars.sampleSelCols,
                                sampleDecCols=[pars.effortCol, pars.distanceCol],
                                sampleIndCol=pars.sampleIndCol, sampleSpecCustCols=pars.sampleSpecCustCols,
                                abbrevCol=pars.sampleAbbrevCol, abbrevBuilder=pars.sampleAbbrev,
                                distanceUnit=pars.distanceUnit, areaUnit=pars.areaUnit,
                                surveyType=pars.surveyType, distanceType=pars.distanceType,
                                clustering=pars.clustering,
                                resultsHeadCols=pars.preResultsHeadCols)

        preResults = preAnlysr.setupResults()
        preAnlysr.shutdown()  # Not really needed actually.

        preResults.fromFile(presFileName)

    # b. Check report generation parameters
    assert isinstance(pars.preReportSortAscend, bool) or len(pars.preReportSortCols) == len(pars.preReportSortAscend)

    # c. Generate specified reports
    oprText = 'generation of {} pre-analysis report(s)'.format(','.join(args.preReports))
    logger.openOperation(oprText)

    preRepPrfx = f'{pars.studyName}{pars.subStudyName}-preanalyses-{reportWord}'
    preReport = PreReport(resultsSet=preResults, lang=pars.studyLang,
                          title=pars.preReportStudyTitle, subTitle=pars.preReportStudySubTitle,
                          anlysSubTitle=pars.preReportAnlysSubTitle, description=pars.preReportStudyDescr,
                          keywords=pars.preReportStudyKeywords, pySources=paramFiles,
                          sampleCols=pars.preReportSampleCols, paramCols=pars.preReportParamCols,
                          resultCols=pars.preReportResultCols, synthCols=pars.preReportSynthCols,
                          sortCols=pars.preReportSortCols, sortAscend=pars.preReportSortAscend,
                          tgtFolder=workDir, tgtPrefix=preRepPrfx,
                          **pars.preReportPlotParams)

    if 'excel' in args.preReports:
        logger.info1('* Excel pre-analysis report to be generated')
        if args.realRun:
            preReport.toExcel(rebuild=pars.preReportRebuild)

    if 'html' in args.preReports:
        logger.info1('* HTML pre-analysis report to be generated')
        if args.realRun:
            preReport.toHtml(rebuild=pars.preReportRebuild,
                             generators=1 if args.threads is None else args.threads)

    logger.closeOperation(oprText)

# 11. Run analyses (if specified to).
Analyser = MCDSAnalyser
resFileName = workDir / f'{pars.studyName}{pars.subStudyName}-analyses-{resultsWord}.xlsx'
if args.analyses:

    # Check analysis spec. file
    analysisSpecFile = pars.analysisSpecFile if 'analysisSpecFile' in vars(pars) else None
    if analysisSpecFile:
        analysisSpecFile = pl.Path(analysisSpecFile)
        if not analysisSpecFile.is_file():
            logger.error(f'Could not find analysis spec. file {analysisSpecFile.as_posix()}')
            sys.exit(2)
    else:
        logger.error('No analysis spec. file specified, can\'t run analyses')
        sys.exit(2)

    oprText = 'analyses'
    logger.openOperation(oprText)

    # a. Create Analyser object.
    anlysr = Analyser(dfMonoCatObs, dfTransects=dfTransects, dSurveyArea=pars.studyAreaSpecs,
                      effortConstVal=pars.passEffort, effortCol=pars.effortCol,
                      transectPlaceCols=pars.transectPlaceCols, passIdCol=pars.passIdCol,
                      sampleSelCols=pars.sampleSelCols, sampleDecCols=[pars.effortCol, pars.distanceCol],
                      anlysSpecCustCols=pars.analysisSpecCustCols,
                      abbrevCol=pars.analysisAbbrevCol, abbrevBuilder=pars.analysisAbbrev,
                      anlysIndCol=pars.analysisIndCol, sampleIndCol=pars.sampleIndCol,
                      distanceUnit=pars.distanceUnit, areaUnit=pars.areaUnit,
                      surveyType=pars.surveyType, distanceType=pars.distanceType,
                      clustering=pars.clustering,
                      resultsHeadCols=pars.resultsHeadCols,
                      ldTruncIntrvSpecs=pars.ldTruncIntrvSpecs, truncIntrvEpsilon=pars.truncIntrvEpsilon,
                      workDir=workDir, runMethod=pars.runAnalysisMethod, runTimeOut=pars.runAnalysisTimeOut,
                      logData=pars.logAnalysisData, logProgressEvery=pars.logAnalysisProgressEvery,
                      defEstimKeyFn=pars.defEstimKeyFn, defEstimAdjustFn=pars.defEstimAdjustFn,
                      defEstimCriterion=pars.defEstimCriterion, defCVInterval=pars.defCVInterval,
                      defMinDist=pars.defMinDist, defMaxDist=pars.defMaxDist,
                      defFitDistCuts=pars.defFitDistCuts, defDiscrDistCuts=pars.defDiscrDistCuts)

    # b. Check analysis specs.
    dfExplAnlysSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols, verdict, reasons = \
        anlysr.explicitParamSpecs(implParamSpecs=analysisSpecFile, dropDupes=True, check=True)

    assert userParamSpecCols == pars.analysisParamCols
    assert verdict
    assert not reasons

    logger.info2(f'Explicit analysis specs:\n{dfExplAnlysSpecs.to_string()}')
    logger.info1(f'From analysis specs, {len(dfExplAnlysSpecs)} analyses to run')
    anlysSpecFileName = f'{pars.studyName}{pars.subStudyName}-analyses-explispecs.xlsx'
    if args.verbose and not args.realRun:
        dfExplAnlysSpecs.to_excel(anlysSpecFileName, index=False)

    if any(col not in dfExplAnlysSpecs.columns for col in pars.analysisSpecCustCols):
        logger.error('Missing custom (pass-through) column(s) in analysis specs: {}'
                     .format(', '.join(col for col in pars.analysisSpecCustCols
                                       if col not in dfExplAnlysSpecs.columns)))
        sys.exit(2)

    # c. Run analyses.
    if args.realRun:
        results = anlysr.run(implParamSpecs=analysisSpecFile, threads=args.threads)
        anlysSpecFileName = workDir / anlysSpecFileName
        dfExplAnlysSpecs.to_excel(anlysSpecFileName, index=False)
    anlysr.shutdown()

    # d. Save results to disk.
    if args.realRun:
        results.toExcel(resFileName)

    logger.closeOperation(oprText)

# 12. Generate analysis reports (if specified to).
FullReport = MCDSResultsFullReport
FilSorReport = MCDSResultsFilterSortReport

if args.reports:

    # a. Load analysis results if not just computed
    if not args.analyses:

        if not resFileName.is_file():
            logger.error(f'Cannot generate analysis reports: results file not found {resFileName.as_posix()}')
            sys.exit(2)

        logger.info(f'Loading analysis results from {resFileName.as_posix()}')

        anlysr = Analyser(dfMonoCatObs, dfTransects=dfTransects, dSurveyArea=pars.studyAreaSpecs,
                          effortConstVal=pars.passEffort, effortCol=pars.effortCol,
                          transectPlaceCols=pars.transectPlaceCols, passIdCol=pars.passIdCol,
                          sampleSelCols=pars.sampleSelCols, sampleDecCols=[pars.effortCol, pars.distanceCol],
                          anlysSpecCustCols=pars.analysisSpecCustCols,
                          abbrevCol=pars.analysisAbbrevCol, abbrevBuilder=pars.analysisAbbrev,
                          anlysIndCol=pars.analysisIndCol, sampleIndCol=pars.sampleIndCol,
                          distanceUnit=pars.distanceUnit, areaUnit=pars.areaUnit,
                          surveyType=pars.surveyType, distanceType=pars.distanceType,
                          clustering=pars.clustering,
                          resultsHeadCols=pars.resultsHeadCols)

        results = anlysr.setupResults()
        anlysr.shutdown()  # Not really needed actually.

        results.fromFile(resFileName)

    # b. Generate specified reports
    oprText = 'generation of {} analysis report(s)'.format(','.join(args.reports))
    logger.openOperation(oprText)

    repPrfx = f'{pars.studyName}{pars.subStudyName}-analyses-{reportWord}'

    # * Auto-filtered reports.
    if ('excel' in args.reports and 'full' not in args.reports['excel'] and 'filsorReportSchemes' in dir()) \
       or ('html' in args.reports and 'full' not in args.reports['html'] and 'filsorAnlysReportSchemes' in dir()):

        assert isinstance(pars.filsorReportSortAscend, bool) \
               or len(pars.filsorReportSortCols) == len(pars.filsorReportSortAscend)

        filsorReport = FilSorReport(resultsSet=results, lang=pars.studyLang,
                                    title=pars.anlysFilsorReportStudyTitle,
                                    subTitle=pars.anlysFilsorReportStudySubTitle,
                                    anlysSubTitle=pars.anlysFilsorReportAnlysSubTitle,
                                    description=pars.anlysFilsorReportStudyDescr,
                                    keywords=pars.anlysFilsorReportStudyKeywords, pySources=paramFiles,
                                    sampleCols=pars.filsorReportSampleCols, paramCols=pars.filsorReportParamCols,
                                    resultCols=pars.filsorReportResultCols, synthCols=pars.filsorReportSynthCols,
                                    sortCols=pars.filsorReportSortCols, sortAscend=pars.filsorReportSortAscend,
                                    filSorSchemes=pars.filsorReportSchemes,
                                    tgtFolder=workDir, tgtPrefix=repPrfx,
                                    **pars.filsorReportPlotParams)

        if 'excel' in args.reports and 'full' not in args.reports['excel'] and 'filsorReportSchemes' in dir():
            logger.info1('* Auto-filtered Excel analysis report to be generated: all schemes')
            if args.realRun:
                filsorReport.toExcel(rebuild=pars.filsorReportRebuild)

        if 'html' in args.reports  and 'full' not in args.reports['html'] and 'filsorAnlysReportSchemes' in dir():
            logger.info1('* Auto-filtered HTML analysis report(s) to be generated: {}'
                         .format(', '.join(filsorAnlysReportSchemes.keys())))
            if args.realRun:
                for scheme in filsorAnlysReportSchemes.values():
                    filsorReport.toHtml(filSorScheme=scheme, rebuild=pars.filsorReportRebuild)

    # * Full reports.
    if ('excel' in args.reports and ('full' in args.reports['excel'] or 'filsorReportSchemes' not in dir())) \
       or ('html' in args.reports and 'full' in args.reports['html']):

        assert isinstance(pars.fullReportSortAscend, bool) \
               or len(pars.fullReportSortCols) == len(pars.fullReportSortAscend)

        fullReport = FullReport(resultsSet=results, lang=pars.studyLang,
                                title=pars.anlysFullReportStudyTitle,
                                subTitle=pars.anlysFullReportStudySubTitle,
                                anlysSubTitle=pars.anlysFullReportAnlysSubTitle,
                                description=pars.anlysFullReportStudyDescr,
                                keywords=pars.anlysFullReportStudyKeywords, pySources=paramFiles,
                                sampleCols=pars.fullReportSampleCols, paramCols=pars.fullReportParamCols,
                                resultCols=pars.fullReportResultCols, synthCols=pars.fullReportSynthCols,
                                sortCols=pars.fullReportSortCols, sortAscend=pars.fullReportSortAscend,
                                tgtFolder=workDir, tgtPrefix=repPrfx,
                                **pars.fullReportPlotParams)

        if 'excel' in args.reports and ('full' in args.reports['excel'] or 'filsorReportSchemes' not in dir()):
            logger.info1('* Full Excel analysis report to be generated')
            if args.realRun:
                fullReport.toExcel(rebuild=pars.fullReportRebuild)

        if 'html' in args.reports and 'full' in args.reports['html']:
            logger.info1('* Full HTML analysis report to be generated')
            if args.realRun:
                fullReport.toHtml(rebuild=pars.fullReportRebuild,
                                  generators=1 if args.threads is None else args.threads)

    # if not any(rep in dir() for rep in ['fullReport', 'filsorReport']):  # This doesn't work ... !!??
    if not any(['fullReport' in dir(), 'filsorReport' in dir()]):
        logger.error('Neither full nor filter & sort analysis report to be generated (whatever format)')
        sys.exit(2)

    logger.closeOperation(oprText)

# 13. Run opt-analyses (if specified to).
OptAnalyser = MCDSTruncationOptanalyser
oresFileName = workDir / f'{pars.studyName}{pars.subStudyName}-optanalyses-{resultsWord}.xlsx'
if args.optAnalyses:

    # Check analysis spec. file
    optAnalysisSpecFile = pars.optAnalysisSpecFile if 'optAnalysisSpecFile' in vars(pars) else None
    if optAnalysisSpecFile:
        optAnalysisSpecFile = pl.Path(optAnalysisSpecFile)
        if not optAnalysisSpecFile.is_file():
            logger.error(f'Could not find opt-analysis spec. file {optAnalysisSpecFile.as_posix()}')
            sys.exit(2)
    else:
        logger.error('No opt-analysis spec. file specified, can\'t run opt-analyses')
        sys.exit(2)

    oprText = 'opt-analyses'
    logger.openOperation(oprText)

    # a. Create OptAnalyser object.
    optAnlysr = OptAnalyser(dfMonoCatObs, dfTransects=dfTransects, dSurveyArea=pars.studyAreaSpecs,
                            effortConstVal=pars.passEffort, effortCol=pars.effortCol,
                            transectPlaceCols=pars.transectPlaceCols, passIdCol=pars.passIdCol,
                            sampleSelCols=pars.sampleSelCols, sampleDecCols=[pars.effortCol, pars.distanceCol],
                            sampleDistCol=pars.distanceCol, anlysSpecCustCols=pars.optAnalysisSpecCustCols,
                            abbrevCol=pars.analysisAbbrevCol, abbrevBuilder=pars.analysisAbbrev,
                            anlysIndCol=pars.analysisIndCol, sampleIndCol=pars.sampleIndCol,
                            distanceUnit=pars.distanceUnit, areaUnit=pars.areaUnit,
                            surveyType=pars.surveyType, distanceType=pars.distanceType,
                            clustering=pars.clustering,
                            resultsHeadCols=pars.optResultsHeadCols,
                            ldTruncIntrvSpecs=pars.ldTruncIntrvSpecs, truncIntrvEpsilon=pars.truncIntrvEpsilon,
                            workDir=workDir, logData=pars.logOptAnalysisData,
                            runMethod=pars.runOptAnalysisMethod, runTimeOut=pars.runOptAnalysisTimeOut,
                            logAnlysProgressEvery=pars.logOptAnalysisProgressEvery,
                            logOptimProgressEvery=pars.logOptimisationProgressEvery,
                            backupOptimEvery=pars.backupOptimisationsEvery,
                            defEstimKeyFn=pars.defEstimKeyFn, defEstimAdjustFn=pars.defEstimAdjustFn,
                            defEstimCriterion=pars.defEstimCriterion, defCVInterval=pars.defCVInterval,
                            defExpr2Optimise=pars.defExpr2Optimise, defMinimiseExpr=pars.defMinimiseExpr,
                            defOutliersMethod=pars.defOutliersMethod,
                            defOutliersQuantCutPct=pars.defOutliersQuantCutPct,
                            defFitDistCutsFctr=pars.defFitDistCutsFctr,
                            defDiscrDistCutsFctr=pars.defDiscrDistCutsFctr,
                            defSubmitTimes=pars.defSubmitTimes, defSubmitOnlyBest=pars.defSubmitOnlyBest,
                            dDefSubmitOtherParams=pars.dDefSubmitOtherParams,
                            dDefOptimCoreParams=dict(core=pars.defCoreEngine, maxIters=pars.defCoreMaxIters,
                                                     termExprValue=pars.defCoreTermExprValue,
                                                     algorithm=pars.defCoreAlgorithm,
                                                     maxRetries=pars.defCoreMaxRetries))

    # b. Check opt-analysis specs.
    dfExplOptAnlysSpecs, userParamSpecCols, intParamSpecCols, unmUserParamSpecCols, verdict, reasons = \
        optAnlysr.explicitParamSpecs(implParamSpecs=optAnalysisSpecFile, dropDupes=True, check=True)

    assert userParamSpecCols == pars.optAnalysisParamCols
    assert verdict
    assert not reasons

    logger.info2(f'Explicit opt-analysis specs:\n{dfExplOptAnlysSpecs.to_string()}')
    optUserParamSpecCols = optAnlysr.zoptr4Specs.optimisationParamSpecUserNames(userParamSpecCols, intParamSpecCols)
    sbAnlysNeedOpt = dfExplOptAnlysSpecs[optUserParamSpecCols].apply(optAnlysr.analysisNeedsOptimisationFirst,
                                                                     axis='columns')
    logger.info1('From opt-analysis specs, {} / {} opt-analyses to run with truncation optimisation first ...'
                 .format(sbAnlysNeedOpt.sum(), len(dfExplOptAnlysSpecs)))
    logger.info1('... implying possibly up to {} auto-analyses in the background if only full "auto" specs'
                 .format(sbAnlysNeedOpt.sum() * pars.defCoreMaxIters * pars.defSubmitTimes))
    optAnlysSpecFileName = f'{pars.studyName}{pars.subStudyName}-optanalyses-explispecs.xlsx'
    if args.verbose and not args.realRun:
        dfExplOptAnlysSpecs.to_excel(optAnlysSpecFileName, index=False)

    if any(col not in dfExplOptAnlysSpecs.columns for col in pars.optAnalysisSpecCustCols):
        logger.error('Missing custom (pass-through) column(s) in opt-analysis specs: {}'
                     .format(', '.join(col for col in pars.optAnalysisSpecCustCols
                                       if col not in dfExplOptAnlysSpecs.columns)))
        sys.exit(2)

    # c. Check if recovery possible (not 100% reliable), if specified
    if args.recoverOpts and not list(workDir.glob('optr-resbak-*.pickle.xz')):
        logger.error('No optimisation backup file found, can\'t recover ; you must start from scratch')
        sys.exit(2)
    else:
        logger.info('Backup files are there, recovery is very likely possible: let\'s try !')

    # d. Run opt-analyses.
    if args.realRun:
        optResults = optAnlysr.run(implParamSpecs=optAnalysisSpecFile,
                                   threads=args.threads, recoverOptims=args.recoverOpts)
        optAnlysSpecFileName = workDir / optAnlysSpecFileName
        dfExplOptAnlysSpecs.to_excel(optAnlysSpecFileName, index=False)
    optAnlysr.shutdown()

    # e. Save results to disk.
    if args.realRun:
        optResults.toExcel(oresFileName)

    logger.closeOperation(oprText)

# 14. Generate opt-analysis reports (if specified to).
if args.optReports:

    # a. Load opt-analysis results if not just computed
    if not args.optAnalyses:

        if not oresFileName.is_file():
            logger.error(f'Cannot generate opt-analysis reports: results file not found {oresFileName.as_posix()}')
            sys.exit(2)

        logger.info(f'Loading opt-analysis results from {oresFileName.as_posix()}')

        optAnlysr = OptAnalyser(dfMonoCatObs, dfTransects=dfTransects, dSurveyArea=pars.studyAreaSpecs,
                                effortConstVal=pars.passEffort, effortCol=pars.effortCol,
                                transectPlaceCols=pars.transectPlaceCols, passIdCol=pars.passIdCol,
                                sampleSelCols=pars.sampleSelCols, sampleDecCols=[pars.effortCol, pars.distanceCol],
                                sampleDistCol=pars.distanceCol, anlysSpecCustCols=pars.optAnalysisSpecCustCols,
                                abbrevCol=pars.analysisAbbrevCol, abbrevBuilder=pars.analysisAbbrev,
                                anlysIndCol=pars.analysisIndCol, sampleIndCol=pars.sampleIndCol,
                                distanceUnit=pars.distanceUnit, areaUnit=pars.areaUnit,
                                surveyType=pars.surveyType, distanceType=pars.distanceType,
                                clustering=pars.clustering,
                                resultsHeadCols=pars.optResultsHeadCols)

        optResults = optAnlysr.setupResults()
        optAnlysr.shutdown()  # Not really needed actually.

        optResults.fromFile(oresFileName)

    # b. Generate specified reports
    assert isinstance(pars.filsorReportSortAscend, bool) \
           or len(pars.filsorReportSortCols) == len(pars.filsorReportSortAscend)

    oprText = 'generation of {} opt-analysis report(s)'.format(','.join(args.optReports))
    logger.openOperation(oprText)

    optRepPrfx = f'{pars.studyName}{pars.subStudyName}-optanalyses-{reportWord}'

    # * Auto-filtered reports.
    if ('excel' in args.optReports and 'full' not in args.optReports['excel'] and 'filsorReportSchemes' in dir()) \
       or ('html' in args.optReports and 'filsorOptAnlysReportSchemes' in dir()):

        assert isinstance(pars.filsorReportSortAscend, bool) \
               or len(pars.filsorReportSortCols) == len(pars.filsorReportSortAscend)

        filsorOptReport = FilSorReport(resultsSet=optResults, lang=pars.studyLang,
                                       title=pars.optAnlysFilsorReportStudyTitle,
                                       subTitle=pars.optAnlysFilsorReportStudySubTitle,
                                       anlysSubTitle=pars.optAnlysFilsorReportAnlysSubTitle,
                                       description=pars.optAnlysFilsorReportStudyDescr,
                                       keywords=pars.optAnlysFilsorReportStudyKeywords, pySources=paramFiles,
                                       sampleCols=pars.filsorReportSampleCols, paramCols=pars.filsorReportParamCols,
                                       resultCols=pars.filsorReportResultCols, synthCols=pars.filsorReportSynthCols,
                                       sortCols=pars.filsorReportSortCols, sortAscend=pars.filsorReportSortAscend,
                                       filSorSchemes=pars.filsorReportSchemes,
                                       tgtFolder=workDir, tgtPrefix=optRepPrfx,
                                       **pars.filsorReportPlotParams)

        if 'excel' in args.optReports and 'full' not in args.optReports['excel'] and 'filsorReportSchemes' in dir():
            logger.info1('* Auto-filtered Excel opt-analysis report to be generated: all schemes')
            if args.realRun:
                filsorOptReport.toExcel(rebuild=pars.filsorReportRebuild)

        if 'html' in args.optReports and 'filsorOptAnlysReportSchemes' in dir():
            logger.info1('* Auto-filtered HTML opt-analysis report(s) to be generated: {}'
                         .format(', '.join(filsorOptAnlysReportSchemes.keys())))
            if args.realRun:
                for scheme in filsorOptAnlysReportSchemes.values():
                    filsorOptReport.toHtml(filSorScheme=scheme, rebuild=pars.filsorReportRebuild)

    # * Full reports.
    if ('excel' in args.optReports and ('full' in args.optReports['excel'] or 'filsorReportSchemes' not in dir())) \
       or ('html' in args.optReports and 'full' in args.optReports['html']):

        assert isinstance(pars.fullReportSortAscend, bool) \
               or len(pars.fullReportSortCols) == len(pars.fullReportSortAscend)

        fullOptReport = FullReport(resultsSet=optResults, lang=pars.studyLang,
                                   title=pars.optAnlysFullReportStudyTitle,
                                   subTitle=pars.optAnlysFullReportStudySubTitle,
                                   anlysSubTitle=pars.optAnlysFullReportAnlysSubTitle,
                                   description=pars.optAnlysFullReportStudyDescr,
                                   keywords=pars.optAnlysFullReportStudyKeywords, pySources=paramFiles,
                                   sampleCols=pars.fullReportSampleCols, paramCols=pars.fullReportParamCols,
                                   resultCols=pars.fullReportResultCols, synthCols=pars.fullReportSynthCols,
                                   sortCols=pars.fullReportSortCols, sortAscend=pars.fullReportSortAscend,
                                   tgtFolder=workDir, tgtPrefix=optRepPrfx,
                                   **pars.fullReportPlotParams)

        if 'excel' in args.optReports \
           and ('full' in args.optReports['excel'] or 'filsorReportSchemes' not in dir()):
            logger.info1('* Full Excel opt-analysis report to be generated')
            if args.realRun:
                fullOptReport.toExcel(rebuild=pars.fullReportRebuild)

        if 'html' in args.optReports and 'full' in args.optReports['html']:
            logger.info1('* Full HTML opt-analysis report to be generated')
            if args.realRun:
                fullOptReport.toHtml(rebuild=pars.fullReportRebuild,
                                     generators=1 if args.threads is None else args.threads)

    # if not any(rep in dir() for rep in ['fullOptReport', 'filsorOptReport']):  # This doesn't work ... !!??
    if not any(['fullOptReport' in dir(), 'filsorOptReport' in dir()]):
        logger.error('Neither full nor filter & sort opt-analysis report to be generated (whatever format)')
        sys.exit(2)

    logger.closeOperation(oprText)

if not args.realRun and not emptyRun:
    logger.info('Checks done, seems you can now really run this, through -u / --realrun :-)')

# 15. Done.
sys.exit(0)
