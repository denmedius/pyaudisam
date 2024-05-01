# coding: utf-8
# PyAuDiSam: Automation of Distance Sampling analyses with Distance software (http://distancesampling.org/)

# Copyright (C) 2021 Jean-Philippe Meuret, Sylvain Sainnier

# This program is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see https://www.gnu.org/licenses/.

# Common tools for automated unit, integration and validation tests

from types import SimpleNamespace as DotDict
import re
import pathlib as pl
import shutil
import difflib

import pandas as pd

import pyaudisam as ads

from conftest import pTestDir, pTmpDir, pLogFile


# More useful test folders
pRefInDir = pTestDir / 'refin'
pRefOutDir = pTestDir / 'refout'

# Temporary work folder default (see setupWorkFolder below).
pWorkDir = pTmpDir / 'work'

# Setup local logger
_logger = ads.logger('uiv.tst')


def setupLogger(name, level=ads.DEBUG, otherLoggers=None):
    """Create logger for tests and configure logging"""
    otherLoggers = otherLoggers or {'ads': ads.INFO2, 'ads.eng': ads.INFO, 'ads.exr': ads.INFO}
    for dLvl in [dict(name='matplotlib', level=ads.WARNING)] \
                + [dict(name=nm, level=lvl) for nm, lvl in otherLoggers.items()] \
                + [dict(name='uiv.tst', level=level)]:
        _ = ads.logger(dLvl['name'], level=dLvl['level'])

    return ads.logger(name, level)


def logPlatform():
    """Show testing configuration (traceability)"""
    _logger.info('Testing platform:')
    for k, v in ads.runtime:
        if k != 'pyaudisam':
            _logger.info(f'* {k}: {v}')
    _logger.info(f'PyAuDiSam {ads.__version__} from {pl.Path(ads.__path__[0]).resolve().as_posix()}')


def logBegin(what):
    """Log beginning of tests"""
    _logger.info(f'Testing pyaudisam: {what} ...')
    _logger.info('Current folder: ' + pl.Path().absolute().as_posix())
    _logger.info('Computation platform:')
    for k, v in ads.runtime.items():
        _logger.info(f'* {k}: {v}')


def logEnd(what, rc=None):
    """Log end of tests"""
    sts = {-1: 'Not run', 0: 'Success', None: None}.get(rc, 'Error')
    msg = f'see {pLogFile.as_posix()}' if sts is None else f'{sts} (code: {rc})'
    _logger.info(f'Done testing pyaudisam: {what} => {msg}.\n')


def setupWorkDir(dirName='work', cleanup=True):
    global pWorkDir
    pWorkDir = pTmpDir / dirName
    if cleanup:
        cleanupWorkDir()
    pWorkDir.mkdir(parents=True, exist_ok=True)


def cleanupWorkDir():
    if pWorkDir.is_dir():
        shutil.rmtree(pWorkDir, ignore_errors=True)  # Note: avoid any Ruindows shell or explorer inside this folder !


def sampleAbbrev(sSample):
    """Short string for sample 'identification'

    WARNING: Because we compare analysis results produced by API ou CLI runs
             to others produced by CLI or API, for simplicity, this function MUST be exactly the same
             as the one in valtests-de-params.py"""

    abrvSpe = ''.join(word[:4].title() for word in sSample['Espèce'].split(' ')[:2])
    sampAbbrev = '{}-{}-{}-{}'.format(abrvSpe, sSample.Passage.replace('+', ''),
                                      sSample.Adulte.replace('+', ''), sSample['Durée'])
    return sampAbbrev


def analysisAbbrev(sAnlys):
    """Short string for analysis 'identification'

    WARNING: Because we compare analysis results produced by API ou CLI runs
             to others produced by CLI or API, for simplicity, this function MUST be exactly the same
             as the one in valtests-de-params.py"""

    abbrevs = [sampleAbbrev(sAnlys)]
    abbrevs += [sAnlys['FonctionClé'][:3].lower(), sAnlys['SérieAjust'][:3].lower()]
    dTroncAbrv = {'l': 'TrGche' if 'TrGche' in sAnlys.index else 'TroncGche',
                  'r': 'TrDrte' if 'TrDrte' in sAnlys.index else 'TroncDrte',
                  'm': 'NbTrches' if 'NbTrches' in sAnlys.index else 'NbTrModel'
                                  if 'NbTrModel' in sAnlys.index else 'NbTrchMod',
                  'd': 'NbTrDiscr'}
    for trAbrv, name in dTroncAbrv.items():
        if name in sAnlys.index and not pd.isnull(sAnlys[name]):
            nmAbrv = sAnlys[name][0].lower() if isinstance(sAnlys[name], str) else str(int(sAnlys[name]))
            abbrevs.append(trAbrv + nmAbrv)

    return '-'.join(abbrevs)


def listUniqueStrings(re2FindAll, lines):
    """List unique values of strings matching in a set of lines
    with a given regexp pattern with 1 capturing () couple
    that defines the string values to search for and return through <pattern>.findall
    (ex: r'xyz([2-7e-p]+).')"""

    if isinstance(re2FindAll, str):
        re2FindAll = re.compile(re2FindAll)

    uniqStrs = []
    for unStrsInLine in [set(re2FindAll.findall(line)) for line in lines]:
        for strng in unStrsInLine:
            if strng not in uniqStrs:
                uniqStrs.append(strng)

    return uniqStrs


def selectLines(re2Search, lines):
    """Select lines matching (re.search) a regexp pattern"""

    if isinstance(re2Search, str):
        re2Search = re.compile(re2Search)

    return [line for line in lines if re2Search.search(line)]


def replaceStrings(froms, tos, lines):
    """Replace strings in text lines, inplace the list"""

    froms2Tos = dict(zip(froms, tos))
    repLines = 0
    for lineInd in range(len(lines)):
        line = lines[lineInd]
        for from_, to_ in froms2Tos.items():
            lines[lineInd] = lines[lineInd].replace(from_, to_)
        if line != lines[lineInd]:
            repLines += 1

    return repLines


def replaceRegExps(re2Search, repl, lines):
    """Replace subtrings matching a regexp pattern by another string, in text lines,
    inplace the list (the replacement can also be a function: see re.sub)"""

    if isinstance(re2Search, str):
        re2Search = re.compile(re2Search)

    repLines = 0
    for lineInd in range(len(lines)):
        line = lines[lineInd]
        lines[lineInd] = re2Search.sub(repl, line)
        if line != lines[lineInd]:
            repLines += 1

    return repLines


def removeLines(re2Search, lines):
    """Remove lines where a given regexp pattern is found (re.search), inplace the list"""

    if isinstance(re2Search, str):
        re2Search = re.compile(re2Search)

    ind2Remove = [ind for ind in range(len(lines)) if re2Search.search(lines[ind])]
    for ind in reversed(ind2Remove):
        del lines[ind]

    return len(ind2Remove)


def unifiedDiff(expectedLines, realLines, logger=None, subject='text'):
    """Run difflib.unified_diff on 2 text line sets and extract resulting diff blocks
    as a list of DotDict(startLines=DotDict(expected=<line number>, real=<line number>),
                         expectedLines=list(<expected lines>), realLines=list(<actual lines>))"""

    if logger:
        logger.info(f'Unified diff of {subject}:')

    blocks = []
    block = None
    for diffLine in difflib.unified_diff(expectedLines, realLines, n=0):

        if logger:
            logger.info(diffLine.rstrip('\n'))

        if diffLine[0] == '-':
            if diffLine.strip() == '---':
                continue
            block.expectedLines.append(diffLine[1:].rstrip('\n'))
            continue

        if diffLine[0] == '+':
            if diffLine.strip() == '+++':
                continue
            block.realLines.append(diffLine[1:].rstrip('\n'))
            continue

        if diffLine[0] == '@':
            if block:
                blocks.append(block)
            block = DotDict(startLines=DotDict(expected=-1, real=-1), expectedLines=[], realLines=[])
            fields = diffLine.split(' ')[1:-1]
            if ',' in fields[0]:
                fields[0] = fields[0].split(',')[0]
            block.startLines.expected = - int(fields[0])
            if ',' in fields[1]:
                fields[1] = fields[1].split(',')[0]
            block.startLines.real = int(fields[1])

    if block:
        blocks.append(block)

    return blocks


def checkAnalysisFolders(paths, anlysKind='analysis'):
    _logger.info(f'Checking {anlysKind} folders (minimal) ...')
    for path in paths:
        assert {fpn.name for fpn in pl.Path(path).iterdir()} \
               == {'cmd.txt', 'data.txt', 'log.txt', 'output.txt', 'plots.txt', 'stats.txt'}
    _logger.info(f'... done checking {anlysKind} folders (minimal).')
