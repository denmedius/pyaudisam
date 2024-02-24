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

import pathlib as pl

import pandas as pd

import pyaudisam as ads


# Useful test folders
pTestDir = pl.Path(__file__).parent
pRefInDir = pTestDir / 'refin'
pRefOutDir = pTestDir / 'refout'

# Temporary work folder.
pTmpDir = pTestDir / 'tmp'
pTmpDir.mkdir(exist_ok=True)

# Setup local logger
_logger = ads.logger('uiv.tst')


# Create logger for tests and configure logging.
def setupLogger(name, level=ads.DEBUG, otherLoggers={'ads': ads.INFO}):
    logLevels = [dict(name='matplotlib', level=ads.WARNING),
                 dict(name='ads', level=otherLoggers.get('ads', ads.INFO))] \
                + [dict(name=nm, level=lvl) for nm, lvl in otherLoggers.items() if nm != 'ads'] \
                + [dict(name='uiv.tst', level=level),
                   dict(name=name, level=level)]
    dateTime = pd.Timestamp.now().strftime('%Y%m%d%H%M')
    fpnLogFile = pTmpDir / f'{name}.{dateTime}.log'
    ads.log.configure(loggers=logLevels, handlers=[fpnLogFile], reset=True)

    return ads.logger(name)


# Show testing configuration (traceability).
def logPlatform():
    _logger.info('Testing platform:')
    for k, v in ads.runtime:
        if k != 'pyaudisam':
            _logger.info(f'* {k}: {v}')
    _logger.info(f'PyAuDiSam {ads.__version__} from {pl.Path(ads.__path__[0]).resolve().as_posix()}')


# Log beginning of tests
def logBegin(what):
    _logger.info(f'Testing pyaudisam: {what} ...')


# Log end of tests
def logEnd(what, rc=None):
    sts = {-1: 'Not run', 0: 'Success', None: None}.get(rc, 'Error')
    msg = 'see pytest report' if sts is None else f'{sts} (code: {rc})'
    _logger.info(f'Done testing pyaudisam: {what} => {msg}.')
