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

# Pytest configuration file for all automated unit, integration and validation tests

import sys
import time
import pathlib as pl
import typing
import pytest

# The test root folder.
pTestDir = pl.Path(__file__).parent

# Temporary work folder root.
pTmpDir = pTestDir / 'tmp'
pTmpDir.mkdir(exist_ok=True)

# Update PYTHONPATH for pyaudisam package to be importable.
sys.path.insert(0, pTestDir.parent.as_posix())

# Configure the logging system.
from pyaudisam import log

_logLevels = [dict(name='matplotlib', level=log.WARNING),
              dict(name='ads', level=log.INFO)]
_dateTime = time.strftime('%y%m%d.%H%M', time.localtime())
pLogFile = pTmpDir / f'pytest.{_dateTime}.log'
log.configure(loggers=_logLevels, handlers=[pLogFile], reset=True)

# A plugin to make test report available to fixture ini/finalisation
_phase_report_key = pytest.StashKey[typing.Dict[str, pytest.CollectReport]]()

@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):

    # Execute all other hooks to obtain the report object
    rep = yield

    # Store test results for each phase of a call, which can
    # be "setup", "call", "teardown"
    item.stash.setdefault(_phase_report_key, {})[rep.when] = rep

    return rep

# An auto-use function-scope fixture for logging its begin and end
_logr = log.logger('uiv.tst')

@pytest.fixture(autouse=True, scope='function')
def inifinalizeFunction(request):

    # Code that will run before the test function
    _logr.info(f'Starting {request.node.nodeid} ...')

    yield  # The test function will be run at this point

    # Code that will run after the test function
    report = request.node.stash[_phase_report_key]
    if report['setup'].failed:
        status = 'NOT SETUP'
        details = report['setup'].longreprtext + '\n'
    elif 'call' not in report:
        status = 'SKIPPED'
        details = ''
    else:
        status = report['call'].outcome.upper()
        details = report['call'].longreprtext + '\n'

    _logr.info(f'Done ({status}) with {request.node.nodeid}\n{details}')
