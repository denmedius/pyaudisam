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

# Complementary automated unit and integration tests for "utils" and "logger" submodules
# (some bits for improving code coverage ; see other test_* modules for the remainder)

# To run : simply run "pytest" and check standard output + ./tmp/unt-ans.{datetime}.log for details

import pytest

import pyaudisam as ads
from pyaudisam import utils

import unintval_utils as uivu


# Mark module
pytestmark = pytest.mark.unintests

# Setup local logger.
logger = uivu.setupLogger('unt.utl', level=ads.DEBUG8)

KWhat2Test = 'utils, logger'


###############################################################################
#                         Actions to be done before any test                  #
###############################################################################
def testBegin():
    uivu.logBegin(what=KWhat2Test)


###############################################################################
#                                Test Cases                                   #
###############################################################################

def testUtils():

    # loadPythonData: path without suffix + non-existing file
    inPath = 'path/to/non-existing-data-file'
    path, data = utils.loadPythonData(inPath)

    assert path.as_posix() == inPath + '.py'
    assert data is None

    logger.info0('PASS testUtils: loadPythonData')


def testLogger():

    # .info<all levels>
    logger.info0('Logger test message: level=INFO0')
    logger.info1('Logger test message: level=INFO1')
    logger.info2('Logger test message: level=INFO2')
    logger.info3('Logger test message: level=INFO3')
    logger.info4('Logger test message: level=INFO4')
    logger.info5('Logger test message: level=INFO5')
    logger.info6('Logger test message: level=INFO6')
    logger.info7('Logger test message: level=INFO7')
    logger.info8('Logger test message: level=INFO8')

    # .debug<all levels>
    logger.debug0('Logger test message: level=DEBUG0')
    logger.debug1('Logger test message: level=DEBUG1')
    logger.debug2('Logger test message: level=DEBUG2')
    logger.debug3('Logger test message: level=DEBUG3')
    logger.debug4('Logger test message: level=DEBUG4')
    logger.debug5('Logger test message: level=DEBUG5')
    logger.debug6('Logger test message: level=DEBUG6')
    logger.debug7('Logger test message: level=DEBUG7')
    logger.debug8('Logger test message: level=DEBUG8')

    logger.info0('PASS testLogger')


###############################################################################
#                         Actions to be done after all tests                  #
###############################################################################
def testEnd():
    uivu.logEnd(what=KWhat2Test)
