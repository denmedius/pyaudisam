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

# Module version
__version__ = '0.9.3rc'

import os
import sys
import platform


# Infos about run-time (Python version, dependency library versions, ... + also updated by sub-modules)
_osEdition = (' ' + platform.win32_edition()) if sys.platform.startswith('win32') else ''
runtime = dict(os=f'{platform.system()}{_osEdition} {platform.version()} ({platform.architecture()[0]})',
               processor=f'{platform.processor()}, {os.cpu_count()} CPUs',
               python=f'{sys.implementation.name} ({sys.platform}) R{sys.version}')

# Transparent sub-module exports (in order not to care about them, and only import the top = pyaudisam package one)
from . import log
from .log import logger, DEBUG, DEBUG0, DEBUG1, DEBUG2, DEBUG3, DEBUG4, DEBUG5, DEBUG6, DEBUG7, DEBUG8, \
                         INFO,  INFO0,  INFO1,  INFO2,  INFO3,  INFO4, INFO5,  INFO6,  INFO7,  INFO8, \
                         WARNING, ERROR, CRITICAL

from .data import DataSet, FieldDataSet, MonoCategoryDataSet, SampleDataSet, ResultsSet

from .engine import DSEngine, MCDSEngine

from .executor import Executor

from .analysis import DSAnalysis, MCDSAnalysis, MCDSPreAnalysis

from .analyser import Analyser, DSAnalyser, MCDSAnalysisResultsSet, MCDSAnalyser, \
                      FilterSortSchemeIdManager, MCDSPreAnalysisResultsSet, MCDSPreAnalyser

from .optimisation import Interval, DSOptimisation, MCDSTruncationOptimisation, \
                          MCDSZerothOrderTruncationOptimisation

from .optimiser import DSParamsOptimiser, MCDSTruncationOptimiser, MCDSZerothOrderTruncationOptimiser

from .optanalyser import MCDSTruncationOptanalyser, MCDSTruncOptanalysisResultsSet

from .report import MCDSResultsPreReport, MCDSResultsFullReport, MCDSResultsFilterSortReport

from .utils import loadPythonData

# Sort and update runtime last bits
runtime.update(pyaudisam=f'{__version__}')
runtime.update({'DS engine': runtime.pop('DS engine')})