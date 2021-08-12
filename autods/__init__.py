# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment

# Requirements:
#  python 3.8+
#  pandas 0.25+
#  matplotlib 3.1+
#  jinja2
#  zoopt 4.0+
#  ... to be completed.

# Module version
__version__ = 0.9


# Sub-module exports
from . import log
from .log import logger, DEBUG, DEBUG0, DEBUG1, DEBUG2, DEBUG3, DEBUG4, \
                         INFO,  INFO0,  INFO1,  INFO2,  INFO3,  INFO4, \
                         WARNING, ERROR, CRITICAL

from .data import DataSet, FieldDataSet, MonoCategoryDataSet, SampleDataSet, ResultsSet

from .engine import DSEngine, MCDSEngine

from .executor import Executor

from .analysis import DSAnalysis, MCDSAnalysis, MCDSPreAnalysis

from .analyser import Analyser, DSAnalyser, MCDSAnalyser, MCDSAnalysisResultsSet, MCDSPreAnalyser

from .optimisation import Interval, DSOptimisation, MCDSTruncationOptimisation, \
                          MCDSZerothOrderTruncationOptimisation

from .optimiser import DSParamsOptimiser, MCDSTruncationOptimiser, MCDSZerothOrderTruncationOptimiser

from .optanalyser import MCDSTruncationOptanalyser, MCDSTruncOptanalysisResultsSet

from .report import MCDSResultsPreReport, MCDSResultsFullReport

