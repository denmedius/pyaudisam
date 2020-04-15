# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment

# Require:
#  python 3.7
#  pandas 0.25
#  matplotlib 3.1
#  jinja2
#  zoopt
#  ... to be completed.

# Module version
__version__ = 0.9

# Module exports
from .engine import DSEngine, MCDSEngine
from .executor import Executor
from .analysis import MCDSAnalysis, MCDSPreAnalysis
from .analyser import DSAnalyser, MCDSAnalyser #, MCDSPreAnalyser, MCDSParamsOptimiser
from .data import DataSet, FieldDataSet, IndividualsDataSet, SampleDataSet, ResultsSet, MCDSResultsSet
from .report import MCDSResultsPreReport, MCDSResultsFullReport
