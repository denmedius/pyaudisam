# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment

# Require:
#  python 3.6
#  pandas 0.25
#  matplotlib 3.1
#  jinja2
#  ... to be completed.

# Module version
__version__ = 0.9

# Module exports
from .engine import MCDSEngine
from .analysis import MCDSAnalysis#, MCDSPreAnalyser, MCDSAnalyser
from .data import SampleDataSet, MCDSResultsSet#, FieldDataSet, AnalysesSpecsSet
from .report import MCDSResultsPreReport, MCDSResultsFullReport