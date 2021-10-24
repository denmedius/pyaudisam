# Python module for AUtomated DIstance SAMpling analyses

This is module interfaces **distance sampling** analysis engines from [Distance software](http://distancesampling.org/),
and possibly others in the future.

It is intended for making it easier :
* to run numerous analysis with many (many) parameter variants on many field observation samples
  (possibly using some optimisation techniques for automated computation of right and left distance truncations),
* to select the best analysis variant results through a mostly automated process, based on customisable statistical quality indicators,
* to produce partly customisable reports in spreadsheet format (numerical results only), and in HTML format (more complete, with full-featured plots like in Distance, and more).

As for now, only the Windows MCDS.exe engine and Point Transect analyses are supported.

## Warning !

This is **work-in-progress**,
* not yet usable without good python skills and Distance Sampling
  through [Distance software](http://distancesampling.org/) knowledge,
* the mostly automated filtering of numerous analysis results is not yet fully implemented !

So you'd better wait for version 1.0.0 that will come with more (!) documentation and some real life examples :-)

## Requirements

* python 3.8+
* pandas 0.25+
* matplotlib 3.1+
* jinja2 2.10+
* zoopt 0.4+

## Installation

You can install **pyaudisam** from [PyPI](https://pypi.org/project/pyaudisam/):

$ pip install pyaudisam

## Usage

For the moment, you can have an idea of how to use it by running tests/valtests.ipynb notebook :-)
