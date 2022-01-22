# Python module for AUtomated DIstance SAMpling analyses

This is module interfaces **distance sampling** analysis engines from [Distance software](http://distancesampling.org/),
and possibly others in the future.

It is intended for making it easier :
* to run numerous analysis with many (many) parameter variants on many field observation samples
  (possibly using some optimisation techniques for automated computation of right and left distance truncations),
* to select the best analysis variant results through a mostly automated process, based on customisable statistical
  quality indicators,
* to produce partly customisable reports in spreadsheet format (numerical results only), and in HTML format
  (more complete, with full-featured plots like in Distance, and more).

As for now, only the Windows MCDS.exe engine and Point Transect analyses are supported.

## Warning !

This is **work-in-progress**,
* not yet usable without good python skills and Distance Sampling
  through [Distance software](http://distancesampling.org/) knowledge,
* not yet documented (even if the code _is_), no real example available.

So you'd better wait for version 1.0.0 that will come with more documentation and some real life examples :-)

## Requirements

The module itself:
* python 3.8+
* pandas 0.25+
* matplotlib 3.1+
* jinja2 2.10+
* zoopt 0.4+

Tests:
* pytest
* plotly (sometimes)

## Installation

You can install **pyaudisam** from [PyPI](https://pypi.org/project/pyaudisam/)
in your current python environment (conda or venv, whatever):

  <code>
  $ pip install pyaudisam
  </code>

TODO: Publish also on [Conda Forge](https://conda-forge.org/), probably following
      [this recipe](https://jacobtomlinson.dev/posts/2020/publishing-open-source-python-packages-on-github-pypi-and-conda-forge/#conda-forge).  

## Usage

As a python package, **pyaudisam** can be used through its API : for the moment, you can have an idea of how to use 
it by playing with the fully functional [jupyter notebook](https://jupyter.org/) 'tests/valtests.ipynb' (see below 
Running tests for how to obtain and run it).

But there's also a command-line interface: run it with the -h/--help option ...

  <code>
  $ python -m pyaudisam --help
  </code>

## Documentation

TODO:
* a concrete quick-start guide with a real life use case and relevant data to play with,
* a guide for building the module API documentation ([sphinx](https://www.sphinx-doc.org/) should work out of the box
  as [reStructured text](https://en.wikipedia.org/wiki/ReStructuredText) has been used in docstrings),

## Running tests

You first need to clone the [source tree](https://github.com/denmedius/pyaudisam) or download and install
a [source package](https://pypi.org/project/pyaudisam/#files): once done, look in the 'tests' sub-folder, 
everything's in :
* some tests are fully automated : after installing pytest, simply run it:

  <code>
  $ pytest
  </code>

* some other tests not: they are implemented as [jupyter notebooks](https://jupyter.org/) that you must run step by 
  step (as long as no one has fully automated them :-)
