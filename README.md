# Python module for AUtomated DIstance SAMpling analyses

This module interfaces **distance sampling** analysis engines from [Distance software](https://distancesampling.org/), and possibly others in the future ; thus, it has been designed in order to make it easier :
* to run (in parallel) numerous analysis with many (many) parameter variants on many field observation samples
  (possibly using some optimisation techniques for automated computation of right and left distance truncations),
* to select the best analysis variant results through a mostly automated process, based on customisable statistical
  quality indicators,
* to produce partly customisable reports in spreadsheet (numerical results only) and HTML formats
  (more complete, with full-featured plots like in Distance, and more).

As for now, only the Windows MCDS.exe V6 engine and Point Transect analyses are supported.

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

`$ pip install pyaudisam`

TODO: Publish also on [Conda Forge](https://conda-forge.org/), probably following
      [this recipe](https://jacobtomlinson.dev/posts/2020/publishing-open-source-python-packages-on-github-pypi-and-conda-forge/#conda-forge).  

## Usage

As a python package, **pyaudisam** can be used through its API : for the moment, you can have an idea of how to use 
it by playing with the fully functional [jupyter](https://jupyter.org/) notebook [tests/valtests.ipynb](./tests/valtests.ipynb) (see below [Running tests](#running-tests) for how to obtain and run it).

But there's also a command-line interface: run it with the -h/--help option, and read the quick-start guide (see below).

`python -m pyaudisam --help`

## Documentation

* a short ["how it works" guide](./docs/how-it-works/how-it-works-en.md) to understand the basics (also in [French](https://github.com/denmedius/pyaudisam/tree/main/docs/how-it-works/how-it-works-fr.md)),
* a concrete ["quick-start" guide](./docs/howto-acdc19-nat/howto.md) with a real life use case and relevant field data to play with,
* another similar but shorter concrete ["quick-start guide" (in French)](https://sylbor63.pagesperso-orange.fr/ds/acdc19/materiau-public.zip) with the full field data set of the "ACDC 2019" birding study.

TODO:
* complete the quick start guides above by other small and focused articles to explain some mandatory details:
  - how to build a sample or analysis specification workbook (see a French language first draft in [analyser.py:273](./pyaudisam/analyser.py)),
  - ...
* write a technical documentation of the whole module,
* write a guide for building the module API documentation ([sphinx](https://www.sphinx-doc.org/) should work out of the box as [reStructured text](https://en.wikipedia.org/wiki/ReStructuredText) has been used in docstrings),

## Running tests

You first need to clone the [source tree](https://github.com/denmedius/pyaudisam) or download and install
a [source package](https://pypi.org/project/pyaudisam/#files): once done, look in the _tests_ sub-folder, 
everything's inside :
* some tests are fully automated : after installing pytest, simply run it:

  `pytest`

* some other tests not: they are implemented as [jupyter notebooks](https://jupyter.org/) (see [tests/unintests.ipynb](./tests/unintests.ipynb) and [tests/unintests.ipynb](./tests/unintests.ipynb']) that you must run step by step (as long as no one has fully automated them :-).

## Building

To build pyaudisam [PyPI](https://pypi.org/project/pyaudisam/) source and binary packages, you first need to clone the [source tree](https://github.com/denmedius/pyaudisam) or download and install a [source package](https://pypi.org/project/pyaudisam/#files): once done, it's as simple as:

`python -m build`

Note: Don't care about warnings about pyaudisam.mcds and pyaudisam.report being recognised as importable,
 but being absent from setuptools' packages configuration ... these folders simply contain
 pyaudisam config. and data files, no python code at all.

## Contributing

Merge requests are very welcome !

And if you are lacking ideas, here are some good ones ;-)

### To do list

* finish tests automation (move [tests/unintests.ipynb](./tests/unintests.ipynb) and [tests/unintests.ipynb](./tests/unintests.ipynb') notebooks code to pytest scripts),
* make pyaudisam work under Linux / Mac OS (all python: OK, but calling MCDS.exe):
  - or: through some kind of external client-server interface to MCDS.exe (that runs only under Windows),
  - or: by porting MCDS to Linux (closed Fortran source, but old, so might be obtained through a polite request ; BUT, needs an IMSL license, which is horribly expensive).
  - or: by rewriting MCDS from scratch,
  - or: by rewriting MCDS using the [MRDS Distance package](https://distancesampling.org/), meaning so kind of interface to R,
* build a GUI for pyaudisam command-line,
* add support line transects (only point transects for the moment),
* add support for the co-variates feature of MCDS,
* ...

### Some hints

Some formal things that won't change as long as I'm the main maintainer :-)
* this code is not blacked or isorted or fully conform to pep8,
* the identifier naming scheme used is old-fashioned: camel case everywhere.
