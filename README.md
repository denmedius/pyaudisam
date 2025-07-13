# PYthon module for AUtomated DIstance SAMpling analyses

This module interfaces **distance sampling** analysis engines from [Distance software](https://distancesampling.org/), and possibly others in the future ; thus, it has been designed in order to make it easier :
* to run (in parallel) numerous [Distance Sampling](https://en.wikipedia.org/wiki/Distance_sampling) analyses with many (many) parameter variants on many field observation samples
  (possibly using some optimisation techniques for automated computation of right and left distance truncations),
* to select the best analysis variant results through a mostly automated process, based on customisable statistical
  quality indicators,
* to produce partly customisable reports in spreadsheet (numerical results only) and HTML formats
  (more complete, with full-featured plots like in Distance, and more).

As for now, only the Windows MCDS.exe 6.x (Distance 6 to 7.3) and 7.4 (Distance 7.4 and 7.5 at least) engine and Point Transect analyses are supported, and so, it runs only under Windows.

## Requirements

The module itself was actually tested extensively with:
* python 3.12.3
* numpy 1.26.4
* pandas 2.2.2
* openpyxl 3.1.2
* xlrd 2.0.1 (only for .xls format support)
* odfpy 1.4.1
* jinja2 3.1.4
* matplotlib 3.8.4
* packaging 24.0
* zoopt 0.4.2

It probably works as is with earlier versions, but not below python 3.9 as specified in [setup.py](https://github.com/denmedius/pyaudisam/blob/main/setup.py) and pandas 2.1 (but you'll need to run the whole test suite first to make sure).

If you need Python 3.8 compatibility, you can:
* or: use the 1.1.0 release (but you'll be limited to pandas 1.x),
* or: tweak the latest source release, but at your own risks, because for sure you'll have to do some fixes (hint: run the whole test suite to see what's happening).

As for testing dependencies:
* pytest, pytest-cov,
* plotly (sometimes, in old notebooks).

## Installation

You can install **pyaudisam** from [PyPI](https://pypi.org/project/pyaudisam/) in your current python environment (conda or venv, whatever):

`pip install pyaudisam`

Or from a downloaded source package:

`pip install pyaudisam-1.2.0.tar.gz`

Or from a downloaded wheel package:

`pip install pyaudisam-1.2.0-py3-none-any.whl`

Or even directly from GitHub:

* `pip install git+https://github.com/pypa/sampleproject.git@1.2.0`
* `pip install git+https://github.com/pypa/sampleproject.git@main`

## Usage

As a **python package**, pyaudisam can be used through its python API.

But there's also a **command-line interface**: try and run it with the -h/--help option.

`python -m pyaudisam --help`

Whichever method, the best way to go is to read the concrete quick-start guide : see [Documentation](#documentation) below
(but be aware that you'll need to install an external Distance Sampling engine, like MCDS, to run analyses with paudisam).

## Documentation

* a short ["how it works" guide](https://github.com/denmedius/pyaudisam/blob/main/docs/how-it-works/how-it-works-en.md) to understand the basics (also in [French](https://github.com/denmedius/pyaudisam/blob/main/docs/how-it-works/how-it-works-fr.md)),
* a concrete ["quick-start" guide](https://github.com/denmedius/pyaudisam/blob/main/docs/howto-acdc19-nat/howto.md) with a real life use case and relevant field data to play with,
* another similar but shorter concrete ["quick-start guide" (in French)](http://jpmeuret.free.fr/ds/acdc19/materiau-public.zip) (command-line only) with the full field data set of the "ACDC 2019" birding study.

Note: You can also get a detailed idea of how to use pyaudisam python API by playing with the fully functional [jupyter](https://jupyter.org/) notebook [tests/valtests.ipynb](https://github.com/denmedius/pyaudisam/blob/main/tests/valtests.ipynb) (see below [Running tests](#running-tests) for how to obtain and run it).

## Testing

You first need to clone the [source tree](https://github.com/denmedius/pyaudisam) or download and install a [source package](https://pypi.org/project/pyaudisam/#files): once done, look in the _tests_ sub-folder, everything's inside.

Then, you need to install test dependencies:

`pip install pyaudisam[test]`

Some tests are fully automated, simply run:

`pytest`

For code coverage during tests, simply run:

`pytest --cov`

Or even, if you want an HTML report with annotated code coverage:

`pytest --cov --cov-report html`

Notes:
* With pandas 2.2, you'll face the following warning a few times when running some tests: as the future behaviour is unknown, we don't know how to fix this, so we left things as is => when using later versions of pandas, when the current behavior is really deprecated, you may have to fix things yourself.
  ```
  FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.
  In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes.
  To retain the old behavior, exclude the relevant entries before the concat operation.
  ```
* All the test suite has been fully automated (even if not 100% covering code) ; but the old test suite implemented as [jupyter notebooks](https://jupyter.org/) is still available (see [tests/unintests.ipynb](https://github.com/denmedius/pyaudisam/blob/main/tests/unintests.ipynb) and [tests/valtests.ipynb](https://github.com/denmedius/pyaudisam/blob/main/tests/valtests.ipynb)).

## Building

To build pyaudisam [PyPI](https://pypi.org/project/pyaudisam/) source and binary packages, you need:
* a source tree (clone the [source tree](https://github.com/denmedius/pyaudisam) or download and extract a [source package](https://pypi.org/project/pyaudisam/#files)),
* a python environment where pyaudisam works,
* the `build` module (to install through pip as an example).

Then, it's as simple as:

`python -m build`

You'll get 2 files in the `dist` folder (ex. for version 1.2.0) :
* the wheel package: `pyaudisam-1.2.0-py3-none-any.whl`
* the source package: `pyaudisam-1.2.0.tar.gz`

## Contributing

Merge requests are very welcome !

And if you are lacking ideas, here are some good ones below ;-)

### To do list

* documentation:
  * complete the quick start guides above by other small and focused articles to explain some mandatory details:
    * how to build a sample or analysis specification workbook (see a short draft in [analyser.py:273](https://github.com/denmedius/pyaudisam/blob/main/pyaudisam/analyser.py)),
    * ...
  * write a technical documentation of the whole module and sub-modules,
  * write a guide for building the module API documentation ([sphinx](https://www.sphinx-doc.org/) should work out of the box as [reStructured text](https://en.wikipedia.org/wiki/ReStructuredText) has been used in docstrings),
* code quality and tests:
  * add more tests for improving code coverage (thanks to HTML coverage report),
  * configure and run pylint, and follow its useful advices, 
  * main: split \_Application.\_run in feature sub-functions for clarity,
* features:
  * add support for line transects (only point transects for the moment),
  * add support for the co-variates feature of MCDS,
  * integrate the notebook prototype of "final reports" (workbook, HTML, and OpenDoc text formats) to automate most of the work of producing a publication-grade "full results appendix" for a Distance Sampling study (based on the auto-filtered report, but with semi-automated diagnosis at sample and analysis level in order to help in the final choice for each sample),
  * add more features for selecting sample data before running analyses (to avoid the need of creating multiple data sets, run multiple analysis sessions, and then re-aggregate results and reports manually): exclude some specific transects, pre-truncate data above some fixed distance, ...
* packaging:
  * publish also pyaudisam on [Conda Forge](https://conda-forge.org/), probably following [this recipe](https://jacobtomlinson.dev/posts/2020/publishing-open-source-python-packages-on-github-pypi-and-conda-forge/#conda-forge),
* platform support:
  * make pyaudisam work under Linux / macOS (all python: OK, but ... calling MCDS.exe, that runs exclusively under Windows):
    * or: through some kind of external client-server interface to MCDS.exe (that runs only under Windows),
    * or: by porting MCDS to Linux (closed Fortran source, but old, so might be obtained through a polite request to [this Distance Sampling forum](https://groups.google.com/g/distance-sampling) ;
      BUT, you'll need an IMSL license, which is horribly expensive).
    * or: by rewriting MCDS from scratch, or by porting the [MRDS Distance package](https://distancesampling.org/) to Python,
    * or: by rewriting MCDS using the [MRDS Distance package](https://distancesampling.org/), meaning some kind of interface to R,
* user interface:
  * build a GUI for pyaudisam command-line (with some kind of "project" concept, and parameter set template, and ...),
* ...

### Known issues

* AnalysisResultsSet.toOpenDoc sometimes produces broken header rows (the 3-row multi-index header is not rendered as it is on the right side),
* The new undocumented MCDS 7.4 result column names are not translated correctly (switched fr and en translations) (minor, as not used actually for the moment),
* The "Details" table header is not translated in auto-filtered reports (whereas the "Synthesis" one is),
* Too many decimals rendered for the Max/Min dist figures in HTML reports when the distance unit is "meter",
* The colorisation of HTML and workbook auto-filtered reports is of no use as it is now (need for a full rework).

### Release notes

You can [read them here](https://github.com/denmedius/pyaudisam/blob/main/docs/release-notes.md) :-)

### Some hints

Some formal things that I don't plan to change (let's concentrate on substantive content) :-)
* this code is not blacked or isorted or fully conform to pep8 (but it's clean, commented, and it works),
* the identifier naming scheme used is old-fashioned: camel case everywhere.
