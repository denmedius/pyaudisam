PyAuDiSam release notes
=======================

# 1.1.0 (may 2024)

* full non-regression pytest-automated suite (90% coverage),
* added support for MCDS 7.4 / Distance 7.5 (backward-compatible with MCDS 6.2 / Distance 6.x - 7.3),
* easier customization of MCDS detection through MCDS_PATH environment variable,
* logging of actual MCDS version used (rather than simply its installation folder),
* improved packaging (add test dependencies, fix package building warnings),
* fixed analysis results column set possibly varying with analysis params specs, due to auto-cleanup of all-empty columns by DataSet.dfData,
* minor fixes and improvements:
  - fixed AFS-Steps sheet index in auto-filtered workbook reports for easier loading through pandas,
  - moved most of \__main__ code to a new 'main' module, to make it more easily auto-testable,
  - improved configuration and documentation about code coverage measurements during pytests,

# 1.0.2 (February 2024)

* added MCDSPreAnalyser.computeSampleStats() to compute distance stats for samples to pre-analyse,
* fixed BalQual1 indicator computation (has been failing systematically for a long time !?),
* start again porting the notebook-based test suite to pytest (~20 % done),
* minor fixes and improvements to reports,
* stricter requirements.txt to reflect actually tested software platform.

# 1.0.1 (May 2023)

* new how-to and how it works documentation with concrete runnable scenarios and relevant field data.
* command line mode minor fixes,
* improved packaging.

# 0.9.3 (January 2022)

* improvements, fixes and non-regression tests (notebook) for auto-filtered reports,
* new command line mode (__main__ sub-module),
* improved packaging.

# 0.9.1 (October 2021)

The first really working version with its initial features:
* point transect Distance Sampling withMCDS 6.2 under Windows 7+,
* automated parallel run and reporting of multiple combinations of samples and analysis parameters,
* optional auto-determination of the left and right truncation distances (through a parallelized optimisation system),
* automated filtering and sorting of results for easier choice of the right analysis to select (among the numerous variants run) at the end for each sample.
