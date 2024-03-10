Distance Sampling with pyaudisam
--------------------------------

* [pyaudisam@github.com](https://github.com/denmedius/pyaudisam)
* [pyaudisam@pipy.org](https://pypi.org/project/pyaudisam/)
* Author: Jean-Philippe Meuret <fefeqe22.vucuqu82 (at) murena.io>


# I. Purpose

Show how to use **pyaudisam** for:
* running many simple [**conventional Distance Sampling**](http://distancesampling.org/) analyses (with MCDS V6),
* on a real life field survey dataset,
* using as many combination / variants of analysis parameters as needed (without fearing the number: semi-automated post-processing :-),
* with possibly automated truncation distance determination (through an optimisation technique),
* automatically sort and filter the bunch of results in order to find the "best" one(s), using an algorithm that mimics the way human analysts work,
* automatically build relevant Excel and HTML reports (with everything like Distance software, and more).

The dataset used here is an anonymized extract from a **2019 point transect birding field survey**, for which **2 articles** have been **published** in "Le Grand-Duc" French ornithology periodical of the "Ligue pour la Protection des 
Oiseaux" (aka "LPO"), Auvergne region, France :
1. MEURET J.-P., GUELIN F., 2023. - L'avifaune de plateaux agricoles granitiques au sud de Clermont-Ferrand : estimation des populations d'oiseaux nicheurs communs au printemps 2019 par la méthode du Distance Sampling et comparaison de plusieurs méthodes et durées d'inventaire. Le Grand-Duc, 91 : 27-83,
2. DULPHY J.-P., GUELIN F., 2023. - L’avifaune de plateaux agricoles granitiques au sud de Clermont-Ferrand : trente ans après, estimation des populations de quelques espèces des années 1990 par Distance Sampling et Indices Ponctuels d'Abondance. Le Grand-Duc, 91 : 84-96.

These 2 articles can be **freely downloaded** from the [Faune AuRA bio-database and website](https://www.faune-aura.org/index.php?m_id=20283).

No more than an anecdote, but we called this ornithological study "ACDC 2019" because :
* of course, it was run in 2019,
* the surveyed area is located some kilometers on the south-east of another area that we had surveyed for bird again a few years before,
* this previous surveyed area had a square shape,
* so we named the 2019 study area "the other right beside square", that is in French "L'Autre Carré D'à Côté" (even if the new area is quite far from a square shape ;-),
* so ... nothing to do with the well-known hard rock band of the 70s and 80s: if you feel disappointed, you can stop reading here and get back to your daily noisy routine ;-).

The **dataset** is actually an **extract**, for lighter and faster computations:
* from the full **"Naturalist"** dataset, where field surveys have been run using the ["Naturalist" smartphone app.](https://play.google.com/store/apps/details?id=ch.biolovision.naturalist)
* where only 4 species have been kept.

Here, we will demonstrate how to use pyaudisam in order to :
* run a set of specified Distance Sampling pre-analyses (as a first Distance Sampling glance to the data set),
* run Distance Sampling analyses using multiple combinations of analysis parameters, possibly with automated left and right truncation distance determination (through an optimisation technique),
* same, but without distance truncation optimisation,
* automatically post-process this bunch of results to filter the best analyses to be kept at the end for each sample (with the opportunity at the end for the user to check and fix if needed the automatic filtering),
* export 1 file for each specified sample, ready to import into [Distance Software 6+](http://distancesampling.org/), for any "manual" analysis in order to check some more analysis parameter combination not yet explored through pyaudisam.

And this demonstration will use pyaudisam in 2 ways:
* as a **command line tool**, in a terminal,
* as a **python module**, through a python notebook.

Note: You can download the whole anonymized dataset of the ACDC 2019 study from [this place](http://jpmeuret.free.fr/ds/acdc19/materiau-public.zip):
* with all the bird species this time (not only 4 of them),
* with the "Papyrus" sub-dataset: sighting that were collected on the same point transects by a second birder team, using the same protocol but this time by recording the sightings through pen and paper, on a satellite photo print for each point transect (we wanted to compare survey results with this good old method ... to the new "Naturalist smartphone app." one ... see articles about that above),
* with a concatenated dataset (Naturalist + Papyrus),
* the instructions for using pyaudisam to do DS analyses for these 3 sub-datasets are in the `a-lire-dabord.md` file (French language),
* the study settings python script is a bit more sophisticated, making use of the special parameter `-s` option.


# II. Provided files

* [howto.md](./howto.md): This document !
* [ACDC2019-108-points-300m-circles.kml](./ACDC2019-108-points-300m-circles.kml): 
  Geolocations of the 108 surveyed point transects, with the limits of the covered area (geoportail.gouv.fr or google-earth can import and display it),
* [ACDC2019-Nat-ObsIndivDist.xlsx](./ACDC2019-Nat-ObsIndivDist.xlsx): The bird sightings collected on the field through the Naturalist smartphone app. by a 10 member team (each row describes 1 individual bird, and especially gives its estimated distance from the birder) ... that is : the dataset !
* [ACDC2019-Samples.xlsx](./ACDC2019-Samples.xlsx): The specification of the set of samples to be pre-analysed (here, it's some kind of implicit specification, that needs to be explicitated by pyaudisam in order to get the full and explicit list of thus defined combinations ; the columns of each sheet are used as lists of possible values for the so-named "property" of the samples, and the explicitation process actually produces all the possible combinations listed ; pyaudisam also supports explicit specification sheets, and a mix of the 2 kinds also ; see below for the result of this explicitation),
* [ACDC2019-OptAnalysesToDo.xlsx](./ACDC2019-OptAnalysesToDo.xlsx): The specification of the set of DS analyses (with truncation distance parameters automatically computed for some analyses ; same implicit specification of analyses parameter combinations as for the samples just above, with also explicit specification for other combinations ... pyaudisam knows to explicitate it all ... see below for the result of this process),
* [acdc-2019-nat-ds-params.py](./acdc-2019-nat-ds-params.py): The set of pyaudisam parameters to use for running the example analyses (... etc) ; roughly speaking, if there were some kind of GUI for pyaudisam, part of these would be global settings with default values, and the remainder would be properties for some kind of "study" concept in this GUI (some related to the input data set, some to samples to analyse and some to the analyses to be run) ; these parameters can be used with the command line or with the provided example jupyter notebook,
* [acdc-2019-nat-ds-run.ipynb](./acdc-2019-nat-ds-run.ipynb): The example [jupyter](https://jupyter.org/) notebook demonstrating how to use pyaudisam as a python package and run just the same as through the command line.

Note: If you have downloaded a source package of pyaudisam, you can find these files in the `docs\howto-acdc19-nat` extract sub-folder (otherwise, you can download them one by one by clicking on to their link above, and then on the "Download raw file" button (top right).

# III. Now, let's go !

## A. Pre-requisites

We highly recommend you read first the [short "how it works" guide](../how-it-works/how-it-works-en.md) in order to understand better what's not explicitly explained below :-)

Warning: Only works under Windows (10 or newer, probably 7 and 8 also).

1. Download and install "Distance for Windows" release 6 or newer: see the [distancesampling.org](http://distancesampling.org/)
2. Download and install [Python 3.8](https://www.python.org/downloads/) (not tested with newer release, may be some issues to be expected)
3. Install pyaudisam 1.0.0 or newer (with its dependencies) : see [instructions at pypi.org](https://pypi.org/project/pyaudisam/)
4. Open a Windows terminal (bash: through https://gitforwindows.org/ or WSL ; Cmd also works, and PowerShell too)
5. cd to the folder of this very document
6. If needed, activate the python environnement where you just installed pyaudisam and its dependencies
7. Check pyaudisam installation : the following command should dump some basic help about pyaudisam command line options into the terminal (and also other technical information)

    `python -m pyaudisam -h`

Note: As an alternative to installing pyaudisam in your python environment (through conda or pip) after activating it, you can also apply the commands of the following chapters (just as they are, no change) after first:
1. extracting a source package of pyaudisam 1.0.0 or newer after downloading it from [pypi.org](https://pypi.org/project/pyaudisam/),
2. set the PYTHONPATH env. variable to the absolute path of the root folder of the extracted tree (example: /c/my/dev/pyaudisam-1.0.0 for ... the 1.0.0 release).


## B. pyaudisam as a command line tool

### 1. Background and tips

For running any pyaudisam processing on the "ACDC 2019 Nat" dataset, you'll always use a **command line starting by**:

`python -m pyaudisam -p acdc-2019-nat-ds-params.py -w <work folder>` ...

* `-p acdc-2019-nat-ds-params.py` gives to pyaudisam the **standard and study-specific settings** to use (it's a python script),
* `-w <work folder>` gives to pyaudisam the **work folder**, i.e. the folder to produce output file into,
* not used here, but you could also use the `-s key1=value1, key2=value2, ...` option to give pyaudisam "special parameters" to be used to customise "from outside" what's happening inside the script given to pyaudisam through the `-p` option (for some kind of genericity, may be for sub-studies, or ...) .

Whatever the command line, if the **`-u` flag option** is not there, pyaudisam will actually produce nothing, apart from a simple console report to explain what it should have done if ... `-u` had been there ; a handy way of checking before jumping in !

The "work folder" given to pyaudisam through the `-w` option can be used in 2 ways:
* as is, if the **`-n` flag option** is used : all the produced files will go in the given work folder, and may be some already present files (produced by a previous command with the same target work folder) will be used as input ; in this case, you may use a single folder, or a folder and sub-folder couple (like acdc19/preanlys),
* if the `-n` flag option is not used, as the root folder for an automatically date-and-time-stamped sub-folder created on the go for pyaudisam output at command run time ; the auto-generated date and time stamp is of the form '[0-9]{6}-[0-9]{6}' (regular expression syntax),
* except when the work folder has itself this form `.+/.*[0-9]{6}-[0-9]{4,6}$`: in this case, it is used as is, just as if the `-n` flag option was present in the command line,
* note: whatever happen, if `-u` is used, the work folder, and sub-folder if any, are automatically created if not there.

So, repeated execution of pyaudisam commands using always the same work folder (`-w` option) **won't overwrite or mix-up** their output files, unless:
* you use the `-n` flag option,
* you use a date and time stamped work sub-folder.
And then, when overwriting / mixing-up, if all is about the same analyses / samples, the results would be simply updated, just like the report ; but re-running analyses will also generate a new whole set of special sub-folders, the previous subset becoming then useless and abandoned ... but you can well clean up things by removing them afterwards using their creation time as a selection criterium :-)

Note that this last case is useful for generating reports from the analysis results produced by a previous command !

Last: In order to get more details in the console during execution (will it be fake, without the `-u` flag option, or real, with it), use the `-v` flag option (verbose mode).


### 2. Run pre-analyses for checking samples

As a first Distance Sampling study step, we'll start here by running **pre-analyses** on a designed set of **samples**, in order to get a first idea of which samples are eligible for real Distance Sampling analyses, or not (i.e. a first DS glance to the data set).

Before running the pre-analyses, you have to manually **design the list of samples** you want to check :
* a sample is by definition 1 sub-set of the dataset which is meaningful in the study,
* here, it can be 1 particular combination of the bird species, seasonal pass, bird sex, and transect duration properties of the sightings (see below the column matching for provided ACDC2019-Nat-ObsIndivDist.xlsx and ACDC2019-Samples.xlsx workbook files):
  - bird species ("Espèce" column): we kept only 4 species from the original field dataset, to make computations and checks shorter, and produced data lighter,
  - seasonal pass ("Passage" column): as every common breeding bird species has it own breeding periods, we decided to run 2 seasonal pass on each point transect (roughly 1 in April = "a" pass, 1 in May => "b" pass) ; "a+b" identifies the concatenated sighting subset "a" + "b",
  - bird sex ("Adulte" column): "m" for males, "a" for others ("autres" in French) ; for most species, breeding females are quite inconspicuous, contrary to males, who keep singing for hours early in the morning, so it's far more efficient to count them and estimate the whole breeding population as twice their number (or exactly their number if the population unit is breeding pairs) ; "m+a" identifies the concatenated sighting subset "m" + "a"
  - transect duration ("Durée" column): for this study, we run 10mn sessions on each point transect, but also marked separately the sightings of the 5 first minutes ; this way, we got 2 sub-datasets: 1 for 5mn transects, 1 for 10mn transects (one of the objectives of this study was to compare DS estimations for these 2 durations, 'cause we were suspecting that 5mn was too short for some species to contacts most nearby individuals) ; there's no concatenated subset here, as the 10mn subset includes the 5mn one,
* you should always select a wide sample set, in order not to miss unexpectedly workable ones (some sample may suffer very few sightings, but with a good statistical shape which makes them suitable for good distance sampling analyses, contrary to some more opulent but too biased samples),
* but of course not unreasonably wide: more is much work to achieve for checking !

Happily, the **ACDC2019-Samples.xlsx workbook** has been prepared for you, for a quick start (but you can tweak it as you like) as a specification of the samples to be pre-analysed (as explained above, it's using here "all-implicit" specification of the combinations, though its `*_impl` sheets, but you could add more explicit combinations through another sheet, not ending by `_impl` ; no written doc. for this for the moment, but you can explore ACDC2019-OptAnalysesToDo.xlsx to get an example of the mix of implicit and explict specification sheets ... here for analyses parameters, but it works the same).


#### a. Run pre-analyses

To **run the pre-analyses** for all the samples specified in `ACDC2019-Samples.xlsx`, on the sightings listed in `ACDC2019-Nat-ObsIndivDist.xlsx`, in the work folder `./pranlys` :
1. don't run, but check what's going to happen:

    `python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./pranlys -e`

2. actually run:

    `python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./pranlys -e -u`

Once finished (1 or 2 seconds), this run produces the following files in this folder `./pranlys/<date>-<hour>` (created if not yet present):
* the (1-sheet) workbook of explicitated specs of the 10 pre-analysed samples (from input implicit specs in `ACDC2019-Samples.xlsx): `ACDC2019-Nat-samples-explispecs.xlsx`,
* the workbook of 10 pre-analysis results (`all-results` sheet), + other sheets to recall computation specs (`samples`, `models`, pre-`analyser` setting) and `runtime` = Computing platform tech. info: `ACDC2019-Nat-preanalyses-results.xlsx`,
* for each pre-analysis (1 by sample), one sub-folder with all files output by MCDS,
* the execution log, for traceability.


#### b. Report pre-analyses

Then, we want to look at the results, just like we'd do in Distance GUI : to enable this, we must generate the **pre-analysis report(s)**, by post-processing the results we just produced.

But first, we've got to remind of the `./pranlys/<date>-<hour>` folder where the results were written ; it's easy: we just have to copy it from the console log ; let's say it's `YYMMDD-HHMMSS`.

Now, let's run this command, for both an Excel and HTML report (first without the ending `-u` to check, then with it to really go):

`python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./pranlys/YYMMDD-HHMMSS -t excel,html -u`

Note: Don't care about the following openpyxl warning in the console at the end: `CSSWarning: Unhandled color format: 'transparent'`

Once finished (maybe 10 or 20s seconds), this run produces the following files in the previous pre-analysis folder `./pranlys/YYMMDD-HHMMSS`:
* the Excel report workbook: `ACDC2019-Nat-preanalyses-report.xlsx` (showing nearly the same as the HTML report, except for the missing DS plots ; mostly useful for post-processing with some other tools),
* the files of the HTML report, as a mini web-site:
    - the main page `ACDC2019-Nat-preanalyses-report.hml` (with its `report.css` style sheet, and `.svg` icons),
    - inside each pre-analysis sub-folder : a second-level page for the relevant analysis, `index.html`, along with the `.png` plots (the same plots as produced by Distance, with some improvements)
* the execution log, for traceability.

The main page of the HTML report displays first a "super-synthesis" table, with 1 row for each pre-analysis run ; each of these rows shows 6 columns:
* on the left, 3 data columns for:
    - sample identification info. / specs,
    - pre-analysis identification info.,
    - exit code of MCDS: 1 is OK, 2 means some warning(s), 3 means some error(s),
    - pre-analysis parameters for the first model that gave a result (without any error, no more constraint),
    - some key-figures about the sample (number of individuals, max distance, DS effort),
    - some of the usual quality indicators (Khi2, KS, AIC, DCv),
    - the main DS results of the analysis: detection probability, estimated density and number of individuals, each with 95% confidence interval, EDR/ESW,
* on the right, 3 columns showing the plots to be able to quickly check how the sample is usable for DS:
    - first, the raw histogram of distances (with 3 different distance bin widths),
    - then, the 2 usual Detection Probability plots (density aka PDF, and integrated density, both over the distance) that you can find in Distance software,
* you can click on the index column at the extreme left side of the table: you'll open the "details" page for the relevant analysis, with more details (actually an exact copy of the report that Distance would show for the analysis).

Note: The information displayed in the first 3 columns is fully customizable (see acdc-2019-ds-params.preReport{Sample|Params|Results}Cols).

In this main HTML page, there's also below some traceability tables (Samples, Models, Analyser, Computing platform) to recall, if needed later, any detail about how these results were computed.

As for the Excel workbook, it features multiple sheets:
* "Synthesis": a customizable selection of the numerical results for each analysis (see acdc-2019-ds-params.preReportSynthCols),
* "Details": all possible numerical results output by MCDS for each analysis,
* finally, some traceability sheets (actually matching with the last tables of the main HTML table `ACDC2019-Nat-preanalyses-report.hml` : Samples, Models, Analyser, Computing platform) to recall, if needed later, any detail about how these results were computed.


#### c. Run and report in 1 command

Of course, we can in one only command run the pre-analyses and generate the report(s) ; example, for the HTML report only here:

`python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./pranlys -e -t excel,html -u`

Last: Talking about sample selection for later in-depth analyses, let's say that:
* if a particular sample can't get any results during pre-analysis (i.e. when all tried models failed), after checking that the model choosing fallback strategy is correct (not to restrictive), it does not mean that this sample is completely unusable ; because sometimes, after removing outliers, it's OK ; here, this can be achieved through distance truncation ... but pre-analyses don't do that for the moment, so you'll probably have to check yourself with Distance ... or make a bet and run analyses anyway !
* samples with very poor results (after looking at the figures and plots) could also get better results after distance truncation, so don't eliminate them if they contain at least 20 sightings: one never knows, if the distance distribution is clean, it might work.


### 3. Run and report designed analyses (with optimised distance truncation ... or without)

After taking a first and quick glance to the usability of the data set and decided of the samples we'd like to study, it's time for running more serious DS analyses:
* run analyses using multiple combinations of analysis parameters, possibly with automated left and right truncation distance determination (through an optimisation technique),
* automatically post-process this bunch of results to filter the best analyses to be kept at the end for each sample (with the opportunity at the end for the user to check and fix if needed the automatic filtering).


#### a. Run analyses

For this, you first have to specify the combinations of analysis parameters you want 
to try ; it's done in a similar way as for specifying the samples for pre-analyses: 
through a workbook with sheets containing implicit specs (ending with `_impl`) and sheets containing explicit specs (the other ones) ; pyaudisam knows how to explicitate such a workbook by auto-generating and combining the explicitation result of all the sheets ... but it might be complicated to master at first.

Happily, the **`ACDC2019-OptAnalysesToDo.xlsx` workbook** has been prepared for you, for a quick start (but you can tweak it as you like) as a specification of the analyses to be run.

To **run all the analyses** specified in `ACDC2019-OptAnalysesToDo.xlsx`, on the sightings listed in `ACDC2019-Nat-ObsIndivDist.xlsx`, in the work folder `./optanlys` :
1. don't run, but check what's going to happen (note the figures at the end ... an estimation of up to 21000 analyses to be run, because of the `auto` distance truncation variants in the analyses spec. file ; keep in mind that each analysis, i.e. 1 MCDS run, may last around 1s on 1 processor):

    `python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./optanlys -o`

2. actually run:

    `python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./optanlys -o -u`

Notes:
* pyaudisam knows how to automatically use the multiple processors / cores provided by the system: MCDS runs are done parallely (here using N parallel processes, where N is the number of processors reported by the system + 4) !
* While optimisations are running (the first stage), you'll get some follow-up information in the console every 20 "to-be-optimised analysis", with an estimation of the end time of this stage.
* On my powerful 6 hyper-threading core i7 10850H laptop, this optimisation stage lasts around 20mn, that is around 1000 MCDS runs per minute.
* While the final actual analysis run (now we've got all the parameters for all the analysis variants to run) ... lasts only around 20 seconds.

Once finished, this run produces the following files in this folder `./optanlys/<date>-<hour>` (created if not yet present):
* the explicitated specs of the 272 run analyses (specified through `ACDC2019-OptAnalysesToDo.xlsx`): `ACDC2019-Nat-optanalyses-explispecs.xlsx` (note that it's a bit more than explicitation, as it also contains the optimised distance truncation parameters : the `auto` specs were replaced by the computed values),
* the workbook of the 272 analysis results (`all-results` sheet), + other sheets to recall computation specs (`analyses` specs, `analyser` settings) and `runtime` = computing plateform tech. info: `ACDC2019-Nat-optanalyses-results.xlsx`,
  (note: for the "optimised" analyses, thousands of MCDS calls are achieved ... here around 18 000 ... but their run-folders are auto-cleaned up, to trace remains at all)
* for each analysis, one sub-folder with all files output by MCDS,
* the execution log, for traceability,
* and some other miscellaneous files (like intermediate results files for restarting optimisations in case the whole process was interrupted, and some debugging ones that should not remain, this need to be fixed ;-).


#### b. Filter and report analyses

Then, we want to look at the results, just like we'd do in Distance GUI, and also to **filter** and sort them, as they are very numerous ... what a hassle to examine all these analyses and decide **which is the best** for each sample !

Fortunately, pyaudisam helps quite much here (this is one of its main purposes): to enable this, we must generate the **analysis report(s)**, by post-processing the results we just produced.

But first, we've got to remind of the `./optanlys/<date>-<hour>` folder where the results were written ; it's easy: we just have to copy it from the console log ; let's say it's `YYMMDD-HHMMSS`.

Note: Once created and filled-up by pyaudisam, don't rename the folder in any way (change `optanlys` or the actual `YYMMDD-HHMMSS` for something else), as there are references to this path part in the results files: doing so would prevent you from producing the reports ; on the other hand, you can well rename the parent folder: these references are relative ones.

Now, let's run this command, for both an Excel and one HTML report (first without the ending `-u` to check, then with it to really go):

`python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./optanlys/YYMMDD-HHMMSS -f html:r925,excel -u`

Once finished (a few minutes), this run produces the following files in this folder `./optanlys/YYMMDD-HHMMSS`:
* the Excel report workbook: `ACDC2019-Nat-optanalyses-report.xlsx`
     (1 sheet per available "filter & sort scheme", a "**filter**" in short, i.e. a set of filter and sort parameters : see `acdc-2019-ds-params.filsorReportSchemes` for the actual parameter values, and the [short "how it works" guide](../how-it-works/how-it-works-en.md) for an explanation about these "filters"),
* the files of the HTML report for the selected filter `ExAicMQua-r925m8q3d12` (the only one with `r925` contained in its name :-), as a mini web-site:
    - the main page `ACDC2019-Nat-optanalyses-report.ExAicMQua-r925m8q3d12.hml` (with its `report.css` style sheet, and `.svg` icons),
    - inside each selected (by the filter) analysis sub-folder : a second-level page for the relevant analysis, `index.html`, along with the `.png` plots (the same plots as produced by Distance, with some improvements)
* the execution log, for traceability.

Whatever report type, Excel or HTML, they show various result tables featuring 1 row per analysis that have been kept by the relevant filter, that is the N "best" ones when considering their "combined quality indicator" (N is actually the last figure found at the end of the considered filter Id : ex. N=12 for ExAicMQua-r925m8q3d12) ; for easier examination, these kept analyses are grouped by sample and sorted over this quality indicator, better = higher first.

The main page of the HTML report displays first a "super-synthesis" table, with 1 row for each kept analysis (the N best ones per sample) ; each of these rows features 6 columns:
* on the left, 3 data columns for:
    - sample identification info. / specs,
    - some key-figures about the sample (number of individuals, max distance),
    - main analysis parameters (model, adjustment series, left and right truncation distances if any, number of cuts for model fitting),
    - analysis identification info.,
    - exit code of MCDS: 1 is OK, 2 means some warning(s), 3 means some error(s),
    - some key-figures about the analysis done (number and rate of kept observations, ...),
    - some of the usual quality indicators (Khi2, KS, AIC, DCv),
    - the "combined quality indicators" Qual Bal 1, 2 & 3 (but only Qual Bal 3 is actually used)
    - the main DS results of the analysis: detection probability, estimated density and number of individuals, each with 95% confidence interval, EDR/ESW,
* on the right, 3 columns showing the plots needed to quickly check how the sample is usable for DS:
    - first, the Quantile-Quantile plot comparing the cumulated distribution functions of the field data versus the model-fitted data,
    - then, the 2 usual Detection Probability plots (density aka PDF, and integrated density, both over the distance) that you can find in Distance software,
* you can click on the index column at the extreme left side of the table: you'll open the "details" page for the relevant analysis, with more details (actually an exact copy of the report that Distance would show for the analysis).

Note: The data displayed in the first 3 columns is fully customizable (see acdc-2019-ds-params.filsorReport{Sample|Params|Results}Cols).

In this main HTML page, there's also below:
* a "synthesis" table with a selection of the numerical results for each analysis ("Main figures")
* a "details" table with all possible numerical results output by MCDS for each analysis (the "All details")
* finally, some traceability tables (actually matching with the last sheets of the analysis results workbook `ACDC2019-Nat-optanalyses-results.xlsx`) to recall, if needed later, any detail about how these results were computed.

As for the Excel workbook, it features multiple sheets:
* 1 sheet per available Auto-Filter-and-Sort (AFS) **"filter"**, each with the same customizable column set as the "Synthesis" table of the relevant main "HTML" page (see acdc-2019-ds-params.filsorReportSynthCols),
* "AFS-Steps" : a traçability sheet for all filters, listing all filters steps and parameters (the same table as in the main HTML table, but for all filters here),
* "Synthesis": a customizable selection of the numerical results for **all** analyses run, not only the kept ones, whatever filter (see acdc-2019-ds-params.filsorReportSynthCols),
* "Details": all possible numerical results output by MCDS for **all** analyses run, not only the kept ones, whatever filter,
* finally, some other traceability sheets (actually matching with the last tables of the main HTML table `ACDC2019-Nat-preanalyses-report.hml` : Samples, Models, Analyser, Computing platform) to recall, if needed later, any detail about how these results were computed.


#### c. Run and report in 1 command

Of course, in one command only, we can run the analyses and generate the filtered report(s) ; example, for the 2 ExAicMQua-r925m8q3d12 and MFTA-ExAicMQua-r975m8q3d8 HTML filtered reports:

`python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./optanlys -o -f html:r925,html:r975 -u`

In this particular case, we'll get 2 "main" HTML pages `ACDC2019-Nat-optanalyses-report.ExAicMQua-r925m8q3d12.hml` and `ACDC2019-Nat-optanalyses-report.ExAicMQua-r975m8q3d8.hml` (1 for each filter specified in the command line option -f).


#### d. Full (rather than "filtered") reports

This is generally not what you want, as you trust pyaudisam filtering system and don't want to examine all the analyses yourself one by one :-).

But in case you really want to get exhaustive **reports**, showing the resuls **of all the analyses** run, without any filtering (beware of the total number !), you can go with the "**full**" Excel and HTML report types:

`python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./optanlys/YYMMDD-HHMMSS -f html:full,excel:full -u`

The produced Excel report workbook: `ACDC2019-Nat-optanalyses-report.xlsx` is actually just the same as the "filtered" one (see above), except that it **does not contain any "filtered" sheet** ; which means that it not actually usefull, as it is really a subset of the filtered report. Note that here, due to some specific settings in `acdc-2019-ds-params.py`, especially about row sorting, the results are not in the same order (sorted by increasing left and right truncation distance, and then only by decreasing Qual Bal 3 indicator) ; but you can change the settings !

Warning: If you generate the "full" Excel report (`-f excel:full`) after generating the "filtered" one (only `-f excel`), the first one will be overwritten by the second one, as they share the same name `ACDC2019-Nat-optanalyses-report.xlsx` (but not a big deal, as `-f excel:full` is probably useless).

Notes: When filters are available in `-p acdc-2019-nat-ds-params.py`, using `excel:full` as a report specifier prevents the "filtered" sheets (1 for each filter) to be generated ; with simply `excel`, you'll get both types of report in the same workbook: the filtered one (1 sheet for each filter) + the full one (`synthesis` and `details` sheets).

As for the HTML report `ACDC2019-Nat-optanalyses-report.html`, it shows the same structure and organisation as a filtered report (for 1 filter), but the tables display 1 row per run analysis **among all of them** (might result quite big a report !), rather than 1 row per **retained** analysis after filtering (the N best ones); here also, the analysis results are grouped by sample and sorted by descending combined quality indicator (Qual Bal 3 here).

And again, even if the full report shows a different name (`ACDC2019-Nat-optanalyses-report.html`), generating the "full" HTML report (`-f html:full`) after some "filtered" one(s) will overwrite it (them) partially (actually, only the navigation links from the analysis-specific pages to the main page, but it is enough to break it ; this is for sure a bug to fix one day :-).


### 4. Run designed analyses (without optimised distance truncation)

If you don't need / want any automated truncation distance computations for your analyses, you've got 2 options:
1. follow the instruction above (chapter 3), but simply avoid using the `auto` keyword when building your opt-analyses spec. file (see above the `ACDC2019-OptAnalysesToDo.xlsx` workbook) to specify which combinations of analyses parameters you want to try,
2. you can also go another simpler way (no need to change your analysis spec. file `ACDC2019-OptAnalysesToDo.xlsx`): see below.

General hint: The generated files will no more contain the "optanalyses" keyword, but the simpler "analyses" keyword.


#### a. Run analyses

To **run all the analyses** specified in `ACDC2019-OptAnalysesToDo.xlsx`, except for the ones with `auto` distance truncation parameter values (they'll be automatically ignored), on the sightings listed in `ACDC2019-Nat-ObsIndivDist.xlsx`, in the work folder `./anlys` (as always, run first without `-u`):

`python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./anlys -a -u`


#### b. Filter and report analyses

To **generate** both HTML and Excel **filtered reports** (the HTML one only for the `MFTA-ExAicMQua-r950m8q3d10` filter), from the results of the analyses just run:

1. retrieve the `./anlys/<date>-<hour>` folder where the analysis results were just written: easy, copy it from the console log ; let's say it's `YYMMDD-HHMMSS`.

2. generate the (filtered) reports (as always, run first without `-u` to check what's going on):

`python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./anlys/YYMMDD-HHMMSS -r excel,html:r950 -u`


#### c. Run and report in 1 command

You can also run and report analyses the same way in one command only:

`python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./anlys -a -r excel,html:r950 -u`


#### d. Full (rather than "filtered") report

This is generally not what you want (see above), but in case you want it, you can go with the "**full**" report types:

`python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./anlys/YYMMDD-HHMMSS -r excel:full,html:full -u`


### 5. Export sample data files ready for manual analyses through Distance

In some cases, you might have doubts about some analysis results produced and reported above, or some needed / not needed combination of analysis parameters to try, or ... etc.

Then you might need to fall back to the Distance software to run things manually for a few analyses / samples.

But for this, you need to import the field observations into Distance.

And here again, pyaudisam can make things simple: it also features a sample export command for Distance !

To **export all the samples** specified in `ACDC2019-Samples.xlsx` from the sightings listed in `ACDC2019-Nat-ObsIndivDist.xlsx` into the `./dist-exp` folder:

`python -m pyaudisam -p acdc-2019-nat-ds-params.py -w ./dist-exp -n -x -u`

Once finished (1 or 2 seconds), this run produces the following files in this `./dist-exp` folder (auto-created if not yet present):
* the 10 .txt files (1 per sample) suitable for direct import into Distance 6+ (no column specification needed, all's in the .txt files),
* the explicitation table for the 10 samples (implicitly specified in `ACDC2019-Samples.xlsx`): `ACDC2019-Nat-samples-explispecs.xlsx`,
* the execution log, for traceability.

Now, you can start Distance and go on for manual analyses !


## C. pyaudisam as a python module

You can also use pyaudisam as a python module in your scripts or notebooks: the [acdc-2019-nat-ds-run.ipynb](./acdc-2019-nat-ds-run.ipynb) jupyter notebook is here to demonstrate how you can do **exactly the same work** as the one explained above in **B.2 to B.5**, but with the confort of a notebook :-).
