# coding: utf-8

# PyAuDiSam: Automation of Distance Sampling analyses with Distance software (http://distancesampling.org/)
#
# Copyright (C) 2021 Jean-Philippe Meuret
#
# This program is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see https://www.gnu.org/licenses/.

# Submodule "report": HTML and Excel report generation from DS results

import sys
import os
import shutil
import re
import pathlib as pl
from packaging import version as pkgver
import copy

import datetime as dt
import codecs

import math
import numpy as np
import pandas as pd

import jinja2
import matplotlib.pyplot as plt
import matplotlib.ticker as pltt
# import seaborn as sb

from . import log, runtime, __version__
from .executor import Executor
from .analyser import MCDSAnalysisResultsSet

runtime.update(matplotlib=sys.modules['matplotlib'].__version__, jinja2=jinja2.__version__)

logger = log.logger('ads.rep')

# Actual package install dir.
KInstDirPath = pl.Path(__file__).parent.resolve()


def _mergeTransTables(base, update):

    """Merge an 'update' translation table into a 'base' one ('update' completes or overwrites 'base').

    Note: Trans. tables are dict(<lang>=dict(<source>: <translation>))
    """

    final = copy.deepcopy(base)
    for lang in update.keys():
        if lang not in final:
            final[lang] = dict()
        final[lang].update(update[lang])
    return final


class ResultsReport:

    """Base for results reports classes (abstract)"""

    # Translation table for output documents (specialized in derived classes, merged with custom instance one).
    DTrans = dict(en={}, fr={})

    def __init__(self, resultsSet, title, subTitle, description, keywords,
                 dCustomTrans=None, lang='en', pySources=[], tgtFolder='.', tgtPrefix='results'):
    
        """Ctor
        
        Parameters:
        :param resultsSet: source results
        :param title: main page title (and <title> tag in HTML header)
        :param subTitle: main page sub-title (under the title, lower font size)
        :param description: main page description text (under the sub-title, lower font size)
        :param keywords: for HTML header <meta name="keywords" ...>
        :param dCustomTrans: custom translations to complete the report standard ones,
                             as a dict(fr=dict(src: fr-trans), en=dict(src: en-trans))
        :param lang: Target language for translation (only 'en' and 'fr' supported)
        :param pySources: path-name of source files to copy in report folder and link in report
        :param tgtFolder: target folder for the report (for _all_ generated files)
        :param tgtPrefix: default target file name for the report
        """

        assert len(resultsSet) > 0, 'Can\'t build reports with nothing inside'
        assert os.path.isdir(tgtFolder), 'Target folder {} doesn\'t seem to exist ...'.format(tgtFolder)
        assert lang in ['en', 'fr'], 'Only en and fr translation supported for the moment'
        
        self.resultsSet = resultsSet
        
        self.trRunFolderCol = resultsSet.dfColTrans.loc[resultsSet.analysisClass.RunFolderColumn, lang]
        self.dfEnColTrans = None  # EN to other languages column name translation

        self.lang = lang
        self.title = title
        self.subTitle = subTitle
        self.description = description
        self.keywords = keywords
        self.pySources = pySources
        self.dTrans = _mergeTransTables(self.DTrans, dict() if dCustomTrans is None else dCustomTrans)
        
        self.tgtPrefix = tgtPrefix
        self.tgtFolder = tgtFolder
        
        self.tmplEnv = None
    
    @staticmethod
    def _libVersions():
        return {'Python': sys.version.split()[0],
                'NumPy': runtime['numpy'],
                'Pandas': runtime['pandas'],
                'ZOOpt': runtime['zoopt'],
                'Matplotlib': runtime['matplotlib'],
                'Jinja': runtime['jinja2']}

    # Translate string
    def tr(self, s):
        return self.dTrans[self.lang].get(s, s)
    
    def trEnColNames(self, colNames, startsWith=None):
        
        """Translate EN-translated column(s) name(s) to self.lang one

        Parameters:
        :param colNames: str (1), list(N) or dict (N keys) of column names
        :param startsWith: if not None or '', only columns starting with it will get processed
        """

        # 1. Build translation function (and translation table if needed)
        # a. If target language is English, no translation needed
        if self.lang == 'en':
            def _trEnColName(cn):
                return cn

        # b. If target language is NOT English, translation is NEEDED
        else:
            if self.dfEnColTrans is None:
                self.dfEnColTrans = self.resultsSet.transTable()
                self.dfEnColTrans.set_index('en', inplace=True, drop=True)

            def _trEnColName(cn):  # Assuming there's only 1 match !
                return self.dfEnColTrans.at[cn, self.lang]
        
        # 2. Translate !
        if isinstance(colNames, str):
            trColNames = _trEnColName(colNames) \
                         if not startsWith or colNames.startswith(startsWith) else None
        elif isinstance(colNames, list):
            trColNames = [_trEnColName(colName) for colName in colNames
                          if not startsWith or colName.startswith(startsWith)]
        elif isinstance(colNames, dict):
            trColNames = {_trEnColName(colName): value for colName, value in colNames.items()
                          if not startsWith or colName.startswith(startsWith)}
        else:
            raise NotImplementedError(f'Unsupported type {type(colNames)} for en-column names to translate spec')
        
        # Done
        return trColNames
        
    # Output file pathname generation.
    def targetFilePathName(self, suffix, prefix=None, tgtFolder=None):
        
        return os.path.join(tgtFolder or self.tgtFolder, (prefix or self.tgtPrefix) + suffix)
    
    def relativeRunFolderUrl(self, runFolderPath):

        return os.path.relpath(runFolderPath, self.tgtFolder).replace(os.sep, '/')
    
    # Install needed attached files for HTML report.
    def installAttFiles(self, attFiles):
        
        # Given attached files.
        for fn in attFiles:
            shutil.copy(KInstDirPath / 'report' / fn, self.tgtFolder)
            
        # Python source files.
        for fpn in self.pySources:
            shutil.copy(fpn, self.tgtFolder)
    
    # Get Jinja2 template environment for HTML reports.
    def getTemplateEnv(self):
        
        # Build and configure jinja2 environment if not already done.
        if self.tmplEnv is None:
            self.tmplEnv = jinja2.Environment(loader=jinja2.FileSystemLoader([KInstDirPath]),
                                              trim_blocks=True, lstrip_blocks=True)
            # self.tmplEnv.filters.update(trace=_jcfPrint2StdOut)  # Template debugging ...

        return self.tmplEnv
    
    def asWorkbook(self, subset=None, rebuild=False):

        """Format as a "generic" workbook format, i.e. as a dict(name=(DataFrame, useIndex))
        where each item is a named worksheet

        Parameters:
        :param subset: Selected list of data categories to include ; None = [] = all
                       (categories in {'specs'})
        :param rebuild: if True force rebuild of report = prevent use of / reset any cache
                        (not used here)
        """
        
        logger.debug(f'ResultsReport.asWorkbook({subset=}, {rebuild=})')

        ddfWbk = dict()

        # Specs (no need to check if 'specs' in subset: we have nothing else than results specs here).
        for spName, dfSpData in self.resultsSet.specs2Tables().items():
            logger.info1(f'* {spName} ...')
            ddfWbk[self.tr(spName)] = (dfSpData, True)

        # Done
        return ddfWbk
 
    def toExcel(self, fileName=None, engine='openpyxl', ext='.xlsx', rebuild=False):

        """Export to a workbook file format (Excel, ODF, ...)

        Parameters:
        :param fileName: target file path name ; None => self.tgtFolder/self.tgtPrefix + ext
        :param engine: Python module to use for exporting (openpyxl=default, or odf)
        :param ext: extension of target file, if not specified in filename
        :param rebuild: if True force rebuild of report = prevent use of / reset any cache
        """
        
        logger.debug(f'ResultsReport.toExcel({fileName=}, {ext=}, {rebuild=})')

        fileName = fileName or os.path.join(self.tgtFolder, self.tgtPrefix + ext)
        
        with pd.ExcelWriter(fileName, engine=engine) as xlsxWriter:
            logger.info(f'Building workbook report {fileName} ...')
            for wstName, (dfWstData, wstIndex) in self.asWorkbook(rebuild=rebuild).items():
                dfWstData.to_excel(xlsxWriter, sheet_name=wstName, index=wstIndex)

        logger.info('... done.')

        return fileName

    def toOpenDoc(self, fileName=None, rebuild=False):

        """Export report to Open Doc worksheet format

        Parameters:
        :param fileName: target file path name ; None => self.tgtFolder/self.tgtPrefix + ext
        :param rebuild: if True force rebuild of report = prevent use of / reset any cache
        """
    
        assert pkgver.parse(pd.__version__).release >= (1, 1), \
               'Don\'t know how to write to OpenDoc format before Pandas 1.1'
        
        return self.toExcel(fileName, engine='odf', ext='.ods', rebuild=rebuild)

    # Final formatting of translated data tables, for HTML or SpreadSheet rendering
    # in the "one analysis at a time" case.
    # (sort, convert units, round values, and style).
    # To be specialized in derived classes (here, we do nothing) !
    # Note: Use trEnColNames method to pass from EN-translated columns names to self.lang-ones
    # Return a pd.DataFrame.Styler
    def finalFormatEachAnalysisData(self, dfTrData, sort=True, convert=True, round_=True, style=True):
        
        return dfTrData.style  # Nothing done here, specialize in derived class if needed.

    # Final formatting of translated data tables, for HTML or SpreadSheet rendering
    # in the "all analyses at once" case.
    # (sort, convert units, round values, and style).
    # To be specialized in derived classes (here, we do nothing) !
    # Note: Use trEnColNames method to pass from EN-translated columns names to self.lang-ones
    # Return a pd.DataFrame.Styler
    def finalFormatAllAnalysesData(self, dfTrData, sort=True, indexer=None, convert=True, round_=True, style=True):
        
        logger.debug(f'ResultsReport.finalFormatAllAnalysesData({sort=}, {indexer=}, {convert=}, {round_=}, {style=})')

        return dfTrData.style  # Nothing done here, specialize in derived class if needed.


class DSResultsDistanceReport(ResultsReport):

    """DS results reports class (Excel and HTML, targeting similar layout as in Distance 6+)"""

    # Translation table.
    DTrans = _mergeTransTables(base=ResultsReport.DTrans,
        update=dict(en={'RunFolder': 'Analysis', 'Synthesis': 'Synthesis',
                        'Details': 'Details', 'Traceability': 'Traceability',
                        'Table of contents': 'Table of contents',
                        'Click on analysis # for details': 'Click on analysis number to get to detailed report',
                        'Main results': 'Results: main figures',
                        'Detailed results': 'Results: all details',
                        'Analysis': 'Analysis',
                        'Download Excel': 'Download as Excel(TM) file',
                        'Summary computation log': 'Summary computation log',
                        'Detailed computation log': 'Detailed computation log',
                        'Previous analysis': 'Previous analysis', 'Next analysis': 'Next analysis',
                        'Back to top': 'Back to global report',
                        'Observation model': 'Observations (fitted)',
                        'Real observations': 'Observations (sampled)',
                        'Fixed bin distance histograms': 'Fixed bin distance histograms',
                        'Distance': 'Distance', 'Distance in': 'Distance in',
                        'Number of observations': 'Number of observations',
                        'Page generated': 'Generated', 'with': 'with',
                        'with icons from': 'with icons from',
                        'and': 'and', 'in': 'in', 'sources': 'sources', 'on': 'on',
                        'Point': 'Point transect', 'Line': 'Line transect',
                        'Radial': 'Radial distance', 'Perpendicular': 'Perpendicular distance',
                        'Radial & Angle': 'Radial distance & Angle',
                        'Clustering': 'With clustering', 'No clustering': 'No clustering',
                        'Meter': 'Meter', 'Kilometer': 'Kilometer', 'Mile': 'Mile',
                        'Inch': 'Inch', 'Feet': 'Feet', 'Yard': 'Yard', 'Nautical mile': 'Nautical mile',
                        'Hectare': 'Hectare', 'Acre': 'Acre', 'Sq. Meter': 'Sq. Meter',
                        'Sq. Kilometer': 'Sq. Kilometer', 'Sq. Mile': 'Sq. Mile',
                        'Sq. Inch': 'Sq. Inch', 'Sq. Feet': 'Sq. Feet', 'Sq. Yard': 'Sq. Yard',
                        'Sq. Nautical mile': 'Sq. Nautical mile',
                        'Traceability tech. details':
                          'Traceability data (more technical details on how this report was produced)',
                        'Order': 'Order', 'Qual Bal': 'Qual Bal', 'Pre-selection': 'Pre-selection'},
                    fr={'DossierExec': 'Analyse', 'Synthesis': 'Synthèse',
                        'Details': 'Détails', 'Traceability': 'Traçabilité',
                        'Table of contents': 'Table des matières',
                        'Click on analysis # for details':
                          "Cliquer sur le numéro de l'analyse pour accéder au rapport détaillé",
                        'Main results': 'Résultats : indicateurs principaux',
                        'Detailed results': 'Résultats : tous les détails',
                        'Analysis': 'Analyse',
                        'Download Excel': 'Télécharger le classeur Excel (TM)',
                        'Summary computation log': 'Résumé des calculs',
                        'Detailed computation log': 'Détail des calculs',
                        'Previous analysis': 'Analyse précédente', 'Next analysis': 'Analyse suivante',
                        'Back to top': 'Retour au rapport global',
                        'Observation model': 'Observations (fitted)',  # No actual translation for plots
                        'Real observations': 'Observations (sampled)',  # Idem
                        'Fixed bin distance histograms': 'Fixed bin distance histograms',  # Idem
                        'Distance': 'Distance', 'Distance in': 'Distance in',  # Idem
                        'Number of observations': 'Number of observations',  # Idem
                        'Page generated': 'Généré', 'with': 'avec',
                        'with icons from': 'avec les pictogrammes de',
                        'and': 'et', 'in': 'dans', 'sources': 'sources', 'on': 'le',
                        'Point': 'Point fixe', 'Line': 'Transect',
                        'Radial': 'Distance radiale', 'Perpendicular': 'Distance perpendiculaire',
                        'Radial & Angle': 'Distance radiale & Angle',
                        'Clustering': 'Avec clustering', 'No clustering': 'Sans clustering',
                        'Meter': 'Mètre', 'Kilometer': 'Kilomètre', 'Mile': 'Mile',
                        'Inch': 'Pouce', 'Feet': 'Pied', 'Yard': 'Yard', 'Nautical mile': 'Mille marin',
                        'Hectare': 'Hectare', 'Acre': 'Acre', 'Sq. Meter': 'Mètre carré',
                        'Sq. Kilometer': 'Kilomètre carré', 'Sq. Mile': 'Mile carré',
                        'Sq. Inch': 'Pouce carré', 'Sq. Feet': 'Pied carré', 'Sq. Yard': 'Yard carré',
                        'Sq. Nautical mile': 'Mille marin carré',
                        'Traceability tech. details':
                          'Données de traçabilité (autres détails techniques sur comment ce rapport a été produit)',
                        'Order': 'Ordre', 'Qual Bal': 'Qual Equi', 'Pre-selection': 'Pré-selection'}))

    @staticmethod
    def noDupColumns(cols, log=True, head='Results cols'):

        """Drop duplicates from a column list, and possibly warn about which"""

        dups = None
        if isinstance(cols, list):
            dups = [col for ind, col in enumerate(cols) if col in cols[:ind]]
            if len(dups) > 0:
                cols = [col for ind, col in enumerate(cols) if col not in cols[:ind]]

        elif isinstance(cols, pd.MultiIndex):
            dups = cols[cols.duplicated()]
            if len(dups) > 0:
                cols = cols.drop_duplicates()

        if log and dups is not None and len(dups) > 0:
            logger.warning(head + ': Dropped {} duplicate(s) {}'
                                  .format(len(dups), ', '.join(str(dup) for dup in dups)))

        return cols

    def __init__(self, resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                 synthCols=None, sortCols=None, sortAscend=None, dCustomTrans=None, lang='en',
                 plotImgFormat='png', plotImgSize=(640, 400), plotImgQuality=90,
                 plotLineWidth=2, plotDotWidth=6, plotFontSizes=dict(title=11, axes=10, ticks=9, legend=10),
                 pySources=[], tgtFolder='.', tgtPrefix='results', logProgressEvery=5):
                       
        """Ctor
        
        Parameters:
        :param resultsSet: source results
        :param title: main page title (and <title> tag in HTML header)
        :param subTitle: main page sub-title (under the title, lower font size)
        :param description: main page description text (under the sub-title, lower font size)
        :param anlysSubTitle: analysis pages title
        :param keywords: for HTML header <meta name="keywords" ...>
        :param synthCols: for synthesis table (Excel format only, "Synthesis" tab)
        :param sortCols: sorting columns for report tables
        :param sortAscend: sorting order for report tables, as a bool or list of bools, of len(synthCols)
        :param dCustomTrans: custom translations to complete the report standard ones,
                             as a dict(fr=dict(src: fr-trans), en=dict(src: en-trans))
        :param lang: Target language for translation
        :param plotImgFormat: png, svg and jpg all work with Matplotlib 3.2.1+
        :param plotImgSize: size of the image generated for each plot = (width, height) in pixels
        :param plotImgQuality: JPEG format quality (%) ; ignored if plotImgFormat not in ('jpg', 'jpeg')
        :param plotLineWidth: width (unit: pixel) of drawn lines (observation histograms, fitted curves)
        :param plotDotWidth: width (unit: pixel) of drawn dots / points (observation distances)
        :param plotFontSizes: font sizes (unit: point) for plots (dict with keys from title, axes, ticks, legend)
        :param pySources: path-name of source files to copy in report folder and link in report
        :param tgtFolder: target folder for the report (for _all_ generated files)
        :param tgtPrefix: default target file name for the report
        :param logProgressEvery: every such nb of details pages, log some elapsed time stats
                                 and end of generation forecast
        """
    
        assert synthCols is None or isinstance(synthCols, list) or isinstance(synthCols, pd.MultiIndex), \
               'Synthesis columns must be specified as None (all), or as a list of tuples, or as a pandas.MultiIndex'
        
        assert logProgressEvery > 0, 'logProgressEvery must be positive'

        super().__init__(resultsSet, title, subTitle, description, keywords,
                         dCustomTrans=dCustomTrans, lang=lang, pySources=pySources,
                         tgtFolder=tgtFolder, tgtPrefix=tgtPrefix)
        
        self.synthCols = self.noDupColumns(synthCols, head='Synthesis columns')
        self.sortCols = self.noDupColumns(sortCols, head='Sorting columns')
        self.sortAscend = sortAscend

        assert sortAscend is None or isinstance(sortAscend, bool) or len(sortAscend) == len(self.sortCols), \
               'Some duplicated sort columns were removed, or sortAscend is too long or short, ' \
               'such that sortAscend and sortCols are not compatible => please fix these params'
        
        self.plotImgFormat = plotImgFormat
        self.plotImgSize = plotImgSize
        self.plotImgQuality = plotImgQuality
        self.plotLineWidth = plotLineWidth
        self.plotDotWidth = plotDotWidth
        self.plotFontSizes = plotFontSizes

        self.anlysSubTitle = anlysSubTitle

        self.logProgressEvery = logProgressEvery
        
    def checkNeededColumns(self):

        """Side check as soon as possible : Are all report needed columns available ?
        (now that computed columns have been ... post-computed through self.resultsSet.dfFilSorData calls)

        :raise: AssertionError if not the case
        """
        raise NotImplementedError('Abstract method DSResultsDistanceReport.checkNeededColumns must not be called')

    # Static attached files for HTML report.
    AttachedFiles = ['report.css', 'fa-feather-alt.svg', 'fa-angle-up.svg', 'fa-file-excel.svg',
                     'fa-file-excel-hover.svg', 'fa-arrow-left-hover.svg', 'fa-arrow-left.svg',
                     'fa-arrow-right-hover.svg', 'fa-arrow-right.svg',
                     'fa-arrow-up-hover.svg', 'fa-arrow-up.svg']
    
    # Plot ... data to be plotted, and draw resulting figure to image files.
    PlotImgPrfxQqPlot = 'qqplot'
    PlotImgPrfxDetProb = 'detprob'
    PlotImgPrfxProbDens = 'probdens'
    PlotImgPrfxDistHist = 'disthist'
    StripPlotAlpha, StripPlotJitter = 0.5, 0.3
    RefDistHistBinWidths = [10, 20, 40]  # unit = Distance unit
    HistBinWithRefDist = 600  # unit = Distance unit
    
    def generatePlots(self, plotsData, tgtFolder, rebuild=True, sDistances=None, lang='en',
                      imgFormat='png', imgSize=(640, 400), imgQuality=90, grid=True, transparent=False,
                      colors=dict(background='#f9fbf3', histograms='blue',
                                  multihistograms=['blue', 'green', 'red'], curves='red', dots='green'),
                      widths=dict(lines=2, dots=6), fontSizes=dict(title=11, axes=10, ticks=9, legend=10)):
        
        dPlots = dict()

        # For each plot from extracted plotsData, 
        for title, pld in plotsData.items():
            
            # a. Determine target image file name, and if rebuild not forced, don't regenerate it if already there
            if 'Qq-plot' in title:
                tgtFileName = self.PlotImgPrfxQqPlot
            elif 'Detection Probability' in title:
                sufx = title.split(' ')[-1]  # Assume last "word" is the plot number
                sufx = sufx if sufx.isnumeric() else ''  # But when only 1, there's no number.
                tgtFileName = self.PlotImgPrfxDetProb + sufx
            elif 'Pdf' in title:
                sufx = title.split(' ')[-1]  # Assume last "word" is the plot number
                sufx = sufx if sufx.isnumeric() else ''  # But when only 1, there's no number.
                tgtFileName = self.PlotImgPrfxProbDens + sufx
            else:
                raise NotImplementedError(f'Unsupported plot "{title}" found in loaded plot data')
            tgtFileName = tgtFileName + '.' + imgFormat.lower()

            dPlots[title] = tgtFileName  # Save image URL

            tgtFilePathName = os.path.join(tgtFolder, tgtFileName)                
            if not rebuild and os.path.isfile(tgtFilePathName):
                continue  # Already done, and not to be done again.

            # b. Create the target figure and one-only subplot (note: QQ plots with forced height square shape).
            figHeight = imgSize[1] / plt.rcParams['figure.dpi']
            figWidth = figHeight if 'Qq-plot' in title else imgSize[0] / plt.rcParams['figure.dpi']

            fig = plt.figure(figsize=(figWidth, figHeight))
            axes = fig.subplots()
            
            # c. Plot a figure from the plot data (3 possible types, from title).
            if 'Qq-plot' in title:
                
                n = len(pld['dataRows'])
                df2Plot = pd.DataFrame(data=pld['dataRows'],
                                       columns=[self.tr(s) for s in ['Observation model', 'Real observations']],
                                       index=np.linspace(0.5/n, 1.0-0.5/n, n))
                
                df2Plot.plot(ax=axes, zorder=10, color=[colors['histograms'], colors['curves']],
                             linewidth=widths['lines'], grid=grid,
                             xlim=(pld['xMin'], pld['xMax']), ylim=(pld['yMin'], pld['yMax']))

            elif 'Detection Probability' in title:
                
                # if sDistances is not None:
                #     axes2 = axes.twinx()
                #     sb.stripplot(ax=axes2, zorder=5, x=sDistances, color=colors['dots'], size=widths['dots'],
                #                  alpha=self.StripPlotAlpha, jitter=self.StripPlotJitter)

                df2Plot = pd.DataFrame(data=pld['dataRows'], 
                                       columns=[pld['xLabel'], pld['yLabel'] + ' (sampled)',
                                                pld['yLabel'] + ' (fitted)'])
                df2Plot.set_index(pld['xLabel'], inplace=True)
                
                df2Plot.plot(ax=axes, zorder=10, color=[colors['histograms'], colors['curves']],
                             linewidth=widths['lines'], grid=grid,
                             xlim=(pld['xMin'], pld['xMax']), ylim=(pld['yMin'], pld['yMax']))
                
                aMTicks = axes.get_xticks()
                axes.xaxis.set_minor_locator(pltt.MultipleLocator((aMTicks[1]-aMTicks[0])/5))
                axes.tick_params(which='minor', grid_linestyle='-.', grid_alpha=0.6)
                axes.grid(True, which='minor', zorder=0)

            elif 'Pdf' in title:
                
                # if sDistances is not None:
                #     axes2 = axes.twinx()
                #     sb.stripplot(ax=axes2, zorder=5, x=sDistances, color=colors['dots'], size=widths['dots'],
                #                  alpha=self.StripPlotAlpha, jitter=self.StripPlotJitter)

                df2Plot = pd.DataFrame(data=pld['dataRows'], 
                                       columns=[pld['xLabel'], pld['yLabel'] + ' (sampled)',
                                                pld['yLabel'] + ' (fitted)'])
                df2Plot.set_index(pld['xLabel'], inplace=True)
                
                df2Plot.plot(ax=axes, zorder=10, color=[colors['histograms'], colors['curves']],
                             linewidth=widths['lines'], grid=grid,
                             xlim=(pld['xMin'], pld['xMax']), ylim=(pld['yMin'], pld['yMax']))

                aMTicks = axes.get_xticks()
                axes.xaxis.set_minor_locator(pltt.MultipleLocator((aMTicks[1]-aMTicks[0])/5))
                axes.tick_params(which='minor', grid_linestyle='-.', grid_alpha=0.6)
                axes.grid(True, which='minor', zorder=0)

            else:
                raise NotImplementedError(f'Unsupported plot "{title}" found in loaded plot data')

            # d. Finish plotting.
            axes.legend(df2Plot.columns, fontsize=fontSizes['legend'])
            axes.set_title(label=pld['title'] + ' : ' + pld['subTitle'],
                           fontdict=dict(fontsize=fontSizes['title']), pad=10)
            axes.set_xlabel(pld['xLabel'], fontsize=fontSizes['axes'])
            axes.set_ylabel(pld['yLabel'], fontsize=fontSizes['axes'])
            axes.tick_params(axis='both', labelsize=fontSizes['ticks'])
            axes.grid(True, which='major', zorder=0)
            if not transparent:
                axes.set_facecolor(colors['background'])
                fig.patch.set_facecolor(colors['background'])
                
            # e. Generate an image file for the plot figure (forcing the specified patch background color).
            fig.tight_layout()
            pilArgs = dict(quality=imgQuality) if imgFormat == 'jpg' else dict()
            fig.savefig(tgtFilePathName, bbox_inches='tight', transparent=transparent,
                        facecolor=axes.figure.get_facecolor(), edgecolor='none', pil_kwargs=pilArgs)

            # g. Memory cleanup (does not work in interactive mode ... but OK thanks to plt.ioff above)
            axes.clear()
            fig.clear()
            plt.close(fig)

        # Standard fixed-bin-width super-imposed histograms (multiple bin width, scaled with distMax)
        if sDistances is not None:

            # a. Determine target image file name, and if rebuild not forced, don't regenerate it if already there
            title = 'Standard Distance Histograms'
            tgtFileName = self.PlotImgPrfxDistHist
            tgtFileName = tgtFileName + '.' + imgFormat.lower()

            dPlots[title] = tgtFileName  # Save image URL

            tgtFilePathName = os.path.join(tgtFolder, tgtFileName)                
            if rebuild or not os.path.isfile(tgtFilePathName):  # Do it only if forced to do or not already done.

                # b. Create the target figure and one-only subplot
                figHeight = imgSize[1] / plt.rcParams['figure.dpi']
                figWidth = imgSize[0] / plt.rcParams['figure.dpi']

                fig = plt.figure(figsize=(figWidth, figHeight))
                axes = fig.subplots()
                    
                # c. Plot the figure from the distance data
                # axes2 = axes.twinx()
                # sb.stripplot(ax=axes2, zorder=5, x=sDistances, color=colors['dots'], size=widths['dots'],
                #              alpha=self.StripPlotAlpha, jitter=self.StripPlotJitter)

                distMax = sDistances.max()

                distHistBinWidths = np.array(self.RefDistHistBinWidths, dtype=float)
                distHistBinWidths *= 2 ** int(distMax / self.HistBinWithRefDist)

                for binWidthInd, binWidth in enumerate(distHistBinWidths):
                
                    aDistBins = np.linspace(start=0, stop=binWidth * int(distMax / binWidth),
                                            num=1 + int(distMax / binWidth)).tolist()
                    if aDistBins[-1] < distMax:
                        aDistBins.append(distMax)

                    binWidthRInd = len(distHistBinWidths) - binWidthInd - 1
                    sDistances.plot.hist(ax=axes, bins=aDistBins, fill=None, linewidth=1,
                                         zorder=10*(1+binWidthRInd), rwidth=1-0.15*binWidthRInd,
                                         edgecolor=colors['multihistograms'][binWidthInd])

                axes.set_xlim((0, distMax))
                axes.grid(True, which='minor', zorder=0)
                aMTicks = axes.get_xticks()
                axes.tick_params(which='minor', grid_linestyle='-.', grid_alpha=0.6)
                axes.xaxis.set_minor_locator(pltt.MultipleLocator((aMTicks[1]-aMTicks[0])/5))
                axes.yaxis.set_major_locator(pltt.MaxNLocator(integer=True))

                # d. Finish plotting.
                axes.legend([self.tr('Real observations') + ' ' + str(int(binWidth))
                             for binWidth in distHistBinWidths], fontsize=fontSizes['legend'])
                axes.set_title(label=self.tr('Fixed bin distance histograms'),
                               fontdict=dict(fontsize=fontSizes['title']), pad=10)
                axes.set_xlabel(self.tr('Distance in') + ' ' + self.resultsSet.distanceUnit.lower() + 's',
                                fontsize=fontSizes['axes'])
                axes.set_ylabel(self.tr('Number of observations'), fontsize=fontSizes['axes'])
                axes.tick_params(axis='both', labelsize=fontSizes['ticks'])
                axes.grid(True, which='major', zorder=0)
                if not transparent:
                    axes.set_facecolor(colors['background'])
                    fig.patch.set_facecolor(colors['background'])
                    
                # e. Generate an image file for the plot figure (forcing the specified patch background color).
                fig.tight_layout()
                pilArgs = dict(quality=imgQuality) if imgFormat == 'jpg' else dict()
                fig.savefig(tgtFilePathName, bbox_inches='tight', transparent=transparent,
                            facecolor=axes.figure.get_facecolor(), edgecolor='none', pil_kwargs=pilArgs)

                # f. Memory cleanup (does not work in interactive mode ... but OK thanks to plt.ioff above)
                axes.clear()
                fig.clear()
                plt.close(fig)

        return dPlots
    
    # Top page
    def toHtmlAllAnalyses(self, rebuild=False):
        
        logger.debug(f'DSResultsDistanceReport.toHtmlAllAnalyses({rebuild=})')
        logger.info('Top page ...')
        
        # 1. Generate post-processed and translated synthesis table.
        # a. Add run folder column if not selected (will serve to generate the link to the analysis detailed report)
        synCols = self.synthCols
        if self.resultsSet.analysisClass.RunFolderColumn not in synCols:
            synCols += [self.resultsSet.analysisClass.RunFolderColumn]
        dfSyn = self.resultsSet.dfTransData(self.lang, columns=synCols)
        
        # b. Links to each analysis detailed report.
        idxFmt = '{{n:0{}d}}'.format(1+max(int(math.log10(len(dfSyn))), 1))
        numNavLinkFmt = '<a href="./{{p}}/index.html">{}</a>'.format(idxFmt)

        def numNavLink(sAnlys):
            return numNavLinkFmt.format(p=self.relativeRunFolderUrl(sAnlys[self.trRunFolderCol]), n=sAnlys.name)
       
        # c. Post-format as specified in actual class.
        dfsSyn = self.finalFormatAllAnalysesData(dfSyn, sort=True, indexer=numNavLink,
                                                 convert=True, round_=True, style=True)

        # 2. Generate post-processed and translated detailed table.
        dfDet = self.resultsSet.dfTransData(self.lang)

        # a. Add run folder column if not there (will serve to generate the link to the analysis detailed report)
        detTrCols = list(dfDet.columns)
        if self.trRunFolderCol not in detTrCols:
            detTrCols += [self.trRunFolderCol]
        dfDet = dfDet.reindex(columns=detTrCols)
       
        # b. Links to each analysis detailed report.
        dfsDet = self.finalFormatAllAnalysesData(dfDet, sort=True, indexer=numNavLink,
                                                 convert=False, round_=False, style=True)

        # 3. Generate traceability infos parts (results specs).
        ddfTrc = self.asWorkbook(subset=['specs'])

        # 4.Generate top report page.
        genDateTime = dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        tmpl = self.getTemplateEnv().get_template('mcds/top.htpl')
        xlFileUrl = os.path.basename(self.targetFilePathName(suffix='.xlsx')).replace(os.sep, '/')
        html = tmpl.render(synthesis=dfsSyn.render(),  # escape=False, index=False),
                           details=dfsDet.render(),  # escape=False, index=False),
                           traceability={trcName: dfTrcTable.to_html(escape=False, na_rep='')
                                         for trcName, (dfTrcTable, _) in ddfTrc.items()},
                           title=self.title, subtitle=self.subTitle.replace('\n', '<br>'),
                           description=self.description.replace('\n', '<br>'), keywords=self.keywords,
                           xlUrl=xlFileUrl, tr=self.dTrans[self.lang], 
                           pySources=[pl.Path(fpn).name for fpn in self.pySources],
                           genDateTime=genDateTime, version=__version__, libVersions=self._libVersions(),
                           distanceUnit=self.tr(self.resultsSet.distanceUnit),
                           areaUnit=self.tr(self.resultsSet.areaUnit),
                           surveyType=self.tr(self.resultsSet.surveyType),
                           distanceType=self.tr(self.resultsSet.distanceType),
                           clustering=self.tr('Clustering' if self.resultsSet.clustering else 'No clustering'))
        html = re.sub('(?:[ \t]*\n){2,}', '\n'*2, html)  # Cleanup blank line series to one only.

        # Write top HTML to file.
        htmlPathName = self.targetFilePathName(suffix='.html')
        with codecs.open(htmlPathName, mode='w', encoding='utf-8-sig') as tgtFile:
            tgtFile.write(html)

        return htmlPathName
    
    def getRawTransData(self, **kwargs):

        """Retrieve input translated raw data for HTML pages specific to each analysis

        Parameters:
        :param kwargs: Args relevant to derived classes (none here).

        :return: 2 dataFrames, for synthesis (synthCols) and detailed (all) column sets,
                 + None (other implementations may use this place for something relevant)
        """

        # Generate translated synthesis table.
        synthCols = self.synthCols
        if self.resultsSet.analysisClass.RunFolderColumn not in synthCols:
            synthCols += [self.resultsSet.analysisClass.RunFolderColumn]
        dfSynthRes = self.resultsSet.dfTransData(self.lang, columns=synthCols)

        # Generate translated detailed table.
        dfDetRes = self.resultsSet.dfTransData(self.lang)

        # Side check as soon as possible : Are all report-needed columns available ?
        self.checkNeededColumns()

        return dfSynthRes, dfDetRes, None

    def toHtmlEachAnalysis(self, rebuild=False, generators=0, topSuffix='.html', **kwargs):
        
        """Generate HTML page specific to each analysis

        Parameters:
        :param rebuild: if True, rebuild from scratch (data extraction + plots) ;
                        otherwise, use any cached data or existing plot image file
        :param generators: Number of parallel (process) generators to use :
                           - None => no parallelism used, sequential execution,
                           - 0 => auto-number, based on the actual number of CPUs onboard,
                           - > 0 => the actual number to use (Note: 1 means no parallelism,
                            but some asynchronism though, contrary to None).
        :param topSuffix: Suffix for top page HTML file
        :param kwargs: Other args relevant to derived classes.
        """

        # Get source translated raw data to format
        dfSynthRes, dfDetRes, _ = self.getRawTransData(rebuild=rebuild, **kwargs)

        # Generate translated synthesis and detailed tables.
        logger.info(f'Analyses pages ({len(dfSynthRes)}) ...')
        executor = Executor(processes=generators)
        nExpWorkers = executor.expectedWorkers()
        if nExpWorkers > 1:
            logger.info(f'... through at most {nExpWorkers} parallel generators ...')

        # 1. 1st pass : Generate previous / next list (for navigation buttons) with the sorted order if any
        dfSynthRes = self.finalformatEachAnalysisData(dfSynthRes, sort=True, indexer=True,
                                                      convert=False, round_=False, style=False).data
        sCurrUrl = dfSynthRes[self.trRunFolderCol]
        sCurrUrl = sCurrUrl.apply(lambda path: self.targetFilePathName(tgtFolder=path, prefix='index', suffix='.html'))
        sCurrUrl = sCurrUrl.apply(lambda path: os.path.relpath(path, self.tgtFolder).replace(os.sep, '/'))
        dfAnlysUrls = pd.DataFrame(dict(current=sCurrUrl, previous=np.roll(sCurrUrl, 1), next=np.roll(sCurrUrl, -1)))

        # And don't forget to sort & index detailed results the same way as synthesis ones.
        dfDetRes = self.finalformatEachAnalysisData(dfDetRes, sort=True, indexer=True,
                                                    convert=False, round_=False, style=False).data

        # 2. 2nd pass : Generate
        # a. Stops heavy Matplotlib.pyplot memory leak in generatePlots (WTF !?)
        wasInter = plt.isinteractive()
        if wasInter:
            plt.ioff()

        # b. Generate analysis detailed HTML page, for each analysis, parallely.
        topHtmlPathName = self.targetFilePathName(suffix=topSuffix)
        trCustCols = [col for col in self.resultsSet.transCustomColumns(self.lang) if col in dfDetRes.columns]
        
        # i. Start generation of all pages in parallel (unless specified not)
        genStart = pd.Timestamp.now()  # Start of elapsed time measurement.
        pages = dict()
        for lblRes in dfSynthRes.index:
            
            logger.info1(f'#{lblRes}/{len(dfSynthRes)}: '
                         + ' '.join(f'{k}={v}' for k, v in dfDetRes.loc[lblRes, trCustCols].iteritems()))

            pgFut = executor.submit(self._toHtmlAnalysis, lblRes, dfSynthRes.loc[lblRes],
                                    dfDetRes.loc[lblRes], dfAnlysUrls.loc[lblRes],
                                    topHtmlPathName, trCustCols, rebuild=rebuild)
                                    
            pages[pgFut] = lblRes
        
        if executor.isParallel():
            logger.info1(f'Waiting for generators results ...')
        
        # ii. Wait for end of generation of each page, as it comes first.
        nDone = 0
        for pgFut in executor.asCompleted(pages):

            # If there, it's because it's done (or crashed) !
            exc = pgFut.exception()
            if exc:
                logger.error(f'#{pages[pgFut]}: Exception: {exc}')
            elif executor.isParallel():
                logger.info1(f'#{pages[pgFut]}: Done.')

            # Report elapsed time and number of pages completed until now (once per self.logProgressEvery pages).
            nDone += 1
            if nDone % self.logProgressEvery == 0 or nDone == len(pages):
                now = pd.Timestamp.now()
                elapsedTilNow = now - genStart
                expectedEnd = now
                if nDone < len(pages):
                    expectedEnd += pd.Timedelta(elapsedTilNow.value * (len(pages) - nDone) / nDone)
                logger.info1('{}/{} pages in {} (mean {:.2f}s){}'
                             .format(nDone, len(pages), str(elapsedTilNow.round('S')).replace('0 days ', ''),
                                     elapsedTilNow.total_seconds() / nDone,
                                     ': done.' if nDone == len(pages)
                                               else ': should end around ' + expectedEnd.strftime('%Y-%m-%d %H:%M:%S')
                                                                             .replace(now.strftime('%Y-%m-%d '), '')))

        # iii. Terminate parallel executor.
        executor.shutdown()

        # c. Restore Matplotlib.pyplot interactive mode as it was before.
        if wasInter:
            plt.ion()

    def _toHtmlAnalysis(self, lblRes, sSynthRes, sDetRes, sResNav, topHtmlPathName, trCustCols, rebuild=True):

        # Postprocess synthesis table :
        dfSyn = pd.DataFrame([sSynthRes])
        idxFmt = '{{:0{}d}}'.format(1+max(int(math.log10(len(dfSyn))), 1))
        dfSyn[self.trRunFolderCol] = dfSyn[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
        dfSyn.index = dfSyn.index.map(lambda n: idxFmt.format(n))
        dfsSyn = self.finalformatEachAnalysisData(dfSyn, sort=False, indexer=None,
                                                  convert=True, round_=True, style=True)
        
        # Post-process detailed table :
        dfDet = pd.DataFrame([sDetRes])
        dfDet[self.trRunFolderCol] = dfDet[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
        dfDet.index = dfDet.index.map(lambda n: idxFmt.format(n))
        dfsDet = self.finalformatEachAnalysisData(dfDet, sort=False, indexer=None,
                                                  convert=False, round_=False, style=True)
        
        # Generate analysis report page.
        genDateTime = dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        subTitle = '{} {} : {{}}'.format(self.tr('Analysis'), idxFmt).format(lblRes, self.anlysSubTitle)
        engineClass = self.resultsSet.engineClass
        anlysFolder = sDetRes[self.trRunFolderCol]
        tmpl = self.getTemplateEnv().get_template('mcds/anlys.htpl')
        html = tmpl.render(synthesis=dfsSyn.render(),
                           details=dfsDet.render(),
                           log=engineClass.decodeLog(anlysFolder),
                           output=engineClass.decodeOutput(anlysFolder),
                           plots=self.generatePlots(plotsData=engineClass.decodePlots(anlysFolder), 
                                                    sDistances=engineClass.loadDataFile(anlysFolder).DISTANCE,
                                                    tgtFolder=anlysFolder, lang='en',  # No translation.
                                                    imgFormat=self.plotImgFormat, imgSize=self.plotImgSize,
                                                    imgQuality=self.plotImgQuality,
                                                    widths=dict(lines=self.plotLineWidth, dots=self.plotDotWidth),
                                                    colors=dict(background='#f9fbf3', histograms='blue',
                                                                multihistograms=['blue', 'green', 'red'],
                                                                curves='red', dots='green'),
                                                    fontSizes=self.plotFontSizes,
                                                    rebuild=rebuild),
                           title=self.title, subtitle=subTitle, keywords=self.keywords,
                           navUrls=dict(prevAnlys='../' + sResNav.previous,
                                        nextAnlys='../' + sResNav.next,
                                        back2Top='../' + os.path.basename(topHtmlPathName)),
                           tr=self.dTrans[self.lang], pySources=[pl.Path(fpn).name for fpn in self.pySources],
                           genDateTime=genDateTime, version=__version__, libVersions=self._libVersions(), 
                           distanceUnit=self.tr(self.resultsSet.distanceUnit),
                           areaUnit=self.tr(self.resultsSet.areaUnit),
                           surveyType=self.tr(self.resultsSet.surveyType),
                           distanceType=self.tr(self.resultsSet.distanceType),
                           clustering=self.tr('Clustering' if self.resultsSet.clustering else 'No clustering'))
        html = re.sub('(?:[ \t]*\n){2,}', '\n'*2, html)  # Cleanup blank line series to one only.

        # Write analysis HTML to file.
        htmlPathName = self.targetFilePathName(tgtFolder=anlysFolder, prefix='index', suffix='.html')
        with codecs.open(htmlPathName, mode='w', encoding='utf-8-sig') as tgtFile:
            tgtFile.write(html)

    def toHtml(self, rebuild=False):
    
        """HTML report generation.

        Parameters:
        :param rebuild: if True, rebuild from scratch (data extraction + plots) ;
                        otherwise, use any cached data or existing plot image file

        Note: Parallelism does not work for this class, hence the absence of the generators parameter.
        """

        logger.debug(f'DSResultsDistanceReport.toHtml({rebuild=})')

        # Install needed attached files.
        self.installAttFiles(self.AttachedFiles)
            
        # Generate synthesis report page (all analyses in one page).
        topHtmlPathName = self.toHtmlAllAnalyses(rebuild=rebuild)

        # Generate detailed report pages (one page for each analysis)
        # Note: For some obscure reason, parallelism does not work here (while it does for derived classes !).
        generators = 1
        self.toHtmlEachAnalysis(rebuild=rebuild, generators=generators)

        logger.info('... done.')
        
        return topHtmlPathName

    def asWorkbook(self, subset=None, rebuild=False):

        """Format as a "generic" workbook format, i.e. as a dict(name=(DataFrame, useIndex))
        where each item is a named worksheet

        Parameters:
        :param subset: Selected list of data categories to include ; None = [] = all
                       (categories in {'specs'})
        :param rebuild: if True force rebuild of report = prevent use of / reset any cache
                        (not used here)
        """
        
        logger.debug(f'DSResultsDistanceReport.asWorkbook({subset=}, {rebuild=})')

        ddfWbk = dict()

        # Build results worksheets if specified in subset
        if not subset or 'results' in subset:

            # Synthesis
            logger.info1('* synthesis ...')
            synCols = self.synthCols
            if self.resultsSet.analysisClass.RunFolderColumn not in synCols:
                synCols += [self.resultsSet.analysisClass.RunFolderColumn]
            dfSyn = self.resultsSet.dfTransData(self.lang, columns=synCols)
            dfSyn[self.trRunFolderCol] = dfSyn[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
            
            dfsSyn = self.finalFormatAllAnalysesData(dfSyn, sort=True, indexer=True,
                                                     convert=True, round_=True, style=True)

            ddfWbk[self.tr('Synthesis')] = (dfsSyn, True)

            # Details
            logger.info1('* details ...')
            dfDet = self.resultsSet.dfTransData(self.lang)
            dfDet[self.trRunFolderCol] = dfDet[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
            
            dfsDet = self.finalFormatAllAnalysesData(dfDet, sort=True, indexer=True,
                                                     convert=False, round_=False, style=True)

            ddfWbk[self.tr('Details')] = (dfsDet, True)

        # Append inherited worksheets.
        ddfWbk.update(super().asWorkbook(subset=subset, rebuild=rebuild))

        # Done
        return ddfWbk
 

class MCDSResultsDistanceReport(DSResultsDistanceReport):

    """A specialized report for MCDS analyses, targeting similar layout as in Distance 6+,
    with actual output formatting."""

    DTrans = _mergeTransTables(base=DSResultsDistanceReport.DTrans,
        update=dict(en={'Study type:': "<strong>Study type</strong>:",
                        'Units used:': "<strong>Units used</strong>:",
                        'for distances': 'for distances',
                        'for areas': 'for areas',
                        'Estimator selection criterion:': '<strong>Adjustment term selection criterion</strong>:',
                        'Confidence value interval:': '<strong>Confidence value interval</strong>:',
                        'If not listed in table below, please': 'If not listed in table below, please',
                        'BE AWARE that different values have been used among analyses':
                          '<strong>BE AWARE</strong> that different values have been used among analyses',
                        'note that all analyses have been run with the same value':
                          'note that <strong>ALL</strong> analyses have been run with the same value',
                        'see detailed table below': 'see detailed table below',
                        'see details for each analysis': 'see details for each analysis',
                        'Note: Some figures rounded, but not converted':
                           "<strong>Note</strong>: Most figures have been rounded for readability,"
                           " but 'CoefVar Density' have been further modified : converted to %",
                        'Note: All figures untouched, as output by MCDS':
                           "<strong>Note</strong>: All values have been left untouched,"
                           " as output by MCDS (no rounding, no conversion)",
                        'samples': 'Samples', 'analyses': 'Analyses', 'models': 'Models',
                        'analyser': 'Analyser', 'runtime': 'Computing platform'},
                    fr={'Study type:': "<strong>Type d'étude</strong>:",
                        'Units used:': "<strong>Unités utilisées</strong>:",
                        'for distances': 'pour les distances',
                        'for areas': 'pour les surfaces',
                        'Estimator selection criterion:':
                          "<strong>Critère de sélection des termes d'ajustement</strong>:",
                        'Confidence value interval:': '<strong>Intervalle de confiance</strong>:',
                        'If not listed in table below, please': 'Si non présent dans la table ci-dessous,',
                        'BE AWARE that different values have been used among analyses':
                          'faites <strong>ATTENTION</strong>,'
                          ' différentes valeurs ont été utilisés suivant les analyses',
                        'note that all analyses have been run with the same value':
                          'notez que <strong>TOUTES</strong> les analyses ont été faites avec la même valeur',
                        'see detailed table below': 'voir table de détails ci-dessous',
                        'see details for each analysis': 'voir détails de chaque analyse',
                        'Note: Some figures rounded, but not converted':
                           "<strong>N.B.</strong> Presque toutes les valeurs ont été arrondies pour la lisibilité,"
                           " mais seul 'CoefVar Densité' a été autrement modifié : converti en %",
                        'Note: All figures untouched, as output by MCDS':
                           "<strong>N.B.</strong> Aucune valeur n'a été convertie ou arrondie,"
                           " elles sont toutes telles que produites par MCDS",
                        'samples': 'Echantillons', 'analyses': 'Analyses', 'models': 'Modèles',
                        'analyser': 'Analyseur', 'runtime': 'Plateforme de calcul'}))
    
    RightTruncCol = ('encounter rate', 'right truncation distance (w)', 'Value')

    def __init__(self, resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                 synthCols=None, sortCols=None, sortAscend=None, dCustomTrans=None, lang='en',
                 plotImgFormat='png', plotImgSize=(640, 400), plotImgQuality=90,
                 plotLineWidth=2, plotDotWidth=5, plotFontSizes=dict(title=12, axes=10, ticks=9, legend=10),
                 pySources=[], tgtFolder='.', tgtPrefix='results'):

        """
        Parameters:
        :param resultsSet: source results
        :param title: main page title (and <title> tag in HTML header)
        :param subTitle: main page sub-title (under the title, lower font size)
        :param description: main page description text (under the sub-title, lower font size)
        :param anlysSubTitle: analysis pages title
        :param keywords: for HTML header <meta name="keywords" ...>
        :param synthCols: for synthesis table (Excel format only, "Synthesis" tab)
        :param sortCols: sorting columns for report tables
        :param sortAscend: sorting order for report tables, as a bool or list of bools, of len(synthCols)
        :param dCustomTrans: custom translations to complete the report standard ones,
                             as a dict(fr=dict(src: fr-trans), en=dict(src: en-trans))
        :param lang: Target language for translation
        :param plotImgFormat: png, svg and jpg all work with Matplotlib 3.2.1+
        :param plotImgSize: size of the image generated for each plot = (width, height) in pixels
        :param plotImgQuality: JPEG format quality (%) ; ignored if plotImgFormat not in ('jpg', 'jpeg')
        :param plotLineWidth: width (unit: pixel) of drawn lines (observation histograms, fitted curves)
        :param plotDotWidth: width (unit: pixel) of drawn dots / points (observation distances)
        :param plotFontSizes: font sizes (unit: point) for plots (dict with keys from title, axes, ticks, legend)
        :param pySources: path-name of source files to copy in report folder and link in report
        :param tgtFolder: target folder for the report (for _all_ generated files)
        :param tgtPrefix: default target file name for the report
         """
    
        assert isinstance(resultsSet, MCDSAnalysisResultsSet), 'resultsSet must be a MCDSAnalysisResultsSet'

        super().__init__(resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                         synthCols=synthCols, sortCols=sortCols, sortAscend=sortAscend,
                         dCustomTrans=dCustomTrans, lang=lang,
                         plotImgFormat=plotImgFormat, plotImgSize=plotImgSize, plotImgQuality=plotImgQuality,
                         plotLineWidth=plotLineWidth, plotDotWidth=plotDotWidth, plotFontSizes=plotFontSizes,
                         pySources=pySources, tgtFolder=tgtFolder, tgtPrefix=tgtPrefix)
        
    # Styling colors
    CChrGray = '#869074'
    CBckGreen, CBckGray = '#e0ef8c', '#dae3cb'
    CSclGreen, CSclOrange, CSclRed = '#cbef8c', '#f9da56', '#fe835a'
    CChrInvis = '#e8efd1'  # body background
    ScaledColors = [CSclGreen, CSclOrange, CSclRed]
    ScaledColorsRvd = list(reversed(ScaledColors))
    
    DExCodeColors = dict(zip([1, 2, 3], ScaledColors))
    
    @classmethod
    def colorExecCodes(cls, sCodes):
        return ['background-color: ' + cls.DExCodeColors.get(c, cls.DExCodeColors[3]) for c in sCodes]
    
    @classmethod
    def scaledColorV(cls, v, thresholds, colors):  # len(thresholds) == len(colors) - 1
        if pd.isnull(v):
            return cls.CBckGray
        for ind, thresh in enumerate(thresholds):
            if v > thresh:
                return colors[ind]
        return colors[-1]
    
    @classmethod
    def scaledColorS(cls, sValues, thresholds, colors):
        return ['background-color: ' + cls.scaledColorV(v, thresholds, colors) for v in sValues]
    
    @staticmethod
    def isNull(o):
        return not isinstance(o, list) and pd.isnull(o)
    
    @staticmethod
    def shortenDistCuts(distCuts):
        if isinstance(distCuts, int) or isinstance(distCuts, float):
            return distCuts
        short = str(distCuts)
        if short in ['None', 'nan']:
            return None
        else:
            return short.translate(str.maketrans({c: '' for c in '[] '})).replace('.0,', ',')

    @staticmethod
    def _roundNumber(v, ndigits=0):
        """round number with built-in round function, unless NaN, which fails to int() if ndigits=0"""
        return v if pd.isnull(v) else round(v, ndigits)

    # Final formatting of translated data tables, for HTML or SpreadSheet rendering
    # in the "all analyses at once" case.
    # (sort, generate index, convert units, round values, and style).
    # Note: Use trEnColNames method to pass from EN-translated columns names to self.lang-ones
    # Return a pd.DataFrame.Styler
    def finalFormatAllAnalysesData(self, dfTrData, sort=True, indexer=None, convert=True, round_=True, style=True):
        
        logger.debug(f'MCDSResultsDistanceReport.finalFormatAllAnalysesData'
                     f'({sort=}, {indexer=}, {convert=}, {round_=}, {style=})')

        # Sorting
        df = dfTrData
        if sort:
        
            # If no sorting order was specified, generate one simple one,
            # through a temporarily sample num. column and Delta AIC column (so ... it MUST be there)
            # (assuming analyses have been run as grouped by sample)
            if not self.sortCols:
            
                # Note: Ignoring all-NaN sample id columns, for a working groupby
                sampleIdCols = [col for col in self.resultsSet.transSampleColumns(self.lang)
                                if col in df.columns and not df[col].isna().all()]
                df.insert(0, column='#Sample#', value=df.groupby(sampleIdCols, sort=False).ngroup())

                sortCols = ['#Sample#'] + [col for col in self.trEnColNames(['Delta AIC']) if col in df.columns]
                sortAscend = True
                
            # Otherwise, use the one specified.
            else:
            
                # ... after some cleaning up in case some sort columns are not present.
                sortCols = list()
                sortAscend = list() if isinstance(self.sortAscend, list) else self.sortAscend
                for ind, col in enumerate(self.resultsSet.transColumns(self.sortCols, self.lang)):
                    if col in df.columns:
                        sortCols.append(col)
                        if isinstance(self.sortAscend, list):
                            sortAscend.append(self.sortAscend[ind])
                assert not isinstance(sortAscend, list) or len(sortCols) == len(sortAscend)

            # Sort
            df.sort_values(by=sortCols, ascending=sortAscend, inplace=True)
            
            # Remove temporary sample num. column if no sorting order was specified
            if not self.sortCols:
                df.drop(columns=['#Sample#'], inplace=True)

        # Standard 1 to N index + optional post-formatting (ex. for synthesis <=> details navigation).
        if indexer:
            df.index = range(1, len(df) + 1)
            if callable(indexer):
                df.index = df.apply(indexer, axis='columns')

        # Converting to other units, or so.
        if convert:
            
            # for col in self.trEnColNames(['Density', 'Min Density', 'Max Density']): # 'CoefVar Density',
            #     if col in df.columns:
            #         df[col] *= 1000000 / 10000 # ha => km2
            
            col = self.trEnColNames('CoefVar Density')
            if col in df.columns:
                kVarDens = 100.0
                df[col] *= kVarDens  # [0, 1] => %
            
            for col in self.trEnColNames(['Fit Dist Cuts', 'Discr Dist Cuts']):
                if col in df.columns:
                    df[col] = df[col].apply(self.shortenDistCuts)
            
        # Reducing float precision
        if round_:
            
            # Use built-in round for more accurate rounding than np.round
            # a. Fixed list of columns: enumerate their English names.
            dColDecimals = {**{col: 4 for col in ['Delta CoefVar Density']},
                            **{col: 3 for col in ['Effort', 'PDetec', 'Min PDetec', 'Max PDetec']},
                            **{col: 2 for col in ['Delta AIC', 'Chi2 P', 'KS P', 'CvM Uw P', 'CvM Cw P',
                                                  'Density', 'Min Density', 'Max Density',
                                                  'Qual Chi2+', 'Qual KS+', 'Qual DCv+']},
                            **{col: 1 for col in ['AIC', 'EDR/ESW', 'Min EDR/ESW', 'Max EDR/ESW',
                                                  'Number', 'Min Number', 'Max Number',
                                                  'CoefVar Density', 'CoefVar Number', 'Obs Rate']},
                            **{col: 0 for col in ['Left Trunc Dist', 'Right Trunc Dist',
                                                  'Left Trunc', 'Right Trunc']}}
                                                     
            for col, dec in self.trEnColNames(dColDecimals).items():
                if col in df.columns:
                    df[col] = df[col].apply(self._roundNumber, ndigits=dec)

            # b. Dynamic lists of columns: select their names through a startswith criterion.
            for col in df.columns:
                if col.startswith(self.tr('Qual Bal')):
                    df[col] = df[col].apply(self._roundNumber, ndigits=2)
                if col.startswith(self.tr('Order')) or col.startswith(self.tr('Pre-selection')):
                    df[col] = df[col].apply(self._roundNumber, ndigits=0)
            
            # Don't use df.round ... because it does not work, at least with pandas 1.0.x up to 1.1.2 !?!?!?
            # df = df.round(decimals={ col: dec for col, dec in self.trEnColNames(dColDecimals).items() \
            #                                   if col in df.columns })
            
        # Styling
        return self.styleAllAnalysesData(df, convert=convert, round_=round_, style=style)

    @staticmethod
    def _trimTrailingZeroesFormat(v):
        return '' if pd.isnull(v) else format(v, 'g') if pd.api.types.is_numeric_dtype(v) else v

    def styleAllAnalysesData(self, df, convert=True, round_=True, style=True):
    
        dfs = df.style
        
        if style:
        
            # Remove trailing (useless) zeroes in floats when rounding requested.
            if round_:
                dfs.format(self._trimTrailingZeroesFormat,
                           subset=[col for col in df.columns if df[col].dtype is np.dtype('float')])

            # Left align all-text columns
            cols = [col for col in df.columns
                    if df[col].dropna().apply(lambda v: isinstance(v, str)).all()]
            if cols:
                dfs.set_properties(subset=cols, **{'text-align': 'left'})

            # Green background for the 0-value Delta AIC rows
            col = self.trEnColNames('Delta AIC')
            if col in df.columns and df[col].max() > 0:  # if all delta AIC == 0, no need to stress it.
                dfs.set_properties(subset=pd.IndexSlice[df[df[col] == 0].index, :],
                                   **{'background-color': self.CBckGreen})
               
            # Red/Orange/Green color code for exec. codes for normal codes
            col = self.trEnColNames('ExCod')
            if col in df.columns:
                dfs.apply(self.colorExecCodes, subset=[col], axis='columns')
            
            # Red/Orange/Green color code for DCV based on thresholds
            col = self.trEnColNames('CoefVar Density')
            if col in df.columns:
                kVarDens = 100.0 if convert else 1.0
                dfs.apply(self.scaledColorS, subset=[col], axis='columns',
                          thresholds=[v * kVarDens for v in [0.3, 0.2]], colors=self.ScaledColorsRvd)
            
            # Red/Orange/Green color code for DCV based on thresholds
            col = self.trEnColNames('KS P')
            if col in df.columns:
                dfs.apply(self.scaledColorS, subset=[col], axis='columns',
                          thresholds=[0.7, 0.2], colors=self.ScaledColors)
            
            # Red/Orange/Green color code for DCV based on thresholds
            col = self.trEnColNames('Chi2 P')
            if col in df.columns:
                dfs.apply(self.scaledColorS, subset=[col], axis='columns',
                          thresholds=[0.7, 0.2], colors=self.ScaledColors)
            
            # Greyed foreground for rows with bad exec codes
            col = self.trEnColNames('ExCod')
            if col in df.columns:
                dfs.set_properties(subset=pd.IndexSlice[df[~df[col].isin([1, 2])].index, :],
                                   **{'color': self.CChrGray})
            
            # NaN cells are set to transparent foreground / no shadow (to hide NaNs).
            dfs.where(self.isNull, 'color: transparent').where(self.isNull, 'text-shadow: none')
        
        return dfs

    # Final formatting of translated data tables, for HTML or SpreadSheet rendering
    # in the "all analyses at once" case.
    # (sort, convert units, round values, and style).
    # Note: Use trEnColNames method to pass from EN-translated columns names to self.lang-ones
    # Return a pd.DataFrame.Styler
    def finalformatEachAnalysisData(self, dfTrData, sort=True, indexer=None, convert=True, round_=True, style=True):
    
        return self.finalFormatAllAnalysesData(dfTrData, sort=sort, indexer=indexer,
                                               convert=convert, round_=round_, style=style)

    @staticmethod
    def float2str(v):  # Workaround to_html non-transparent default float format (!?)
        return format(v, 'g')

    @staticmethod
    def series2VertTable(ser):
        return re.sub('\n *', '',  ser.to_frame().to_html(header=False,
                                                          float_format=MCDSResultsDistanceReport.float2str,
                                                          na_rep=''))
    
    def plotImageHtmlElement(self, runFolder, plotImgPrfx, plotHeight):
        
        if plotImgPrfx in [self.PlotImgPrfxQqPlot, self.PlotImgPrfxDistHist]:
            plotFileName = '{}.{}'.format(plotImgPrfx, self.plotImgFormat)
            if os.path.isfile(os.path.join(runFolder, plotFileName)):
                return '<img src="./{}/{}" style="height: {}px" />' \
                       .format(self.relativeRunFolderUrl(runFolder), plotFileName, plotHeight)
        else:
            for plotInd in range(3, 0, -1):
                plotFileName = '{}{}.{}'.format(plotImgPrfx, plotInd, self.plotImgFormat)
                if os.path.isfile(os.path.join(runFolder, plotFileName)):
                    return '<img src="./{}/{}" style="height: {}px" />' \
                           .format(self.relativeRunFolderUrl(runFolder), plotFileName, plotHeight)
            plotFileName = '{}.{}'.format(plotImgPrfx, self.plotImgFormat)
            if os.path.isfile(os.path.join(runFolder, plotFileName)):
                return '<img src="./{}/{}" style="height: {}px" />' \
                       .format(self.relativeRunFolderUrl(runFolder), plotFileName, plotHeight)
        
        return f'No {plotImgPrfx} plot produced'
        
    def asWorkbook(self, subset=None, rebuild=False):

        """Format as a "generic" workbook format, i.e. as a dict(name=(DataFrame, useIndex))
        where each item is a named worksheet

        Parameters:
        :param subset: Selected list of data categories to include ; None = [] = all
                       (categories in {'specs'})
        :param rebuild: if True force rebuild of report = prevent use of / reset any cache
                        (not used here)
        """
        
        logger.debug(f'MCDSResultsDistanceReport.asWorkbook({subset=}, {rebuild=})')

        ddfWbk = dict()

        baseWbk = super().asWorkbook(subset=subset, rebuild=rebuild)

        # Build results worksheets if specified in subset
        if not subset or 'samples' in subset:

            # Format sample list if not already there.
            if self.tr('samples') not in baseWbk:

                logger.info1('* samples ...')

                # But first, relocate synthesis and details sheets before the future 'samples' sheet if present.
                synthShName = self.tr('Synthesis')
                if synthShName in baseWbk:
                    ddfWbk[synthShName] = baseWbk.pop(synthShName)

                detShName = self.tr('Details')
                if detShName in baseWbk:
                    ddfWbk[detShName] = baseWbk.pop(detShName)

                # Build this missing 'samples' sheet.
                dfSamples = self.resultsSet.listSamples().copy()
                dfSamples.reset_index(inplace=True)
                dfSamples.columns = self.resultsSet.transColumns(dfSamples.columns, self.lang)
                dfSamples.set_index(self.resultsSet.transColumn(self.resultsSet.sampleIndCol, self.lang),
                                    inplace=True)

                # Add it at the end of the workbook
                ddfWbk[self.tr('samples')] = (dfSamples, True)

        # Append inherited worksheets at the end.
        ddfWbk.update(baseWbk)

        # Done
        return ddfWbk

    def toHtml(self, rebuild=False, generators=0):
        
        """HTML report generation.

        Parameters:
        :param rebuild: if True, rebuild from scratch (data extraction + plots) ;
                        otherwise, use any cached data or existing plot image file
        :param generators: Number of parallel (process) generators to use :
                           - 0 => auto-number, based on the actual number of CPUs onboard,
                           - > 0 => the actual number to use
                           Note: Parallelism works well for this class, hence the default 0.
        """

        logger.debug(f'MCDSResultsDistanceReport.toHtml({rebuild=}, {generators=})')

        # Install needed attached files.
        self.installAttFiles(self.AttachedFiles)
            
        # Generate full report detailed pages (one for each analysis)
        # (done first to have plot image files generated for top report page generation right below).
        self.toHtmlEachAnalysis(rebuild=rebuild, generators=generators)
        
        # Generate top = main report page (one for all analyses).
        topHtmlPathName = self.toHtmlAllAnalyses(rebuild=rebuild)

        logger.info('... done.')
                
        return topHtmlPathName


class MCDSResultsPreReport(MCDSResultsDistanceReport):

    """A specialized pre-report for MCDS analyses, with actual output formatting.

    HTML mode gives a specialized main page layout, with a super-synthesis table (with plots)
    in place of the synthesis and detailed tables of MCDSResultsDistanceReport ;
    detailed (linked) pages unchanged from MCDSResultsDistanceReport.

    Designed for showing results of fully automatic pre-analyses, in order to give the user
    hints about which actual analyses are to be done, and with what parameter values.
    """

    # Translation table.
    DTrans = _mergeTransTables(base=MCDSResultsDistanceReport.DTrans,
        update=dict(en={'Quick-view results': 'Results: the essence',
                        'SampleParams': 'Sample & Model',
                        'Results1': 'Results (1/2)', 'Results2': 'Results (2/2)',
                        'DistHist': 'Standard distance histogram',
                        'ProbDens': 'Detection probability density (PDF)',
                        'DetProb': 'Detection probability'},
                    fr={'Quick-view results': 'Résultats : l\'essentiel',
                        'SampleParams': 'Echantillon & Modèle',
                        'Results1': 'Résultats (1/2)', 'Results2': 'Résultats (2/2)',
                        'DistHist': 'Histogramme standard des distances',
                        'ProbDens': 'Densité de probabilité de détection (DdP)',
                        'DetProb': 'Probabilité de détection'}))

    def __init__(self, resultsSet, title, subTitle, anlysSubTitle, description, keywords, 
                 sampleCols, paramCols, resultCols, synthCols=None,
                 sortCols=None, sortAscend=None, dCustomTrans=None, lang='en',
                 superSynthPlotsHeight=288, plotImgFormat='png', plotImgSize=(640, 400), plotImgQuality=90,
                 plotLineWidth=1, plotDotWidth=4, plotFontSizes=dict(title=11, axes=10, ticks=9, legend=10),
                 pySources=[], tgtFolder='.', tgtPrefix='results'):

        """Ctor
        
        Parameters:
        :param resultsSet: source results
        :param title: main page title (and <title> tag in HTML header)
        :param subTitle: main page sub-title (under the title, lower font size)
        :param description: main page description text (under the sub-title, lower font size)
        :param anlysSubTitle: analysis pages title
        :param keywords: for HTML header <meta name="keywords" ...>
        :param sampleCols: for main page table, 1st column (top)
        :param paramCols: for main page table, 1st column (bottom)
        :param resultCols: for main page table, 2nd and 3rd columns
        :param synthCols: for synthesis table (Excel format only, "Synthesis" tab)
        :param dCustomTrans: custom translations to complete the report standard ones,
                             as a dict(fr=dict(src: fr-trans), en=dict(src: en-trans))
        :param lang: Target language for translation
        :param superSynthPlotsHeight: Display height (in pixels) of the super-synthesis table plots
        :param plotImgFormat: png, svg and jpg all work with Matplotlib 3.2.1+
        :param plotImgSize: size of the image generated for each plot = (width, height) in pixels
        :param plotImgQuality: JPEG format quality (%) ; ignored if plotImgFormat not in ('jpg', 'jpeg')
        :param plotLineWidth: width (unit: pixel) of drawn lines (observation histograms, fitted curves)
        :param plotDotWidth: width (unit: pixel) of drawn dots / points (observation distances)
        :param plotFontSizes: font sizes (unit: point) for plots (dict with keys from title, axes, ticks, legend)
        :param pySources: path-name of source files to copy in report folder and link in report
        :param tgtFolder: target folder for the report (for _all_ generated files)
        :param tgtPrefix: default target file name for the report
        """

        super().__init__(resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                         synthCols=synthCols, sortCols=sortCols, sortAscend=sortAscend,
                         dCustomTrans=dCustomTrans, lang=lang,
                         plotImgFormat=plotImgFormat, plotImgSize=plotImgSize, plotImgQuality=plotImgQuality,
                         plotLineWidth=plotLineWidth, plotDotWidth=plotDotWidth, plotFontSizes=plotFontSizes,
                         pySources=pySources, tgtFolder=tgtFolder, tgtPrefix=tgtPrefix)
        
        self.sampleCols = self.noDupColumns(sampleCols, head='Sample columns')
        self.paramCols = self.noDupColumns(paramCols, head='Parameter columns')
        self.resultCols = self.noDupColumns(resultCols, head='Result columns')
        self.superSynthPlotsHeight = superSynthPlotsHeight

    def checkNeededColumns(self):

        """Side check as soon as possible : Are all report needed columns available ?
        (now that computed columns have been ... post-computed through self.resultsSet.dfFilSorData calls)

        :raise: AssertionError if not the case
        """

        assert all(col in self.resultsSet.columns for col in self.sampleCols), \
               'Missing super-synthesis sample columns in resultsSet: {}' \
               .format(', '.join('|'.join(col) for col in self.sampleCols if col not in self.resultsSet.columns))
        assert all(col in self.resultsSet.columns for col in self.paramCols), \
               'Missing super-synthesis parameters columns in resultsSet: {}' \
               .format(', '.join('|'.join(col) for col in self.paramCols if col not in self.resultsSet.columns))
        assert all(col in self.resultsSet.columns for col in self.resultCols), \
               'Missing super-synthesis results columns in resultsSet: {}' \
               .format(', '.join('|'.join(col) for col in self.resultCols if col not in self.resultsSet.columns))

    # Final formatting of translated data tables, for HTML or SpreadSheet rendering
    # in the "all analyses at once" case.
    # (sort, convert units, round values, and style).
    # Note: Use trEnColNames method to pass from EN-translated columns names to self.lang-ones
    # Return a pd.DataFrame.Styler
    def finalFormatAllAnalysesData(self, dfTrData, sort=True, indexer=None, convert=True, round_=True, style=True):
        
        logger.debug(f'MCDSResultsPreReport.finalFormatAllAnalysesData'
                     f'({sort=}, {indexer=}, {convert=}, {round_=}, {style=})')

        df = dfTrData

        # Sorting
        if sort:
            
            # If no sorting order was specified, generate one simple one, through a temporarily sample num. column
            # (assuming analyses have been run as grouped by sample)
            if not self.sortCols:
            
                # Note: Ignoring all-NaN sample id columns, for a working groupby
                sampleIdCols = [col for col in self.resultsSet.transSampleColumns(self.lang)
                                if col in df.columns and not df[col].isna().all()]
                df.insert(0, column='#Sample#', value=df.groupby(sampleIdCols, sort=False).ngroup())

                sortCols = ['#Sample#']
                sortAscend = True
                
            # Otherwise, use the one specified.
            else:
            
                # ... after some cleaning up in case some sort columns are not present.
                sortCols = list()
                sortAscend = list() if isinstance(self.sortAscend, list) else self.sortAscend
                for ind, col in enumerate(self.resultsSet.transColumns(self.sortCols, self.lang)):
                    if col in df.columns:
                        sortCols.append(col)
                        if isinstance(self.sortAscend, list):
                            sortAscend.append(self.sortAscend[ind])
                assert not isinstance(sortAscend, list) or len(sortCols) == len(sortAscend)
                assert len(sortCols) > 0

            # Sort
            df.sort_values(by=sortCols, ascending=sortAscend, inplace=True)
            
            # Remove temporary sample num. column if no sorting order was specified
            if not self.sortCols:
                df.drop(columns=['#Sample#'], inplace=True)
        
        # Standard 1 to N index for synthesis <=> details navigation.
        if indexer:
            df.index = range(1, len(df) + 1)

        # Converting to other units, or so.
        if convert:
            
            col = self.trEnColNames('CoefVar Density')
            if col in df.columns:
                kVarDens = 100.0
                df[col] *= kVarDens  # [0, 1] => %
            
        # Reducing float precision
        if round_:
            
            # Use built-in round for more accurate rounding than np.round
            # a. Fixed list of columns: simply enumerate their English names.
            dColDecimals = {**{col: 3 for col in ['PDetec', 'Min PDetec', 'Max PDetec']},
                            **{col: 2 for col in ['Delta AIC', 'Chi2 P', 'KS P', 'CvM Uw P', 'CvM Cw P',
                                                  'Density', 'Min Density', 'Max Density',
                                                  'Qual Chi2+', 'Qual KS+', 'Qual DCv+']},
                            **{col: 1 for col in ['AIC', 'EDR/ESW', 'Min EDR/ESW', 'Max EDR/ESW',
                                                  'Number', 'Min Number', 'Max Number',
                                                  'CoefVar Density', 'CoefVar Number', 'Obs Rate']},
                            **{col: 0 for col in ['Left Trunc', 'Right Trunc']}}
            for col, dec in self.trEnColNames(dColDecimals).items():
                if col in df.columns:
                    df[col] = df[col].apply(self._roundNumber, ndigits=dec)

            # b. Dynamic lists of columns: select their names through a startswith criterion.
            for col in df.columns:
                if col.startswith(self.tr('Qual Bal')):
                    df[col] = df[col].apply(self._roundNumber, ndigits=2)

            # Don't use df.round ... because it does nothing, at least with pandas to 1.1.2 !?!?!?
            # df = df.round(decimals={col: dec for col, dec in self.trEnColNames(dColDecimals).items()
            #                         if col in df.columns})

        # Styling
        return self.styleAllAnalysesData(df, convert=convert, round_=round_, style=style)

    # Top page
    def toHtmlAllAnalyses(self, rebuild=False):
        
        logger.info('Top page ...')
        logger.debug(f'MCDSResultsPreReport.toHtmlAllAnalyses({rebuild=})')

        # Generate the table to display from raw results 
        # (index + 5 columns : sample, params, results, ProbDens plot, DetProb plot)
        # 1. Get translated detailed results
        dfDet = self.resultsSet.dfTransData(self.lang)

        # 2. List estimator selection criterion and CV interval values used
        #    (these values MUST be notified in the report, even if not in the selected columns to show)
        estimSelCrits = dfDet[self.trEnColNames('Mod Chc Crit')].unique().tolist()
        confIntervals = dfDet[self.trEnColNames('Conf Interv')].unique().tolist()

        # 3. Post-format results (styling not used later, so don't do it).
        dfsDet = self.finalFormatAllAnalysesData(dfDet, sort=True, indexer=True,
                                                 convert=True, round_=True, style=False)
        dfDet = dfsDet.data

        # 4. Translate sample, parameter and result columns
        sampleTrCols = self.resultsSet.transColumns(self.sampleCols, self.lang)
        paramTrCols = self.resultsSet.transColumns(self.paramCols, self.lang)
        result1TrCols = self.resultsSet.transColumns(self.resultCols, self.lang)

        midResInd = len(result1TrCols) // 2 + len(result1TrCols) % 2
        result2TrCols = result1TrCols[midResInd:]
        result1TrCols = result1TrCols[:midResInd]
        
        # 5. Fill target table index and columns
        dfSupSyn = pd.DataFrame(dict(SampleParams=dfDet[sampleTrCols + paramTrCols].apply(self.series2VertTable,
                                                                                          axis='columns'),
                                     Results1=dfDet[result1TrCols].apply(self.series2VertTable, axis='columns'),
                                     Results2=dfDet[result2TrCols].apply(self.series2VertTable, axis='columns'),
                                     DistHist=dfDet[self.trRunFolderCol].apply(self.plotImageHtmlElement,
                                                                               plotImgPrfx=self.PlotImgPrfxDistHist,
                                                                               plotHeight=self.superSynthPlotsHeight),
                                     ProbDens=dfDet[self.trRunFolderCol].apply(self.plotImageHtmlElement,
                                                                               plotImgPrfx=self.PlotImgPrfxProbDens,
                                                                               plotHeight=self.superSynthPlotsHeight),
                                     DetProb=dfDet[self.trRunFolderCol].apply(self.plotImageHtmlElement,
                                                                              plotImgPrfx=self.PlotImgPrfxDetProb,
                                                                              plotHeight=self.superSynthPlotsHeight)))
        
        idxFmt = '{{n:0{}d}}'.format(1+max(int(math.log10(len(dfSupSyn))), 1))
        numNavLinkFmt = '<a href="./{{p}}/index.html">{}</a>'.format(idxFmt)

        def numNavLink(sAnlys):
            return numNavLinkFmt.format(p=self.relativeRunFolderUrl(sAnlys[self.trRunFolderCol]), n=sAnlys.name)
        dfSupSyn.index = dfDet.apply(numNavLink, axis='columns')
        
        # 6. Translate table columns.
        dfSupSyn.columns = [self.tr(col) for col in dfSupSyn.columns]

        # 7. Generate traceability infos parts (results specs).
        ddfTrc = self.asWorkbook(subset=['specs', 'samples'])

        # 8. Generate top report page.
        genDateTime = dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        tmpl = self.getTemplateEnv().get_template('mcds/pretop.htpl')
        xlFileUrl = os.path.basename(self.targetFilePathName(suffix='.xlsx')).replace(os.sep, '/')
        html = tmpl.render(supersynthesis=dfSupSyn.to_html(escape=False),
                           traceability={trcName: dfTrcTable.to_html(escape=False, na_rep='')
                                         for trcName, (dfTrcTable, _) in ddfTrc.items()},
                           title=self.title, subtitle=self.subTitle.replace('\n', '<br>'),
                           description=self.description.replace('\n', '<br>'), keywords=self.keywords,
                           xlUrl=xlFileUrl, tr=self.dTrans[self.lang],
                           pySources=[pl.Path(fpn).name for fpn in self.pySources],
                           genDateTime=genDateTime, version=__version__, libVersions=self._libVersions(), 
                           distanceUnit=self.tr(self.resultsSet.distanceUnit),
                           areaUnit=self.tr(self.resultsSet.areaUnit),
                           surveyType=self.tr(self.resultsSet.surveyType),
                           distanceType=self.tr(self.resultsSet.distanceType),
                           clustering=self.tr('Clustering' if self.resultsSet.clustering else 'No clustering'),
                           estimSelCrits=estimSelCrits, confIntervals=[str(v) for v in confIntervals])
        html = re.sub('(?:[ \t]*\n){2,}', '\n'*2, html)  # Cleanup blank lines series to one only

        # 9. Write top HTML to file.
        htmlPathName = self.targetFilePathName(suffix='.html')
        with codecs.open(htmlPathName, mode='w', encoding='utf-8-sig') as tgtFile:
            tgtFile.write(html)

        return htmlPathName


class MCDSResultsFullReport(MCDSResultsDistanceReport):

    """A specialized full report for MCDS analyses, with actual output formatting.

    HTML mode gives a mix of Distance and PreReport main page layout,
    with a super-synthesis table (with plots), a synthesis table, and a detailed table ;
    detailed table unchanged from MCDSResultsDistanceReport
    detailed (linked) pages unchanged from MCDSResultsDistanceReport.
    """

    # Translation table.
    DTrans = _mergeTransTables(base=MCDSResultsDistanceReport.DTrans,
        update=dict(en={'Quick-view results': 'Results: the essence',
                        'SampleParams': 'Sample & Model',
                        'Results1': 'Results (1/2)', 'Results2': 'Results (2/2)',
                        'QqPlot': 'Quantile-quantile plot',
                        'ProbDens': 'Detection probability density (PDF)',
                        'DetProb': 'Detection probability'},
                    fr={'Quick-view results': 'Résultats : l\'essentiel',
                        'SampleParams': 'Echantillon & Modèle',
                        'Results1': 'Résultats (1/2)', 'Results2': 'Résultats (2/2)',
                        'QqPlot': 'Diagramme quantile-quantile',
                        'ProbDens': 'Densité de probabilité de détection (DdP)',
                        'DetProb': 'Probabilité de détection'}))
    
    def __init__(self, resultsSet, title, subTitle, anlysSubTitle, description, keywords, 
                 sampleCols, paramCols, resultCols, synthCols=None, sortCols=None, sortAscend=None,
                 dCustomTrans=None, lang='en',
                 superSynthPlotsHeight=288, plotImgFormat='png', plotImgSize=(640, 400), plotImgQuality=90,
                 plotLineWidth=1, plotDotWidth=4, plotFontSizes=dict(title=11, axes=10, ticks=9, legend=10),
                 pySources=[], tgtFolder='.', tgtPrefix='results'):

        """Ctor
        
        Parameters:
        :param resultsSet: source results
        :param title: main page title (and <title> tag in HTML header)
        :param subTitle: main page sub-title (under the title, lower font size)
        :param description: main page description text (under the sub-title, lower font size)
        :param anlysSubTitle: analysis pages title
        :param keywords: for HTML header <meta name="keywords" ...>
        :param sampleCols: for main page table, 1st column (top)
        :param paramCols: for main page table, 1st column (bottom)
        :param resultCols: for main page table, 2nd and 3rd columns
        :param synthCols: for synthesis table (Excel format only, "Synthesis" tab)
        :param sortCols: sorting columns for report tables
        :param sortAscend: sorting order for report tables, as a bool or list of bools, of len(synthCols)
        :param dCustomTrans: custom translations to complete the report standard ones,
                             as a dict(fr=dict(src: fr-trans), en=dict(src: en-trans))
        :param lang: Target language for translation
        :param superSynthPlotsHeight: Display height (in pixels) of the super-synthesis table plots
        :param plotImgFormat: png, svg and jpg all work with Matplotlib 3.2.1+
        :param plotImgSize: size of the image generated for each plot = (width, height) in pixels
        :param plotImgQuality: JPEG format quality (%) ; ignored if plotImgFormat not in ('jpg', 'jpeg')
        :param plotLineWidth: width (unit: pixel) of drawn lines (observation histograms, fitted curves)
        :param plotDotWidth: width (unit: pixel) of drawn dots / points (observation distances)
        :param plotFontSizes: font sizes (unit: point) for plots (dict with keys from title, axes, ticks, legend)
        :param pySources: path-name of source files to copy in report folder and link in report
        :param tgtFolder: target folder for the report (for _all_ generated files)
        :param tgtPrefix: default target file name for the report
        """

        super().__init__(resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                         synthCols=synthCols, sortCols=sortCols, sortAscend=sortAscend, 
                         dCustomTrans=dCustomTrans, lang=lang,
                         plotImgFormat=plotImgFormat, plotImgSize=plotImgSize, plotImgQuality=plotImgQuality,
                         plotLineWidth=plotLineWidth, plotDotWidth=plotDotWidth, plotFontSizes=plotFontSizes,
                         pySources=pySources, tgtFolder=tgtFolder, tgtPrefix=tgtPrefix)
        
        self.sampleCols = self.noDupColumns(sampleCols, head='Sample columns')
        self.paramCols = self.noDupColumns(paramCols, head='Parameter columns')
        self.resultCols = self.noDupColumns(resultCols, head='Result columns')
        self.superSynthPlotsHeight = superSynthPlotsHeight

    def checkNeededColumns(self):

        """Side check as soon as possible : Are all report needed columns available ?
        (now that computed columns have been ... post-computed through self.resultsSet.dfFilSorData calls)

        :raise: AssertionError if not the case
        """

        assert all(col in self.resultsSet.columns for col in self.sampleCols), \
               'Missing super-synthesis sample columns in resultsSet: {}' \
               .format(', '.join('|'.join(col) for col in self.sampleCols if col not in self.resultsSet.columns))
        assert all(col in self.resultsSet.columns for col in self.paramCols), \
               'Missing super-synthesis parameters columns in resultsSet: {}' \
               .format(', '.join('|'.join(col) for col in self.paramCols if col not in self.resultsSet.columns))
        assert all(col in self.resultsSet.columns for col in self.resultCols), \
               'Missing super-synthesis results columns in resultsSet: {}' \
               .format(', '.join('|'.join(col) for col in self.resultCols if col not in self.resultsSet.columns))

    # Top page (based on results.dfTransData).
    def toHtmlAllAnalyses(self, rebuild=False):
        
        logger.info('Top page ...')
        
        # 1. Get source translated raw data to format
        dfSynRes, dfDetRes, _ = self.getRawTransData(rebuild=rebuild)

        # 2. List estimator selection criterion and CV interval values used
        #    (these values MUST be notified in the report, even if not in the selected columns to show)
        estimSelCrits = dfDetRes[self.trEnColNames('Mod Chc Crit')].unique().tolist()
        confIntervals = dfDetRes[self.trEnColNames('Conf Interv')].unique().tolist()

        # 3. Super-synthesis: Generate post-processed and translated table.
        #    (index + 5 columns : sample + params, results, Qq plot, ProbDens plot, DetProb plot)
        # a. Get translated and post-formatted detailed results
        dfDet = dfDetRes.copy()  # Also needed as is below.
        
        # b. Styling not used for super-synthesis, so don't do it.
        dfsDet = self.finalFormatAllAnalysesData(dfDet, sort=True, indexer=True,
                                                 convert=True, round_=True, style=False)
        dfDet = dfsDet.data

        # c. Translate sample, parameter and result columns
        sampleTrCols = self.resultsSet.transColumns(self.sampleCols, self.lang)
        paramTrCols = self.resultsSet.transColumns(self.paramCols, self.lang)
        result1TrCols = self.resultsSet.transColumns(self.resultCols, self.lang)

        midResInd = len(result1TrCols) // 2 + len(result1TrCols) % 2
        result2TrCols = result1TrCols[midResInd:]
        result1TrCols = result1TrCols[:midResInd]
        
        # d. Fill target table index and columns
        dfSupSyn = pd.DataFrame(dict(SampleParams=dfDet[sampleTrCols + paramTrCols].apply(self.series2VertTable,
                                                                                          axis='columns'),
                                     Results1=dfDet[result1TrCols].apply(self.series2VertTable, axis='columns'),
                                     Results2=dfDet[result2TrCols].apply(self.series2VertTable, axis='columns'),
                                     QqPlot=dfDet[self.trRunFolderCol].apply(self.plotImageHtmlElement,
                                                                             plotImgPrfx=self.PlotImgPrfxQqPlot,
                                                                             plotHeight=self.superSynthPlotsHeight),
                                     ProbDens=dfDet[self.trRunFolderCol].apply(self.plotImageHtmlElement,
                                                                               plotImgPrfx=self.PlotImgPrfxProbDens,
                                                                               plotHeight=self.superSynthPlotsHeight),
                                     DetProb=dfDet[self.trRunFolderCol].apply(self.plotImageHtmlElement,
                                                                              plotImgPrfx=self.PlotImgPrfxDetProb,
                                                                              plotHeight=self.superSynthPlotsHeight)))
        
        idxFmt = '{{n:0{}d}}'.format(1+max(int(math.log10(len(dfSupSyn))), 1))
        numNavLinkFmt = '<a href="./{{p}}/index.html">{}</a>'.format(idxFmt)

        def numNavLink(sAnlys):
            return numNavLinkFmt.format(p=self.relativeRunFolderUrl(sAnlys[self.trRunFolderCol]), n=sAnlys.name)
        dfSupSyn.index = dfDet.apply(numNavLink, axis='columns')
        
        # e. Translate table columns.
        dfSupSyn.columns = [self.tr(col) for col in dfSupSyn.columns]

        # 4. Synthesis: Generate post-processed and translated table.
        # a. Add run folder column if not selected (will serve to generate the link to the analysis detailed report)
        dfSyn = dfSynRes
        dfSyn[self.trRunFolderCol] = dfSyn[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
        
        # b. Links to each analysis detailed report.
        idxFmt = '{{n:0{}d}}'.format(1+max(int(math.log10(len(dfSyn))), 1))
        numNavLinkFmt = '<a href="./{{p}}/index.html">{}</a>'.format(idxFmt)

        def numNavLink(sAnlys):
            return numNavLinkFmt.format(p=sAnlys[self.trRunFolderCol], n=sAnlys.name)
       
        # c. Post-format as specified in actual class.
        dfsSyn = self.finalFormatAllAnalysesData(dfSyn, sort=True, indexer=numNavLink,
                                                 convert=True, round_=True, style=True)

        # 5. Details: Generate post-processed and translated table.
        dfDet = dfDetRes

        # a. Add run folder column if not there (will serve to generate the link to the analysis detailed report)
        detTrCols = list(dfDet.columns)
        if self.trRunFolderCol not in detTrCols:
            detTrCols += [self.trRunFolderCol]
        dfDet[self.trRunFolderCol] = dfDet[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
        dfDet = dfDet.reindex(columns=detTrCols)
       
        # b. Links to each analysis detailed report.
        dfsDet = self.finalFormatAllAnalysesData(dfDet, sort=True, indexer=numNavLink,
                                                 convert=False, round_=False, style=True)

        # 6. Generate traceability infos parts (results specs).
        ddfTrc = self.asWorkbook(subset=['specs', 'samples'])

        # Generate top report page.
        genDateTime = dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        tmpl = self.getTemplateEnv().get_template('mcds/fulltop.htpl')
        xlFileUrl = os.path.basename(self.targetFilePathName(suffix='.xlsx')).replace(os.sep, '/')
        html = tmpl.render(supersynthesis=dfSupSyn.to_html(escape=False),
                           synthesis=dfsSyn.render(),  # escape=False, index=False),
                           details=dfsDet.render(),  # escape=False, index=False),
                           traceability={trcName: dfTrcTable.to_html(escape=False, na_rep='')
                                         for trcName, (dfTrcTable, _) in ddfTrc.items()},
                           title=self.title, subtitle=self.subTitle.replace('\n', '<br>'),
                           description=self.description.replace('\n', '<br>'), keywords=self.keywords,
                           xlUrl=xlFileUrl, tr=self.dTrans[self.lang],
                           pySources=[pl.Path(fpn).name for fpn in self.pySources],
                           genDateTime=genDateTime, version=__version__, libVersions=self._libVersions(),
                           distanceUnit=self.tr(self.resultsSet.distanceUnit),
                           areaUnit=self.tr(self.resultsSet.areaUnit),
                           surveyType=self.tr(self.resultsSet.surveyType),
                           distanceType=self.tr(self.resultsSet.distanceType),
                           clustering=self.tr('Clustering' if self.resultsSet.clustering else 'No clustering'),
                           estimSelCrits=estimSelCrits, confIntervals=[str(v) for v in confIntervals])
        html = re.sub('(?:[ \t]*\n){2,}', '\n'*2, html)  # Cleanup blank lines series to one only

        # 7. Write top HTML to file.
        htmlPathName = self.targetFilePathName(suffix='.html')
        with codecs.open(htmlPathName, mode='w', encoding='utf-8-sig') as tgtFile:
            tgtFile.write(html)

        return htmlPathName


class MCDSResultsFilterSortReport(MCDSResultsFullReport):

    """A specialized filtered and sorted full report for MCDS analyses, with actual output formatting
    and above all auto-filtered and sorted results aiming at showing the few best results to the user
    among which to (manually) select THE best (for each sample).

    Just like MCDSResultsFullReport, but for filtered and sorted results for 1-only scheme.
    """

    # Translation table.
    DTrans = _mergeTransTables(base=MCDSResultsFullReport.DTrans,
        update=dict(en={'Scheme': 'Scheme', 'Step': 'Step',
                        'Property': 'Property', 'Value': 'Value',
                        'AFS': 'AFS', 'AFSM': 'AFSM', 'Steps': 'Steps',
                        'Filter & Sort steps': 'Filter and sort steps'},
                    fr={'Scheme': 'Méthode', 'Step': 'Etape',
                        'Property': 'Propriété', 'Value': 'Valeur',
                        'AFS': 'FTA', 'AFSM': 'MFTA', 'Steps': 'Etapes',
                        'Filter & Sort steps': 'Etapes de filtrage et tri'}))

    ResClass = MCDSAnalysisResultsSet

    def __init__(self, resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                 sampleCols, paramCols, resultCols, synthCols=None, sortCols=None, sortAscend=None,
                 dCustomTrans=None, lang='en',
                 filSorSchemes=[dict(method=ResClass.filterSortOnExecCode)],
                 superSynthPlotsHeight=288, plotImgFormat='png', plotImgSize=(640, 400), plotImgQuality=90,
                 plotLineWidth=1, plotDotWidth=4, plotFontSizes=dict(title=11, axes=10, ticks=9, legend=10),
                 pySources=[], tgtFolder='.', tgtPrefix='results'):

        """Ctor
        
        Parameters:
        :param resultsSet: source results (an instance of MCDSAnalysisResultsSet, or subclass,
                                           named ResClass or R below)
        :param title: main page title (and <title> tag in HTML header) for the HTML report (1 scheme only)
        :param subTitle: main page sub-title (under the title, lower font size) for the HTML report (1 scheme only) ;
                         any {fsMeth} placeholder will get replaced by the method name of the used filter sort scheme 
        :param description: main page description text (under the sub-title, lower font size)
                            for the HTML report (1 scheme only) ; any {fsMeth} placeholder formatted as in subTitle
        :param anlysSubTitle: analysis pages title
        :param keywords: for HTML header <meta name="keywords" ...>
        :param sampleCols: for main page table, 1st column (top)
        :param paramCols: for main page table, 1st column (bottom)
        :param resultCols: for main page table, 2nd and 3rd columns
        :param synthCols: Subset and order of columns to keep at the end (before translation)
                          as the synthesis table of each filter-sort sub-report (None = [] = all)
                          Warning: No need to specify here pre-selection and final selection columns,
                                   as they'll be added automatically, and relocated at a non-customisable place.
        :param sortCols: sorting columns for report tables ??? which ones ???
        :param sortAscend: sorting order for report tables, as a bool or list of bools,
                           of len(synthCols) ??? which ones ???
        :param dCustomTrans: custom translations to complete the report standard ones,
                             as a dict(fr=dict(src: fr-trans), en=dict(src: en-trans))
        :param lang: Target language for translation
        :param filSorSchemes: filter and sort schemes to apply in order to generate each sub-report
                 as a list of dict(method= <results set class>.filterSortOnXXX method to use,
                                   deduplicate= dict(dupSubset=, dDupRounds=) of deduplication params
                                       (if not or partially given, see RCLS.filterSortOnXXX defaults)
                                   filterSort= dict of other <method>-specific params,
                                   preselCols= target columns for generating auto-preselection ones,
                                               containing [1, preselNum] ranks ; default: []
                                   preselAscs= Rank direction to use for each column (list),
                                               or all (single bool) ; default: True
                                               (True means that lower values are "better" ones)
                                   preselThrhs= Eliminating threshold for each column (list),
                                                or all (single number) ; default: 0.2
                                                (eliminated above if preselAscs True, below otherwise)
                                   preselNum= number of (best) pre-selections to keep for each sample) ;
                                              default: 5
                 example: [dict(method=R.filterSortOnExecCode,  # let R = MCDSTruncOptanalysisResultsSet
                                preselCols=[R.CLCmbQuaBal1, R.CLCmbQuaBal2], preselAscs=False,
                                preselThrhs=0.2, preselNum=5),
                           dict(method=R.filterSortOnExCAicMulQua,
                                deduplicate=dict(dupSubset=[R.CLNObs, R.CLEffort, R.CLDeltaAic, R.CLChi2,
                                                            R.CLKS, R.CLCvMUw, R.CLCvMCw, R.CLDCv]),
                                                 dDupRounds={R.CLDeltaAic: 1, R.CLChi2: 2, R.CLKS: 2,
                                                             R.CLCvMUw: 2, R.CLCvMCw: 2, R.CLDCv: 2})
                                filterSort=dict(sightRate=92.5, nBestAIC=3, nBestQua=1,
                                                whichBestQua=[R.CLGrpOrdClTrChi2KSDCv, R.CLGrpOrdClTrQuaBal3],
                                                nFinalRes=12, whichFinalQua=R.CLCmbQuaBal3, ascFinalQua=False),
                                preselCols=[R.CLCmbQuaBal1, R.CLDCv], preselAscs=[False, True],
                                preselThrhs=[0.2, 0.5], preselNum=3)]        
        :param superSynthPlotsHeight: Display height (in pixels) of the super-synthesis table plots
        :param plotImgFormat: png, svg and jpg all work with Matplotlib 3.2.1+
        :param plotImgSize: size of the image generated for each plot = (width, height) in pixels
        :param plotImgQuality: JPEG format quality (%) ; ignored if plotImgFormat not in ('jpg', 'jpeg')
        :param plotLineWidth: width (unit: pixel) of drawn lines (observation histograms, fitted curves)
        :param plotDotWidth: width (unit: pixel) of drawn dots / points (observation distances)
        :param plotFontSizes: font sizes (unit: point) for plots (dict with keys from title, axes, ticks, legend)
        :param pySources: path-name of source files to copy in report folder and link in report
        :param tgtFolder: target folder for the report (for _all_ generated files)
        :param tgtPrefix: default target file name for the report
        """

        super().__init__(resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                         sampleCols=sampleCols, paramCols=paramCols, resultCols=resultCols, synthCols=synthCols,
                         sortCols=sortCols, sortAscend=sortAscend, dCustomTrans=dCustomTrans, lang=lang,
                         superSynthPlotsHeight=superSynthPlotsHeight,
                         plotImgFormat=plotImgFormat, plotImgSize=plotImgSize, plotImgQuality=plotImgQuality,
                         plotLineWidth=plotLineWidth, plotDotWidth=plotDotWidth, plotFontSizes=plotFontSizes,
                         pySources=pySources, tgtFolder=tgtFolder, tgtPrefix=tgtPrefix)
        
        self.filSorSchemes = filSorSchemes

    def asWorkbook(self, subset=None, rebuild=False):

        """Format as a "generic" workbook format, i.e. as a dict(name=(DataFrame, useIndex))
        where each item is a named worksheet

        Parameters:
        :param subset: Selected list of data categories to include ; None = [] = all
                       (categories in {'specs', 'samples', 'results'})
        :param rebuild: If True, force rebuild of filtered and sorted sub-report
                        => prevent use of / reset results set filter & sort cache
        """
        
        logger.debug(f'MCDSResultsFilterSortReport.asWorkbook({subset=}, {rebuild=})')

        ddfWbk = dict()

        logger.info('Formatting FilterSort sub-reports as a workbook ...')

        # Build results worksheets if specified in subset
        if not subset or 'results' in subset:

            # TODO: Add better formatting (color, ... etc)

            # For each filter and sort scheme:
            repLog = list()
            for scheme in self.filSorSchemes:

                logger.info1('* filter & sort "{}" scheme ...'
                             .format(self.resultsSet.filSorSchemeId(scheme)))

                # Apply it
                filSorSchId, dfFilSorRes, filSorSteps = \
                    self.resultsSet.dfFilSorData(scheme=scheme, rebuild=rebuild,
                                                 columns=self.synthCols, lang=self.lang)

                # Store results in workbook
                ddfWbk['-'.join([self.tr('AFSM'), filSorSchId])] = (dfFilSorRes, False)

                # Update all-scheme log
                repLog += filSorSteps

            # Log of opérations, for traceability.
            logger.info1('* filter & sort steps ...')

            indexCols = [self.tr(col) for col in ['Scheme', 'Step']]
            dataCols = [self.tr(col) for col in ['Property', 'Value']]
            dfFilSorHist = pd.DataFrame(repLog, columns=indexCols + dataCols)
            dfFilSorHist.set_index(indexCols, inplace=True)

            ddfWbk['-'.join([self.tr('AFS'), self.tr('Steps')])] = (dfFilSorHist, True)

        # Append inherited worksheets.
        ddfWbk.update(super().asWorkbook(subset=subset, rebuild=rebuild))

        # Done
        logger.info('... done.')

        return ddfWbk

    def getRawTransData(self, filSorScheme=dict(method=ResClass.filterSortOnExecCode),
                        rebuild=False):

        """Retrieve input translated raw data to be formatted

        :return: 2 dataFrames, for synthesis (synCols) and detailed (all) column sets,
                 + the id of the applied scheme and the log of its application. 
        """

        # Generate translated synthesis table.
        synthCols = self.synthCols
        if self.resultsSet.analysisClass.RunFolderColumn not in synthCols:
            synthCols += [self.resultsSet.analysisClass.RunFolderColumn]
        filSorSchId, dfFilSorSynRes, filSorSteps = \
            self.resultsSet.dfFilSorData(scheme=filSorScheme, rebuild=rebuild, columns=synthCols, lang=self.lang)

        # Generate translated detailed table.
        _, dfFilSorDetRes, _ = \
            self.resultsSet.dfFilSorData(scheme=filSorScheme, rebuild=rebuild, lang=self.lang)

        # Side check as soon as possible : Are all report needed columns available ?
        self.checkNeededColumns()

        return dfFilSorSynRes, dfFilSorDetRes, (filSorSchId, filSorSteps)

    def toHtmlAllAnalyses(self, filSorScheme=dict(method=ResClass.filterSortOnExecCode),
                          rebuild=False):

        """Top page for a given scheme.
        """

        logger.info('Top page ...')
        
        # 1. Get source translated raw data to format (post-compute, filter and sort, extract)
        dfSynRes, dfDetRes, (filSorSchId, filSorSteps) = \
            self.getRawTransData(filSorScheme=filSorScheme, rebuild=rebuild)

        # 2. List estimator selection criterion and CV interval values used
        #    (these values MUST be notified in the report, even if not in the selected columns to show)
        estimSelCrits = dfDetRes[self.trEnColNames('Mod Chc Crit')].unique().tolist()
        confIntervals = dfDetRes[self.trEnColNames('Conf Interv')].unique().tolist()

        # 3. Super-synthesis: Format filtered and translated data.
        #    (index + 5 columns : sample + params, results, Qq plot, ProbDens plot, DetProb plot)
        dfDet = dfDetRes.copy()

        # a. Styling not used for super-synthesis, so don't do it.
        dfsDet = self.finalFormatAllAnalysesData(dfDet, sort=True, indexer=True,
                                                 convert=True, round_=True, style=False)
        dfDet = dfsDet.data

        # b. Translate sample, parameter and result columns
        sampleTrCols = self.resultsSet.transColumns(self.sampleCols, self.lang)
        paramTrCols = self.resultsSet.transColumns(self.paramCols, self.lang)
        result1TrCols = self.resultsSet.transColumns(self.resultCols, self.lang)

        midResInd = len(result1TrCols) // 2 + len(result1TrCols) % 2
        result2TrCols = result1TrCols[midResInd:]
        result1TrCols = result1TrCols[:midResInd]
        
        # c. Fill target table index and columns
        dfSupSyn = \
            pd.DataFrame(dict(SampleParams=dfDet[sampleTrCols + paramTrCols].apply(self.series2VertTable,
                                                                                   axis='columns'),
                              Results1=dfDet[result1TrCols].apply(self.series2VertTable, axis='columns'),
                              Results2=dfDet[result2TrCols].apply(self.series2VertTable, axis='columns'),
                              QqPlot=dfDet[self.trRunFolderCol].apply(self.plotImageHtmlElement,
                                                                      plotImgPrfx=self.PlotImgPrfxQqPlot,
                                                                      plotHeight=self.superSynthPlotsHeight),
                              ProbDens=dfDet[self.trRunFolderCol].apply(self.plotImageHtmlElement,
                                                                        plotImgPrfx=self.PlotImgPrfxProbDens,
                                                                        plotHeight=self.superSynthPlotsHeight),
                              DetProb=dfDet[self.trRunFolderCol].apply(self.plotImageHtmlElement,
                                                                       plotImgPrfx=self.PlotImgPrfxDetProb,
                                                                       plotHeight=self.superSynthPlotsHeight)))
        
        idxFmt = '{{n:0{}d}}'.format(1+max(int(math.log10(len(dfSupSyn))), 1))
        numNavLinkFmt = '<a href="./{{p}}/index.html">{}</a>'.format(idxFmt)

        def numNavLink(sAnlys):
            return numNavLinkFmt.format(p=self.relativeRunFolderUrl(sAnlys[self.trRunFolderCol]), n=sAnlys.name)
        dfSupSyn.index = dfDet.apply(numNavLink, axis='columns')
        
        # d. Translate table columns.
        dfSupSyn.columns = [self.tr(col) for col in dfSupSyn.columns]

        # 4. Synthesis: Format filtered and translated data.
        # a. Add run folder column if not selected (will serve to generate the link to the analysis detailed report)
        dfSyn = dfSynRes
        dfSyn[self.trRunFolderCol] = dfSyn[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
        
        # b. Links to each analysis detailed report.
        idxFmt = '{{n:0{}d}}'.format(1 + max(int(math.log10(len(dfSyn))), 1))
        numNavLinkFmt = '<a href="./{{p}}/index.html">{}</a>'.format(idxFmt)

        def numNavLink(sAnlys):
            return numNavLinkFmt.format(p=sAnlys[self.trRunFolderCol], n=sAnlys.name)
       
        # c. Post-format as specified in actual class.
        dfsSyn = self.finalFormatAllAnalysesData(dfSyn, sort=True, indexer=numNavLink,
                                                 convert=True, round_=True, style=True)

        # 5. Details: Format filtered and translated data.
        dfDet = dfDetRes

        # a. Add run folder column if not there (will serve to generate the link to the analysis detailed report)
        detTrCols = list(dfDet.columns)
        if self.trRunFolderCol not in detTrCols:
            detTrCols += [self.trRunFolderCol]
        dfDet[self.trRunFolderCol] = dfDet[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
        dfDet = dfDet.reindex(columns=detTrCols)
       
        # b. Links to each analysis detailed report.
        dfsDet = self.finalFormatAllAnalysesData(dfDet, sort=True, indexer=numNavLink,
                                                 convert=False, round_=False, style=True)

        # 6. Generate traceability infos section.
        # a. Log of opérations.
        indexCols = [self.tr(col) for col in ['Scheme', 'Step']]
        dataCols = [self.tr(col) for col in ['Property', 'Value']]
        dfFilSorHist = pd.DataFrame(filSorSteps, columns=indexCols + dataCols)
        dfFilSorHist.set_index(indexCols, inplace=True)
        ddfTrc = {self.tr('Filter & Sort steps'): (dfFilSorHist, True)}

        # b. Results specs
        ddfTrc.update(self.asWorkbook(subset=['specs', 'samples']))

        # 7. Generate top report page.
        genDateTime = dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        tmpl = self.getTemplateEnv().get_template('mcds/fulltop.htpl')
        xlFileUrl = os.path.basename(self.targetFilePathName(suffix='.xlsx')).replace(os.sep, '/')
        html = tmpl.render(supersynthesis=dfSupSyn.to_html(escape=False),
                           synthesis=dfsSyn.render(),  # escape=False, index=False),
                           details=dfsDet.render(),  # escape=False, index=False),
                           traceability={trcName: dfTrcTable.to_html(escape=False, na_rep='')
                                         for trcName, (dfTrcTable, _) in ddfTrc.items()},
                           title=self.title, keywords=self.keywords,
                           subtitle=self.subTitle.format(fsId=filSorSchId.split('@')[0]).replace('\n', '<br>'),
                           description=self.description.format(fsId=filSorSchId.split('@')[0]).replace('\n', '<br>'),
                           xlUrl=xlFileUrl, tr=self.dTrans[self.lang],
                           pySources=[pl.Path(fpn).name for fpn in self.pySources],
                           genDateTime=genDateTime, version=__version__, libVersions=self._libVersions(), 
                           distanceUnit=self.tr(self.resultsSet.distanceUnit),
                           areaUnit=self.tr(self.resultsSet.areaUnit),
                           surveyType=self.tr(self.resultsSet.surveyType),
                           distanceType=self.tr(self.resultsSet.distanceType),
                           clustering=self.tr('Clustering' if self.resultsSet.clustering else 'No clustering'),
                           estimSelCrits=estimSelCrits, confIntervals=[str(v) for v in confIntervals])
        html = re.sub('(?:[ \t]*\n){2,}', '\n'*2, html)  # Cleanup blank lines series to one only

        # 8. Write top HTML to file.
        filSorSchId = self.resultsSet.filSorSchemeId(filSorScheme)
        htmlPathName = self.targetFilePathName(suffix=f'.{filSorSchId}.html')
        with codecs.open(htmlPathName, mode='w', encoding='utf-8-sig') as tgtFile:
            tgtFile.write(html)

        return htmlPathName

    def toHtml(self, filSorScheme=dict(method=ResClass.filterSortOnExecCode),
               rebuild=False, generators=0):

        """HTML report generation for a given scheme.

        Parameters:
        :param filSorScheme: the 1 (and only) scheme to use for building the report (see ctor)
        :param rebuild: if True, rebuild from scratch (data extraction + plots) ;
                        otherwise, use any cached data or existing plot image files
        :param generators: Number of parallel (process) generators to use :
                           - None => no parallelism used, sequential execution,
                           - 0 => auto-number, based on the actual number of CPUs onboard,
                           - > 0 => the actual number to use (Note: 1 means no parallelism,
                            but some asynchronism though, contrary to None).

        Note: Parallelism didn't work for this class (WTF ?), at least with Matplotlib 3.1 ;
              (actually, it seems to work only the first time, and only when rebuild == False
               ... and maybe no matplotlib drawing actually done ; but then, we get:
              Exception: Can't pickle <function sync_do_first ...>) ;
              but it works with Matplotlib 3.4.2
        """

        logger.debug(f'MCDSResultsFilterSortReport.toHtml({rebuild=}, {filSorScheme=})')

        # Install needed attached files.
        self.installAttFiles(self.AttachedFiles)
            
        # Generate full report detailed pages (one for each analysis)
        # (done first to have plot image files generated for top report page generation right below).
        filSorSchId = self.resultsSet.filSorSchemeId(filSorScheme)
        self.toHtmlEachAnalysis(filSorScheme=filSorScheme, rebuild=rebuild, generators=generators,
                                topSuffix=f'.{filSorSchId}.html')
        
        # Generate top = main report page (one for all analyses).
        topHtmlPathName = self.toHtmlAllAnalyses(filSorScheme=filSorScheme, rebuild=rebuild)

        logger.info('... done.')
                
        return topHtmlPathName


if __name__ == '__main__':

    sys.exit(0)
