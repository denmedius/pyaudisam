# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Report : HTML and Excel report generation from DS results
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

# Warning: Only MCDS engine, and Point Transect analyses supported for the moment


import sys
import os, shutil
import re
import pathlib as pl
import copy

import datetime as dt
import codecs

import numpy as np
import pandas as pd

import jinja2
import matplotlib.pyplot as plt
import matplotlib.ticker as pltt

import logging

logger = logging.getLogger('autods')

# Actual package install dir.
KInstDirPath = pl.Path(__file__).parent.resolve()


# Base for results reports classes (abstract)
class ResultsReport(object):

    # Translation table for output documents.
    DTrans = dict(en={ }, fr={ })

    def __init__(self, resultsSet, title, subTitle, description, keywords, dCustomTrans=dict(),
                       lang='en', attachedDir='.', tgtFolder='.', tgtPrefix='results'):
    
        assert len(resultsSet) > 0, 'Can\'t build reports with nothing inside'
        assert os.path.isdir(tgtFolder), 'Target folder {} doesn\'t seem to exist ...'.format(tgtFolder)
        
        self.resultsSet = resultsSet
        
        self.trRunFolderCol = resultsSet.dfAnalysisColTrans.loc[resultsSet.analysisClass.RunFolderColumn, lang]
        self.dfEnColTrans = None # EN to other languages column name translation

        self.lang = lang
        self.title = title
        self.subTitle = subTitle
        self.description = description
        self.keywords = keywords
        
        self.dTrans = copy.deepcopy(dCustomTrans)
        for lang in self.DTrans.keys():
            if lang not in self.dTrans:
                self.dTrans[lang] = dict()
            self.dTrans[lang].update(self.DTrans[lang])
        
        self.attachedDir = attachedDir
        
        self.tgtPrefix = tgtPrefix
        self.tgtFolder = tgtFolder
        
        self.tmplEnv = None
        
    # Translate string
    def tr(self, s):
        return self.dTrans[self.lang].get(s, s)
    
    # Translate EN-translated column(s) name(s) to self.lang one
    # * colNames : str (1), list(N) or dict (N keys) of column names
    def trEnColNames(self, colNames):
        
        if self.lang == 'en':
            return colNames
        
        if self.dfEnColTrans is None:
            self.dfEnColTrans = self.resultsSet.transTable()
            self.dfEnColTrans.set_index('en', inplace=True, drop=True)
            
        def trEnColName(cn): # Assuming there's only 1 match !
            return self.dfEnColTrans.at[cn, self.lang]
        
        if isinstance(colNames, str):
            trColNames = trEnColName(colNames)
        elif isinstance(colNames, list):
            trColNames = [trEnColName(k) for k in colNames]
        elif isinstance(colNames, dict):
            trColNames = { trEnColName(k) : v for k, v in colNames.items() }
        # Otherwise ... boom: trColNames undefined !
        
        return trColNames
        
    # Output file pathname generation.
    def targetFilePathName(self, suffix, prefix=None, tgtFolder=None):
        
        return os.path.join(tgtFolder or self.tgtFolder, (prefix or self.tgtPrefix) + suffix)
    
    def relativeRunFolderUrl(self, runFolderPath):
        return os.path.relpath(runFolderPath, self.tgtFolder).replace(os.sep, '/')
    
    # Install needed attached files for HTML report.
    def installAttFiles(self, attFiles):
        
        for fn in attFiles:
            shutil.copy(KInstDirPath / 'report' / fn, self.tgtFolder)        
    
    # Get Jinja2 template environment for HTML reports.
    def getTemplateEnv(self):
        
        # Build and configure jinja2 environnement if not already done.
        if self.tmplEnv is None:
            self.tmplEnv = jinja2.Environment(loader=jinja2.FileSystemLoader([KInstDirPath]),
                                              trim_blocks=True, lstrip_blocks=True)
            #self.tmplEnv.filters.update(trace=_jcfPrint2StdOut) # Template debugging ...

        return self.tmplEnv
    
    # Final formatting of translated data tables, for HTML or SpreadSheet rendering
    # in the "one analysis at a time" case.
    # (sort, convert units, round values, and style).
    # To be specialized in derived classes (here, we do nothing) !
    # Note: Use trEnColNames method to pass from EN-translated columns names to self.lang-ones
    # Return a pd.DataFrame.Styler
    def finalFormatEachAnalysisData(self, dfTrData, sort=True, convert=True, round=True, style=True):
        
        return dfTrData.style # Nothing done here, specialize in derived class if needed.

    # Final formatting of translated data tables, for HTML or SpreadSheet rendering
    # in the "all analyses at once" case.
    # (sort, convert units, round values, and style).
    # To be specialized in derived classes (here, we do nothing) !
    # Note: Use trEnColNames method to pass from EN-translated columns names to self.lang-ones
    # Return a pd.DataFrame.Styler
    def finalFormatAllAnalysesData(self, dfTrData, sort=True, convert=True, round=True, style=True):
        
        return dfTrData.style # Nothing done here, specialize in derived class if needed.

    
# Results full reports class (Excel and HTML, targeting similar result as in Distance 6+)
class ResultsFullReport(ResultsReport):

    # Translation table.
    DTrans = dict(en={ 'RunFolder': 'Analysis', 'Synthesis': 'Synthesis', 'Details': 'Details',
                       'Synthesis table': 'Synthesis table',
                       'Click on analysis # for details': 'Click on analysis number to get to detailed report',
                       'Detailed results': 'Detailed results',
                       'Download Excel': 'Download as Excel(TM) file',
                       'Summary computation log': 'Summary computation log',
                       'Detailed computation log': 'Detailed computation log',
                       'Previous analysis': 'Previous analysis', 'Next analysis': 'Next analysis',
                       'Back to top': 'Back to global report',
                       'Page generated with': 'Page générée via', 'with icons from': 'avec les pictogrammes de',
                       'and': 'et', 'in': 'dans', 'sources': 'sources', 'on': 'le' },
                  fr={ 'DossierExec': 'Analyse', 'Synthesis': 'Synthèse', 'Details': 'Détails',
                       'Synthesis table': 'Tableau de synthèse',
                       'Click on analysis # for details': 'Cliquer sur le numéro de l\'analyse pour accéder au rapport détaillé',
                       'Detailed results': 'Résultats en détails',
                       'Download Excel': 'Télécharger le classeur Excel (TM)',
                       'Summary computation log': 'Résumé des calculs', 'Detailed computation log': 'Détail des calculs',
                       'Previous analysis': 'Analyse précédente', 'Next analysis': 'Analyse suivante',
                       'Back to top': 'Retour au rapport global',
                       'Page generated with': 'Page generated with', 'with icons from': 'with icons from',
                       'and': 'and', 'in': 'in', 'sources': 'sources', 'on': 'on' })

    def __init__(self, resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                       synthCols=None, dCustomTrans=dict(), lang='en',
                       attachedDir='.', tgtFolder='.', tgtPrefix='results'):
    
        assert synthCols is None or isinstance(synthCols, list) or isinstance(synthCols, pd.MultiIndex), \
               'synthesis columns must be specified as None (all), or as a list of tuples, or as a pandas.MultiIndex'
        
        super().__init__(resultsSet, title, subTitle, description, keywords,
                         dCustomTrans=dCustomTrans, lang=lang,
                         attachedDir=attachedDir, tgtFolder=tgtFolder, tgtPrefix=tgtPrefix)
        
        self.synthCols = synthCols

        self.anlysSubTitle = anlysSubTitle
        
    # Attached files for HTML report.
    AttachedFiles = ['autods.css', 'fa-feather-alt.svg', 'fa-angle-up.svg', 'fa-file-excel.svg', 'fa-file-excel-hover.svg',
                     'fa-arrow-left-hover.svg', 'fa-arrow-left.svg', 'fa-arrow-right-hover.svg', 'fa-arrow-right.svg',
                     'fa-arrow-up-hover.svg', 'fa-arrow-up.svg']
    
    # Plot ... data to be plot, and draw resulting figure to image files.
    PlotImgPrfxQqPlot = 'qqplot'
    PlotImgPrfxDetProb = 'detprob'
    PlotImgPrfxProbDens = 'probdens'
    
    @classmethod
    def generatePlots(cls, plotsData, tgtFolder, imgFormat='png', figSize=(14, 7),
                      grid=True, bgColor='#f9fbf3', transparent=False, trColors=['blue', 'red']):
        
        imgFormat = imgFormat.lower()
        
        # Stops heavy Matplotlib memory leak in XXXReport.generatePlots below (WTF !?)
        wasInter = plt.isinteractive()
        if wasInter:
            plt.ioff()

        # For each plot, 
        dPlots = dict()
        for title, pld in plotsData.items():
            
            # Create the target figure and one-only subplot.
            fig = plt.figure(figsize=figSize)
            axes = fig.subplots()
            
            # Plot a figure from the plot data (3 possible types, from title).
            if 'Qq-plot' in title:
                
                tgtFileName = cls.PlotImgPrfxQqPlot
                
                n = len(pld['dataRows'])
                df2Plot = pd.DataFrame(data=pld['dataRows'],
                                       columns=['If the fit was perfect ...', 'Real observations'],
                                       index=np.linspace(0.5/n, 1.0-0.5/n, n))
                
                df2Plot.plot(ax=axes, color=trColors, grid=grid,
                             xlim=(pld['xMin'], pld['xMax']),
                             ylim=(pld['yMin'], pld['yMax']))

            elif 'Detection Probability' in title:
                
                tgtFileName = cls.PlotImgPrfxDetProb + title.split(' ')[-1] # Assume last "word" is the hist. number
                
                df2Plot = pd.DataFrame(data=pld['dataRows'], 
                                       columns=[pld['xLabel'], pld['yLabel'] + ' (sampled)',
                                                pld['yLabel'] + ' (fitted)'])
                df2Plot.set_index(pld['xLabel'], inplace=True)
                
                df2Plot.plot(ax=axes, color=trColors, grid=grid,
                             xlim=(pld['xMin'], pld['xMax']), 
                             ylim=(pld['yMin'], pld['yMax']))
                
                aMTicks = axes.get_xticks()
                axes.xaxis.set_minor_locator(pltt.MultipleLocator((aMTicks[1]-aMTicks[0])/5))
                axes.tick_params(which='minor', grid_linestyle='-.', grid_alpha=0.6)
                axes.grid(True, which='minor')
        
            elif 'Pdf' in title:
                
                tgtFileName = cls.PlotImgPrfxProbDens + title.split(' ')[-1] # Assume last "word" is the Pdf number
                
                df2Plot = pd.DataFrame(data=pld['dataRows'], 
                                       columns=[pld['xLabel'], pld['yLabel'] + ' (sampled)',
                                                pld['yLabel'] + ' (fitted)'])
                df2Plot.set_index(pld['xLabel'], inplace=True)
                
                df2Plot.plot(ax=axes, color=trColors, grid=grid,
                             xlim=(pld['xMin'], pld['xMax']), 
                             ylim=(pld['yMin'], pld['yMax']))
        
                aMTicks = axes.get_xticks()
                axes.xaxis.set_minor_locator(pltt.MultipleLocator((aMTicks[1]-aMTicks[0])/5))
                axes.tick_params(which='minor', grid_linestyle='-.', grid_alpha=0.6)
                axes.grid(True, which='minor')
                
            # Finish plotting.
            axes.legend(df2Plot.columns, fontsize=14)
            axes.set_title(label=pld['title'] + ' : ' + pld['subTitle'],
                           fontdict=dict(fontsize=18), pad=20)
            axes.set_xlabel(pld['xLabel'], fontsize=14)
            axes.set_ylabel(pld['yLabel'], fontsize=14)
            axes.tick_params(axis = 'both', labelsize=12)
            axes.grid(True, which='major')
            if not transparent:
                axes.set_facecolor(bgColor)
                axes.figure.patch.set_facecolor(bgColor)
                
            # Generate an image file for the plot figure (forcing the specified patch background color).
            tgtFileName = tgtFileName + '.' + imgFormat
            fig.savefig(os.path.join(tgtFolder, tgtFileName),
                                box_inches='tight', transparent=transparent,
                                facecolor=axes.figure.get_facecolor(), edgecolor='none')

            # Memory cleanup (does not work in interactive mode ... but OK thanks to plt.ioff above)
            axes.clear()
            fig.clear()
            plt.close(fig)

            # Save image URL.
            dPlots[title] = tgtFileName
                
        # Restore pyplot interactive mode as it was before entering this function.
        if wasInter:
            plt.ion()

        return dPlots
    
    # Top page
    def toHtmlAllAnalyses(self):
        
        logger.info('Top page ...')
        
        # Generate post-processed and translated synthesis table.
        # a. Add run folder column if not selected (will serve to generate the link to the analysis detailed report)
        synCols = self.synthCols
        if self.resultsSet.analysisClass.RunFolderColumn not in synCols:
            synCols += [self.resultsSet.analysisClass.RunFolderColumn]
        dfSyn = self.resultsSet.dfTransData(self.lang, subset=synCols)
        dfSyn[self.trRunFolderCol] = dfSyn[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
        
        # b. Links to each analysis detailled report.
        dfSyn.index = \
            dfSyn.apply(lambda an: '<a href="./{p}/index.html">{n:04d}</a>' \
                                   .format(p=an[self.trRunFolderCol], n=an.name+1), axis='columns')
       
        # c. Post-format as specified in actual class.
        dfsSyn = self.finalFormatAllAnalysesData(dfSyn)

        # Generate post-processed and translated detailed table.
        dfDet = self.resultsSet.dfTransData(self.lang)

        # a. Add run folder column if not there (will serve to generate the link to the analysis detailed report)
        detTrCols = list(dfDet.columns)
        if self.trRunFolderCol not in detTrCols:
            detTrCols += [self.trRunFolderCol]
        dfDet[self.trRunFolderCol] = dfDet[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
        dfDet = dfDet.reindex(columns=detTrCols)
       
        # b. Links to each analysis detailed report.
        dfDet.index = \
            dfDet.apply(lambda an: '<a href="./{p}/index.html">{n:04d}</a>' \
                                   .format(p=an[self.trRunFolderCol], n=an.name+1), axis='columns')
        
        # c. Post-format as specified in actual class.
        dfsDet = self.finalFormatAllAnalysesData(dfDet, convert=False, round=False)

        # Generate top report page.
        genDateTime = dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        tmpl = self.getTemplateEnv().get_template('mcds/top.htpl')
        xlFileUrl = os.path.basename(self.targetFilePathName(suffix='.xlsx')).replace(os.sep, '/')
        html = tmpl.render(synthesis=dfsSyn.render(), #escape=False, index=False),
                           details=dfsDet.render(), #escape=False, index=False),
                           title=self.title, subtitle=self.subTitle,
                           description=self.description, keywords=self.keywords,
                           xlUrl=xlFileUrl, tr=self.dTrans[self.lang], genDateTime=genDateTime)
        html = re.sub('(?:[ \t]*\\\n){2,}', '\n'*2, html) # Cleanup blank line series to one only.

        # Write top HTML to file.
        htmlPathName = self.targetFilePathName(suffix='.html')
        with codecs.open(htmlPathName, mode='w', encoding='utf-8-sig') as tgtFile:
            tgtFile.write(html)

        return htmlPathName
    
    # Analyses pages.
    PlotImgFormat = 'png'
    def toHtmlEachAnalysis(self):
        
        # Generate translated synthesis and detailed tables.
        dfSynthRes = self.resultsSet.dfTransData(self.lang, subset=self.synthCols)
        dfDetRes = self.resultsSet.dfTransData(self.lang)

        logger.info('Analyses pages ({}) ...'.format(len(dfSynthRes)))

        # 1. 1st pass : Generate previous / next list (for navigation buttons)
        #    with the sorted order if any
        dfSynthRes = self.finalformatEachAnalysisData(dfSynthRes, convert=False, round=False, style=False).data
        sCurrUrl = dfSynthRes[self.trRunFolderCol]
        sCurrUrl = sCurrUrl.apply(lambda path: self.targetFilePathName(tgtFolder=path, prefix='index', suffix='.html'))
        sCurrUrl = sCurrUrl.apply(lambda path: os.path.relpath(path, self.tgtFolder).replace(os.sep, '/'))
        dfAnlysUrls = pd.DataFrame(dict(current=sCurrUrl, previous=np.roll(sCurrUrl, 1), next=np.roll(sCurrUrl, -1)))

        # 2. 2nd pass : Generate
        topHtmlPathName = self.targetFilePathName(suffix='.html')
        tmpl = self.getTemplateEnv().get_template('mcds/anlys.htpl')
        engineClass = self.resultsSet.engineClass
        trCustCols = self.resultsSet.transCustomColumns(self.lang)
        
        for lblAnlys in dfSynthRes.index:
            
            sAnlysCustCols = dfDetRes.loc[lblAnlys, trCustCols]
            logger.info('  #{}'.format(lblAnlys) + ' '.join(['{}={}'.format(k, v) for k, v in sAnlysCustCols.iteritems()]))
        
            anlysFolder = dfDetRes.at[lblAnlys, self.trRunFolderCol]

            # Postprocess synthesis table :
            dfSyn = dfSynthRes.loc[lblAnlys].to_frame().T
            dfSyn.index = dfSyn.index.map(lambda n: '{:03d}'.format(n+1))
            dfsSyn = self.finalformatEachAnalysisData(dfSyn)
            
            # Postprocess detailed table :
            dfDet = dfDetRes.loc[lblAnlys].to_frame().T
            dfDet.index = dfDet.index.map(lambda n: '{:03d}'.format(n+1))
            dfsDet = self.finalformatEachAnalysisData(dfDet, convert=False, round=False)
            
            # Generate analysis report page.
            genDateTime = dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            subTitle = 'Analyse {:03d} : {}'.format(lblAnlys+1, self.anlysSubTitle)
            sAnlysUrls = dfAnlysUrls.loc[lblAnlys]
            html = tmpl.render(synthesis=dfsSyn.render(),
                               details=dfsDet.render(),
                               log=engineClass.decodeLog(anlysFolder),
                               output=engineClass.decodeOutput(anlysFolder),
                               plots=self.generatePlots(engineClass.decodePlots(anlysFolder),
                                                        anlysFolder, imgFormat=self.PlotImgFormat),
                               title=self.title, subtitle=subTitle, keywords=self.keywords,
                               navUrls=dict(prevAnlys='../'+sAnlysUrls.previous,
                                            nextAnlys='../'+sAnlysUrls.next,
                                            back2Top='../'+os.path.basename(topHtmlPathName)),
                               tr=self.dTrans[self.lang], genDateTime=genDateTime)
            html = re.sub('(?:[ \t]*\\\n){2,}', '\n'*2, html) # Cleanup blank line series to one only.

            # Write analysis HTML to file.
            htmlPathName = self.targetFilePathName(tgtFolder=anlysFolder, prefix='index', suffix='.html')
            with codecs.open(htmlPathName, mode='w', encoding='utf-8-sig') as tgtFile:
                tgtFile.write(html)
        
    # HTML report generation (based on results.dfTransData).
    def toHtml(self):
        
        # Install needed attached files.
        self.installAttFiles(self.AttachedFiles)
            
        # Generate full report page for each analysis
        topHtmlPathName = self.toHtmlAllAnalyses()

        # Generate report page for each analysis
        self.toHtmlEachAnalysis()

        logger.info('... done.')
                
        return topHtmlPathName

    # Génération du rapport Excel.
    def toExcel(self):
        
        xlsxPathName = os.path.join(self.tgtFolder, self.tgtPrefix + '.xlsx')
        
        with pd.ExcelWriter(xlsxPathName) as xlsxWriter:
            
            # Synthesis
            dfSyn = self.resultsSet.dfTransData(self.lang, subset=self.synthCols)
            dfSyn[self.trRunFolderCol] = dfSyn[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
            
            # ... Convert run folder columns to hyperlink if present
            def toHyperlink(path):
                return '=HYPERLINK("file:///{path}", "{path}")'.format(path=path)
            if self.resultsSet.analysisClass.RunFolderColumn in self.synthCols:                
                dfSyn[self.trRunFolderCol] = dfSyn[self.trRunFolderCol].apply(toHyperlink)
            
            dfsSyn = self.finalFormatAllAnalysesData(dfSyn)
            
            dfsSyn.to_excel(xlsxWriter, sheet_name=self.tr('Synthesis'), index=True)
            
            # Details
            dfDet = self.resultsSet.dfTransData(self.lang)
            
            dfSyn[self.trRunFolderCol] = dfSyn[self.trRunFolderCol].apply(self.relativeRunFolderUrl)
            dfDet[self.trRunFolderCol] = dfDet[self.trRunFolderCol].apply(toHyperlink)
            
            dfsDet = self.finalFormatAllAnalysesData(dfDet, convert=False, round=False)
            
            dfsDet.to_excel(xlsxWriter, sheet_name=self.tr('Details'), index=True)

        return xlsxPathName

# A specialized full report for MCDS analyses, with actual output formating
class MCDSResultsFullReport(ResultsFullReport):

    DCustTrans = \
        dict(en={ 'Note: Some figures rounded or converted': 
                     "<strong>Note</strong>: Densities are expressed per square km,"
                     " and most figures have been rounded for readability",
                  'Note: All figures untouched, as output by MCDS': 
                     "<strong>Note</strong>: All values have been left untouched,"
                     " as outuput by MCDS (no rounding, no conversion)" },
             fr={ 'Note: Some figures rounded or converted':
                      "<strong>N.B.</strong> Les densités sont exprimées par km carré, et presque toutes les valeurs"
                      " ont été arrondies pour la lisibilité",
                  'Note: All figures untouched, as output by MCDS':
                      "<strong>N.B.</strong> Aucune valeur n'a été convertie ou arrondie,"
                      " elles sont toutes telles que produites par MCDS" })
    
    def __init__(self, resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                       synthCols=None, dCustomTrans=None, lang='en', attachedDir='.', tgtFolder='.', tgtPrefix='results'):
    
        super().__init__(resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                         synthCols, self.DCustTrans if dCustomTrans is None else dCustomTrans,
                         lang, attachedDir, tgtFolder, tgtPrefix)
        
    # Styling colors
    cChrGray = '#869074'
    cBckGreen, cBckGray = '#e0ef8c', '#dae3cb'
    cSclGreen, cSclOrange, cSclRed = '#cbef8c', '#f9da56', '#fe835a'
    cChrInvis = '#e8efd1' # body background
    scaledColors = [cSclGreen, cSclOrange, cSclRed]
    scaledColorsRvd = list(reversed(scaledColors))
    
    dExCodeColors = dict(zip([1, 2, 3], scaledColors))
    
    @classmethod
    def colorExecCodes(cls, sCodes):
        return ['background-color: ' + cls.dExCodeColors.get(c, cls.dExCodeColors[3]) for c in sCodes]
    
    @classmethod
    def scaledColorV(cls, v, thresholds, colors): # len(thresholds) == len(colors) - 1
        if pd.isnull(v):
            return cls.cBckGray
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
            return short.translate(str.maketrans({c:'' for c in '[] '})).replace('.0,', ',')

    # Final formatting of translated data tables, for HTML or SpreadSheet rendering
    # in the "all analyses at once" case.
    # (sort, convert units, round values, and style).
    # Note: Use trEnColNames method to pass from EN-translated columns names to self.lang-ones
    # Return a pd.DataFrame.Styler
    def finalFormatAllAnalysesData(self, dfTrData, sort=True, convert=True, round=True, style=True):
        
        # Sorting
        df = dfTrData
        if sort:
            # Temporarily add a sample Id column for sorting by (assuming analysis have been run as grouped by sample)
            sampleIdCols = self.resultsSet.transSampleColumns(self.lang)
            df.insert(0, column='#Sample#', value=df.groupby(sampleIdCols, sort=False).ngroup())
            
            # Sort
            sortCols = ['#Sample#'] + [col for col in self.trEnColNames(['Delta AIC']) if col in df.columns]
            df.sort_values(by=sortCols, ascending=True, inplace=True)
            
            # Clean-up
            df.drop(columns=['#Sample#'], inplace=True)
        
        # Converting to other units, or so.
        kVarDens = 1.0
        if convert:
            
            for col in self.trEnColNames(['Density', 'Min Density', 'Max Density']): # 'CoefVar Density', 
                if col in df.columns:
                    df[col] *= 1000000 / 10000 # ha => km2
            
            col = self.trEnColNames('CoefVar Density')
            if col in df.columns:
                kVarDens = 100.0
                df[col] *= kVarDens # [0, 1] => %
            
            for col in self.trEnColNames(['Fit Dist Cuts', 'Discr Dist Cuts']):
                if col in df.columns:
                    df[col] = df[col].apply(self.shortenDistCuts)
            
        # Reducing float precision
        if round:
            
            KDColDecimals = { **{ col: 3 for col in ['PDetec', 'Min PDetec', 'Max PDetec'] },
                              **{ col: 2 for col in ['Delta AIC', 'Chi2 P', 'KS P'] },
                              **{ col: 1 for col in ['AIC', 'EDR/ESW', 'Min EDR/ESW', 'Max EDR/ESW',
                                                     'Density', 'Min Density',
                                                     'Max Density', 'CoefVar Density',
                                                     'Left Trunc Dist', 'Right Trunc Dist'] } }
            df = df.round(decimals={ col: dec for col, dec in self.trEnColNames(KDColDecimals).items() \
                                              if col in df.columns })
        
        # Styling
        dfs = df.style
        if style:
            
            col = self.trEnColNames('Delta AIC')
            if col in df.columns and df[col].max() > 0: # if all delta AIC == 0, no need to stress it.
                dfs.set_properties(subset=pd.IndexSlice[df[df[col] == 0].index, :],
                                   **{'background-color': self.cBckGreen})
               
            col = self.trEnColNames('ExCod')
            if col in df.columns:
                dfs.apply(self.colorExecCodes, subset=[col], axis='columns')
            
            col = self.trEnColNames('CoefVar Density')
            if col in df.columns:
                dfs.apply(self.scaledColorS, subset=[col], axis='columns',
                          thresholds=[v * kVarDens for v in [0.3, 0.2]], colors=self.scaledColorsRvd)
            
            col = self.trEnColNames('KS P')
            if col in df.columns:
                dfs.apply(self.scaledColorS, subset=[col], axis='columns',
                          thresholds=[0.7, 0.2], colors=self.scaledColors)
            
            col = self.trEnColNames('Chi2 P')
            if col in df.columns:
                dfs.apply(self.scaledColorS, subset=[col], axis='columns',
                          thresholds=[0.7, 0.2], colors=self.scaledColors)
            
            col = self.trEnColNames('ExCod')
            if col in df.columns:
                dfs.set_properties(subset=pd.IndexSlice[df[~df[col].isin([1, 2])].index, :],
                                   **{'color': self.cChrGray})
            
            dfs.where(self.isNull, 'color: transparent').where(self.isNull, 'text-shadow: none')
        
        return dfs

    # Final formatting of translated data tables, for HTML or SpreadSheet rendering
    # in the "one analysis at a time" case.
    # (sort, convert units, round values, and style).
    # Note: Use trEnColNames method to pass from EN-translated columns names to self.lang-ones
    # Return a pd.DataFrame.Styler
    finalformatEachAnalysisData = finalFormatAllAnalysesData


# A specialized pre-report for MCDS analyses, with actual output formating
# (HTML only, targeting very simple mono-table synthesis for fully automatic pre-analyses,
#  in order to give the user hints about what analyses are to be done, and with what parameter values).
class MCDSResultsPreReport(MCDSResultsFullReport):

    # Translation table.
    DTrans = dict(en={ 'RunFolder': 'Analysis', 'Synthesis': 'Synthesis', 'Details': 'Details',
                       'Synthesis table': 'Synthesis table',
                       'Click on analysis # for details': 'Click on analysis number to get to detailed report',
                       'Sample': 'Sample', 'Parameters': 'Parameters', 'Results': 'Results',
                       'ProbDens': 'Detection probability density (PDF)',
                       'DetProb': 'Detection probability',
                       'Detailed results': 'Detailed results',
                       'Download Excel': 'Download as Excel(TM) file',
                       'Summary computation log': 'Summary computation log',
                       'Detailed computation log': 'Detailed computation log',
                       'Previous analysis': 'Previous analysis', 'Next analysis': 'Next analysis',
                       'Back to top': 'Back to global report',
                       'Page generated with': 'Page générée via', 'with icons from': 'avec les pictogrammes de',
                       'and': 'et', 'in': 'dans', 'sources': 'sources', 'on': 'le' },
                  fr={ 'DossierExec': 'Analyse', 'Synthesis': 'Synthèse', 'Details': 'Détails',
                       'Synthesis table': 'Tableau de synthèse',
                       'Click on analysis # for details': 'Cliquer sur le numéro de l\'analyse pour accéder au rapport détaillé',
                       'Sample': 'Echantillon', 'Parameters': 'Paramètres', 'Results': 'Résultats',
                       'ProbDens': 'Densité de probabilité de détection (DdP)',
                       'DetProb': 'Probabilité de détection',
                       'Detailed results': 'Résultats en détails',
                       'Download Excel': 'Télécharger le classeur Excel (TM)',
                       'Summary computation log': 'Résumé des calculs', 'Detailed computation log': 'Détail des calculs',
                       'Previous analysis': 'Analyse précédente', 'Next analysis': 'Analyse suivante',
                       'Back to top': 'Retour au rapport global',
                       'Page generated with': 'Page generated with', 'with icons from': 'with icons from',
                       'and': 'and', 'in': 'in', 'sources': 'sources', 'on': 'on' })

    DCustTrans = \
        dict(en={ 'Note: Some figures rounded or converted': 
                     "<strong>Note</strong>: Densities are expressed per square km,"
                     " and most figures have been rounded for readability",
                  'Note: All figures untouched, as output by MCDS': 
                     "<strong>Note</strong>: All values have been left untouched,"
                     " as outuput by MCDS (no rounding, no conversion)" },
             fr={ 'Note: Some figures rounded or converted':
                      "<strong>N.B.</strong> Les densités sont exprimées par km carré, et presque toutes les valeurs"
                      " ont été arrondies pour la lisibilité",
                  'Note: All figures untouched, as output by MCDS':
                      "<strong>N.B.</strong> Aucune valeur n'a été convertie ou arrondie,"
                      " elles sont toutes telles que produites par MCDS" })
    
    def __init__(self, resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                       sampleCols, paramCols, resultCols, anlysSynthCols=None, 
                       plotsHeight=256, lang='en', attachedDir='.', tgtFolder='.', tgtPrefix='results'):

        super().__init__(resultsSet, title, subTitle, anlysSubTitle, description, keywords,
                         synthCols=anlysSynthCols, dCustomTrans=self.DCustTrans, lang=lang,
                         attachedDir=attachedDir, tgtFolder=tgtFolder, tgtPrefix=tgtPrefix)
        
        self.sampleCols = sampleCols
        self.paramCols = paramCols
        self.resultCols = resultCols
        self.plotsHeight = plotsHeight

    # Final formatting of translated data tables, for HTML or SpreadSheet rendering
    # in the "one analysis at a time" case.
    # (sort, convert units, round values, and style).
    # Note: Use trEnColNames method to pass from EN-translated columns names to self.lang-ones
    # Return a pd.DataFrame.Styler
    def finalFormatAllAnalysesData(self, dfTrData, sort=True, convert=True, round=True, style=True):
        
        # Sorting
        df = dfTrData
        if sort:
            
            pass # No sorting to be done here
        
        # Converting to other units, or so.
        kVarDens = 1.0
        if convert:
            
            for col in self.trEnColNames(['Density', 'Min Density', 'Max Density']):
                if col in df.columns:
                    df[col] *= 1000000 / 10000 # ha => km2
            
            col = self.trEnColNames('CoefVar Density')
            if col in df.columns:
                kVarDens = 100.0
                df[col] *= kVarDens # [0, 1] => %
            
        # Reducing float precision
        if round:
            
            dColDecimals = { **{ col: 3 for col in ['PDetec', 'Min PDetec', 'Max PDetec'] },
                             **{ col: 2 for col in ['Delta AIC', 'Chi2 P', 'KS P'] },
                             **{ col: 1 for col in ['AIC', 'EDR/ESW', 'Min EDR/ESW', 'Max EDR/ESW',
                                                    'Density', 'Min Density', 'Max Density', 'CoefVar Density'] } }
            df = df.round(decimals={ col: dec for col, dec in self.trEnColNames(dColDecimals).items() if col in df.columns })
        
        # Styling
        dfs = df.style
        if style:
            
            pass # No styling to be done here.
        
        return dfs

    @staticmethod
    def series2VertTable(ser):
        
        def float2str(v): # Workaround to_html non transparent default float format (!?)
            return '{:g}'.format(v)
        return re.sub('\\\n *', '', ser.to_frame().to_html(header=False, float_format=float2str))
    
        #return ''.join('<p>{}: {}</p>'.format(k, v) for k, v in dictOrSeries.items())
        
    def plotImageHtmlElement(self, runFolder, plotImgPrfx):
        
        for plotInd in range(3, 0, -1):
            plotFileName = '{}{}.{}'.format(plotImgPrfx, plotInd, self.PlotImgFormat)
            if os.path.isfile(os.path.join(runFolder, plotFileName)):
                return '<img src="./{}/{}" style="height: {}px" />' \
                       .format(self.relativeRunFolderUrl(runFolder), plotFileName, self.plotsHeight)
        
        return '{} plot image file not found'.format(plotImgPrfx)
        
    # Top page
    def toHtmlAllAnalyses(self):
        
        logger.info('Top page ...')
        
        # Generate the table to display from raw results 
        # (index + 5 columns : sample, params, results, ProbDens plot, DetProb plot)
        # 1. Get translated and post-formated detailed results
        dfDet = self.resultsSet.dfTransData(self.lang)
        dfsDet = self.finalFormatAllAnalysesData(dfDet)
        dfDet = dfsDet.data

        # 2. Translate sample, parameter and result columns
        dTransResCol = self.resultsSet.transTable()
        sampleTrCols = [dTransResCol[self.lang].get(col, str(col)) for col in self.sampleCols]
        paramTrCols = [dTransResCol[self.lang].get(col, str(col)) for col in self.paramCols]
        resultTrCols = [dTransResCol[self.lang].get(col, str(col)) for col in self.resultCols]
        
        # 3. Fill target table index and columns
        dfSyn = pd.DataFrame(dict(Sample=dfDet[sampleTrCols].apply(self.series2VertTable, axis='columns'),
                                  Parameters=dfDet[paramTrCols].apply(self.series2VertTable, axis='columns'),
                                  Results=dfDet[resultTrCols].apply(self.series2VertTable, axis='columns'),
                                  ProbDens=dfDet[self.trRunFolderCol].apply(self.plotImageHtmlElement,
                                                                            plotImgPrfx=self.PlotImgPrfxProbDens),
                                  DetProb=dfDet[self.trRunFolderCol].apply(self.plotImageHtmlElement,
                                                                           plotImgPrfx=self.PlotImgPrfxDetProb)))
        
        dfSyn.index = \
            dfDet.apply(lambda an: '<a href="./{p}/index.html">{n:04d}</a>' \
                                   .format(p=self.relativeRunFolderUrl(an[self.trRunFolderCol]), n=an.name+1),
                        axis='columns')
        
        # 4. Translate table columns.
        dfSyn.columns = [self.tr(col) for col in dfSyn.columns]

        self.dfSyn = dfSyn
        
        # Generate top report page.
        genDateTime = dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        tmpl = self.getTemplateEnv().get_template('mcds/pretop.htpl')
        xlFileUrl = os.path.basename(self.targetFilePathName(suffix='.xlsx')).replace(os.sep, '/')
        html = tmpl.render(synthesis=dfSyn.to_html(escape=False),
                           title=self.title, subtitle=self.subTitle,
                           description=self.description, keywords=self.keywords,
                           xlUrl=xlFileUrl, tr=self.dTrans[self.lang], genDateTime=genDateTime)
        html = re.sub('(?:[ \t]*\\\n){2,}', '\n'*2, html) # Cleanup blank lines series to one only

        # Write top HTML to file.
        htmlPathName = self.targetFilePathName(suffix='.html')
        with codecs.open(htmlPathName, mode='w', encoding='utf-8-sig') as tgtFile:
            tgtFile.write(html)

        return htmlPathName
    
    # HTML report generation (based on results.dfTransData).
    def toHtml(self):
        
        # Install needed attached files.
        self.installAttFiles(self.AttachedFiles)
            
        # Generate full report page for each analysis
        # (done first to have plot image files generated for top report page generation just below).
        self.toHtmlEachAnalysis()
        
        # Generate top report page.
        topHtmlPathName = self.toHtmlAllAnalyses()

        logger.info('... done.')
                
        return topHtmlPathName


if __name__ == '__main__':

    sys.exit(0)
