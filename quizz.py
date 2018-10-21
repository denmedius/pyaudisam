# -*- coding: utf-8 -*-

# Code de génération d'une page de quizz d'application "des Oiseaux à l'Oreille"
#
# A partir de textes markdown d'annonce (question), d'indices et de solution (réponse),
# pour chaque exercice = 1 enregistrement sonore (MP3) et son contexte (date, lieu, auteur, ...)

import os
import sys
import codecs

from collections import namedtuple, OrderedDict as odict

import datetime as dt

import jinja2

import pandas as pd

import lxml.html


# Descripteurs.
KIdPremEtape = 'lancement'
DescripteurQuiz = namedtuple('DescripteurQuiz', ['id', 'titre', 'exercices'])
DescripteurEtape = namedtuple('DescripteurEtape', ['index', 'id', 'titre'])

def _rootIndentMD(mdText):
    """Remove extra common heading blanks on every line + 1st line if empty
    """
    headSpaces = None
    for line in mdText.split('\n'):
        #print('"{}"'.format(line), end='')
        if line.strip(): # Ignore space-only lines
            if headSpaces is None:
                headSpaces = line[:-len(line.lstrip())]
                #print('start "{}"'.format(headSpaces), end='')
            else:
                nSpacesLeft = len(headSpaces)
                while nSpacesLeft > 0 and not line.startswith(headSpaces[:nSpacesLeft]):
                    nSpacesLeft -= 1
                headSpaces = headSpaces[:nSpacesLeft]
                #print('next {}, "{}"'.format(nSpacesLeft, headSpaces), end='')
        #print()
    mdText = '\n'.join([line[len(headSpaces):] for line in mdText.split('\n')])
    if len(mdText) and mdText[0] == '\n':
        mdText = mdText[1:]
    return mdText

class DescripteurExercice(object):
    def __init__(self, index, id, titre, lieu, date, altitude, auteur, licence,
                       enregistrement, duree, milieux, etapes):
        self.index = index
        self.id = id
        self.titre = titre
        self.lieu = lieu
        self.date = date
        self.altitude = altitude
        self.auteur = auteur
        self.licence = licence
        self.enregistrement = enregistrement
        self.duree = duree
        self.milieux = _rootIndentMD(milieux)
        self.etapes = { id : _rootIndentMD(text) for id, text in etapes.items() }
        
# Styles CSS
_KStylesCss = """
    html, body {
     margin: 0;
     padding: 0;
    }
    body {
     background-color: #e8efd1;
     font-family: Arial, Helvetica, sans-serif;
     font-size: 100%;
    }
    h1 {
     font-size: 480%; 
     color: #244c0c; 
     text-align: center;
    }
    h2 {
     font-size: 240%; 
     color: #244c0c; 
    }
    h3 {
     font-size: 160%; 
     color: #244c0c;
    }
    h4 {
     font-size: 120%; 
     color: #244c0c; 
    }
    h5 {
     font-size: 100%; 
     color: #244c0c; 
    }
    h6 {
     font-size: 80%; 
     color: #244c0c; 
    }
    p {
     color: #244c0c; 
    }
    ul,ol,li,td {
     color: #244c0c; 
    }
    a:link {
     color: #2f7404;
     font-weight: bold;
     text-decoration:underline;
    }
    a:visited {
     color: #379000;
     font-weight:bold;
     text-decoration:underline;
    }
    a:active,
    a:hover {
     color: #bd5a35;
     text-decoration:underline;
    }

    table a:link {
     color: #244c0c;
     font-weight: bold;
     text-decoration:none;
    }
    table a:visited {
     color: #546122;
     font-weight:bold;
     text-decoration:none;
    }
    table a:active,
    table a:hover {
     color: #bd5a35;
     text-decoration:underline;
    }

    table {
     font-family:Arial, Helvetica, sans-serif;
     color:#244c0c;
     text-shadow: 1px 1px 0px #fff;
     background:#eaebec;
     margin: 15px 8px 0 8px;
     border: #ccc 1px solid;

     -moz-border-radius:3px;
     -webkit-border-radius:3px;
     border-radius:3px;

     -moz-box-shadow: 0 1px 2px #d1d1d1;
     -webkit-box-shadow: 0 1px 2px #d1d1d1;
     box-shadow: 0 1px 2px #d1d1d1;
    }
    table th {
     text-align: left;
     padding: 0 8px 0 8px;
     border-top: 1px solid #f9fbf3;
     border-bottom: 1px solid #dee5ca;

     background: #bcc380;
     background: -webkit-gradient(linear, left top, left bottom, from(#bcc380), to(#e4eac8));
     background: -moz-linear-gradient(top, #bcc380, #e4eac8);
    }
    table th:first-child {
     text-align: left;
     padding-left: 10px;
    }
    table tr:first-child th:first-child {
     -moz-border-radius-topleft:3px;
     -webkit-border-top-left-radius:3px;
     border-top-left-radius:3px;
    }
    table tr:first-child th:last-child {
     -moz-border-radius-topright:3px;
     -webkit-border-top-right-radius:3px;
     border-top-right-radius:3px;
    }
    table tr {
     text-align: left;
     padding: 0 12px 0 0;
    }
    table td:first-child {
     text-align: left;
     padding-left: 10px;
     border-left: 0;
    }
    table td {
     padding: 8px 8px 8px 10px;
     border-top: 1px solid #ffffff;
     border-bottom: 1px solid #dee5ca;
     border-left: 1px solid #dee5ca;

     background: #f9fbf3;
     background: -webkit-gradient(linear, left top, left bottom, from(#f8f9f6), to(#f9fbf3));
     background: -moz-linear-gradient(top,  #f8f9f6,  #f9fbf3);
    }
    table tr.even td {
     background: #f6f6f6;
     background: -webkit-gradient(linear, left top, left bottom, from(#f8f8f8), to(#f6f6f6));
     background: -moz-linear-gradient(top,  #f8f8f8,  #f6f6f6);
    }
    table tr:last-child td {
     border-bottom:0;
    }
    table tr:last-child td:first-child {
     -moz-border-radius-bottomleft:3px;
     -webkit-border-bottom-left-radius:3px;
     border-bottom-left-radius:3px;
    }
    table tr:last-child td:last-child {
     -moz-border-radius-bottomright:3px;
     -webkit-border-bottom-right-radius:3px;
     border-bottom-right-radius:3px;
    }
    table tr:hover td {
     background: #f3f4eb;
     background: -webkit-gradient(linear, left top, left bottom, from(#f3f4eb), to(#eeefe9));
     background: -moz-linear-gradient(top, #f3f4eb, #eeefe9); 
    }
    #toTopBtn {
      display: none;
      position: fixed;
      bottom: 15px;
      right: 15px;
      z-index: 99;
      border: none;
      border-radius: 10px;
      outline: none;
      opacity: .25;
      background-color: white;
      cursor: pointer;
    }
    #toTopBtn:hover {
      opacity: .75;
    }
"""

# Javascripts
_KTopJsScript = """
    // Show or hide some element (id) through :
    // * a Show link (<a id="<id>+'s'" ...),
    // * a Hide link embedded into the element.
    function show(id)
    {
      document.getElementById(id).style.display = "block";
      document.getElementById(id+'s').style.display = "none";
    }
    function hide(id)
    {
      document.getElementById(id).style.display = "none";
      document.getElementById(id+'s').style.display = "block";
    }

    // Back to top floating button managment
    // When the user scrolls down100 px from the top of the document, show the button
    window.onscroll = function()
    {
        if (document.body.scrollTop > 100 || document.documentElement.scrollTop > 100)
            document.getElementById("toTopBtn").style.display = "block";
        else
            document.getElementById("toTopBtn").style.display = "none";
    }

    // When the user clicks on the button, scroll to the top of the document
    function scrollToTop()
    {
        document.body.scrollTop = 0;
        document.documentElement.scrollTop = 0;
    }
"""

_KBotJsScript = """
    // Translate 'markdown-text' divs content to HTML at page load time
    window.onload=function(){
      // this function is the reverse version of escapeHTML found at 
      // https://github.com/evilstreak/markdown-js/blob/master/src/render_tree.js
      function unescapeHTML( text ) {
          return text.replace( /&amp;/g, "&" )
                     .replace( /&lt;/g, "<" )
                     .replace( /&gt;/g, ">" )
                     .replace( /&quot;/g, "\\"" )
                     .replace( /&#39;/g, "'" );
        }
      // based on https://gist.github.com/paulirish/1343518
      (function(){
          [].forEach.call( document.querySelectorAll('[markdown-text]'), function fn(elem){
              elem.innerHTML = (new showdown.Converter()).makeHtml(unescapeHTML(elem.innerHTML));
          });
      }());
    }
"""

# Modèle jinja2 de page pour un groupe d'espèces.
_KHtmlQuizz = """
    <!DOCTYPE HTML>
    <head>
        <meta charset="utf-8">
        <title>{{titre}}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <meta name="author" content="Jean-Philippe Meuret"/>
        <meta name="copyright" content="Jean-Philippe Meuret 2018"/>
        <meta name="license" content="CC BY NC SA"/>
        <meta name="description" content="{{title}}"/>
        <meta name="keywords" content="chant, cri, oiseau, ornithologie, oreille, identification, quiz, {{keywords}}"/>
        <meta name="datetime" contents="{{genDateTime}}"/>
        <style type="text/css" media="screen">
          {{stylesCss}}
        </style>
        <script type="text/javascript" src="{{dossierAttache}}/showdown.min.js"></script>
        <script>
          {{topJsScript}}
        </script>
    </head>

    <body>
    
      <h1>{{titre}}</h1>
      <h3 style="text-align: center">{{sousTitre}}</h3>
        
      <div style="margin-left: 15px">
        
        <table>
          <tr>
            <td>
              <h2>Table des matières</h2>
              <div style="margin-left: 10px">
                <ol style="list-style-type: upper-roman">
                  <li><a href="#preambule">Préambule</a></li>
                  {% for quiz in quizz %}
                    <li><a href="#{{quiz.id}}">{{quiz.titre}}</a></li>
                    <ol style="list-style-type: decimal">
                      {% for etape in etapes %}
                        <li><a href="#{{quiz.id}}.{{etape.id}}">{{etape.titre}}</a></li>
                      {% endfor %}
                    </ol>
                  {% endfor %}
                  <li><a href="#licence">Licence / Auteur</a></li>
                  <li><a href="#remerciements">Remerciements</a></li>
                  <li><a href="#attributions">Emprunts / Attributions</a></li>
                </ol>
              </div>
            </td>
            <td style="align: right">
              {% for tocImg in images.tocImg %}
                <img src="{{dossierAttache}}/{{tocImg.img}}"/>
                <h6 style="text-align: right; margin: 0 0 0 0; padding: 0 0 0 0">{{tocImg.legend}}</h6>
              {% endfor %}
            </td>
          </tr>
        </table>

        <h2 id="preambule">Préambule</h2>
        <div markdown-text style="margin-left: 10px">
          
{{preambule}}
          
        </div>

        {% for quiz in quizz %}
        
          <h2 id="{{quiz.id}}">{{quiz.titre}}</h2>
          <div style="margin-left: 10px">
            
            {% for etape in etapes %}
        
              <h3 id="{{quiz.id}}.{{etape.id}}">{{etape.index}}. {{etape.titre}}</h3>

              <div style="margin-left: 10px">
              
              {% if etape.id != 'lancement' %}
                
              <div >
              
                <a href="javascript:show('d{{quiz.id}}.{{etape.id}}')" id="d{{quiz.id}}.{{etape.id}}s">
                  <img height="16" src="{{dossierAttache}}/fa-eye-regular.svg" alt="Montrer"
                       title="cliquez pour voir la suite ... mais uniquement si vous avez bien cherché !"/>
                </a>
                <div id="d{{quiz.id}}.{{etape.id}}" style="display: none">
                  <a href="javascript:hide('d{{quiz.id}}.{{etape.id}}')">
                    <img height="16" src="{{dossierAttache}}/fa-eye-slash-regular.svg" alt="Cacher"
                         title="cliquez pour cacher la suite"/>
                  </a>
                  
              {% endif %}
  
                  {% for exr in quiz.exercices %}
            
                    <h4 id="{{quiz.id}}.{{exr.id}}">{{exr.index}}. {{exr.titre}}</h4>
                    <div style="margin-left: 10px">
                    
                      <p>{{exr.lieu}}, altitude {{exr.altitude}} m, {{exr.date}} ({{exr.duree}}),
                         par {{exr.auteur}} (licence {{exr.licence}})</p>
                      
                      <div markdown-text>
{{exr.milieux}}
                      </div>
      
                      <table>
                        <tr>
                          <td>
                            <audio controls>
                              <source src="{{dossierSons}}/{{exr.enregistrement}}" type="audio/mp3" preload="none"/>
                            </audio>
                          </td>
                          <td>
                            <a href="{{dossierSons}}/{{exr.enregistrement}}" target="_blank">Téléchargement</a>
                          </td>
                        </tr>
                      </table>
                      
                      <!-- div contents is markdown : Must be 1st-column aligned ! -->
                      <div markdown-text>
{{exr.etapes[etape.id]}}
                      </div>
                    
                    </div>
                
                  {% endfor %}
                  
            {% endfor %}
        
            {% for etape in etapes %}
        
              {% if etape.id != 'lancement' %}
              
                </div> <!-- id="d{{quiz.id}}.{{etape.id}}" -->
                
              </div> <!-- style="margin-left: 10px" -->
                      
              {% endif %}
                  
            {% endfor %}
              
            </div>
              
          </div>
          
        {% endfor %}
        
        <h2 id="licence">Licence / Auteur</h2>
        <div style="margin-left: 10px">
        
          <p>Ce document est publié sous la licence
             <b><a href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr" target="_blank">
             Creative Commons BY NC SA 4.0</a></b>
             <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr" target="_blank">
               <img height="20" src="{{dossierAttache}}/by-nc-sa.eu.svg" alt="Creative Commons BY NC SA"/>
             </a>
             par <a href="http://jpmeuret.free.fr/" target="_blank">Jean-Philippe Meuret</a>,
             <a href="http://www.lpo-auvergne.org" target="_blank">LPO Auvergne</a> ({{effort}} heures de travail).</p>
          <p>Vous pouvez (et même devez ;-) le diffuser sans en demander l'autorisation, à qui vous voulez,
             dans sa forme originale ou après modifications, par extraits ou dans son intégralité, pourvu que :</p>
          <ul>
              <li>vous en citiez l'auteur initial (voir ci-dessus) : BY,</li>
              <li>la diffusion n'ait pas un but commercial : NC,</li>
              <li>la diffusion se fasse avec la même licence (CC BY NC SA) : SA.</li>
          </ul>
          <p>Attention cependant aux licences potentiellement plus restrictives :</p>
          <ul>
              <li>des enregistrements (Voir ci-dessus la licence et l'auteur de chacun),</li>
              <li>des photos et dessins d'illustration (Voir légende associée à chacun).</li>
          </ul>
             
        </div>

        <h2 id="remerciements">Remerciements</h2>
        <div style="margin-left: 10px">
        
          <p>La plupart des enregistrements utilisés ici (mais pas tous !)
             proviennent du site <a href="https://www.xeno-canto.org/" target="_blank">xeno-canto.org</a> :
             un grand merci aux ornithologues qui ont bien voulu partager leurs trouvailles
             et ainsi rendre cette publication possible.</p>
             
          {{remerciements}}
                      
        </div>
        
        <h2 id="attributions">Emprunts / Attributions</h2>
        <div style="margin-left: 10px">
        
          <p>Les icônes des petits yeux 'Cacher / Montrer les espèces d'arrière plan'
             et le chevron vertical du bouton de retour en haut de page sont l'oeuvre de
             <a href="https://fontawesome.com/" target="_blank">Font Awesome</a>,
             et sont distribuées selon la licence
             <a href="https://creativecommons.org/licenses/by/4.0/deed.fr" target="_blank">CC BY 4.0</a> ;
             seule leur couleur - noire à l'origine - a été modifiée (en vert).</p>
          {{attributions}}
        </div>
        
        <h6>
            Page générée via <a href="https://www.python.org/" target="_blank">Python 3</a>,
            <a href="https://pandas.pydata.org/" target="_blank">Pandas</a>
            et <a href="http://jinja.pocoo.org/" target="_blank">Jinja 2</a>
            dans <a href="http://jupyter.org/" target="_blank">Jupyter Notebook</a>
            (sources : <a href="./quizz.py" target="_blank">quizz.py</a>
             et <a href="./{{notebook}}" target="_blank">{{notebook}}</a>),
            le {{genDateTime}}.
        </h6>

      </div>

      <button onclick="scrollToTop()" id="toTopBtn" title="Remonter" alt="Remonter">
        <img width="64" height="64" src="{{dossierAttache}}/fa-angle-up.svg"/>
      </button>

      <script>
        {{botJsScript}}
      </script>
      
    </body>
"""

# Fonction principale de générationde la page.
def buildHtmlPage(titre, sousTitre, description, motsClef, preambule, quizz, etapes, remerciements, effort,
                  attributions, images, dossierSons, dossierAttache,
                  notebook='Quizz.ipynb', prefixeFicCible='quizz'):
    preambule = jinja2.Template(preambule).render(dossierAttache=dossierAttache)
    html = jinja2.Template(_KHtmlQuizz) \
                  .render(titre=titre, sousTitre=sousTitre, description=description, motsClef=motsClef,
                          preambule=preambule, quizz=quizz, etapes=etapes,
                          remerciements=remerciements, attributions=attributions,
                          dossierAttache=dossierAttache, dossierSons=dossierSons, images=images, effort=effort,
                          stylesCss=_KStylesCss, topJsScript=_KTopJsScript, botJsScript=_KBotJsScript,
                          notebook=notebook, genDateTime=dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S'))

    chemFicCible = os.path.join('.', '{}{}.html'.format(prefixeFicCible, '.local' if dossierSons != '.' else ''))
    with codecs.open(chemFicCible, mode='w', encoding='utf-8-sig') as ficCible:
        ficCible.write(html)
        
    return chemFicCible
