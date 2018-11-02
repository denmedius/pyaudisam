# -*- coding: utf-8 -*-

# Code de génération d'une page de quizz d'application "des Oiseaux à l'Oreille"
#
# A partir de textes markdown d'annonce (question), d'indices et de solution (réponse),
# pour chaque exercice = 1 enregistrement sonore (MP3) et son contexte (date, lieu, auteur, ...)

import os
import sys
import re
import codecs

from collections import namedtuple, OrderedDict as odict

import datetime as dt

import jinja2

import pandas as pd

import lxml.html


# Descripteurs.
DescripteurEtape = namedtuple('DescripteurEtape', ['index', 'id', 'titre'])

class Descripteur(object):
    
    def _rootIndentMD(self, mdText):
        
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
        
        headSpaces = headSpaces or ''
        mdText = '\n'.join([line[len(headSpaces):] for line in mdText.split('\n')])
        if len(mdText) and mdText[0] == '\n':
            mdText = mdText[1:]
            
        return mdText

class DescripteurQuizz(Descripteur):
    
    def __init__(self, id, titre, intro, publier, exercices, anecdotes):
        
        self.id = id
        self.titre = titre
        self.intro = self._rootIndentMD(intro)
        self.publier = publier
        self.exercices = exercices
        self.anecdotes = anecdotes

class DescripteurEnregistrement(Descripteur):
    
    def __init__(self, index, id, titre, auteur, licence, fichierSon, dossierSons):
        
        self.index = index
        self.id = id
        self.titre = titre
        self.auteur = auteur
        self.licence = licence
        self.fichierSon = fichierSon
        self.idSon, self.siteSon = self._sourceSon()
        self.urlTchSon, self.urlPageSon = self._urlsSon(dossierSons)

    _KXcNumPattern = re.compile('.*-XC([0-9]+)[-\.]')
    def _sourceSon(self):
        
        idSon, siteSon = None, None
        
        mo = self._KXcNumPattern.match(self.fichierSon)
        if mo:
            idSon = mo.group(1)
            siteSon = 'XC' # TODO: Support autres sources que XC ?
        
        return idSon, siteSon
    
    _KXcNumPattern = re.compile('.*-XC([0-9]+)[-\.]')
    def _urlsSon(self, dossierSons):
        
        # Publi. locale par défaut
        urlTchSon = dossierSons + '/' + self.fichierSon
        urlPageSon = None
        
        if self.siteSon == 'XC': # TODO: Support autres sources que XC ?
            urlPageSon = 'https://www.xeno-canto.org/' + self.idSon
        
        # Mais si publi. sur l'internet demandée, et son xeno-canto non modifié ...
        if dossierSons == '.' and self.idSon:
            if self.siteSon == 'XC': # TODO: Support autres sources que XC ?
                if self.fichierSon.find('-extrait') < 0:
                    urlTchSon = urlPageSon + '/download'
        
        return urlTchSon, urlPageSon
    
    def lecteurHtml(self, dossierSons):
        
        return """<span>
                  <audio controls>
                    <source src="{dos}/{fic}" type="audio/mpeg" />
                  </audio>
                  <a style="font-size: 125%" href="{url}" target="_blank">{url}</a>
                  </span>
               """.format(dos=dossierSons, fic=self.fichierSon, url=self.urlPageSon)
    
class DescripteurExercice(DescripteurEnregistrement):
    
    def __init__(self, index, id, titre, lieu, date, heure, altitude, auteur, licence,
                       fichierSon, dossierSons, duree, milieux, etapes):
        
        super().__init__(index, id, titre, auteur, licence, fichierSon, dossierSons)
        
        self.lieu = lieu
        self.date = date
        self.heure = heure
        self.altitude = altitude
        self.duree = duree
        self.milieux = self._rootIndentMD(milieux)
        self.etapes = { id : self._rootIndentMD(text) for id, text in etapes.items() }
        
class DescripteurAnecdote(DescripteurEnregistrement):
    
    def __init__(self, index, id, titre, auteur, licence, fichierSon, dossierSons, texte):
        
        super().__init__(index, id, titre, auteur, licence, fichierSon, dossierSons)
        
        self.texte = self._rootIndentMD(texte)

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
          elem.innerHTML = (new showdown.Converter({'strikethrough':'true'})).makeHtml(unescapeHTML(elem.innerHTML));
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
    <link rel="stylesheet" type="text/css" href="{{dossierAttache}}/quizz.css">
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
                    <li>
                      {% if quiz.publier[etape.id] %}
                        <a href="#{{quiz.id}}.{{etape.id}}">{{etape.titre}}</a>
                      {% else %}
                        {{etape.titre}} (... pour bientôt)
                      {% endif %}
                    </li>
                  {% endfor %}
                </ol>
                <ol style="list-style-type: upper-latin">
                  {% if quiz.anecdotes %}
                  <li><a href="#{{quiz.id}}.Anecdotes">Pour l'anecdote ...</a></li>
                  <ol style="list-style-type: none">
                    {% for anecd in quiz.anecdotes %}
                      <li><a href="#{{quiz.id}}.{{anecd.id}}">{{anecd.index}}. {{anecd.titre}}</a></li>
                    {% endfor %}
                  </ol>
                  {% endif %}
                </ol>
              {% endfor %}
              <li><a href="#licence">Licence / Auteur</a></li>
              <li><a href="#remerciements">Remerciements</a></li>
              <li><a href="#attributions">Emprunts / Attributions</a></li>
            </ol>
          </div>
        </td>
        <td style="align: right">
          {% for img in images.imgTMat %}
            <img src="{{dossierAttache}}/{{img.img}}"/>
            <h6 style="text-align: right; margin: 0 0 0 0; padding: 0 0 0 0">{{img.legend}}</h6>
          {% endfor %}
        </td>
      </tr>
    </table>

    <h2 id="preambule" style="margin-bottom: 0">Préambule</h2>
    <div markdown-text class="chapter" style="margin-left: 10px">

{{preambule}}

    </div>
    
    {% for quiz in quizz %}

      <h2 id="{{quiz.id}}">{{quiz.titre}}</h2>
      <div style="margin-left: 10px">

    <div markdown-text class="chapter">
{{quiz.intro}}
    </div>

        {% for etape in etapes %}
        
          {% set etapes_loop = loop %}
          
          <h3 id="{{quiz.id}}.{{etape.id}}">{{etape.index}}. {{etape.titre}}</h3>

          <div style="margin-left: 10px; margin-top: 20px">

            {% if quiz.publier[etape.id] %}
  
              {% if not etapes_loop.first %}

                <a href="javascript:show('d{{quiz.id}}.{{etape.id}}')" id="d{{quiz.id}}.{{etape.id}}s">
                  <img height="32" src="{{dossierAttache}}/fa-eye-regular.svg" alt="Montrer"
                       title="cliquez pour voir la suite si elle est déjà publiée"/>
                  cliquez pour voir la suite (MAIS ... uniquement si vous avez bien cherché !)
                </a>
                <div id="d{{quiz.id}}.{{etape.id}}" style="display: none">
                  <a href="javascript:hide('d{{quiz.id}}.{{etape.id}}')">
                    <img height="32" src="{{dossierAttache}}/fa-eye-slash-regular.svg" alt="Cacher"
                         title="cliquez pour cacher la suite"/>
                    cliquez pour cacher la suite
                  </a>

              {% endif %}

            {% else %}

                <p>Patience ... et cherchez encore, en attendant la publication !</p>
              
                <div style="display: none">
                
            {% endif %}

            {% if quiz.publier[etape.id] %}
  
              {% for exr in quiz.exercices %}
  
                <h4 id="{{quiz.id}}.{{exr.id}}" style="margin-bottom: 0">{{exr.index}}. {{exr.titre}}</h4>
                <div class="chapter" style="margin-left: 10px"> <!-- contenu exercice -->
  
                  <p>{{exr.lieu}} (altitude {{exr.altitude}} m), {{exr.date}} ({{exr.duree}})</p>
  
                  <div markdown-text>
{{exr.milieux}}
                  </div>
  
                  <table>
                    <tr>
                      <td>
                        <audio controls>
                          <source src="{{exr.urlTchSon}}" type="audio/mp3" preload="none"/>
                        </audio>
                      </td>
                      <td>
                        <a href="{{exr.urlTchSon}}" download>
                          <img height="20" src="{{dossierAttache}}/fa-download-solid.svg" alt="Télécharger"
                               title="Télécharger"/>
                        </a>
                      </td>
                      <td>
                        par {{exr.auteur}}
                      </td>
                      <td>
                        licence {{exr.licence}}
                      </td>
                      {% if etapes_loop.last and exr.urlPageSon %}
                      <td>
                        <a href="{{exr.urlPageSon}}" target="_blank">Page du site source</a>
                      </td>
                      {% endif %}
                    </tr>
                  </table>
  
                  <!-- div contents is markdown : Must be 1st-column aligned ! -->
                  <div markdown-text>
{{exr.etapes[etape.id]}}
                  </div>
                  
                </div> <!-- contenu exercice -->
  
              {% endfor %} <!-- exr in quiz.exercices -->

            {% endif %} <!-- quiz.publier[etape.id] -->

        {% endfor %} <!-- etape in etapes -->

        {% for etape in etapes %}

          {% if quiz.publier[etape.id] %}
 
            {% if not loop.first %}

              </div> <!-- id="d{{quiz.id}}.{{etape.id}}" -->
 
            {% endif %}

          {% else %}

            </div> <!-- Patience ... -->

          {% endif %} <!-- exr in quiz.exercices -->

          </div> <!-- contenu de chaque étape -->

        {% endfor %} <!-- etape in etapes -->

        {% if quiz.anecdotes %}

          <h3 id="{{quiz.id}}.Anecdotes" style="padding-top: 20px">A. Pour l'anecdote ...</h4>
          <div style="margin-left: 10px; margin-top: 20px">

          {% for anecd in quiz.anecdotes %}

            <h4 id="{{quiz.id}}.{{anecd.id}}" style="margin-bottom: 0">{{anecd.index}}. {{anecd.titre}}</h4>
            <div class="chapter" style="margin-left: 10px">

              <table>
                <tr>
                  <td>
                    <audio controls>
                      <source src="{{anecd.urlTchSon}}" type="audio/mp3" preload="none"/>
                    </audio>
                  </td>
                  <td>
                    <a href="{{anecd.urlTchSon}}" download>
                      <img height="20" src="{{dossierAttache}}/fa-download-solid.svg" alt="Télécharger"
                           title="Télécharger"/>
                    </a>
                  </td>
                  <td>
                    par {{anecd.auteur}}
                  </td>
                  <td>
                    licence {{anecd.licence}}
                  </td>
                  {% if anecd.urlPageSon %}
                  <td>
                    <a href="{{anecd.urlPageSon}}" target="_blank">Page du site source</a>
                  </td>
                  {% endif %}
                </tr>
              </table>

              <!-- div contents is markdown : Must be 1st-column aligned ! -->
              <div markdown-text>
{{anecd.texte}}
              </div>

          </div>

          {% endfor %} <!-- anecd in quiz.anecdotes -->

        </div>

      {% endif %} <!-- quiz.anecdotes -->

    {% endfor %} <!-- quiz in quizz -->

    <h2 id="licence" style="margin-bottom: 0">Licence / Auteur</h2>
    <div class="chapter" style="margin-left: 10px">

      <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr" target="_blank">
        <img height="48" src="{{dossierAttache}}/by-nc-sa.eu.svg" alt="Creative Commons BY NC SA 4.0"/>
      </a>
      
      <div markdown-text>

Ce document est publié sous la licence
**<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr" target="_blank">
Creative Commons BY NC SA 4.0</a>**
par <a href="http://jpmeuret.free.fr/" target="_blank">Jean-Philippe Meuret</a>,
<a href="http://www.lpo-auvergne.org" target="_blank">LPO Auvergne</a> ({{effort}} heures de travail).

Vous pouvez (et même devez ;-) le diffuser sans en demander l'autorisation, à qui vous voulez,
dans sa forme originale ou après modifications, par extraits ou dans son intégralité, pourvu que :
* vous en citiez l'auteur initial (voir ci-dessus) : BY,
* la diffusion n'ait pas un but commercial : NC,
* la diffusion se fasse avec la même licence (CC BY NC SA) : SA.

Attention cependant aux licences potentiellement plus restrictives :
* des enregistrements (voir ci-dessus la licence et l'auteur de chacun),
* des photos et dessins d'illustration (voir légende associée à chacun).

      </div>

    </div>

    <h2 id="remerciements" style="margin-bottom: 0">Remerciements</h2>
    <div class="chapter" style="margin-left: 10px">

      <div markdown-text>
      
La plupart des enregistrements utilisés ici (mais pas tous !)
provient du site <a href="https://www.xeno-canto.org/" target="_blank">xeno-canto.org</a> :
un grand merci aux ornithologues qui ont bien voulu partager leurs trouvailles
et ainsi rendre cette publication possible.</p>

{{remerciements}}

      </div>
      
    </div>

    <h2 id="attributions" style="margin-bottom: 0">Emprunts / Attributions</h2>
    <div class="chapter" style="margin-left: 10px">

      <div markdown-text>
      
Les icônes des petits yeux 'Cacher / Montrer l'étape suivante' et du bouton de téléchargement,
ainsi que le chevron vertical du bouton de retour en haut de page sont l'oeuvre de
<a href="https://fontawesome.com/" target="_blank">Font Awesome</a>,
et sont distribuées selon la licence
<a href="https://creativecommons.org/licenses/by/4.0/deed.fr" target="_blank">CC BY 4.0</a> ;
seule leur couleur - noire à l'origine - a été modifiée (en vert).
         
{{attributions}}

      </div>
      
    </div>

    <h6>
      Page générée via <a href="https://www.python.org/" target="_blank">Python 3</a>,
      <a href="https://pandas.pydata.org/" target="_blank">Pandas</a>,
      <a href="http://jinja.pocoo.org/" target="_blank">Jinja 2</a>
      et <a href="https://github.com/showdownjs/showdown/" target="_blank">showdown.js</a>
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
                          topJsScript=_KTopJsScript, botJsScript=_KBotJsScript,
                          notebook=notebook, genDateTime=dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S'))

    chemFicCible = os.path.join('.', '{}{}.html'.format(prefixeFicCible, '.local' if dossierSons != '.' else ''))
    with codecs.open(chemFicCible, mode='w', encoding='utf-8-sig') as ficCible:
        ficCible.write(html)
        
    return chemFicCible
