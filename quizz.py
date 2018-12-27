# -*- coding: utf-8 -*-

# Code de génération d'une page de quizz d'application "des Oiseaux à l'Oreille"
#
# A partir de textes markdown d'annonce (question), d'indices et de solution (réponse),
# pour chaque exercice = 1 enregistrement sonore (MP3) et son contexte (date, lieu, auteur, ...)
#
# Auteur : Jean-Philippe Meuret (http://jpmeuret.free.fr/nature.html)
# Licence : CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr)

import os
import sys
import re
import codecs

from collections import namedtuple, OrderedDict as odict

import datetime as dt

import jinja2

import pandas as pd

import requests


# Descripteurs.
DescripteurEtape = namedtuple('DescripteurEtape', ['index', 'id', 'titre'])

class Descripteur(object):
    
    def _rootIndentMD(self, mdText):
        
        """Remove extra common heading blanks on every line + 1st line if empty
        """
        
        if not mdText:
            return ''
        
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

    def _substVars(self, mdText, **vars):
        
        for name, value in vars.items():
            
            mdText = mdText.replace('{{'+name+'}}', str(value))
        
        return mdText
        
class DescripteurQuizz(Descripteur):
    
    def __init__(self, id, titre, intro, publier, exercices, anecdotes, effort):
        
        self.id = id
        self.titre = titre # texte (+ HTML)
        self.intro = self._rootIndentMD(intro) # texte Markdown
        self.publier = publier # { étape (infra, lancement, indices2) : booléen }
        self.exercices = exercices # [ DescripteurExercice(...) ]
        self.anecdotes = anecdotes # [ DescripteurAnecdote(...) ]
        self.effort = effort # { étape (infra, lancement, indices2, reponse, anecdotes) : nb heures passées }

class DescripteurEnregistrement(Descripteur):
    
    def __init__(self, index, id, titre, auteur, licence, fichierSon, dossierSons):
        
        self.index = index
        self.id = id
        self.titre = titre # texte (+ HTML)
        self.auteur = auteur # texte (+ HTML)
        self.licence = licence # texte (+ HTML)
        self.fichierSon = fichierSon # nom fichier son (format mp3) dans ./enregistrements
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
        
        return jinja2.Template("""
                <audio controls style="min-width:720px">
                    <source src="{{dos}}/{{fic}}" type="audio/mpeg" />
                </audio>
                {% if url %}
                  <a href="{{url}}" target="_blank">{{url}}</a>
                {% endif %}
               """).render(dos=dossierSons, fic=self.fichierSon, url=self.urlPageSon)
    
class DescripteurExercice(DescripteurEnregistrement):
    
    def __init__(self, index, id, titre, lieu, date, heure, altitude, auteur, licence,
                       fichierSon, dossierSons, dossierAttache, duree, milieux, etapes):
        
        super().__init__(index, id, titre, auteur, licence, fichierSon, dossierSons)
        
        self.lieu = lieu # Texte (+ HTML)
        self.date = date # Texte
        self.heure = heure # Texte
        self.altitude = altitude # Entier (m)
        self.duree = duree # Texte libre
        self.milieux = self._rootIndentMD(milieux) # Texte Markdown
        dVars = dict(dossierSons=dossierSons, dossierAttache=dossierAttache)
        self.etapes = { id : self._substVars(self._rootIndentMD(text), **dVars) \
                       for id, text in etapes.items() }
        
class DescripteurAnecdote(DescripteurEnregistrement):
    
    def __init__(self, index, id, titre, auteur, licence, fichierSon, dossierSons, dossierAttache, texte):
        
        super().__init__(index, id, titre, auteur, licence, fichierSon, dossierSons)
        
        dVars = dict(dossierSons=dossierSons, dossierAttache=dossierAttache)
        self.texte = self._substVars(self._rootIndentMD(texte), **dVars)


# Appelle de l'API Xeno-Canto avec un numéro d'enregistrement
# et affiche et renvoie qq infos à copier & coller dans le NB,
# ainsi qu'un lecteur audio et un lien cliquable vers la page de l'enregistrement.
# Usage: HTML(infosEnregXC(nr=416636))
def infosEnregXC(nr):
    
    rep = requests.get(url='http://www.xeno-canto.org/api/2/recordings', 
                       params=dict(query='nr:{}'.format(nr)))
    rep.raise_for_status()
    dRep = rep.json()

    assert len(dRep['recordings']) == 1
    dRec = dRep['recordings'][0]

    dRec.update(alt=9999, dur=9999)
    spRec = dRec['rec'].split()
    dRec.update(id_rec='???????', ab_rec=spRec[0][0]+''.join(spRec[1:]))
    dRec.update(date_c=dRec['date'].replace('-', ''), 
                time_c=('0' if len(dRec['time']) < 5 else '') + dRec['time'].replace(':', ''))
    dRec.update(date=dt.datetime.strptime(dRec['date'], '%Y-%m-%d').strftime('%d %B %Y'))
    
    print("""
  lieu="{loc}", altitude={alt},
  date="{date}", heure="{time}", duree="{dur}",
  auteur="<a href=\\\"https://www.xeno-canto.org/contributor/{id_rec}\\\" target=\\\"_blank\\\">{rec}</a>",
  licence="<a href=\\\"https:{lic}\\\" target=\\\"_blank\\\">CC BY-NC-??</a>",
  fichierSon="XXIdEnreg-XXPays-{ab_rec}-{date_c}-{time_c}-XC{id}-mono-vbrX.mp3", # Nom fic. dans ./enregistrements
    """.format(**dRec))

    return 'Lien direct <a href="{url}" target="_blank">{url}</a> (nouvel onglet)'.format(url=dRec['url'])


# Code javascript intégré.
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
    <meta name="author" content="Jean-Philippe Meuret"/>
    <meta name="copyright" content="Jean-Philippe Meuret 2018"/>
    <meta name="license" content="CC BY NC SA"/>
    <meta name="description" content="{{titre}}"/>
    <meta name="keywords" content="oiseau, chant, cri, oreille, identification, quiz, ornithologie, {{motsClef}}"/>
    <meta name="datetime" contents="{{genDateTime}}"/>
    <title>{{titre}}</title>
    <link rel="stylesheet" media="screen" type="text/css" href="{{dossierAttache}}/quizz.css">
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
        <td style="min-width: 320px">
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
                        {{etape.titre}} ... pas encore :-)
                      {% endif %}
                    </li>
                  {% endfor %}
                </ol>
                <ol style="list-style-type: upper-latin">
                  {% if quiz.anecdotes|length > 1 %}
                    <li><a href="#{{quiz.id}}.Anecdotes">Pour l'anecdote ...</a></li>
                    <ol style="list-style-type: none">
                      {% for anecd in quiz.anecdotes %}
                        <li><a href="#{{quiz.id}}.{{anecd.id}}">{{anecd.index}}. {{anecd.titre}}</a></li>
                      {% endfor %}
                    </ol>
                  {% elif quiz.anecdotes|length == 1 %}
                    <li><a href="#{{quiz.id}}.Anecdote">Pour l'anecdote ... {{quiz.anecdotes[0].titre}}</a></li>
                  {% endif %}
                </ol>
              {% endfor %}
              <li><a href="#licence">Licence / Auteur</a></li>
              <li><a href="#attribs-mercis">Remerciements et attributions</a></li>
            </ol>
          </div>
        </td>
        <td style="align: right">
          {% for img in images.imgTMat %}
            <img class="shrinkable" src="{{dossierAttache}}/{{img.img}}"/>
            <p style="text-align: right; margin: 0; padding: 0">{{img.legend}}</p>
          {% endfor %}
        </td>
      </tr>
    </table>

    <img class="center" height="32" style="margin-top: 10px"
         src="{{dossierAttache}}/fa-feather-alt.svg" alt="---" />

    <h2 id="preambule">Préambule</h2>
    <div markdown-text class="chapter" style="margin-left: 10px">

{{preambule}}

    </div>
    
    {% for quiz in quizz %}

      <img class="center" height="32" style="margin-top: 10px"
           src="{{dossierAttache}}/fa-feather-alt.svg" alt="---" />

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
                  <img height="32" src="{{dossierAttache}}/fa-eye-regular.svg" alt="Montrer" />
                  cliquez pour voir la suite (MAIS ... uniquement si vous avez bien cherché !)
                </a>
                <div id="d{{quiz.id}}.{{etape.id}}" style="display: none">
                  <a href="javascript:hide('d{{quiz.id}}.{{etape.id}}')">
                    <img height="32" src="{{dossierAttache}}/fa-eye-slash-regular.svg" alt="Cacher" />
                    cliquez pour cacher la suite
                  </a>

              {% endif %}

            {% else %}

                <p>Patience ... et cherchez encore, en attendant la publication !</p>
              
                <div style="display: none">
                
            {% endif %}

            {% if quiz.publier[etape.id] %}
  
              {% for exr in quiz.exercices %}
  
                <h4 id="{{quiz.id}}.{{exr.id}}">{{exr.index}}. {{exr.titre}}</h4>
                <div class="chapter" style="margin-left: 10px"> <!-- contenu exercice -->
  
                  <p>{{exr.lieu}}{{ ' (altitude %d m)' % exr.altitude if exr.altitude is not none }},
                     {{exr.date}} ({{exr.duree}}).</p>
  
                  {% if exr.milieux %}
                    <div markdown-text>
{{exr.milieux}}
                    </div>
                  {% endif %}
  
                  <table>
                    <tr>
                      <td>
                        <audio controls class="audio-player">
                          <source src="{{exr.urlTchSon}}" type="audio/mpeg" preload="none"/>
                        </audio>
                      </td>
                      <td>
                        <a href="{{exr.urlTchSon}}" download>
                          <img height="20" src="{{dossierAttache}}/fa-download-solid.svg" alt="Télécharger" />
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

          {% if quiz.anecdotes|length > 1 %}

            <h3 id="{{quiz.id}}.Anecdotes" style="padding-top: 20px">A. Pour l'anecdote ...</h3>
            <div style="margin-left: 10px; margin-top: 20px">

          {% else %}

            <h3 id="{{quiz.id}}.Anecdote" style="padding-top: 20px">A. Pour l'anecdote ... {{quiz.anecdotes[0].titre}}</h3>
            
          {% endif %}

          {% for anecd in quiz.anecdotes %}

            {% if quiz.anecdotes|length > 1 %}

              <h4 id="{{quiz.id}}.{{anecd.id}}">{{anecd.index}}. {{anecd.titre}}</h4>
            
            {% endif %}

            <div class="chapter" style="margin-left: 10px">

              <table>
                <tr>
                  <td>
                    <audio controls class="audio-player">
                      <source src="{{anecd.urlTchSon}}" type="audio/mpeg" preload="none"/>
                    </audio>
                  </td>
                  <td>
                    <a href="{{anecd.urlTchSon}}" download>
                      <img height="20" src="{{dossierAttache}}/fa-download-solid.svg" alt="Télécharger" />
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

    <img class="center" height="32" style="margin-top: 10px"
         src="{{dossierAttache}}/fa-feather-alt.svg" alt="---" />

    <h2 id="licence">Licence / Auteur</h2>
    <div class="chapter" style="margin-left: 10px; margin-top: 10px">

      <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr" target="_blank">
        <img height="36" src="{{dossierAttache}}/by-nc-sa.eu.svg" alt="Creative Commons BY NC SA 4.0"/>
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

    <img class="center" height="32" style="margin-top: 10px"
         src="{{dossierAttache}}/fa-feather-alt.svg" alt="---" />

    <h2 id="attribs-mercis">Remerciements et attributions</h2>
    <div class="chapter" style="margin-left: 10px">

      <div markdown-text>
      
La plupart des enregistrements utilisés ici (mais pas tous !)
provient du site <a href="https://www.xeno-canto.org/" target="_blank">xeno-canto.org</a> :
un grand merci aux ornithologues qui ont bien voulu partager leurs trouvailles
et ainsi rendre cette publication possible.</p>

{{attribsEtMercis}}

Merci enfin au projet <a href="https://fontawesome.com/" target="_blank">Font Awesome</a>,
qui produit et distribue gratuitement, sous la licence
<a href="https://creativecommons.org/licenses/by/4.0/deed.fr" target="_blank">CC BY 4.0</a>,
des icônes et pictogrammes "trop stylés", dont
<img height="16" src="{{dossierAttache}}/fa-eye-regular.svg" alt="Icône Montrer" />,
<img height="16" src="{{dossierAttache}}/fa-eye-slash-regular.svg" alt="Icône Cacher" />,
<img height="16" src="{{dossierAttache}}/fa-download-solid.svg" alt="Icône Télécharger" />,
<img height="16" src="{{dossierAttache}}/fa-feather-alt.svg" alt="Icône Séparateur" /> et
<img width="16" height="16" src="{{dossierAttache}}/fa-angle-up.svg" alt="Icône Haut de page" />,
dont j'ai simplement changé la couleur, noire à l'origine, en vert (forcément).
         
      </div>
      
    </div>

    <img class="center" height="32" style="margin-top: 10px"
         src="{{dossierAttache}}/fa-feather-alt.svg" alt="---" />

    <h6 style="margin-bottom: 10px">
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

  <button onclick="scrollToTop()" id="toTopBtn" title="Haut de page">
    <img width="64" height="64" src="{{dossierAttache}}/fa-angle-up.svg"/>
  </button>

  <script>
    {{botJsScript}}
  </script>

</body>
"""

# Fonction principale de génération de la page.
def buildHtmlPage(titre, sousTitre, description, motsClef, preambule, quizz, etapes, attribsEtMercis, effort,
                  images, dossierSons, dossierAttache, notebook='Quizz.ipynb', prefixeFicCible='quizz'):
    
    preambule = jinja2.Template(preambule).render(dossierAttache=dossierAttache, dossierSons=dossierSons)
    html = jinja2.Template(_KHtmlQuizz) \
                  .render(titre=titre, sousTitre=sousTitre, description=description, motsClef=motsClef,
                          preambule=preambule, quizz=quizz, etapes=etapes, attribsEtMercis=attribsEtMercis,
                          dossierAttache=dossierAttache, dossierSons=dossierSons, images=images, effort=effort,
                          topJsScript=_KTopJsScript, botJsScript=_KBotJsScript,
                          notebook=notebook, genDateTime=dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S'))

    chemFicCible = os.path.join('.', '{}{}.html'.format(prefixeFicCible, '.local' if dossierSons != '.' else ''))
    with codecs.open(chemFicCible, mode='w', encoding='utf-8-sig') as ficCible:
        ficCible.write(html)
        
    return chemFicCible
