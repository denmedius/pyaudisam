# -*- coding: utf-8 -*-

# Code de génération d'exposé HTML "Les Oiseaux à l'Oreille" pour un groupe d'espèces
#
# Exposé d'analyse détaillées et de description des différents types de chants et cris
# de chaque espèce du groupe, après quelques informations plus générales sur l'espèce elle-même
# (classification, milieux naturels fréquentés, régime alimentaire, identification visuelle,
#  état des populations ... et autres particularités diverses, le tout vu d'Auvergne).
#
# A partir :
#  * d'un dossier de fichiers sons bien nommés
#  * de textes HTML de description du groupe d'espèces, des espèces elles-mêmes et de leurs manifestations sonores.
#
# Auteur : Jean-Philippe Meuret (http://jpmeuret.free.fr/nature.html)
# Licence : CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr)

import os
import codecs

from collections import namedtuple, OrderedDict as odict

import datetime as dt

import jinja2

import pandas as pd

import lxml.html


# Descriteurs.
DescripteurManif = namedtuple('Descripteur', ['ordre', 'nom'])
DescripteurEspece = namedtuple('Descripteur', ['ordre', 'nom', 'latin', 'genre'])

# Descripteurs des types de manifestations sonores.
_KDTypesManifs = \
    odict([('ch', DescripteurManif(ordre=0, nom='Chant')), 
           ('crch', DescripteurManif(ordre=1, nom='Chants (ou cris ?)')),
           ('t',  DescripteurManif(ordre=2, nom='Tambour')),
           ('m', DescripteurManif(ordre=3, nom='Martellements')),
           ('cr', DescripteurManif(ordre=4, nom='Cris'))])

def _idTypeManif(nom):
    for idTM, descTM in _KDTypesManifs.items():
        if nom == descTM.nom:
            return idTM
    return None

# En sortie, autant de listes (de champs) que de types de manifestation sonores détéctés
def _extraireSon(cheminFichier):
    
    rawFields = cheminFichier[:-4].replace(' ', '').split('-')
    
    # 1ère passe (1 seule liste de champs, correspondant au 1er type de manifestation sonore trouvé).
    fields = list()
    
    fields.append(rawFields[0]) # espece
    
    fields.append(rawFields[1]) # typManif
   
    avecInd = -1 if 'avec' not in rawFields else rawFields.index('avec')
    if avecInd >= 0:
        descManif = '-'.join(rawFields[1:avecInd])
        fields.append(descManif.replace('-', ' ')) # descManif
        fields.append(''.join(rawFields[avecInd+1:-1])) # autresEsp
    else:
        descManif = '-'.join(rawFields[1:-1])
        fields.append(descManif.replace('-', ' ')) # descManif
        fields.append('') # autresEsp
        
    fields.append(rawFields[-1]) # idXC
    
    fields.append(cheminFichier) # FileName
    
    # 2nde passe (gestion multi-manifestations)
    manifs = list()
    #print(':'.join(fields), end='#')
    for field in descManif.split('-'):
        #print(field, end='=')
        if field in _KDTypesManifs:
            manifFields = fields.copy()
            manifFields[1] = field # typManif
            manifs.append(manifFields)
            #print(':'.join(manifFields[:-1]), end='')
    #print('')
    
    return manifs

#(extraireSon('PicCendre-cr-pul-au-nid-Russie-XC194414.mp3'),
# extraireSon('PicVert-cr-t-doux-parade-intime-cp-XC310864.mp3'),
# extraireSon('PicVert-ch-calme-long-avecPicEpeMesBlePinArbLorEurTroMigCorNoiEtoSan-XC216202.mp3'))

def _ordreDescManif(sSon):
    
    # Autres manifs en dernier dernier.
    if not sSon.descManif.startswith(sSon.typManif):
        return 4
    # Manifs atypiques en dernier.
    elif sSon.descManif.find(' atyp') > 0:
        return 3
    # Manifs typiques en premier.
    elif sSon.descManif.find(' typ') > 0:
        return 0
    # Ch comm après les autres.
    elif sSon.descManif.find(' comm') > 0:
        return 2
    # Les autres avant.
    else:
        return 1

def _decouperAutreEspeces(autresEsp):
    
    decAutEsp = list()
    
    nIndCar = 0
    while nIndCar < len(autresEsp):

        # D'abord, 'poss' ou 'prob' ou ... en minuscules
        nIndDebEsp = nIndCar
        while nIndCar < len(autresEsp) and autresEsp[nIndCar].islower():
            nIndCar += 1

        if nIndCar < len(autresEsp):

            # Ensuite : Genre sur 3 lettres
            nIndCar += 3

            # Ensuite : Espèce sur 3 lettres, sauf si 'Sp'
            nIndCar += 2
            if autresEsp[nIndCar-2:nIndCar] != 'Sp':
                nIndCar += 1

        # Et hop !
        decAutEsp.append(autresEsp[nIndDebEsp:nIndCar]) 
    
    return ' '.join(decAutEsp)

#(decouperAutreEspeces('possPicEpeRoiSpFauTNoHirRus'), decouperAutreEspeces('orchestre'), decouperAutreEspeces(''))

# Lecture d'un dossier et "extraction des sons" pour un groupe d'espèces
# (chaque fichier est nommé de manière à décrire son contenu : voire fonction 'extraireSon' ci-dessus)
# Pour chaque fichier, on ne retient qu'une espèce du groupe, mais possiblement plusieurs types de manifestation sonores
# En sortie, DataFrame avec autant de lignes par fichier que de manifestations sonores,
#            trié par espèce (ordre via dEspeces), type de manif. (ordre via _KDTypesManifs)
#            et par description du type de manif (1=commence par '<type> typ', 2=par '<type>', 3=reste)
def _lireDossierSons(cheminDossier, dEspeces):
    
    fichiers = [fic for fic in os.listdir(cheminDossier) if fic.endswith('.mp3')]
    
    dfSons = pd.DataFrame(data=[manif for fic in fichiers for manif in _extraireSon(fic)],
                          columns=['espece', 'typManif', 'descManif', 'autresEsp', 'idXC', 'fichier'])


    dfSons['ordEspece'] = dfSons.espece.apply(lambda esp: 99 if esp not in dEspeces else dEspeces[esp].ordre)
    dfSons.espece = dfSons.espece.apply(lambda esp: 'EspeceInconnue' if esp not in dEspeces else esp)

    dfSons['ordTypManif'] = \
        dfSons.typManif.apply(lambda typ: 99 if typ not in _KDTypesManifs else _KDTypesManifs[typ].ordre)
    dfSons['ordDescManif'] = dfSons.apply(_ordreDescManif, axis=1)
    dfSons.typManif = \
        dfSons.typManif.apply(lambda typ: 'Manif. sonore inconnue' if typ not in _KDTypesManifs \
                                                                   else _KDTypesManifs[typ].nom)

    dfSons.autresEsp = dfSons.autresEsp.apply(_decouperAutreEspeces)
    
    dfSons.sort_values(by=['ordEspece', 'ordTypManif', 'ordDescManif'], inplace=True)
    
    return dfSons.drop(['ordEspece', 'ordTypManif', 'ordDescManif'], axis=1)

# Conversion DataFrame en arbre/liste pour génération page HTML "espèce par espèce" via jinja2
def _arbreEspeces(dfSons, dEspeces, dSpecifsEspeces, urlDossierSons=None):

    especes = list()
    for esp, descEsp in dEspeces.items():
        nomEsp = descEsp.nom
        dSpecifsEsp = dSpecifsEspeces[esp]
        dfSonsEsp = dfSons[dfSons.espece == esp]
        typManifs = list()
        for typManif, descTypManif in _KDTypesManifs.items():
            nomTypManif = descTypManif.nom
            dfTypManif = dfSonsEsp[dfSonsEsp.typManif == nomTypManif]
            sons = list()
            for iSon, sSon in dfTypManif.iterrows():
                lienXC = 'https://www.xeno-canto.org/{}'.format(sSon.idXC[2:])
                sons.append(dict(id=iSon, desc=sSon.descManif, autres=sSon.autresEsp, 
                                 fichier=sSon.fichier, idXC=sSon.idXC,
                                 url=urlDossierSons + '/' + sSon.fichier if urlDossierSons \
                                     else lienXC + '/download',
                                 lienXC=lienXC))
            if len(sons) == 0 and typManif not in dSpecifsEsp['manifs']:
                continue
            specTypManif = dSpecifsEsp['manifs'].get(typManif, '').strip() or '<p>Pas de commentaire particulier.</p>'
            typManifs.append(dict(id=typManif, nom=descTypManif.nom, specifs=specTypManif, sons=sons))
        especes.append(dict(id=esp, nom=nomEsp, latin=descEsp.latin, genre=descEsp.genre,
                            specifs=dSpecifsEsp['specifs'], typManifs=typManifs))
    
    return especes

# Conversion DataFrame en arbre/liste pour génération page HTML "comparaison des espèces" via jinja2
def _arbreTypesManifs(dfSons, dEspeces, urlDossierSons=None):

    typManifs = list()
    for typManif in dfSons.typManif.unique():
        dfTypManif = dfSons[dfSons.typManif == typManif]
        especes = dfTypManif.espece.unique()
        espSons = list() # list(espece => list(sons))
        nMaxSons = 0
        for esp in especes:
            dfEsp = dfTypManif[dfTypManif.espece == esp]
            nMaxSons = max(nMaxSons, len(dfEsp))
            sons = list()
            for iSon, sSon in dfEsp.iterrows():
                lienXC = 'https://www.xeno-canto.org/{}'.format(sSon.idXC[2:])
                sons.append(dict(id=iSon, desc=sSon.descManif, autres=sSon.autresEsp, fichier=sSon.fichier,
                                 url=urlDossierSons + '/' + sSon.fichier if urlDossierSons \
                                     else lienXC + '/download',
                                 idXC=sSon.idXC, lienXC=lienXC))
            espSons.append(sons)
        for indEsp in range(len(especes)):
            for indSon in range(nMaxSons - len(espSons[indEsp])):
                espSons[indEsp].append(dict(desc='', autres='', fichier='', idXC='', lienXC=''))
        sonsEsps = list()
        for indSon in range(nMaxSons):
            sons = dict()
            for indEsp in range(len(especes)):
                sons[especes[indEsp]] = espSons[indEsp][indSon]
            sonsEsps.append(sons)
        typManifs.append(dict(id=_idTypeManif(typManif), nom=typManif, 
                              especes=[dict(id=esp, nom=dEspeces[esp].nom, latin=dEspeces[esp].latin,
                                            genre=dEspeces[esp].genre) for esp in especes],
                              sons=sonsEsps))
    
    return typManifs

# Générateur de table des matières pour le chapitre "Généralités"
def _planGeneralites(html2Parse, tag2List='h3'):
    
    htmlToc = '<ol style="list-style-type: decimal">\n'
    
    doc = lxml.html.fromstring(html2Parse)
    for node in doc.xpath('//' + tag2List):
        htmlToc += '<li><a href="#{id}">{text}</a></li>\n'.format(id=node.attrib['id'], text=node.text)
        
    htmlToc += '</ol>\n'

    return htmlToc

# Javascripts
_KScriptsJs = """
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

# Modèle jinja2 de page pour un groupe d'espèces.
_KHtmlGroupeEspeces = """
<!DOCTYPE HTML>
<head>
    <meta charset="utf-8">
    <meta name="author" content="Jean-Philippe Meuret"/>
    <meta name="copyright" content="Jean-Philippe Meuret 2018"/>
    <meta name="license" content="CC BY NC SA"/>
    <meta name="subtitle" content="{{sousTitre}}"/>
    <meta name="description" content="{{description}}"/>
    <meta name="keywords" content="chant, cri, oiseau, ornithologie, oreille, identification, {{motsCles}}"/>
    <meta name="datetime" contents="{{genDateTime}}"/>
    <title>{{titre}} ... à l'oreille !</title>
    <link rel="stylesheet" media="screen" type="text/css" href="{{dossierAttache}}/chants.css">
    <script>
      {{scriptsJs}}
    </script>
</head>

<body>

  <table id="title">
    <tr>
      <td style="font-size: 480%">{{titre}}</td>
      <td style="font-size: 240%; margin-left:15px">... à l'oreille</td>
    </tr>
    <tr>
      <td colspan="2" style="font-size: 240%; text-align: center">{{sousTitre}}</td>
    </tr>
    <tr>
      <td colspan="2" style="font-size: 120%; text-align: center">{{description}}</td>
    </tr>
  </table>
  
  <div style="margin-left: 15px">
    
    <table style="min-width: 320px; margin-left: auto; margin-right: auto">
      <tr>
        <td>
          <h2>Table des matières</h2>
          <div style="margin-left: 10px">
            <ol style="list-style-type: upper-roman">
                <li><a href="#généralités">Généralités</a></li>
                  {{planGeneralites}}
                <li><a href="#détails">Détails sonores par espèce</a></li>
                <ol style="list-style-type: decimal">
                <li><a href="#glossaire">Glossaire / Abréviations</a></li>
                {% for esp in especes %}
                    <li><a href="#{{esp.id}}">{{esp.nom}}</a> <i>({{esp.latin}})</i></li>
                    <ol style="list-style-type: lower-latin">
                    {% for typMnf in esp.typManifs %}
                        <li><a href="#{{esp.id}}.{{typMnf.id}}">{{typMnf.nom}}</a></li>
                    {% endfor %}
                    </ol>
                {% endfor %}
                </ol>
                <li><a href="#comparaisons">Comparaisons en vis à vis</a></li>
                <ol style="list-style-type: lower-latin">
                {% for typMnf in typesManifs %}
                    <li><a href="#Comp.{{typMnf.id}}">{{typMnf.nom}}</a></li>
                {% endfor %}
                </ol>
                <li><a href="#quizz">Quizz sur concerts naturels</a></li>
                <li><a href="#licence">Licence / Auteur</a></li>
                <li><a href="#remerciements">Remerciements</a></li>
                <li><a href="#attributions">Emprunts / Attributions</a></li>
            </ol>
          </div>
        </td>
        <td style="align: right">
          {% for tocImg in images.tocImg %}
            <img class="shrinkable" src="{{dossierAttache}}/{{tocImg.img}}"/>
            <p style="text-align: right; margin: 0 0 0 0; padding: 0 0 0 0">{{tocImg.legend}}</p>
          {% endfor %}
        </td>
      </tr>
    </table>

    <img class="center" height="32" style="margin-top: 30px"
         src="{{dossierAttache}}/fa-feather-alt.svg" alt="---" />

    <h2 id="généralités">Généralités</h2>
    <div class="chapter" style="margin-left: 10px">
      {{generalites}}
    </div>

    <img class="center" height="32" style="margin-top: 30px"
         src="{{dossierAttache}}/fa-feather-alt.svg" alt="---" />

    <h2 id="détails">Détails sonores par espèce</h2>
    <div style="margin-left: 10px">
    
      <div class="chapter">
      
        <p>Après rappel de quelques particularités de l'espèce (abondance, milieux de prédilection, statut en Auvergne,
           régime alimentaire, biologie de reproduction, moeurs particulières, identification visuelle, ... etc),
           on trouvera ci-après pour chacune d'elle, et pour chacun de ses types de manifestation sonore
           (cri, chant, et même tambour, martellement, pour les pics ...),
           des échantillons sonores aussi typiques que possible,
           avec dans l'ordre, pour chacun d'eux :</p>
        <ul>
            <li>un lecteur audio pour l'écouter en direct (attention : soyez patient, car selon votre navigateur,
                ou la vitesse de votre connexion internet, il se peut que vous deviez attendre
                un petit moment avant de pouvoir jouer tous les enregistrements, en particulier les derniers
                en bas de page ... parfois jusqu'à plusieurs minutes),</li>
            <li>une description rapide, à base d'adjectifs et de qualificatifs abrégés, sans accents ...
                et/ou d'onomatopées
                (Cf. <a href="#glossaire">glossaire</a> ci-dessous pour la signification des abréviations),</li>
            <li>la liste des autres espèces présentes en arrière plan, à deviner d'abord (quiz !)
                avant de cliquer sur le petit oeil
                <img height="16" src="{{dossierAttache}}/fa-eye-regular.svg" alt="Montrer"/> :
                chaque espèce est signalée par un code à 5 ou 6 lettres
                (3 premières lettres du genre français et 3 premières de l'espèce en général, 
                 mais 'Sp' quand l'espèce est inconnue) ; espèces en gros par ordre d'apparition,
                 sauf les autres espèces du groupe, à la fin de la liste,</li>
            <li>le lien vers la page de l'enregistrement source
                sur <a href="https://www.xeno-canto.org/">xeno-canto.org</a>,
                qui vous permettra entre autre de télécharger l'enregistrement sur votre ordinateur
                si vous voulez le décortiquer tranquillement
                (via <a href="http://audacity.fr/" target="_blank">Audacity</a> par exemple),
                ou d'obtenir quelques informations sur son auteur, ou le lieu où il a été "mis en boîte" par exemple.</li>
        </ul>
        {% if ficTableauSynth %}
        <p>Et pour vos sorties sur le terrain, un <a href="{{dossierAttache}}/{{ficTableauSynth}}">tableau de synthèse</a>
           résumant ce qu'il y a retenir pour chaque espèce, et permettant de les comparer d'un seul coup d'oeil.</p>
        {% endif %}
           
        <p>N.B. Faute de temps, les enregistrements sources n'ont pas été coupés et / ou remontés,
           ce qui aurait permis d'isoler plus précisément les manifestations sonores ciblées ;
           à vous de les retrouver : la plupart du temps, c'est la première qu'on entend, mais parfois non ;
           dans ce cas, fiez-vous à la colonne 'Description', qui liste ces manifestations dans l'ordre d'apparition
           (Cf. glossaire pour leur nom de code : cr, ch, crch, t, m).</p>
           
      </div>
      
      <h3 id="glossaire">Glossaire / Abréviations</h3>
      <div class="chapter" style="margin-left: 10px">
        <p>Signification des codes et abréviations utilisés dans la colonne "Description" des tableaux ci-après
           (N.B. Cliquez sur les '?' dans l'entête de cette colonne pour revenir directement ici
            si vous avez un trou de mémoire ;-).</p>
        <ul>
          {% for expr, def in glossaire %}
             <li>{{expr}} : {{def}},</li>
          {% endfor %}
        </ul>
        
      </div>

      {% for esp in especes %}
      
        <img class="center" height="32" style="margin-top: 30px"
             src="{{dossierAttache}}/fa-feather-alt.svg" alt="---" />

        <h3 id="{{esp.id}}">{{esp.nom}} <i>({{esp.latin}})</i></h3>
        <div style="margin-left: 10px">
        
          <div class="chapter">
            {{esp.specifs}}
          </div>
          
          {% for typMnf in esp.typManifs %}
            <h4 id="{{esp.id}}.{{typMnf.id}}">
              {{typMnf.nom}}
              {% if esp.genre == 'm' %}
                du
              {% else %}
                de la
              {% endif %}
              {{esp.nom}}
              <i>({{esp.latin}})</i>
            </h4>
            <div class="chapter" style="margin-left: 10px">
            
              {{typMnf.specifs}}
              {% if typMnf.sons %}
                <table cellspacing='0'> 
                  <thead>
                    <tr>
                      <th><h3 style="margin: 10px 10px 10px 5px">Enregistrement</h3></th>
                      <th>
                        <h3 style="margin: 10px 10px 10px 5px">Description (<a href="#glossaire">?</a>)</h3>
                        
                      </th>
                      <th><h3 style="margin: 10px 10px 10px 5px">Autres espèces</h3></th>
                      <th><h3 style="margin: 10px 10px 10px 5px">Source</h3></th>
                    </tr>
                   </thead>
                   <tbody>
                     {% for son in typMnf.sons %}
                       <tr> 
                         <td>
                           <audio controls>
                             <source src="{{son.url}}" type="audio/mp3" preload="none"/>
                           </audio>
                         </td>
                         <td>{{son.desc}}</td>
                         <td>
                           {% if son.autres %}
                             <div>
                               <a href="javascript:show('d{{son.id}}')" id="d{{son.id}}s">
                                 <img height="16" src="{{dossierAttache}}/fa-eye-regular.svg" alt="Montrer"/></a>
                               <div id="d{{son.id}}" style="display: none">
                                 <a href="javascript:hide('d{{son.id}}')">
                                   <img height="16" src="{{dossierAttache}}/fa-eye-slash-regular.svg" alt="Cacher"/>
                                 </a>
                                 <span>{{son.autres}}</span>
                               </div>
                             </div>
                           {% endif %}
                         </td>
                         <td><a href='{{son.lienXC}}' target='_blank'>{{son.idXC}}</a></td>
                         <!-- <td>{{son.fichier}}</td> -->
                       </tr>
                     {% endfor %}
                   </tbody>
                 </table>
              {% else %}
              <p>Aucun échantillon sonore trouvé, désolé :-(</p>
              {% endif %}
            
            </div>
          
          {% endfor %}
        </div>
          
      {% endfor %}
      
    </div>

    <img class="center" height="32" style="margin-top: 30px"
         src="{{dossierAttache}}/fa-feather-alt.svg" alt="---" />

    <h2 id="comparaisons">Comparaisons sonores en vis à vis</h2>
    <div style="margin-left: 10px">
    
      <div class="chapter">
      
        <p>Rien de neuf ici, mais pour chaque type de manifestation sonore (cris, chants, tambour, ...),
           une présentation en vis à vis des mêmes échantillons sonores pour chaque espèce,
           avec les mêmes informations, pour pouvoir les comparer plus facilement.</p>
           
        {% if ficTableauSynth %}
          <p>Rappel : <a href="{{dossierAttache}}/{{ficTableauSynth}}">tableau de synthèse</a>.</p>
        {% endif %}

      </div>
    
      {% for typMnf in typesManifs %}
        <h3 id="Comp.{{typMnf.id}}">{{typMnf.nom}}</h3>
        <table cellspacing='0'> 
          <thead>
            <tr>
            {% for esp in typMnf.especes %}
              <th><h3 style="margin: 10px 10px 10px 5px">{{esp.nom}}</h3></th>
            {% endfor %}
            </tr>
            <tr>
            {% for esp in typMnf.especes %}
              <td><i>({{esp.latin}})</i></td>
            {% endfor %}
            </tr>
          </thead>
          <tbody>
            {% for iSon in range(typMnf.sons|length) %}
              <tr> 
                {% for esp in typMnf.especes %}
                  <td>
                    {% if typMnf.sons[iSon][esp.id].fichier %}
                      <audio controls>
                        <source src="{{typMnf.sons[iSon][esp.id].url}}" type="audio/mp3" preload="none"/>
                      </audio>
                      <p>{{typMnf.sons[iSon][esp.id].desc}} (<a href="#glossaire">?</a>)</p>
                      {% if typMnf.sons[iSon][esp.id].autres %}
                        <div>
                          <a href="javascript:show('c{{typMnf.sons[iSon][esp.id].id}}')"
                             id="c{{typMnf.sons[iSon][esp.id].id}}s">
                            <img height="16" src="{{dossierAttache}}/fa-eye-regular.svg" alt="Montrer"/></a>
                          <div id="c{{typMnf.sons[iSon][esp.id].id}}" style="display: none">
                            <a href="javascript:hide('c{{typMnf.sons[iSon][esp.id].id}}')">
                              <img height="16" src="{{dossierAttache}}/fa-eye-slash-regular.svg" alt="Cacher"/>
                            </a>
                            <span>{{typMnf.sons[iSon][esp.id].autres}}</span>
                          </div>
                        </div>
                      {% else %}
                        <img height="16" style="opacity: 0" src="{{dossierAttache}}/fa-eye-regular.svg"/>
                      {% endif %}
                      <p style="text-align: right">
                        <a href='{{typMnf.sons[iSon][esp.id].lienXC}}' target='_blank'>
                          {{typMnf.sons[iSon][esp.id].idXC}}
                        </a>
                      </p>
                    {% endif %}
                  </td>
                {% endfor %}
              </tr>
            {% endfor %}
          </tbody>
        </table> 
      {% endfor %}
    
    </div>

    <img class="center" height="32" style="margin-top: 30px"
         src="{{dossierAttache}}/fa-feather-alt.svg" alt="---" />

    <h2 id="quizz">Quizz sur concerts naturels</h2>
    <div class="chapter" style="margin-left: 10px">
    
      <p>Des quizz progressifs et détaillés sont publiés sur des pages dédiées
         (cherchez le mot "quiz" sur <a href="http://jpmeuret.free.fr/nature.html" target="_blank">
          ma page "Nature"</a>).</p>
    
    </div>

    <h2 id="licence">Licence / Auteur</h2>
    <div class="chapter" style="margin-left: 10px">
    
      <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr" target="_blank">
        <img height="48" src="{{dossierAttache}}/by-nc-sa.eu.svg" alt="Creative Commons BY NC SA 4.0"/>
      </a>
  
      <p>Ce document est publié sous la licence
         <b><a href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr" target="_blank">
         Creative Commons BY NC SA 4.0</a></b>
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
          <li>des sons liés au présent document, issus en totalité de
              <a href="https://www.xeno-canto.org/" target="_blank">xeno-canto.org</a>
              (Voir ci-dessus le lien associé à chaque enregistrement pour sa source exacte et son auteur),</li>
          <li>des photos et dessins d'illustration (Voir légende associée à chacun).</li>
      </ul>
         
    </div>

    <h2 id="remerciements">Remerciements</h2>
    <div class="chapter" style="margin-left: 10px">
    
      {{remerciements}}
                  
      <p>Enfin, tous les enregistrements utilisés ici proviennent du site
         <a href="https://www.xeno-canto.org/" target="_blank">xeno-canto.org</a> :
         un très grand merci aux ornithologues qui ont bien voulu partager leurs trouvailles
         et ainsi rendre cette publication possible.</p>
         
    </div>
    
    <h2 id="attributions">Emprunts / Attributions</h2>
    <div class="chapter" style="margin-left: 10px">
    
      {{attributions}}

      <p>Merci enfin au projet <a href="https://fontawesome.com/" target="_blank">Font Awesome</a>,
         qui produit et distribue gratuitement, sous la licence
         <a href="https://creativecommons.org/licenses/by/4.0/deed.fr" target="_blank">CC BY 4.0</a>,
         des icônes et pictogrammes "trop stylés", dont
         <img height="16" src="{{dossierAttache}}/fa-eye-regular.svg" alt="Icône Montrer" />,
         <img height="16" src="{{dossierAttache}}/fa-eye-slash-regular.svg" alt="Icône Cacher" />,
         <img height="16" src="{{dossierAttache}}/fa-feather-alt.svg" alt="Icône Séparateur" /> et
         <img width="16" height="16" src="{{dossierAttache}}/fa-angle-up.svg" alt="Icône Haut de page" />,
         dont j'ai simplement changé la couleur, noire à l'origine, en vert (forcément).</p>
      
    </div>
    
    <h6>
        Page générée via <a href="https://www.python.org/" target="_blank">Python 3</a>,
        <a href="https://pandas.pydata.org/" target="_blank">Pandas</a>
        et <a href="http://jinja.pocoo.org/" target="_blank">Jinja 2</a>
        dans <a href="http://jupyter.org/" target="_blank">Jupyter Notebook</a>
        (sources : <a href="./chants.py" target="_blank">chants.py</a>
         et <a href="./{{notebook}}" target="_blank">{{notebook}}</a>),
        le {{genDateTime}}.
    </h6>

  </div>

  <button onclick="scrollToTop()" id="toTopBtn" title="Remonter" alt="Remonter">
    <img width="64" height="64" src="{{dossierAttache}}/fa-angle-up.svg"/>
  </button>

</body>
"""

KDGlossaire = \
{
  'accel' : 'accéléré',
  'ad' : 'adulte',
  'age' : 'âgé (pour un poussin = en fin d\'élevage)',
  'agress' : 'agressif, agression',
  'al' : 'alarme',
  'alim' : 'alimentaire',
  'ailes' : 'bruit d\'ailes, à l\'envol ou au passage',
  'ch' : 'chant(s)',
  'comm' : 'communautaire, en groupe',
  'cp' : 'couple',
  'cr' : 'cri(s)',
  'crch' : 'cri(s) similaire(s) au(x) chant(s) et/ou réciproquement, on ne sait pas décider si c\'est un chant ou des cris, ou bien c\'est un mélange des 2 (plusieurs individus)',
  'crec' : 'crécelle (son mat, sec, rêche répété rapidement)',
  'debruit' : 'enregistrement duquel on a soustrait une estimation du "bruit de fond" (au mieux générateur de silences étranges, souvent assez déteriorant)',
  'deter' : 'enregistrement déterioré par un traitement numérique quelconque (Cf. fph, debruit, ou autre), qui empêche ou complique la reconnaissance d\'une ou plusieurs espèces, par ex.',
  'disp' : 'dispute',
  'doppler' : 'effet Doppler (la hauteur des sons émis par un oiseau en rapprochement augmente, et quand il s\'éloigne, elle diminue)',
  'dort' : 'dortoir',
  'doul' : 'douleur',
  'elec' : 'électrique',
  'envol' : 'action de s\'envoler',
  'extr' : 'extrème (pour une alarme, par ex.)',
  'fele' : 'fêlé',
  'fph' : 'enregistrement filtré passe-haut (basses fréquences supprimées ou atténuées)',
  'gagn' : 'gagnage = en train de se nourrir',
  'imit' : 'imite',
  'ind, indiv' : 'individu',
  'inq' : 'inquiétude',
  'intim' : 'intime (couple)',
  'jq' : 'juvénile quémandant',
  'juv' : 'juvénile',
  'jvq' : 'juvénile volant quémandant',
  'liq' : 'liquide',
  'm' : 'martellement(s)',
  'rap' : 'râpeux',
  'rep' : 'réponse, se répondent',
  'roule' : 'roulé',
  'par' : 'parade',
  'pose' : 'posé (= pas en vol !)',
  'poss' : 'possible',
  'pours' : 'poursuite',
  'prob' : 'probable',
  'pul' : 'pulli = poussins (au nid si nidicoles)',
  't' : 'tambour (pour les pics, et les grands singes ;-)',
  'tous' : 'tous (les types de cris, par ex.)',
  'tril' : 'trille (alternance très rapide de 2 notes à hauteurs différentes ; un peu comme une crécelle, mais en plus tonal, mélodique)',
  'typ' : 'typique, représentatif de l\'espèce (pour un cri, un chant, un tambour)',
  'vibr' : 'vibré, vibration',
  'vol' : 'en vol'
}

# Fonction principale de génération de la page.
# * titre : général de la page générée (+ balise head/title)
# * sousTitre : qq détails en plus (+ balise head/meta description)
# * description : pour balise head/meta description
# * motsCles : pour balise head/meta keywords (séparés par des virgules)
# * especes : odict(<idEsp> : DescripteurEspece)
# * generalites : <texte html>
# * specificites : dict(<idEsp> : dict(specifs : <texte html>, manifs : dict(<idManif> : <texte html>)))
# * glossaire : dict(<abrév.> : <explication>), à fusionner avec KDGlossaire
# * remerciements : <texte html>
# * effort : nb total d'heures passé pour la construction
# * attributions : <texte html>
# * images : dict(tocImg : list(dict(img=<nom fic. dans dossierAttache>, legend=<légende>)))
# * urlDossierSons : URL locale dossier sons MP3, ou None pour lien direct xeno-canto.org (ou même dossier que HTML généré si pas XC)
# * dossierSons : chemin local dossier des fichiers sons à analyser (enfin ... leur nom)
# * dossierAttache : URL dossier annexe (fichiers attachés autres que les sons)
# * ficTableauSynth : nom du fichier tableau de synthèse (supposé être dans le dossier attaché)
# * notebook : nom du fichier notebook appelant (où se trouve le contenu rédigé), supposé être dans .
# * prefixeFicCible : préfixe du nom de fichier HTML généré (ajout .local si pub locale, rien si pub sur site web)
def buildHtmlPage(titre, sousTitre, description, motsCles,
                  especes, generalites, specificites, glossaire, remerciements, effort,
                  attributions, images, urlDossierSons, dossierSons, dossierAttache,
                  ficTableauSynth=None, notebook='Chants.ipynb', prefixeFicCible='chants'):
    
    dfSons = _lireDossierSons(cheminDossier=dossierSons, dEspeces=especes)

    # Génération du glossaire complet, trié par entrée.
    dGlossCompl = dict(KDGlossaire)
    dGlossCompl.update(glossaire)
    glossCompl = sorted(dGlossCompl.items())
    
    # Première passe dans les textes.
    generalites = jinja2.Template(generalites).render(dossierAttache=dossierAttache)
    for esp in specificites:
        specificites[esp]['specifs'] = \
            jinja2.Template(specificites[esp]['specifs']).render(dossierAttache=dossierAttache)
        for manif in specificites[esp]['manifs']:
            specificites[esp]['manifs'][manif] = \
                jinja2.Template(specificites[esp]['manifs'][manif]).render(dossierAttache=dossierAttache)

    # Dernière passe finale.
    html = jinja2.Template(_KHtmlGroupeEspeces) \
                  .render(titre=titre, sousTitre=sousTitre, description=description, motsCles=motsCles,
                          especes=_arbreEspeces(dfSons, especes, specificites,
                                                urlDossierSons=urlDossierSons),
                          typesManifs=_arbreTypesManifs(dfSons, especes, urlDossierSons=urlDossierSons),
                          generalites=generalites, planGeneralites=_planGeneralites(generalites),
                          glossaire=glossCompl, remerciements=remerciements,
                          attributions=attributions, ficTableauSynth=ficTableauSynth,
                          dossierAttache=dossierAttache, images=images, effort=effort,
                          scriptsJs=_KScriptsJs, notebook=notebook,
                          genDateTime=dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S'))

    chemFicCible = os.path.join('.', '{}{}.html'.format(prefixeFicCible, '.local' if urlDossierSons else ''))
    with codecs.open(chemFicCible, mode='w', encoding='utf-8-sig') as ficCible:
        ficCible.write(html)
        
    return chemFicCible, dfSons
