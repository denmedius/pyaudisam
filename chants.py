# -*- coding: utf-8 -*-

# Code de génération d'exposé HTML "identification à l'oreille" pour un groupe d'espèces
#
# A partir :
#  * d'un dossier de fichiers sons bien nommés
#  * de textes HTML de description du groupe d'espèces, des espèces elles-mêmes et de leurs manifestations sonores.

import os
import codecs

from collections import namedtuple, OrderedDict as odict

import datetime as dt

import jinja2

import pandas as pd

import lxml.html


# Descriteur générique.
Descripteur = namedtuple('Descripteur', ['ordre', 'nom'])

# Descripteurs des types de manifestations sonores.
_KDTypesManifs = \
    odict([('ch', Descripteur(ordre=0, nom='Chant')), 
           ('crch', Descripteur(ordre=1, nom='Chants (ou cris ?)')),
           ('t',  Descripteur(ordre=2, nom='Tambour')),
           ('m', Descripteur(ordre=3, nom='Martellement')),
           ('cr', Descripteur(ordre=4, nom='Cris'))])

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
    dfSons.espece = dfSons.espece.apply(lambda esp: 'Pic inconnu' if esp not in dEspeces else dEspeces[esp].nom)

    dfSons['ordTypManif'] = \
        dfSons.typManif.apply(lambda typ: 99 if typ not in _KDTypesManifs else _KDTypesManifs[typ].ordre)
    dfSons['ordDescManif'] = \
        dfSons.apply(lambda sSon: 2 if not sSon.descManif.startswith(sSon.typManif) \
                                    else 0 if sSon.descManif.startswith(sSon.typManif + ' typ') else 1, axis=1)
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
        dfSonsEsp = dfSons[dfSons.espece == nomEsp]
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
        especes.append(dict(id=esp, nom=nomEsp, specifs=dSpecifsEsp['specifs'], typManifs=typManifs))
    
    return especes

# Conversion DataFrame en arbre/liste pour génération page HTML "comparaison des espèces" via jinja2
def _arbreTypesManifs(dfSons, urlDossierSons=None):

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
        typManifs.append(dict(nom=typManif, especes=list(especes), sons=sonsEsps))
    
    return typManifs

# Générateur de table des matières pour le chapitre "Généralités"
def _planGeneralites(html2Parse, tag2List='h3'):
    
    htmlToc = '<ol style="list-style-type: decimal">\n'
    
    doc = lxml.html.fromstring(html2Parse)
    for node in doc.xpath('//' + tag2List):
        htmlToc += '<li><a href="#{id}">{text}</a></li>\n'.format(id=node.attrib['id'], text=node.text)
        
    htmlToc += '</ol>\n'

    return htmlToc

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
        <title>{{titre}}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <meta name="author" content="Jean-Philippe Meuret"/>
        <meta name="copyright" content="Jean-Philippe Meuret 2018"/>
        <meta name="license" content="CC BY NC SA"/>
        <meta name="description" content="{{title}}"/>
        <meta name="keywords" content="chant, cri, oiseau, ornithologie, oreille, identification, {{keywords}}"/>
        <meta name="datetime" contents="{{genDateTime}}"/>
        <style type="text/css" media="screen">
          {{stylesCss}}
        </style>
        <script>
          {{scriptsJs}}
        </script>
    </head>

    <body>
    
      <h1>{{titre}}</h1>
      <h3 style="text-align: center">{{sousTitre}}</h3>
        
      <div style="margin-left: 15px">
        
        <table>
          <tr>
            <td style="width: 50%">
              <h2>Plan</h2>
              <div style="margin-left: 10px">
                <ol style="list-style-type: upper-roman">
                    <li><a href="#généralités">Généralités</a></li>
                      {{planGeneralites}}
                    <li><a href="#détails">Détails par espèce</a></li>
                    <ol style="list-style-type: decimal">
                    {% for esp in especes %}
                        <li><a href="#{{esp.id}}">{{esp.nom}}</a></li>
                        <ol style="list-style-type: lower-latin">
                        {% for typMnf in esp.typManifs %}
                            <li><a href="#{{esp.id}}{{typMnf.id}}">{{typMnf.nom}}</a></li>
                        {% endfor %}
                        </ol>
                    {% endfor %}
                    </ol>
                    <li><a href="#comparaisons">Comparaisons en vis à vis</a></li>
                    <ol style="list-style-type: lower-latin">
                    {% for typMnf in typesManifs %}
                        <li><a href="#{{typMnf.id}}">{{typMnf.nom}}</a></li>
                    {% endfor %}
                    </ol>
                    <li><a href="#quiz">Quiz sur concerts naturels</a></li>
                    <li><a href="#glossaire">Glossaire / Abréviations</a></li>
                    <li><a href="#licence">Licence / Auteur</a></li>
                    <li><a href="#remerciements">Remerciements</a></li>
                </ol>
              </div>
            </td>
            <td style="align: right">
              <img src="{{dossierAttache}}/{{images.tocImg.img}}"/>
              <h6 style="text-align: right; margin: 0 0 0 0; padding: 0 0 0 0">{{images.tocImg.legend}}</h6>
            </td>
          </tr>
        </table>

        <h2 id="généralités">Généralités</h2>
        <div style="margin-left: 10px">
          {{generalites}}
        </div>

        <h2 id="détails">Détails sonores par espèce</h2>
        <div style="margin-left: 10px">
        
          <p>Pour chaque espèce, et chacun de ses types de manifestation sonore (cri, chant, tambour, martellement, ...),
             on trouve ici des échantillons sonores aussi typiques que possible, avec dans l'ordre, pour chacun d'eux :</p>
          <ul>
              <li>un lecteur audio pour l'écouter en direct,</li>
              <li>une description rapide, à base d'adjectifs et de qualificatifs abrégés, sans accents ...
                  et/ou d'onomatopées
                  (Cf. <a href="#glossaire">glossaire</a> pour la signification des abréviations),</li>
              <li>la liste des autres espèces présentes en arrière plan, à deviner d'abord (quiz !)
                  avant de cliquer sur le petit oeil : chaque espèce est signalée par un code à 5 ou 6 lettres
                  (3 premières lettres du genre et 3 premières de l'espèce en général, 
                   mais 'Sp' quand l'espèce est inconnue) ; espèces en gros par ordre d'apparition,
                   sauf les autres espèces du groupe, à la fin de la liste,</li>
              <li>le lien vers la page de l'enregistrement source
                  sur <a href="https://www.xeno-canto.org/">xeno-canto.org</a>.</li>
          </ul>
          <p>Et pour vos sorties sur le terrain, un <a href="{{dossierAttache}}/{{ficTableauSynth}}">tableau de synthèse</a>
             résumant ce qu'il y a retenir pour chaque espèce, et permettant de les comparer d'un seul coup d'oeil.</p>
             
          <p>N.B. Faute de temps, je n'ai pas coupé ou remonté les enregistrements sources (ils sont pris tels quels),
             ce qui aurait permis d'isoler plus précisément chaque manifestation sonore ciblée ;
             à vous de la retrouver : la plupart du temps, c'est la première, mais parfois non ; dans ce cas,
             fiez-vous à la colonne 'Description', qui liste ces manifestations dans l'ordre d'apparition
             (Cf. glossaire pour leurs noms de code : cr, ch, crch et t).</p>
        
          {% for esp in especes %}
            <h3 id="{{esp.id}}">{{esp.nom}}</h3>
            <div style="margin-left: 10px">
            
              {{esp.specifs}}
              {% for typMnf in esp.typManifs %}
                <h4 id="{{esp.id}}{{typMnf.id}}">{{typMnf.nom}} du {{esp.nom}}</h4>
                <div style="margin-left: 10px">
                
                  {{typMnf.specifs}}
                  {% if typMnf.sons %}
                    <table cellspacing='0'> 
                      <thead>
                        <tr>
                          <th><h3 style="margin: 10px 10px 10px 5px">Enregistrement</h3></th>
                          <th><h3 style="margin: 10px 10px 10px 5px">Description</h3></th>
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

        <h2 id="comparaisons">Comparaisons sonores en vis à vis</h2>
        <div style="margin-left: 10px">
        
          <p>Rien de neuf ici, mais pour chaque type de manifestation sonore (cris, chants, tambour, ...)
             une présentation en vis à vis des mêmes échantillons sonores pour chaque espèce,
             avec les mêmes informations, pour pouvoir les comparer plus facilement.</p>
        
          <p>Rappel : <a href="{{dossierAttache}}/{{ficTableauSynth}}">tableau de synthèse</a>.</p>

          {% for typMnf in typesManifs %}
            <h3 id="{{typMnf.id}}">{{typMnf.nom}}</h3>
            <table cellspacing='0'> 
              <thead>
                <tr>
                {% for esp in typMnf.especes %}
                  <th><h3 style="margin: 10px 10px 10px 5px">{{esp}}</h3></th>
                {% endfor %}
                </tr>
              </thead>
              <tbody>
                {% for iSon in range(typMnf.sons|length) %}
                  <tr> 
                    {% for esp in typMnf.especes %}
                      <td>
                        {% if typMnf.sons[iSon][esp].fichier %}
                          <audio controls>
                            <source src="{{typMnf.sons[iSon][esp].url}}" type="audio/mp3" preload="none"/>
                          </audio>
                          <p>{{typMnf.sons[iSon][esp].desc}}</p>
                          {% if typMnf.sons[iSon][esp].autres %}
                            <div>
                              <a href="javascript:show('c{{typMnf.sons[iSon][esp].id}}')"
                                 id="c{{typMnf.sons[iSon][esp].id}}s">
                                <img height="16" src="{{dossierAttache}}/fa-eye-regular.svg" alt="Montrer"/></a>
                              <div id="c{{typMnf.sons[iSon][esp].id}}" style="display: none">
                                <a href="javascript:hide('c{{typMnf.sons[iSon][esp].id}}')">
                                  <img height="16" src="{{dossierAttache}}/fa-eye-slash-regular.svg" alt="Cacher"/>
                                </a>
                                <span>{{typMnf.sons[iSon][esp].autres}}</span>
                              </div>
                            </div>
                          {% else %}
                            <img height="16" style="opacity: 0" src="{{dossierAttache}}/fa-eye-regular.svg"/>
                          {% endif %}
                          <p style="text-align: right">
                            <a href='{{typMnf.sons[iSon][esp].lienXC}}' target='_blank'>{{typMnf.sons[iSon][esp].idXC}}</a>
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

        <h2 id="quiz">Quiz sur concerts naturels</h2>
        <div style="margin-left: 10px">
        
          <p>En construction : patience ...</p>
        
        </div>

        <h2 id="glossaire">Glossaire / Abréviations</h2>
        <div style="margin-left: 10px">
          <ul>
            <li>ch : chant(s)</li>
            <li>cr : cri(s)</li>
            <li>crch : cri(s) similaire(s) au(x) chant(s) et/ou réciproquement,
                       on ne sait pas décider si c'est un chant ou des cris</li>
            <li>t : tambour (pour les pics, et les grands singes ;-)</li>
            <li>m : martellement(s)</li>
            <li>ailes : bruit d'ailes, à l'envol ou au passage</li>
            <li>inq : inquiétude</li>
            <li>al : alarme</li>
            <li>par : parade</li>
            <li>pours : poursuite</li>
            <li>intim : intime (couple)</li>
            <li>pul : pulli = poussins (au nid si nidicoles)</li>
            <li>jvq : juvénile volant quémandant</li>
            <li>juv : juvénile</li>
            <li>ad : adulte</li>
            <li>cp : couple</li>
            <li>ind, indiv : individu</li>
            <li>typ : typique, représentatif de l'espèce (pour un cri, un chant, un tambour)</li>
            <li>imit : imite</li>
            <li>rep : réponse, se répondent</li>
            <li>pose : posé (= pas en vol !)</li>
            <li>vol : en vol</li>
            <li>age : âgé (pour un poussin = en fin d'élevage)</li>
            <li>elec : électrique</li>
            <li>fele : fêlé</li>
            <li>accel : accéléré</li>
            <li>tous : tous (les types de cris, par ex.)</li>
            <li>prob : probable</li>
            <li>poss : possible</li>
            <li>deter : enregistrement déterioré par un traitement numérique quelconque (Cf. fph, debruit, ou autre),
                        qui empêche ou complique la reconnaissance d'une ou plusieurs espèces, par ex.</li>
            <li>debruit : enregistrement duquel on a soustrait une estimation du "bruit de fond"
                (au mieux générateur de silences étranges, souvent assez déteriorant)</li>
            <li>fph : enregistrement filtré passe-haut (basses fréquences supprimées ou atténuées)</li>
            <li>doppler : effet Doppler (la hauteur des sons émis par un oiseau en rapprochement augmente,
                                         et quand il s'éloigne, elle diminue)</li>
            {{glossaireSpecifique}}
          </ul>
          
        </div>

        <h2 id="licence">Licence / Auteur</h2>
        <div style="margin-left: 10px">
        
          <p>Ce document est publié sous la licence
             <strong><a href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr" target="_blank">
             Creative Commons BY NC SA 4.0</a></strong>
             <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.fr" target="_blank">
               <img height="20" src="{{dossierAttache}}/by-nc-sa.eu.svg" alt="Creative Commons BY NC SA"/>
             </a>
             par <a href="http://jpmeuret.free.fr/" target="_blank">Jean-Philippe Meuret</a>,
             <a href="http://www.lpo-auvergne.org" target="_blank">LPO Auvergne</a> (X heures de travail).</p>
          <p>Vous pouvez (et même devez ;-) le diffuser sans en demander l'autorisation, à qui vous voulez,
             dans sa forme originale ou après modifications, par extraits ou dans son intégralité, pourvu que :</p>
          <ul>
              <li>vous en citiez l'auteur initial (voir ci-dessus) : BY,
              <li>la diffusion n'ait pas un but commercial : NC,
              <li>la diffusion se fasse avec la même licence (CC BY NC SA) : SA.
          </ul>
          <p>Attention cependant aux licences potentiellement plus restrictives des sons liés au présent document,
             issus en totalité de <a href="https://www.xeno-canto.org/" target="_blank">xeno-canto.org</a>
             (Voir ci-dessus le lien associé à chaque enregistrement pour sa source exacte et son auteur).</p>
             
        </div>

        <h2 id="remerciements">Remerciements</h2>
        <div style="margin-left: 10px">
        
          <p>Tous les enregistrements utilisés ici
             proviennent du site <a href="https://www.xeno-canto.org/" target="_blank">xeno-canto.org</a> :
             un très grand merci aux ornithologues qui ont bien voulu partager leurs trouvailles
             et ainsi rendre cette publication possible.</p>
             
          {{remerciements}}
                      
        </div>
        
        <h2 id="licence">Emprunts / Attributions</h2>
        <div style="margin-left: 10px">
        
          <p>Les icônes de petits yeux utilisées ci-dessus sont l'oeuvre de
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
            dans <a href="http://jupyter.org/" target="_blank">Jupyter Notebook</a>,
            le {{genDateTime}}.
        </h6>

      </div>

      <button onclick="scrollToTop()" id="toTopBtn" title="Remonter" alt="Remonter">
        <img width="64" height="64" src="./sources-2018-pics/fa-angle-up.svg"/>
      </button>

    </body>
"""

# Fonction principale de générationde la page.
def buildHtmlPage(titre, sousTitre, description, motsClef,
                  especes, generalites, specificites, glossaire, remerciements,
                  attributions, images, urlDossierSons, dossierSons, dossierAttache,
                  ficTableauSynth=None, prefixeFicCible='chants'):
    
    dfSonsPics = _lireDossierSons(cheminDossier=dossierSons, dEspeces=especes)

    html = jinja2.Template(_KHtmlGroupeEspeces) \
                  .render(titre=titre, sousTitre=sousTitre, description=description, motsClef=motsClef,
                          especes=_arbreEspeces(dfSonsPics, especes, specificites,
                                                urlDossierSons=urlDossierSons),
                          typesManifs=_arbreTypesManifs(dfSonsPics, urlDossierSons=urlDossierSons),
                          generalites=generalites, planGeneralites=_planGeneralites(generalites),
                          glossaireSpecifique=glossaire, remerciements=remerciements,
                          attributions=attributions, ficTableauSynth=ficTableauSynth,
                          dossierAttache=dossierAttache, images=images,
                          stylesCss=_KStylesCss, scriptsJs=_KScriptsJs,
                          genDateTime=dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S'))

    chemFicCible = os.path.join('.', '{}{}.html'.format(prefixeFicCible, '.local' if urlDossierSons else ''))
    with codecs.open(chemFicCible, mode='w', encoding='utf-8-sig') as ficCible:
        ficCible.write(html)
        
    return chemFicCible, dfSonsPics
    
