Comment ça marche ...
---------------------

# I. Fonctionnement résumé de pyaudisam

Dans le premier mode dit de "**pré-analyse**",
* en entrée, on lui fournit
    - l'ensemble des données individualisées (1 données = 1 contact), chaque donnée précisant par exemple : l'espèce, le passage saisonnier concerné (1er ou 2nd), le "sexe et âge" de l'individu contacté (Mâle adulte, autre adulte = femelle ou indéterminé, ou juvénile), et bien sûr, la distance observateur - oiseau ; les données des échantillons choisis en seront extraites automatiquement,
    - la liste des échantillons à pré-analyser : les combinaisons espèce x durée d'inventaire (5mn ou 10mn) x sexe et age (Mâle adulte, autre adulte ou juvénile) x passage,
* le logiciel effectue ensuite une série d'analyses simples (1 analyse = 1 appel à MCDS) pour chacun de ces échantillons, sans aucune troncature, en suivant la stratégie suivante : essayer successivement les modèles disponibles et leurs séries d'ajustement associées, dans un ordre spécifié, et en s'arrêtant à la première analyse qui aboutit sans erreur (la série d'analyses effectuées est donc souvent limitée à une seule) ; si aucune analyse n'aboutit, on considère que l'échantillon est simplement inexploitable en Distance Sampling (mais c'est discutable : s'il a assez de données, avec une bonne troncature, on pourrait peut-être le "récupérer"),
* en sortie, il produit un rapport de synthèse présentant l'essentiel des résultats chiffrés et graphiques de l'analyse faite (avec succès) pour chaque échantillon, sous la forme d'un tableau à 1 ligne par échantillon (voir ci-dessous qq détails) ; dans ce tableau, le principal, c'est l'histogramme des distances (en fait plusieurs superposés, avec différentes largeurs de tranches) et les courbes de modélisations calculées par MCDS, qui permettent assez rapidement d'évaluer pour chaque échantillon :
    - les chances d'obtenir des densités de bonne qualité statistique,
    - éventuellement les troncatures spécifiques qu'il serait raisonnable / intéressant d'essayer lorsque l'on va vraiment lancer les analyses avec de multiples combinaisons de paramètres (troncature à gauche en particulier, par ex. pour les espèces à faible nombre de données à courte distance, comme Alauda arvensis, Prunella modularis, ...).

Dans le second mode dit "d'**analyse**",
* en entrée, on lui fournit
    - l'ensemble des données individualisées (le même que pour le mode "pré-analyse" : les données des échantillons choisis en seront extraites automatiquement),
    - la liste des échantillons à analyser : comme dans le mode "pré-analyse", mais à priori un sous-ensemble, les pré-analyses étant faites pour filtrer les échantillons non exploitables,
    - les combinaisons de paramètres d'analyse à couvrir pour chaque échantillon ; cette fois, c'est différent du mode pré-analyse : il faut spécifier ce que l'on veut faire : modèle x série d'ajustement x troncature en distance à gauche x troncature en distance à droite x nombre de tranches de distance à considérer pour la modélisation (N.B. pour ces 3 derniers paramètres, un mode "automatique" permet de ne pas avoir à spécifier les valeurs, et de laisser le logiciel les chercher lui-même par optimisation numérique de la qualité statistique des résultats de l'analyse DS), 
* le logiciel enchaîne ensuite automatiquement les grandes étapes suivantes :
    - pour les combinaisons de paramètres "à troncatures automatiques", détermination automatique des distances de troncatures à gauche et à droite par une technique d'optimisation (lancement de nombreuses analyses en faisant varier les dites troncatures dans des limites paramétrées par le nombre de données ignorées, considérées comme des "outliers", ... et en ne retenant que le meilleur jeu de troncatures (à gauche et/ou à droite) selon un indicateur / critère de "qualité combinée" à partir des indicateurs produits par MCDS et des paramètres d'analyses (voir ci-dessous pour un 1er survol, et chapitre II pour tous les détails) ;
    - exécution des analyses individuelles correspondant à chacune des combinaisons de paramètres (à troncatures maintenant toutes fixées) pour chaque échantillon : comme lors d'analyses manuelles via le logiciel Distance 7, c'est son "moteur" de calcul MCDS V6 qui est utilisé, en mode "Conventional Distance Sampling" (pas de co-variable) ; les résultats ainsi produits alimentent ainsi une table dont chaque ligne correspond à une analyse, c'est à dire à l'une des combinaisons de paramètres pour chaque échantillon ; ces résultats sont ceux produits directement par MCDS V6, et complétés par l'indicateur de "qualité combinée" spécifique à pyaudisam
    - tri et filtrage automatique des résultats selon leur "qualité combinée" : pour chaque échantillon, suivant une suite d'étapes proche de ce que ferait "manuellement" un analyste humain cherchant le "meilleur résultat" d'analyse parmi ceux correspondant à de multiples combinaisons de paramètres,
        + ne conserver que les analyses à résultats non "identiques", à un epsilon près (pour les critères statistiques bruts en sortie de MCDS, les densités et probabilité de détection calculées, ainsi que leurs intervalles de confiance)
        + ne conserver que les analyses sans erreur rapportée par MCDS (analyses à "warnings" conservées),
        + pour les analyses à troncatures strictement identiques, ne conserver que les N ayant les meilleurs AIC (N paramétrable),
        + pour les analyses à troncatures proches, ne conserver que celle (1 seule) ayant le meilleur Khi2, le meilleur KS à Khi2 identique, le meilleur DCv à Khi2 et KS identiques (tests du Khi2 et de KS =  Kolmogorov-Smirnov, DCv = variabilité à 95% de la densité estimée),
        + ne conserver que les analyses "tronquées en distance" à moins de X % (i.e. pour lesquelles les troncatures en distance à gauche et/ou à droite n'ont pas éliminé plus de X % des oiseaux contactés au total) (X paramétrable),
        + ne conserver que les analyses restantes ayant les meilleurs critères de "qualité combinée", pourvu qu'ils dépassent la valeur minimale Y (paramétrable),
        + ne conserver enfin que les (au plus) P (paramétrable) meilleures analyses, pour limiter la taille du rapport, mais en conservant tout de même suffisamment de matière pour manuellement vérifier la sélection automatique et la corriger si besoin (Cf. "dernière étape manuelle" ci -dessous) ;
* en sortie, il produit un rapport (Cf. chapitre IV) présentant notamment une synthèse des résultats chiffrés et graphiques de chaque analyse conservée lors du filtrage ci-dessus, sous la forme d'un tableau de synthèse (trié par échantillon et dans l'ordre décroissant du critère de "qualité combinée") ;
* la dernière étape, manuelle celle-ci, consiste à examiner, pour chaque échantillon, les résultats des quelques "meilleures" analyses retenues, et à choisir laquelle conserver finalement : le filtrage automatique décrit ci-dessus n'étant pas parfait, il convient de vérifier systématiquement que la "meilleure" analyse automatiquement proposée est effectivement "meilleure" que les suivantes du classement ; pour la très grande majorité des échantillons, c'est bien le cas, et de toute manière, les écarts de critères qualité et de résultats sont faibles dans ce classement ; mais parfois, l'analyste humain doit préférer une autre "meilleure" analyse, pour corriger les imperfections et les manques du logiciel, en lui-même, mais aussi vis à vis des biais et manques du côté des données de terrain, grâce à sa connaissance du secteur géographique inventorié, des milieux naturels présents et des espèces ciblées, lui donnent des arguments pour parfois corriger l'automatisme ; les cas suivants semblent les plus fréquents :
    - l'histogramme des distances n'est pas régulier, présente des "trous" et / ou des "bosses", et le logiciel a sélectionné des analyses où il y a manifestement sur-modélisation, c'est-à-dire une modélisation qui cherche à s'approcher au mieux de l'enveloppe de l'histogramme, alors que ces "trous" et / ou "bosses" devraient être ignorées,
    - préférence naturelle pour une modélisation en demi-normale,
    - ...

Soulignons l'importance, dans le processus de tri et filtrage automatique décrit ci-dessus, de l'indicateur / critère "qualité combinée" calculé à partir des indicateurs produits par MCDS et des paramètres d'analyses (Cf. chapitre II pour tous les détails) : construit dans l'objectif idéal d'automatiquement orienter autant que possible la sélection des meilleures analyses, il simule en quelque sorte et de manière très simple les préférences d'un analyste humain expérimenté (selon l'état actuel de nos connaissances pratiques du Distance Sampling) :
* le modèle Half-Normal est préférable, suivi de peu par le modèle Uniform ; le modèle Hazard-Rate est moins recherché (pour s'éloigner souvent des réalités de terrain ornithologique : "épaule" longue à courte distance),
* la sur-modélisation ("over-fitting") des histogrammes de distance est à éviter autant que possible, donc les analyses avec un minimum de paramètres d'ajustement du modèle sont préférées,
* les troncatures éliminant peu ou pas de données sont préférées,
* les analyses à faible variabilité de l'estimation de densité (intervalle de confiance à 95%) sont préférées.

Le chapitre suivant donne les détails de la formule de calcul de cet indicateur


# II. Critère de qualité "combinée"

Ce critère, calculé par pyaudisam entre 0 (mauvais) et 1 (excellent), combine après normalisation si nécessaire, certains paramètres d'analyse, ainsi que les critères et résultats bruts produits par MCDS:

| Critère / résultat                                           | MCDS Méthode de normalisation (= passage dans [0, 1]) |
|--------------------------------------------------------------|------------------------------------------------------ |
| Test du Khi2 (Khi2)                                          | déjà dans [0, 1]                                      |
| Test de Kolmogorov-Smirnov (KS)                              | déjà dans [0, 1]                                      |
| Test de Cramer-von-Mises à pondération uniforme (CvM-Uw)     | déjà dans [0, 1]                                      |
| Tests de Cramer-von-Mises à pondération cosinusoïde (CvM-Cw) | déjà dans [0, 1]                                      |
| Taux d'individus (oiseaux) conservé après troncatures        | simple division par 100 |
| Modèle utilisé Half-Normal, Uniform ou Hazard-Rate <br>(le 1er considéré comme le plus conforme à la réalité ornithologique de terrain, <br>le 2ème un peu moins, le 3ème encore moins)                                                         | Half-Normal => 1,0 <br> Uniform => 0,9 <br> Hazard-Rate => 0,6 |
| Variabilité à 95% de la densité d'oiseaux estimée (DCv)      | exp(-63 * DCv2.8)                                     |
| Nombre de paramètres de la série d'ajustement du modèle (polynomiale ou cosinusoïde) | exp(-0.17 * NbParams2)        |

La valeur finale est le produit de ces 8 valeurs normalisées, lui-même élevé à la puissance 1/8 (conservation de dimensionnalité).


# III. Extrait d'un rapport de synthèse des pré-analyses de pyaudisam

Les rapports de synthèse des pré-analyses produits par pyaudisam au format HTML présentent, pour chaque échantillon "pré-analysé" (espèce, passage(s) : a, b ou a+b, durée d'inventaire : 5mn ou 10mn, type d'effectif : m, ou m+a) :
* un tableau à 3 colonnes permettant d'identifier l'échantillon, la pré-analyse, les paramètres de la première modélisation DS ayant fonctionné pour cet échantillon, quelques chiffres-clés décrivant l'échantillon (nombre d'individus considéré, distance max. de contact, effort DS), et enfin les résultats principaux de la pré-analyse (densité estimée et intervalle de confiance à 95%, nombre estimé et intervalle de confiance à 95%, EDR,) ainsi que quelques uns des indicateurs habituels de qualité statistique de la modélisation DS (AIC, Khi2, KS, CoefVar Densité) ; les informations affichées dans ces 3 colonnes sont paramétrables,
* 3 graphiques permettant de juger rapidement "à l'oeil" de l'exploitabilité de l'échantillon (à gauche : histogrammes des distances de contact, avec 3 largeurs différentes de tranches de distances), et de la qualité de cette première modélisation DS ayant fonctionné (au milieu : densité de probabilité de détection modélisée, superposée à l'histogramme des données de terrain, le tout en fonction de la distance ; à droite : intégrale correspondante, modélisée, superposée à l'histogramme des données de terrain, le tout en fonction de la distance).

Les unités sont précisées en entête du rapport (non visibles dans l'extrait ci-dessous : ici, distances en m, et surfaces en km2).

Pour l'exemple, voici un [extrait d'un tel rapport](./preanlys/ACDC2019-Nat-preanalyses-report.html) présentant 2 échantillons (2 espèces) du jeu de données Naturalist.

En cliquant dans la colonne de gauche (sur le numéro de ligne du tableau), on accède à une page de détails de la pré-analyse concernée, en tout points identique à ce que le logiciel Distance 7 produirait (il s'agit d'ailleurs de l'intégralité des informations produites par MCDS, comme dans Distance 7).

En bas de la page principale du rapport HTML, on trouve également des tableaux de traçabilité listant les divers paramètres et conditions d'exécution des analyses.

Un rapport au format Excel peut également être généré : il ne contient aucun graphique, mais présente sous forme tabulaire (Cf. [exemple](./preanlys/ACDC2019-Nat-preanalyses-report.xlsx)):
* d'une part le détail des résultats de chaque pré-analyse effectuée, en totalité de ce que MCDS produit (hormis les graphiques et rapports textuels),
* d'autre part une synthèse (colonnes paramétrables) de ces résultats, sous la forme d'un simple extrait (paramétrable) des colonnes du tableau de détails.


# IV. Extrait d'un rapport d'analyses de pyaudisam

Les rapports de synthèse des analyses produits par pyaudisam au format HTML présentent, pour chaque échantillon analysé (espèce, passage(s) : a, b ou a+b, durée d'inventaire : 5mn ou 10mn, type d'effectif : m, ou m+a), les "meilleures" analyses à l'aune de l'indicateur de qualité "combinée" (Cf. chapitre II).

Ils ont une forme assez similaire aux rapports de pré-analyse (voir ci-dessus), si ce n'est que :
* on n'a plus 1 ligne par échantillon pré-analysé (toutes espèces contactées, tous passages, ...), mais N lignes par échantillon sélectionné (les 30 espèces, l'unique combinaison de passages b ou a+b, l'unique combinaison de type d'effectif m ou m+a, les 2 combinaisons de durées 5mn et 10mn), correspondant au N meilleures analyses (N étant un des paramètres du schémas de filtrage et tri choisi pour le rapport),
* par ligne, le tableau à 3 colonnes donne plus de détails : en particulier les troncatures utilisées, et l'indicateur de qualité "combinée" ("Qual Equi 3" ici, mais c'est aussi paramétrable), critère de classement des analyses par échantillon (la meilleure en premier, en haut) ; les informations à afficher dans ces 3 colonnes sont paramétrables,
* des 3 graphiques permettant de juger rapidement "à l'oeil" de la qualité des résultats de l'analyse, le premier, à gauche a été remplacé logiquement par le diagramme quantile - quantile de comparaison modélisation - données de terrain ; pas de changement en revanche au milieu et à droite : respectivement : densité de probabilité de détection modélisée, superposée à l'histogramme des données de terrain, le tout en fonction de la distance, et à droite : intégrale correspondante, modélisée, superposée à l'histogramme des données de terrain, le tout en fonction de la distance).

De même, les unités sont précisées en entête du rapport (non visibles dans l'extrait ci-dessous : ici, distances en m, et surfaces en km2).

En cliquant dans la colonne de gauche (sur le numéro de ligne du tableau), on accède à une page de détails de la pré-analyse concernée, en tout points identique à ce que le logiciel Distance 7 produirait (il s'agit d'ailleurs de l'intégralité des informations produites par MCDS, comme dans Distance 7).

Pour l'exemple, voici un [extrait d'un tel rapport](./optanlys/ACDC2019-Nat-optanalyses-report.ExAicMQua-r925m8q3d12.html) présentant les 3 meilleures analyses de l'échantillon Sylvia atricapilla Mâles Naturalist 10mn.

Sous ce tableau de synthèse, 2 autres tableaux listant les analyses dans le même ordre présentent respectivement 
* une sélection des colonnes de paramètres, indicateurs statistiques et résultats de chaque analyse (sélection enrichie comparée à celle du tableau de synthèse ci-dessus, soit environ 45 colonnes),
* la totalité de colonnes paramètres, indicateurs statistiques et résultats de chaque analyse, telles que produites par MCDS (environ 120 colonnes), en cas de besoin pointu.

On peut également produire un rapport HTML "complet" (alias "full"), qui présente sous la même forme exactement les résultats de **TOUTES** les analyses effectuées, pas seulement les N meilleures de chaque échantillon (attention : selon le nombre de combinaisons de paramètres d'analyse testé, ce rapport peut être conséquent !).

En bas de la page principale du rapport HTML, on trouve également des tableaux de traçabilité listant les divers paramètres et conditions d'exécution des analyses.

Un rapport au format Excel peut également être généré : il ne contient aucun graphique, mais présente sous forme tabulaire (Cf. [exemple](./optanlys/ACDC2019-Nat-optanalyses-report.xlsx)):
* pour chacun des schémas de filtrage et tri mis en place (pas seulement 1 seul comme dans le rapport HTML), une feuille donnant, groupés par échantillon et triés dans l'ordre décroissant du critère de qualité combinée choisi, les résultats principaux (colonnes paramétrables) des N meilleures analyses (Qual Bal 3 par exemple)
* le détail des résultats de chaque analyse effectuée (pas seulement les N meilleures par échantillon), en totalité de ce que MCDS produit (hormis les graphiques et rapports textuels),
* une synthèse des résultats (colonnes paramétrables) de chaque pré-analyse effectuée, sous la forme d'un simple extrait (paramétrable) des colonnes du tableau de détails,
