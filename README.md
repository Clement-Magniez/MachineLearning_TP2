# Sujet #

Dans ce TD, on essayera de prédire la température 7 jours à l'avance à l'aide d'un CNN et d'un RNN, en comparant leurs performances respectives.

On note qu'il y a deux façons de procéder. On peut, pour chaque modèle, l'entrainer à prédire 1 jour à l'avance et faire une prédiction à 7 jours avec 7 prédictions à un jour, réutilisant les précédantes. On peut aussi entrainer le modèle à prédire directement à 7 jours, ce qui est plus stable et plus économe en évaluations (surtout pour le CNN). Ce sera donc l'approche prise dans ce travail.

# Format des données #

Comme le formatage des données, dépend des hyperparamètres pour le CNN, c'est la classe CNN (et RNN pour l'homogénéité du code) qui implémente cette fonction, dans le fichier CNN.py. Pour les deux modèles, les données sont normalisées. On garde les valeurs de variance et de moyenne pour retransformer les prédictions en températures.

Pour le RNN, les données d'entrée pour prédire un certain jour J sont les températures de l'année, du premier janvier à J-7. Pour le CNN, on fixe un paramètre "time frame" qui détermine la largeur de la fenêtre temporelle prise en compte. Pour prédire J, on a donc en entrée les tempéatures de J-7-time_frame à J-7. Quelques experiences sur time_frame ont montré qu'il avait peu d'influence, il vaudra donc 8 dans le reste du TP.

# Comparaison des dynamiques de convergence #

Après avoir choisi à la main un learning rate et un nomre d'epochs cohérents, on fait varier différents paramètres des réseaux, en utilisant comme critère la loss sur le dataset de test (en bleu). On utilise la même loss (MSE) pour les deux modèles, de sorte à ce que leurs loss respectives soient comparables.

Nombre de neurones du MLP du CNN, et kernel size:
# im cnn
On ne constate pas de différences significatives. La convergence est stable et régulière d'un run à l'autre, la loss converge vers 2.75 dans touts les cas. Les performances sont légèrement meilleures sur le dataset d'entrainement, rien de notable.

Nombre de neurones de l'unique couche du RNN, sur deux runs différents:
# im rnn diverge et im rnn converge
Quand on utilise 30 neurones sur la couche cachée, selon l'initialistion le réseau peut complètement diverger. Les résultats numériques sont assez variables mais en général, on observe un overfitting conséquent pour 20 et 30 neurones, mais des meilleures performances sur le dataset de test dans le "creux du V". Sur cette image, 2.4 pour 20hid et 30hid contre 2.7 pour 10hid, les performances sont donc globalement meilleures que le CNN.


Avec deux couches cachées, de même taille:
# im rnn 2 couches
Dans ce cas, augmenter le nombre de paramètres n'est pas bénéfiques aux performances: l'entrainement est trop instable, peut-etre à cause du bruit important dans les données. On obtient pour 2x10 neurones un résultat comparable à 1x20 et 1x30 en terme de meilleure loss sur le dataset de test, mais un meilleur comportement général. Pour 20 et 30, les résultats ne sont pas probants.

# Résultats finaux #

RNN 2x10 top tier.

