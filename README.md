# MallowMeter

<br>
<div align="center">
    <img src="https://raw.githubusercontent.com/ASionville/MallowMeter/refs/heads/main/images_readme/mosaique.jpg" width="40%">
</div>

## Contexte et objectifs

Avez-vous déjà fixé un marshmallow en vous demandant : "Mais QUEL est son degré exact de cuisson?" Nous aussi ! C'est pourquoi nous avons consacré des années de recherche intensive (bon, peut-être quelques semaines) à résoudre ce problème crucial qui empêchait l'humanité d'avancer.

MallowMeter est né d'une question existentielle : pourquoi certains marshmallows sont parfaitement dorés tandis que d'autres finissent carbonisés comme s'ils avaient tenté de voyager trop près du soleil ? Notre équipe de "mallowlogues" a donc développé un système révolutionnaire capable de classifier automatiquement les marshmallows en 4 catégories cruciales (de 0 "blanc comme neige" à 3 "transformé en charbon").

En utilisant des algorithmes de vision par ordinateur plus sophistiqués que nécessaire, nous analysons méticuleusement la couleur et la texture de ces délicieuses friandises. Pourquoi ? Parce que la science des marshmallows est une affaire sérieusement pas sérieuse !

Notre mission : sauver les soirées camping d'une catastrophe marshmallowesque et offrir enfin à l'humanité l'outil qu'elle ignorait désespérément attendre.

## Installation

### Prérequis
- Python 3.7 ou supérieur
- Pour les librairies, voir [`requirements.txt`](requirements.txt)

### Configuration

1. Clonez ce dépôt :
```bash
git clone https://github.com/ASionville/MallowMeter.git
cd MallowMeter
```

2. Installez les dépendances :
```bash
pip install -r [`requirements.txt`](requirements.txt)
```

## Utilisation

### Vérification de la base de données
```bash
python verif_bdd.py
```
Cette commande analysera la base de données et générera des visualisations des caractéristiques extraites.

### Exécution des modèles
Le programme principal supporte désormais les arguments en ligne de commande, ce qui permet de configurer facilement les tests sans modifier le code source.

#### Options générales
```bash
python main.py --model [knn|cnn] --seed INT --ratio_test FLOAT --log
```

- `--model`: Spécifie le modèle à utiliser (knn ou cnn)
- `--seed`: Graine pour la reproductibilité des résultats (défaut: 42)
- `--ratio_test`: Proportion des données pour l'ensemble de test (défaut: 0.2)
- `--log`: Active les logs détaillés pendant l'exécution

#### Exécution du modèle KNN
```bash
python main.py --model knn --seed 42 --ratio_test 0.2 --k 5 --force_recompute
```

Options spécifiques au KNN:
- `--k`: Nombre de voisins à considérer (défaut: 5)
- `--force_recompute`: Force le recalcul des caractéristiques même si elles existent déjà

#### Exécution du modèle CNN
```bash
python main.py --model cnn --seed 42 --ratio_test 0.2 --enrichissement 10 --num_epochs 50 --patience 5 --min_delta 0.001 --batch_size 16 --learning_rate 0.001
```

Options spécifiques au CNN:
- `--enrichissement`: Niveau d'enrichissement des données (défaut: 10)
- `--num_epochs`: Nombre maximum d'époques d'entraînement (défaut: 50)
- `--patience`: Nombre d'époques sans amélioration avant arrêt (défaut: 5)
- `--min_delta`: Amélioration minimale considérée significative (défaut: 0.001)
- `--batch_size`: Taille des batchs (défaut: 16)
- `--learning_rate`: Taux d'apprentissage (défaut: 0.001)

### Exemples d'utilisation

Exécuter KNN avec 7 voisins sur 30% de données de test:
```bash
python main.py --model knn --k 7 --ratio_test 0.3
```

Exécuter CNN avec un fort enrichissement et plus de patience:
```bash
python main.py --model cnn --enrichissement 30 --patience 10 --num_epochs 100
```

## Modèles utilisés

### KNN (K-Nearest Neighbors)
Le modèle KNN utilise des caractéristiques extraites des images en Lab :
1. **Skewness b** : Asymétrie de la composante b
2. **Écart-type L** : Variabilité de la composante L
3. **Moyenne b** : Moyenne de la composante b
4. **Écart-type b** : Variabilité de la composante b
5. **Contraste b** : Contraste de la composante b
6. **Homogénéité L** : Homogénéité de la composante L
6. **Homogénéité b** : Homogénéité de la composante b
7. **Entropie L** : Mesure de désordre de la composante L
8. **Entropie b** : Mesure de désordre de la composante b
9. **Énergie L** : Uniformité de la composante L
10. **Énergie b** : Uniformité de la composante b

### CNN (Convolutional Neural Network)
Architecture :
- 4 couches de convolution avec batch normalization et max pooling
- 3 couches fully connected avec dropout pour éviter le surapprentissage
- Entrée : images converties en espace colorimétrique Lab
- Early stopping pour optimiser l'apprentissage

## Résultats

Les résultats sont générés dans les dossiers :
- `evaluation/boites_a_moustaches/` : Visualisations des caractéristiques extraites
- `evaluation/knn/` : Résultats du KNN
- `evaluation/cnn/` : Résultats du CNN
- `evaluation/results_knn.csv` et `evaluation/results_cnn.csv` : Résultats quantitatifs

Les performances varient selon les paramètres utilisés :
- KNN : Les meilleures performances sont généralement obtenues avec k=5
- CNN : L'enrichissement des données améliore considérablement les performances, surtout avec un enrichissement de niveau 20 ou 30

## Crédits

Ce projet a été réalisé par :
- [Maxime JOURNOUD](https://github.com/DCucube)
- [Lucas LESCURE](https://github.com/FluffLescure)
- [Aubin SIONVILLE](https://github.com/ASionville)
- [Ruben VERCHERE](https://github.com/RubenV)


Nous tenons à remercier l'équipe pédagogique de TSE pour leur accompagnement

---
© 2025 MallowMeter - Télécom Saint-Étienne
