# MallowMeter - Cadrage et Analyse Fonctionnelle

## Objectif

MallowMeter est un outil de mesure de degré du cuisson des marshmallows destiné aux gourmands de tous âges.

## Etapes importantes

### Base d'images

- [ ] ~~Trouver en ligne une base d'images couleurs de marshmallows non cuits et cuits à différents degrés~~
- [X] Créer une base d'images couleurs de marshmallows non cuits et cuits à différents degrés.
    - [X] Prendre les photos
    - [X] Rogner les images pour isoler les marshmallows
    - [X] Redimensionner les images pour avoir une taille standard
    - [X] Et. les images pour entraîner un modèle de classification
        - Pas cuit
        - Un peu cuit
        - Bien cuit
        - Trop cuit

### Pré-traitement :

- Détection des contours (Sobel, Canny...)

### Descripteurs

- Couleur
*    - Histogrammes R, G, B
*    - Histogramme Swain & Ballard (R, G, B)
*    - Histogrammes H, S, V
*    - Histogramme Swain & Ballard (H, S, V)
*    - Moyenne canal R, G, B
*    - Ecart-type canal R, G, B
*    - Moyenne 3D (R, G, B)
*    - Ecart-type 3D (R, G, B)
    - Moyenne canal L, a, b
    - Ecart-type canal L, a, b
    - Moyenne 3D (L, a, b)
    - Ecart-type 3D (L, a, b)
    - Rapport de pixels clairs VS foncés
    - Distance en LAB par rapport à un échantillon de référence
- Texture
    - Matrice de co-occurrence
        - Contraste
        - Energie
        - Homogénéité
    - Entropie de la texture (Entropie++ -> Grillé++ ?)
    - Histogramme de gradients orientés (HOG)
    - Features de Tamura
        - Contraste
        - Directionnalité
        - Rugosité
        - Régularité

### Classifieurs

Du plus simple au plus complexe:
- KNN (K plus proches voisins)
- GMM (Modèle Gaussien)
- CNN (Convolutional Neural Network)