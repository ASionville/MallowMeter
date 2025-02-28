import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from load_database import load_dataset
from utils import load_or_compute_features

def verif_bdd():
    dataset_path = "dataset/"
    data, labels, ids = load_dataset(dataset_path)
    print("Données chargées avec succès")
    
    # Utiliser load_or_compute_features pour éviter de recalculer si le fichier existe
    # et récupérer les labels depuis le fichier pkl
    features, labels_from_pkl = load_or_compute_features(data, labels, force_recompute=False)
    print("Caractéristiques extraites ou chargées avec succès")
    
    # Utiliser les labels du fichier pkl au lieu des labels chargés via load_dataset
    labels = labels_from_pkl
    print(f"Utilisation des labels depuis le fichier de features: {len(labels)} échantillons")

    # Liste mise à jour des caractéristiques avec leurs indices et titres corrects
    feature_params = {
        # 0: ('l_range', 'Étendue L'),
        # 1: ('b_range', 'Étendue b'),
        # 2: ('l_min', 'Minimum L'),
        # 3: ('l_max', 'Maximum L'),
        # 4: ('b_min', 'Minimum b'),
        # 5: ('b_max', 'Maximum b'),
        # 6: ('l_skewness', 'Skweness L'),
        0: ('b_skewness', 'Skewness b'),
        # 8: ('l_mean', 'Moyenne L'),
        1: ('l_std', 'Écart-type L'),
        2: ('b_mean', 'Moyenne b'),
        3: ('b_std', 'Écart-type b'),
        # 12: ('l_contrast_h', 'Contraste L'),
        4: ('b_contrast', 'Contraste b'),
        5: ('l_homogeneity', 'Homogénéité L'),
        6: ('b_homogeneity', 'Homogénéité b'),
        7: ('l_entropy', 'Entropie L'),
        8: ('b_entropy', 'Entropie b'),
        9: ('l_energy', 'Énergie L'),
        10: ('b_energy', 'Énergie b'),
    }

    # On peut limiter le nombre de graphiques pour éviter de générer trop d'images
    # Par exemple, afficher seulement les caractéristiques principales
    selected_features = range(11)
    
    for index in selected_features:
        feature_name, title = feature_params[index]
        plot_feature_boxplot(features, labels, index, feature_name, title)

def plot_feature_boxplot(features, labels, feature_index, feature_name, title):
    df = pd.DataFrame({
        f'{title}': features[:, feature_index].cpu().numpy(),
        'Classe': labels.cpu().numpy()
    })
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    palette = {0: '#f0ead2', 1: '#f4d35e', 2: '#f77f00', 3: '#d62828'}
    sns.boxplot(x='Classe', y=f'{title}', data=df, hue='Classe', palette=palette, legend=False)
    plt.title(f'Boites à moustaches : {title}')
    plt.savefig(f'evaluation/boites_a_moustaches/{feature_name}.png')
    plt.close()

if __name__ == "__main__":
    verif_bdd()
