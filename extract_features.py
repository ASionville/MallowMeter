import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Calculer la matrice de co-occurrence pour un canal donné
def co_occurrence_matrix(channel, distance=1, direction="horizontal"): # Direction : "horizontal" ou "vertical"
    # Boucle sur chaque pixel de l'image en fonction de la distance et de l'angle
    comat = torch.zeros(256, 256, device=channel.device)
    if direction == "horizontal":
        for i in range(channel.size(0)):
            for j in range(channel.size(1) - distance):
                comat[channel[i, j].long(), channel[i, j + distance].long()] += 1
    elif direction == "vertical":
        for i in range(channel.size(0) - distance):
            for j in range(channel.size(1)):
                comat[channel[i, j].long(), channel[i + distance, j].long()] += 1
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")
    return comat / comat.sum()  

def rgb_to_lab(image):
    # Convertir une image RGB en Lab
    r, g, b = image[0], image[1], image[2]
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0

    # Transformation linéaire
    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b

    # Transformation non-linéaire
    x = torch.where(x > 0.008856, x ** (1/3), 7.787 * x + 16/116)
    y = torch.where(y > 0.008856, y ** (1/3), 7.787 * y + 16/116)
    z = torch.where(z > 0.008856, z ** (1/3), 7.787 * z + 16/116)

    l = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    return torch.stack([l, a, b])

def extract_features(images):
    features = []
    for image in tqdm(images, desc="Extraction des features"):
        lab_image = rgb_to_lab(image)
        l, b = lab_image[0], lab_image[2]

        # Étendue, min/max, moments (skewness)
        # l_range = l.max() - l.min()
        # b_range = b.max() - b.min()
        
        # l_min, l_max = l.min(), l.max()
        # b_min, b_max = b.min(), b.max()

        # l_skewness = ((l - l.mean()) ** 3).mean() / (l.std() ** 3)
        b_skewness = ((b - b.mean()) ** 3).mean() / (b.std() ** 3)

        # Moyenne et écart-type des composantes L et b
        l_mean, l_std = l.mean(), l.std()
        b_mean, b_std = b.mean(), b.std()

        # Matrices de co-occurrence
        l_comat_1 = co_occurrence_matrix(l, distance=1, direction="horizontal")
        b_comat_1 = co_occurrence_matrix(b, distance=1, direction="horizontal")

        # Contraste, homogénéité, entropie, énergie (Tamura)
        # l_contrast = (l_comat_1 * (torch.arange(l_comat_1.size(0)) ** 2).float()).sum()
        b_contrast = (b_comat_1 * (torch.arange(b_comat_1.size(0)) ** 2).float()).sum()

        lomogeneity = (l_comat_1 / (1 + (torch.arange(l_comat_1.size(0)) ** 2).float())).sum()
        bomogeneity = (b_comat_1 / (1 + (torch.arange(b_comat_1.size(0)) ** 2).float())).sum()

        l_entropy = -(l_comat_1 * torch.log(l_comat_1 + 1e-10)).sum()
        b_entropy = -(b_comat_1 * torch.log(b_comat_1 + 1e-10)).sum()

        l_energy = (l_comat_1 ** 2).sum()
        b_energy = (b_comat_1 ** 2).sum()

        features.append(torch.tensor([
            # l_range,
            # b_range,
            # l_min,
            # l_max,
            # b_min,
            # b_max,
            # l_skewness,
            b_skewness,
            # l_mean,
            l_std,
            b_mean,
            b_std,
            # l_contrast,
            b_contrast,
            lomogeneity,
            bomogeneity,
            l_entropy,
            b_entropy,
            l_energy,
            b_energy,
        ], device=image.device))

    return torch.stack(features)

def faire_matrices_de_distances():
    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from load_database import load_dataset
    from utils import load_or_compute_features
    
    # Créer le dossier pour stocker les matrices de distances s'il n'existe pas
    os.makedirs("evaluation/matrices_de_distances", exist_ok=True)
    
    # Charger les données
    dataset_path = "dataset/"
    data, labels, ids = load_dataset(dataset_path)
    print("Données chargées avec succès")
    
    # Charger ou calculer les caractéristiques
    features, labels = load_or_compute_features(data, labels, force_recompute=False)

    print("Caractéristiques extraites ou chargées avec succès")
    
    # Liste des caractéristiques avec leurs indices et titres
    feature_params = {
        0: ('b_skewness', 'Skewness b'),
        1: ('l_std', 'Écart-type L'),
        2: ('b_mean', 'Moyenne b'),
        3: ('b_std', 'Écart-type b'),
        4: ('b_contrast', 'Contraste b'),
        5: ('lomogeneity', 'Homogénéité L'),
        6: ('bomogeneity', 'Homogénéité b'),
        7: ('l_entropy', 'Entropie L'),
        8: ('b_entropy', 'Entropie b'),
        9: ('l_energy', 'Énergie L'),
        10: ('b_energy', 'Énergie b'),
    }
    
    # Convertir les features en numpy pour faciliter les calculs
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Pour chaque caractéristique, calculer la matrice de distance
    for idx, (feature_name, title) in feature_params.items():
        print(f"Calcul de la matrice de distance pour {title}...")
        
        # Extraire les valeurs de cette caractéristique pour toutes les images
        feature_values = features_np[:, idx].reshape(-1, 1)
        
        # Calculer la matrice de distance L2
        n_samples = feature_values.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                # Distance L2 (euclidienne) entre les valeurs de caractéristique
                distance_matrix[i, j] = np.sqrt(np.sum((feature_values[i] - feature_values[j]) ** 2))
        
        # Visualiser la matrice de distance
        plt.figure(figsize=(10, 8))
        
        # Tri des échantillons par classe pour mieux visualiser les blocs
        sort_idx = np.argsort(labels_np)
        sorted_distance_matrix = distance_matrix[sort_idx][:, sort_idx]
        sorted_labels = labels_np[sort_idx]
        
        # Créer la heatmap
        ax = sns.heatmap(sorted_distance_matrix, cmap='viridis')
        
        # Ajouter des lignes pour délimiter les classes
        class_boundaries = np.where(np.diff(sorted_labels) != 0)[0]
        for boundary in class_boundaries:
            plt.axhline(y=boundary + 0.5, color='red', linestyle='-', linewidth=1)
            plt.axvline(x=boundary + 0.5, color='red', linestyle='-', linewidth=1)
        
        plt.title(f'Matrice de distance L2 - {title}')
        plt.tight_layout()
        
        # Sauvegarder l'image
        plt.savefig(f'evaluation/matrices_de_distances/{feature_name}.png')
        plt.close()
    
    print("Toutes les matrices de distance ont été générées et enregistrées.")

if __name__ == "__main__":
    faire_matrices_de_distances()