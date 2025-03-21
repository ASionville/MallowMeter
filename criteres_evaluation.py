import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

def calculate_accuracy(predictions, targets):
    """
    Calcule l'exactitude (accuracy): (TP+TN)/Total
    
    Args:
        predictions: Les prédictions du modèle
        targets: Les valeurs cibles réelles
        
    Returns:
        float: L'exactitude entre 0 et 1
    """
    if isinstance(predictions, torch.Tensor):
        return (predictions == targets).float().mean().item()
    else:
        return np.mean(predictions == targets)

def calculate_precision_recall_jaccard(conf_matrix):
    """
    Calcule la précision, le rappel et l'indice de Jaccard pour chaque classe
    
    Args:
        conf_matrix: La matrice de confusion
        
    Returns:
        tuple: (précisions, rappels, indices_jaccard)
    """
    num_classes = conf_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    jaccard = np.zeros(num_classes)
    
    for i in range(num_classes):
        # True Positives: conf_matrix[i,i]
        tp = conf_matrix[i, i]
        
        # False Positives: sum(conf_matrix[:,i]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp
        
        # False Negatives: sum(conf_matrix[i,:]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        
        # Precision: TP/(TP+FP)
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall: TP/(TP+FN)
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Jaccard Index: TP/(TP+FP+FN)
        jaccard[i] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
    return precision, recall, jaccard

def plot_confusion_matrix(conf_matrix, class_names=None, save_path=None, normalized=False, title='Matrice de confusion'):
    """
    Trace et sauvegarde une matrice de confusion
    
    Args:
        conf_matrix: La matrice de confusion
        class_names: Les noms des classes (optionnel)
        save_path: Chemin pour sauvegarder l'image (optionnel)
        normalized: Si True, les valeurs sont affichées en proportions (format pourcentage)
        title: Titre de la figure
    """
    plt.figure(figsize=(10, 8))
    
    # Format des annotations : entiers ou pourcentage selon normalized
    fmt = '.2%' if normalized else 'd'
    
    sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.title(title)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def get_normalized_confusion_matrix(targets, predictions, num_classes):
    """
    Calcule une matrice de confusion normalisée (proportions par ligne)
    
    Args:
        targets: Les valeurs cibles réelles
        predictions: Les prédictions du modèle
        num_classes: Nombre de classes
        
    Returns:
        array: Matrice de confusion normalisée
    """
    # Calculer la matrice de confusion brute
    conf_matrix = confusion_matrix(targets, predictions, labels=range(num_classes))
    
    # Normaliser par ligne (proportions par classe réelle)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    # Éviter la division par zéro
    row_sums = np.where(row_sums == 0, 1, row_sums)
    normalized_conf_matrix = conf_matrix / row_sums
    
    return normalized_conf_matrix

def evaluate_model(predictions, targets, class_names=None, save_dir=None, model_name="modele"):
    """
    Évalue complètement le modèle et génère tous les critères d'évaluation
    
    Args:
        predictions: Les prédictions du modèle
        targets: Les valeurs cibles réelles
        class_names: Les noms des classes (optionnel)
        save_dir: Répertoire pour sauvegarder les visualisations (optionnel)
        model_name: Nom du modèle pour la sauvegarde (optionnel)
        
    Returns:
        dict: Un dictionnaire contenant tous les critères d'évaluation
    """
    # Conversion en numpy si nécessaire
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Nombre de classes uniques
    num_classes = len(np.unique(np.concatenate((targets, predictions))))
    
    # Calcul de l'exactitude
    accuracy = calculate_accuracy(predictions, targets)
    
    # Calcul des matrices de confusion (brute et normalisée)
    conf_matrix = confusion_matrix(targets, predictions, labels=range(num_classes))
    norm_conf_matrix = get_normalized_confusion_matrix(targets, predictions, num_classes)
    
    # Calcul des métriques par classe
    precision, recall, jaccard = calculate_precision_recall_jaccard(conf_matrix)
    
    # Sauvegarde de la matrice de confusion multi-classes
    if save_dir:
        # Création du répertoire si nécessaire
        os.makedirs(os.path.join(save_dir), exist_ok=True)
        
        # Sauvegarde de la matrice de confusion brute
        save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
        plot_confusion_matrix(
            conf_matrix, 
            class_names=class_names, 
            save_path=save_path,
            normalized=False,
            title='Matrice de confusion (effectifs)'
        )
        
        # Sauvegarde de la matrice de confusion normalisée
        norm_save_path = os.path.join(save_dir, f"{model_name}_normalized_confusion_matrix.png")
        plot_confusion_matrix(
            norm_conf_matrix, 
            class_names=class_names, 
            save_path=norm_save_path,
            normalized=True,
            title='Matrice de confusion (proportions)'
        )
        
        print(f"Matrices de confusion sauvegardées dans {save_dir}")
    
    return {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'jaccard_per_class': jaccard,
        'confusion_matrix': conf_matrix,
        'normalized_confusion_matrix': norm_conf_matrix
    }

def plot_average_confusion_matrix(confusion_matrices, class_names=None, save_path=None, title='Matrice de confusion moyenne'):
    """
    Trace et sauvegarde une matrice de confusion moyenne
    
    Args:
        confusion_matrices: Liste de matrices de confusion normalisées
        class_names: Les noms des classes (optionnel)
        save_path: Chemin pour sauvegarder l'image (optionnel)
        title: Titre de la figure
    """
    # Calculer la moyenne des matrices
    avg_conf_matrix = np.mean(confusion_matrices, axis=0)
    
    # Tracer la matrice moyenne
    plot_confusion_matrix(
        avg_conf_matrix, 
        class_names=class_names, 
        save_path=save_path, 
        normalized=True,
        title=title
    )
    
    return avg_conf_matrix

def calculate_f1_scores(conf_matrix):
    """
    Calcule les F1 scores par classe et le F1 score global.
    
    Args:
        conf_matrix: La matrice de confusion (peut être moyenne)
        
    Returns:
        tuple: (F1 scores par classe, F1 score global)
    """
    num_classes = conf_matrix.shape[0]
    f1_scores = np.zeros(num_classes)
    
    for i in range(num_classes):
        # True Positives: conf_matrix[i, i]
        tp = conf_matrix[i, i]
        
        # False Positives: sum(conf_matrix[:, i]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp
        
        # False Negatives: sum(conf_matrix[i, :]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1_scores[i] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # F1 score global: moyenne des F1 scores par classe
    global_f1_score = np.mean(f1_scores)
    
    return f1_scores, global_f1_score

if __name__ == "__main__":
    # Image 1: Matrice de confusion moyenne - Enrichissement 1, ratio 90/10
    conf_matrix_1 = np.array([
        [1.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.5000, 0.2500, 0.0000],
        [0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ])

    # Image 2: Matrice de confusion moyenne - Enrichissement 1, ratio 80/20
    conf_matrix_2 = np.array([
        [1.0000, 0.0000, 0.0000, 0.0000],
        [0.2857, 0.4286, 0.2857, 0.0000],
        [0.1111, 0.0000, 0.8889, 0.0000],
        [0.0000, 0.0000, 0.1429, 0.8571]
    ])

    # Image 3: Matrice de confusion moyenne - Enrichissement 5, ratio 90/10
    conf_matrix_3 = np.array([
        [1.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ])

    # Image 4: Matrice de confusion moyenne - Enrichissement 5, ratio 80/20
    conf_matrix_4 = np.array([
        [0.8889, 0.1111, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000, 0.0000],
        [0.0000, 0.2222, 0.7778, 0.0000],
        [0.0000, 0.1429, 0.0000, 0.8571]
    ])

    # Image 5: Matrice de confusion moyenne - Enrichissement 10, ratio 90/10
    conf_matrix_5 = np.array([
        [0.7500, 0.2500, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.8000, 0.2000],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ])

    # Image 6: Matrice de confusion moyenne - Enrichissement 10, ratio 80/20
    conf_matrix_6 = np.array([
        [1.0000, 0.0000, 0.0000, 0.0000],
        [0.1429, 0.8571, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.7778, 0.2222],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ])

    # Image 7: Matrice de confusion moyenne - Enrichissement 10, ratio 70/30
    conf_matrix_7 = np.array([
        [0.9617, 0.0383, 0.0000, 0.0000],
        [0.1238, 0.8027, 0.0735, 0.0000],
        [0.0062, 0.0410, 0.8694, 0.0834],
        [0.0155, 0.0238, 0.1429, 0.8178]
    ])

    # Image 8: Matrice de confusion moyenne - Enrichissement 10, ratio 60/40
    conf_matrix_8 = np.array([
        [0.9603, 0.0397, 0.0000, 0.0000],
        [0.1098, 0.8752, 0.0150, 0.0000],
        [0.0141, 0.0864, 0.7444, 0.1552],
        [0.0276, 0.0112, 0.1058, 0.8553]
    ])

    # Image 9: Matrice de confusion moyenne - Enrichissement 10, ratio 50/50
    conf_matrix_9 = np.array([
        [0.9068, 0.0887, 0.0000, 0.0045],
        [0.0993, 0.8236, 0.0771, 0.0000],
        [0.0141, 0.0939, 0.7387, 0.1532],
        [0.0316, 0.0243, 0.0782, 0.8658]
    ])

    # Image 10: Matrice de confusion moyenne - Enrichissement 20, ratio 70/30
    conf_matrix_10 = np.array([
        [0.9708, 0.0225, 0.0000, 0.0067],
        [0.0882, 0.7584, 0.1533, 0.0000],
        [0.0056, 0.0668, 0.8129, 0.1147],
        [0.0000, 0.0118, 0.1538, 0.8344]
    ])

    # Image 11: Matrice de confusion moyenne - Enrichissement 20, ratio 60/40
    conf_matrix_11 = np.array([
        [0.9157, 0.0718, 0.0125, 0.0000],
        [0.0720, 0.7806, 0.1422, 0.0053],
        [0.0209, 0.0542, 0.7942, 0.1307],
        [0.0112, 0.0059, 0.1062, 0.8766]
    ])

    # Image 12: Matrice de confusion moyenne - Enrichissement 20, ratio 50/50
    conf_matrix_12 = np.array([
        [0.9438, 0.0382, 0.0050, 0.0130],
        [0.0931, 0.8135, 0.0934, 0.0000],
        [0.0106, 0.0418, 0.8375, 0.1102],
        [0.0143, 0.0414, 0.0994, 0.8449]
    ])

    # Image 13: Matrice de confusion moyenne - Enrichissement 30, ratio 70/30
    conf_matrix_enrich30_ratio70_30 = np.array([
        [0.9275, 0.0558, 0.0000, 0.0167],
        [0.0489, 0.7867, 0.1644, 0.0000],
        [0.0091, 0.0792, 0.8505, 0.0612],
        [0.0059, 0.0059, 0.1841, 0.8041]
    ])

    # Image 14: Matrice de confusion moyenne - Enrichissement 30, ratio 60/40
    conf_matrix_enrich30_ratio60_40 = np.array([
        [0.9327, 0.0602, 0.0000, 0.0071],
        [0.0768, 0.8829, 0.0403, 0.0000],
        [0.0169, 0.0553, 0.8452, 0.0827],
        [0.0250, 0.0121, 0.1141, 0.8487]
    ])

    # Image 15: Matrice de confusion moyenne - Enrichissement 30, ratio 50/50
    conf_matrix_enrich30_ratio50_50 = np.array([
        [0.9235, 0.0707, 0.0059, 0.0000],
        [0.0790, 0.8354, 0.0856, 0.0000],
        [0.0098, 0.0948, 0.7756, 0.1199],
        [0.0096, 0.0364, 0.1348, 0.8192]
    ])

    liste_matrices = [conf_matrix_1, conf_matrix_2, conf_matrix_3, conf_matrix_4, conf_matrix_5, conf_matrix_6,
                      conf_matrix_7, conf_matrix_8, conf_matrix_9, conf_matrix_10, conf_matrix_11, conf_matrix_12,
                      conf_matrix_enrich30_ratio70_30, conf_matrix_enrich30_ratio60_40, conf_matrix_enrich30_ratio50_50]
    
    # Liste des enrichissements et ratios correspondants
    enrichissements_ratios = [
        (1, "90/10"), (1, "80/20"), (5, "90/10"), (5, "80/20"),
        (10, "90/10"), (10, "80/20"), (10, "70/30"), (10, "60/40"),
        (10, "50/50"), (20, "70/30"), (20, "60/40"), (20, "50/50"),
        (30, "70/30"), (30, "60/40"), (30, "50/50")
    ]

    for i, (conf_matrix, (enrichissement, ratio)) in enumerate(zip(liste_matrices, enrichissements_ratios), 1):
        f1_scores, global_f1 = calculate_f1_scores(conf_matrix)
        print(f"Image {i}: Enrichissement {enrichissement}, Ratio {ratio}, F1 score global: {global_f1:.4f}")