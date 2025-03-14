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
