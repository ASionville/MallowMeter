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

def plot_confusion_matrix(conf_matrix, class_names=None, save_path=None):
    """
    Trace et sauvegarde une matrice de confusion
    
    Args:
        conf_matrix: La matrice de confusion
        class_names: Les noms des classes (optionnel)
        save_path: Chemin pour sauvegarder l'image (optionnel)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.title('Matrice de confusion')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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
    num_classes = len(np.unique(targets))
    
    # Calcul de l'exactitude
    accuracy = calculate_accuracy(predictions, targets)
    
    # Calcul de la matrice de confusion
    conf_matrix = confusion_matrix(targets, predictions, labels=range(num_classes))
    
    # Calcul des métriques par classe
    precision, recall, jaccard = calculate_precision_recall_jaccard(conf_matrix)
    
    # Sauvegarde de la matrice de confusion multi-classes
    if save_dir:
        # Création du répertoire si nécessaire
        os.makedirs(os.path.join(save_dir), exist_ok=True)
        
        # Sauvegarde de la matrice de confusion
        save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names if class_names else range(num_classes),
                    yticklabels=class_names if class_names else range(num_classes))
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        plt.title('Matrice de confusion')
        
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Matrice de confusion sauvegardée dans {save_path}")
    
    return {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'jaccard_per_class': jaccard,
        'confusion_matrix': conf_matrix
    }
