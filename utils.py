import os
import pickle
import numpy as np
import time
from extract_features import extract_features

# Chemins pour la sauvegarde des features
FEATURES_PATH = "features/"
FEATURES_FILE = os.path.join(FEATURES_PATH, "precomputed_features_old.pkl")

def load_or_compute_features(data, labels, force_recompute=False):
    """
    Charge les features depuis un fichier s'il existe, sinon les calcule et les sauvegarde.
    
    Args:
        data: Données d'entrée (images)
        labels: Étiquettes correspondantes
        force_recompute: Force le recalcul même si un fichier existe
        
    Returns:
        Tuple (features, labels)
    """
    if not force_recompute and os.path.exists(FEATURES_FILE):
        print(f"Chargement des features depuis {FEATURES_FILE}...")
        with open(FEATURES_FILE, 'rb') as f:
            saved_data = pickle.load(f)
            return saved_data['features'], saved_data['labels']
    
    print("Calcul des features...")
    start_time = time.time()
    features = extract_features(data)
    elapsed_time = time.time() - start_time
    print(f"Temps de calcul des features: {elapsed_time:.2f} secondes")
    
    # S'assurer que le répertoire existe
    os.makedirs(FEATURES_PATH, exist_ok=True)
    
    # Sauvegarder les features et labels
    with open(FEATURES_FILE, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)
    
    print(f"Features sauvegardées dans {FEATURES_FILE}")
    return features, labels

def custom_split(features, labels, test_size=0.2, seed=42):
    """
    Effectue un split personnalisé sur les features.
    
    Args:
        features: Features calculées
        labels: Étiquettes correspondantes
        test_size: Taille de l'ensemble de test (proportion)
        seed: Graine pour la reproductibilité
    
    Returns:
        Tuple ((train_features, train_labels), (test_features, test_labels))
    """
    np.random.seed(seed)
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    
    split_idx = int((1 - test_size) * len(features))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    
    return (train_features, train_labels), (test_features, test_labels)
