from load_database import load_dataset, split_dataset, enrichissement_dataset
from utils import load_or_compute_features, custom_split
from models.knn import KNN
from criteres_evaluation import evaluate_model
import numpy as np
import torch
import os
from tqdm import tqdm
import csv

def run_test(seeds=range(10), enrichissements=[0], ratios_test=[0.1, 0.2, 0.3, 0.4, 0.5], k_values=[3, 5, 7], force_recompute=False, log=True):
    """
    Exécute les tests avec calcul des features avant le split.
    
    Args:
        seeds: Liste des graines pour la reproductibilité
        enrichissements: Liste des niveaux d'enrichissement
        ratios_test: Liste des proportions pour l'ensemble de test
        k_values: Liste des valeurs de k à tester pour KNN
        force_recompute: Force le recalcul des features
        log: Active les logs pendant l'exécution
    """
    dataset_path = "dataset/"
    results = []
    
    # Charger les données
    data, labels, ids = load_dataset(dataset_path, num_images_per_class=40)
    if log:
        print("Données chargées avec succès")
    
    # Calculer ou charger les features
    features, feature_labels = load_or_compute_features(data, labels, force_recompute)
    
    # Obtenir les noms des classes uniques
    unique_classes = torch.unique(feature_labels).tolist()
    class_names = [str(i) for i in unique_classes]
    
    # Exécuter les tests pour différentes configurations
    total_iterations = len(seeds) * len(enrichissements) * len(ratios_test) * len(k_values)
    pbar = tqdm(total=total_iterations, desc="Tests")
    
    for seed in seeds:
        for ratio_test in ratios_test:
            # Split personnalisé sur les features
            (train_features, train_labels), (test_features, test_labels) = custom_split(features, feature_labels, test_size=ratio_test, seed=seed)
            
            for enrichissement in enrichissements:
                if enrichissement > 0:
                    # Note: l'enrichissement des features nécessiterait une fonction spécifique
                    # Pour l'instant, on saute simplement cette étape si enrichissement > 0
                    continue
                
                for k in k_values:
                    # Entraînement du modèle KNN
                    knn = KNN(X=train_features, Y=train_labels, k=k, p=2)
                    
                    # Prédiction des étiquettes pour les données de test
                    predictions = knn.predict(test_features)
                    
                    # Évaluation du modèle avec les critères demandés
                    model_name = f"knn_k{k}_ratio{ratio_test}_seed{seed}"
                    eval_results = evaluate_model(
                        predictions=predictions,
                        targets=test_labels,
                        class_names=class_names,
                        save_dir="evaluation",
                        model_name=model_name
                    )
                    
                    accuracy = eval_results['accuracy']
                    
                    if log:
                        print(f"Seed: {seed}, Ratio Test: {ratio_test}, K: {k}, Exactitude: {accuracy:.4f}")
                        print(f"  Précision par classe: {eval_results['precision_per_class']}")
                        print(f"  Rappel par classe: {eval_results['recall_per_class']}")
                        print(f"  Indice de Jaccard par classe: {eval_results['jaccard_per_class']}")
                    
                    results.append([seed, ratio_test, k, accuracy])
                    pbar.update(1)
    
    pbar.close()
    
    # Moyenner les résultats sur les seeds
    results_array = np.array(results)
    mean_results = []
    for ratio_test in ratios_test:
        for k in k_values:
            mask = (results_array[:, 1] == ratio_test) & (results_array[:, 2] == k)
            if np.any(mask):
                mean_accuracy = results_array[mask][:, 3].mean()
                mean_results.append([ratio_test, k, mean_accuracy])
    
    # Afficher les résultats moyens
    print("\nRésultats moyens:")
    for ratio_test, k, mean_accuracy in mean_results:
        print(f"Ratio Test: {ratio_test}, K: {k}, Précision Moyenne: {mean_accuracy:.4f}")
    
    # Écrire les résultats dans un fichier CSV
    with open('test_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Seed", "Ratio Test", "K", "Accuracy"])
        writer.writerows(results)
    
    print(f"Résultats enregistrés dans test_results.csv")

if __name__ == "__main__":
    run_test(
        seeds=range(10),
        enrichissements=[0],
        ratios_test=[0.1],
        k_values=[5],
        force_recompute=False,
        log=False
    )
