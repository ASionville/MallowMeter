from load_database import load_dataset, split_dataset, enrichissement_dataset
from utils import load_or_compute_features, custom_split
from models.knn import KNN
from criteres_evaluation import evaluate_model, plot_average_confusion_matrix
import numpy as np
import torch
import os
from tqdm import tqdm
import csv
from collections import defaultdict
from models.cnn import CNN, train, evaluate, plot_training_history, create_dataloaders, EarlyStopping
import torch.optim as optim
import torch.nn as nn

def run_test_knn(seeds=range(10), ratios_test=[0.1, 0.2, 0.3, 0.4, 0.5], k_values=[3, 5, 7], force_recompute=False, log=True):
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
    
    # Dictionnaires pour stocker les matrices de confusion pour chaque configuration
    conf_matrices = defaultdict(list)  # clé: (k, ratio), valeur: liste de matrices de confusion normalisées
    
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
    total_iterations = len(seeds) * len(ratios_test) * len(k_values)
    pbar = tqdm(total=total_iterations, desc="Tests")
    
    for seed in seeds:
        for ratio_test in ratios_test:
            # Split personnalisé sur les features
            (train_features, train_labels), (test_features, test_labels) = custom_split(features, feature_labels, test_size=ratio_test, seed=seed)
                
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
                    save_dir="evaluation/individual" if log else None,  # Sauvegarder individuellement uniquement si log=True
                    model_name=model_name
                )
                
                accuracy = eval_results['accuracy']
                
                # Stocker la matrice de confusion normalisée pour ce test
                conf_matrices[(k, ratio_test)].append(eval_results['normalized_confusion_matrix'])
                
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
    
    # Créer le répertoire pour les matrices moyennes
    os.makedirs("evaluation/average", exist_ok=True)
    
    for ratio_test in ratios_test:
        for k in k_values:
            # Calculer l'exactitude moyenne
            mask = (results_array[:, 1] == ratio_test) & (results_array[:, 2] == k)
            if np.any(mask):
                mean_accuracy = results_array[mask][:, 3].mean()
                mean_results.append([ratio_test, k, mean_accuracy])
                
                # Calculer et sauvegarder la matrice de confusion moyenne pour ce couple (k, ratio)
                if (k, ratio_test) in conf_matrices:
                    avg_conf_save_path = f"evaluation/average/knn_k{k}_ratio{ratio_test}_avg_conf_matrix.png"
                    avg_title = f"Matrice de confusion moyenne - k = {k}, ratio {int(100-ratio_test*100)}/{int(ratio_test*100)}"
                    avg_conf_matrix = plot_average_confusion_matrix(
                        conf_matrices[(k, ratio_test)], 
                        class_names=class_names,
                        save_path=avg_conf_save_path,
                        title=avg_title
                    )
                    print(f"Matrice de confusion moyenne pour k={k}, ratio={ratio_test} sauvegardée")
    
    # Afficher les résultats moyens
    print("\nRésultats moyens:")
    for ratio_test, k, mean_accuracy in mean_results:
        print(f"Ratio Test: {ratio_test}, K: {k}, Précision Moyenne: {mean_accuracy:.4f}")
    
    # Écrire les résultats dans un fichier CSV
    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Seed", "Ratio Test", "K", "Accuracy"])
        writer.writerows(results)
    
    print(f"Résultats enregistrés dans results.csv")

def run_test_cnn(seeds=range(10), enrichissements=[0, 3, 5], ratios_test=[0.2], 
                num_epochs=50, batch_size=16, learning_rate=0.001, 
                patience=10, min_delta=0.001, log=True):
    """
    Exécute les tests avec CNN directement sur les images.
    
    Args:
        seeds: Liste des graines pour la reproductibilité
        enrichissements: Liste des niveaux d'enrichissement (0 pour pas d'enrichissement)
        ratios_test: Liste des proportions pour l'ensemble de test
        num_epochs: Nombre maximum d'époques d'entraînement (défaut: 50)
        batch_size: Taille des batchs
        learning_rate: Taux d'apprentissage
        patience: Nombre d'époques à attendre avant d'arrêter si pas d'amélioration (early stopping)
        min_delta: Amélioration minimale considérée significative pour early stopping
        log: Active les logs pendant l'exécution
    """
    dataset_path = "dataset/"
    results = []
    
    # Dictionnaires pour stocker les matrices de confusion pour chaque configuration
    conf_matrices = defaultdict(list)
    
    # S'assurer que les répertoires existent
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("evaluation/cnn", exist_ok=True)
    
    # Obtenir le périphérique d'exécution (GPU si disponible, sinon CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique: {device}")
    
    # Obtenir les noms des classes
    class_names = ['0', '1', '2', '3']  # Classes 0, 1, 2, 3
    
    total_iterations = len(seeds) * len(enrichissements) * len(ratios_test)
    pbar = tqdm(total=total_iterations, desc="Tests CNN")
    
    for seed in seeds:
        for enrichissement in enrichissements:
            for ratio_test in ratios_test:
                # Configuration pour ce test
                config_name = f"cnn_enr{enrichissement}_ratio{ratio_test}_seed{seed}"
                
                # Charger les données d'origine
                data, labels, ids = load_dataset(dataset_path)
                if log:
                    print(f"Données chargées: {data.shape}, classes: {torch.unique(labels)}")
                
                # Split des données
                (train_data, train_labels, train_ids), (test_data, test_labels, test_ids) = split_dataset(
                    data, labels, ids, test_size=ratio_test, seed=seed
                )
                
                # Enrichissement des données d'entraînement si nécessaire
                if enrichissement > 0:
                    train_data, train_labels, train_ids = enrichissement_dataset(
                        train_data, train_labels, train_ids, enrichissement=enrichissement
                    )
                    if log:
                        print(f"Données enrichies: {train_data.shape}")
                
                # Création des dataloaders
                train_loader, test_loader = create_dataloaders(
                    train_data, train_labels, test_data, test_labels, batch_size=batch_size
                )
                
                # Initialisation du modèle
                model = CNN(num_classes=len(class_names)).to(device)
                
                # Définition de la fonction de perte et de l'optimiseur
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                
                # Variables pour suivre l'entraînement
                train_losses = []
                train_accuracies = []
                val_losses = []
                val_accuracies = []
                
                # Initialisation de l'early stopping
                checkpoint_path = f"models/saved/{config_name}_checkpoint.pth"
                early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=False)
                
                # Barre de progression pour les époques
                epoch_bar = tqdm(range(1, num_epochs + 1), desc="Entraînement CNN", leave=False)
                
                # Entraînement du modèle
                for epoch in epoch_bar:
                    # Entraînement
                    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
                    train_losses.append(train_loss)
                    train_accuracies.append(train_acc)
                    
                    # Évaluation
                    val_loss, val_acc, _, _ = evaluate(model, device, test_loader, criterion)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_acc)
                    
                    # Mise à jour de la barre de progression
                    epoch_bar.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'train_acc': f'{train_acc:.2f}%',
                        'val_loss': f'{val_loss:.4f}',
                        'val_acc': f'{val_acc:.2f}%'
                    })
                    
                    # Early stopping
                    early_stopping(val_loss, model, path=checkpoint_path)
                    
                    if early_stopping.early_stop:
                        epoch_bar.set_description(f"Early stopping à l'époque {epoch}")
                        break
                
                # Fermer la barre de progression des époques
                epoch_bar.close()
                
                # Charger le meilleur modèle
                model.load_state_dict(torch.load(checkpoint_path))
                
                # Évaluation finale
                _, _, predictions, targets = evaluate(model, device, test_loader, criterion)
                
                # Évaluation du modèle avec les critères demandés
                eval_results = evaluate_model(
                    predictions=predictions,
                    targets=targets,
                    class_names=class_names,
                    save_dir=f"evaluation/cnn/individual" if log else None,
                    model_name=config_name
                )
                
                accuracy = eval_results['accuracy']
                
                # Stocker la matrice de confusion normalisée
                conf_matrices[(enrichissement, ratio_test)].append(eval_results['normalized_confusion_matrix'])
                
                # Sauvegarder le modèle final
                torch.save(model.state_dict(), f"models/saved/{config_name}.pth")
                
                # Tracer l'historique d'entraînement
                plot_training_history(
                    train_losses, train_accuracies, val_losses, val_accuracies,
                    save_dir=f"evaluation/cnn/history" if log else None
                )
                
                if log:
                    print(f"\nSeed: {seed}, Enrichissement: {enrichissement}, ")
                    print(f"\tRatio Test: {ratio_test}, Exactitude: {accuracy:.4f}")
                    print(f"\tPrécision par classe: {eval_results['precision_per_class']}")
                    print(f"\tRappel par classe: {eval_results['recall_per_class']}")
                    print(f"\tIndice de Jaccard par classe: {eval_results['jaccard_per_class']}")
                
                results.append([seed, enrichissement, ratio_test, accuracy])
                pbar.update(1)
    
    pbar.close()
    
    # Moyenner les résultats sur les seeds
    results_array = np.array(results)
    mean_results = []
    
    # Créer le répertoire pour les matrices moyennes
    os.makedirs("evaluation/cnn/average", exist_ok=True)
    
    for enrichissement in enrichissements:
        for ratio_test in ratios_test:
            # Calculer l'exactitude moyenne
            mask = (results_array[:, 1] == enrichissement) & (results_array[:, 2] == ratio_test)
            if np.any(mask):
                mean_accuracy = results_array[mask][:, 3].mean()
                mean_results.append([enrichissement, ratio_test, mean_accuracy])
                
                # Calculer et sauvegarder la matrice de confusion moyenne
                if (enrichissement, ratio_test) in conf_matrices:
                    avg_conf_save_path = f"evaluation/cnn/average/cnn_enr{enrichissement}_ratio{ratio_test}_avg_conf_matrix.png"
                    avg_title = f"Matrice de confusion moyenne - Enrichissement {enrichissement}, ratio {int(100-ratio_test*100)}/{int(ratio_test*100)}"
                    avg_conf_matrix = plot_average_confusion_matrix(
                        conf_matrices[(enrichissement, ratio_test)], 
                        class_names=class_names,
                        save_path=avg_conf_save_path,
                        title=avg_title
                    )
                    print(f"Matrice de confusion moyenne pour enrichissement={enrichissement}, ratio={ratio_test} sauvegardée")
    
    # Afficher les résultats moyens
    print("\nRésultats moyens:")
    for enrichissement, ratio_test, mean_accuracy in mean_results:
        print(f"Enrichissement: {enrichissement}, Ratio Test: {ratio_test}, Précision Moyenne: {mean_accuracy:.4f}")
    
    # Écrire les résultats dans un fichier CSV
    with open('results_cnn.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Seed", "Enrichissement", "Ratio Test", "Accuracy"])
        writer.writerows(results)
    
    print(f"Résultats enregistrés dans results_cnn.csv")

if __name__ == "__main__":
    run_test_knn(
        seeds=range(100),
        ratios_test=[0.1, 0.2, 0.3],
        k_values=[3, 5, 7],
        force_recompute=False,
        log=False
    )

    run_test_cnn(
        seeds=range(10),
        enrichissements=[10, 20, 30],
        ratios_test=[0.3, 0.4, 0.5],
        num_epochs=50,   # 50 époques maximum
        patience=5,      # Arrêt après 5 époques sans amélioration
        min_delta=0.001, # Amélioration minimale considérée significative
        log=True
    )
