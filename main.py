from load_database import load_dataset, split_dataset, enrichissement_dataset
from extract_features import extract_features
from models.knn import KNN
from models.cnn import CNN, train_cnn
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import csv

SEED = 42
ENRICHISSEMENT = 3
RATIO_TEST = 0.3

def run(model_type='knn', seeds=[42, 43, 44], enrichissements=[1, 2, 3, 4, 5], ratios_test=[0.2, 0.3, 0.4, 0.5], log=False):
    # Initialisation des paramètres
    dataset_path = "dataset/"
    results = []

    for seed in seeds:
        for enrichissement in enrichissements:
            for ratio_test in ratios_test:
                np.random.seed(seed)

                # Chargement des données
                data, masks, labels, ids = load_dataset(dataset_path)
                if log:
                    print("Données chargées avec succès")
                
                # Séparation des données en ensembles d'entraînement et de test
                (train_data, train_masks, train_labels, train_ids), (test_data, test_masks, test_labels, test_ids) = split_dataset(data, masks, labels, ids, test_size=ratio_test, seed=seed)
                
                # Enrichissement des données d'entraînement
                train_data, train_masks, train_labels, train_ids = enrichissement_dataset(train_data, train_masks, train_labels, train_ids, enrichissement=enrichissement)
                if log:
                    print(f"Nombre d'échantillons d'entraînement: {len(train_data)} (enrichissement x{enrichissement})")
                    print(f"Nombre d'échantillons de test: {len(test_data)}")

                train_features = extract_features(train_data)
                test_features = extract_features(test_data)

                match model_type:
                    case 'knn':
                        # Entraînement du modèle KNN
                        knn = KNN(X=train_features, Y=train_labels, k=3, p=2)
                        if log:
                            print("Modèle KNN entraîné avec succès")
                        
                        # Prédiction des étiquettes pour les données de test
                        predictions = knn.predict(test_features)
                        accuracy = (predictions == test_labels).float().mean().item()
                        if log:
                            print(f"Précision: {accuracy}")
                    case 'cnn':
                        # Création des jeux de données PyTorch
                        train_dataset = TensorDataset(train_features, train_labels)
                        test_dataset = TensorDataset(test_features, test_labels)

                        # Création des loaders
                        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                        # Création du modèle CNN
                        model = CNN(num_classes=len(train_labels.unique()))
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=0.01)

                        # Entraînement du modèle CNN
                        train_cnn(model, train_loader, criterion, optimizer, num_epochs=100)
                        if log:
                            print("Modèle CNN entraîné avec succès")
                        
                        # Prédiction des étiquettes pour les données de test
                        model.eval()
                        predictions = torch.argmax(model(test_features), dim=1)
                        accuracy = (predictions == test_labels).float().mean().item()
                        if log:
                            print(f"Précision: {accuracy}")
                    case _:
                        raise ValueError("Modèle non supporté")

    # Écrire les résultats dans un fichier CSV
    # with open('results.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Seed", "Enrichissement", "Ratio Test", "Accuracy"])
    #     writer.writerows(results)

if __name__ == "__main__":
    run(model_type='cnn', seeds=[42], enrichissements=[5], ratios_test=[0.2], log=True)