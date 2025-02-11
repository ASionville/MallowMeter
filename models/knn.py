import torch
import matplotlib.pyplot as plt
import numpy as np

def distance_matrix(x, y=None, p=2):
    # Calculer la matrice de distance entre les points x et y en utilisant la distance de Minkowski
    y = x if y is None else y
    return torch.cdist(x, y, p=p)

class KNN:
    def __init__(self, X=None, Y=None, k=3, p=2):
        # Initialiser le modèle KNN avec les données d'entraînement, le nombre de voisins k et la distance p
        self.k = k
        self.p = p
        self.train(X, Y)
    
    def train(self, X, Y):
        # Entraîner le modèle avec les données d'entraînement X et les étiquettes Y
        self.train_pts = X
        self.train_label = Y
        if Y is not None:
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        # Prédire les étiquettes pour les données de test x
        if self.train_pts is None or self.train_label is None:
            raise RuntimeError("KNN wasn't trained. Need to execute KNN.train() first")
        
        # Calculer les distances entre les points de test et les points d'entraînement
        dist = distance_matrix(x, self.train_pts, p=self.p)
        knn = dist.topk(self.k, largest=False)
        votes = self.train_label[knn.indices]

        # Trouver l'étiquette majoritaire parmi les k plus proches voisins
        winner = torch.zeros(votes.size(0), dtype=votes.dtype, device=votes.device)
        count = torch.zeros(votes.size(0), dtype=torch.long, device=votes.device) - 1

        for lab in self.unique_labels:
            vote_count = (votes == lab).sum(1)
            who = vote_count >= count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner

if __name__ == "__main__":
    # Générer des données pour l'exemple
    # Générer des données d'entraînement
    num_classes = 3
    points_per_class = 50
    train_data = []
    train_labels = []

    for i in range(num_classes):
        center = np.random.randn(2) * 5  # Centrer les gaussiennes autour de points différents
        class_data = np.random.randn(points_per_class, 2) + center
        train_data.append(class_data)
        train_labels.append(np.full((points_per_class,), i))

    train_data = np.vstack(train_data)
    train_labels = np.concatenate(train_labels)

    # Générer des données de test
    test_data = []
    for i in range(num_classes):
        center = np.random.randn(2) * 5  # Centrer les gaussiennes autour de points différents
        class_data = np.random.randn(10, 2) + center
        test_data.append(class_data)
    test_data = np.vstack(test_data)

    # Initialiser et entraîner le modèle KNN
    knn = KNN(X=torch.tensor(train_data), Y=torch.tensor(train_labels), k=3)

    # Prédire les étiquettes pour les données de test
    predictions = knn.predict(torch.tensor(test_data))
    print(predictions)

    # Plot des données d'entraînement
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, marker='o', label='Train Data')

    # Plot des données de test
    plt.scatter(test_data[:, 0], test_data[:, 1], c=predictions, marker='x', label='Test Data')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('KNN Classification')
    plt.show()
