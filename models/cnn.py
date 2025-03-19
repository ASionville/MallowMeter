import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        # Couches de convolution
        # Input: 3 canaux (L, a, b) de taille 512x512
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)  # 256x256
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x128
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)  # 64x64
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32x32
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 16x16
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8
        
        # Couches fully connected
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Couches de convolution
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten pour les couches fully connected
        x = x.view(-1, 128 * 8 * 8)
        
        # Couches fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class EarlyStopping:
    """
    Early stopping pour arrêter l'entraînement lorsque la validation loss ne s'améliore plus.
    
    Args:
        patience (int): Nombre d'époques à attendre après une dernière amélioration
        min_delta (float): Amélioration minimale considérée significative
        verbose (bool): Affiche des messages si True
    """
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss, model, path='checkpoint.pt'):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        """Sauvegarder le modèle quand la validation loss diminue"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def train(model, device, train_loader, optimizer, criterion, verbose=False):
    """
    Fonction d'entraînement sans affichage de progression par batch
    
    Args:
        model: Le modèle à entraîner
        device: L'appareil sur lequel effectuer les calculs (CPU/GPU)
        train_loader: Le chargeur de données d'entraînement
        optimizer: L'optimiseur à utiliser
        criterion: La fonction de perte
        verbose: Afficher des informations détaillées si True
    
    Returns:
        Tuple (loss moyenne, précision)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Réinitialiser les gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass et optimisation
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Calculer la loss
            test_loss += criterion(output, target).item()
            
            # Obtenir les prédictions
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            
            # Stocker les prédictions et les cibles pour évaluation future
            predictions.append(pred)
            targets.append(target)
    
    # Concaténer toutes les prédictions et cibles
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # print(f'\nTest set: Average loss: {test_loss:.4f}, ' 'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy, predictions, targets

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies, save_dir=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Evolution')
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.set_title('Accuracy Evolution')
    ax2.grid(True)  
    
    plt.tight_layout()
    
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/training_history.png")
    else:
        plt.show()
    
    plt.close()

def create_dataloaders(train_data, train_labels, test_data, test_labels, batch_size=32):
    # Création des datasets
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    # Création des dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
