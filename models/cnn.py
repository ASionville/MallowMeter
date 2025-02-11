import torch.nn as nn

# Constantes pour l'architecture du modèle
NUM_LAYERS = 2
LAYER_WIDTHS = [32, 64]
ACTIVATION_FUNCTION = nn.ReLU
INPUT_DIM = (1, 28, 28)
NUM_CLASSES = 10

class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = INPUT_DIM[0]
        
        for width in LAYER_WIDTHS:
            self.layers.append(nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1))
            self.layers.append(ACTIVATION_FUNCTION())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            in_channels = width
        
        self.fc1 = nn.Linear(LAYER_WIDTHS[-1] * (INPUT_DIM[1] // 2**NUM_LAYERS) * (INPUT_DIM[2] // 2**NUM_LAYERS), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, LAYER_WIDTHS[-1] * (INPUT_DIM[1] // 2**NUM_LAYERS) * (INPUT_DIM[2] // 2**NUM_LAYERS))
        x = self.fc1(x)
        x = self.fc2(x)
        return self.softmax(x)

def train(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss / len(train_loader)}")
