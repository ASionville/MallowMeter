import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def enrichissement_image(image):
    # Appliquer des transformations aléatoires pour l'enrichissement de données
    if np.random.rand() > 0.5:
        image = transforms.functional.hflip(image)
    if np.random.rand() > 0.5:
        image = transforms.functional.vflip(image)
    angle = int(np.random.choice([0, 90, 180, 270]))
    image = transforms.functional.rotate(image, angle)
    return image

def rgb_to_lab(image):
    # Convertir une image RGB en Lab
    r, g, b = image[0], image[1], image[2]
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0

    # Transformation linéaire
    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b

    # Transformation non-linéaire
    x = torch.where(x > 0.008856, x ** (1/3), 7.787 * x + 16/116)
    y = torch.where(y > 0.008856, y ** (1/3), 7.787 * y + 16/116)
    z = torch.where(z > 0.008856, z ** (1/3), 7.787 * z + 16/116)

    l = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    return torch.stack([l, a, b])

def load_dataset(dataset_path):
    data = []
    masks = []
    labels = []
    ids = []
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    for filename in os.listdir(dataset_path):
        if filename.endswith(".png"):
            # Extraire le numéro de l'échantillon et la classe à partir du nom du fichier
            parts = filename.split('_')
            sample_id = int(parts[0][3:])
            class_label = int(parts[1][0])

            # Charger l'image et appliquer les transformations
            image = Image.open(os.path.join(dataset_path, filename))
            image = transform(image)
            image = rgb_to_lab(image)  # Convertir l'image en Lab

            data.append(image)
            labels.append(class_label)
            ids.append(sample_id)  # Ajouter l'id à la liste

    data = torch.stack(data)
    labels = torch.tensor(labels, dtype=torch.long)
    ids = torch.tensor(ids, dtype=torch.long)  # Convertir ids en tenseur
    return data, labels, ids

def split_dataset(data, labels, ids, test_size=0.2, seed=42):
    np.random.seed(seed)
    num_samples = len(data)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_idx = int((1 - test_size) * num_samples)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_data = data[train_indices]
    train_labels = labels[train_indices]
    train_ids = ids[train_indices]

    test_data = data[test_indices]
    test_labels = labels[test_indices]
    test_ids = ids[test_indices]

    return (train_data, train_labels, train_ids), (test_data, test_labels, test_ids)

def enrichissement_dataset(data, labels, ids, enrichissement=3):
    augmented_data = []
    augmented_labels = []
    augmented_ids = []

    for i in range(len(data)):
        image = data[i]
        class_label = labels[i]
        sample_id = ids[i]

        augmented_data.append(image)
        augmented_labels.append(class_label)
        augmented_ids.append(sample_id)

        for _ in range(enrichissement):
            aug_image = enrichissement_image(image)
            augmented_data.append(aug_image)
            augmented_labels.append(class_label)
            augmented_ids.append(sample_id)

    augmented_data = torch.stack(augmented_data)
    augmented_labels = torch.tensor(augmented_labels, dtype=torch.long)
    augmented_ids = torch.tensor(augmented_ids, dtype=torch.long)

    return augmented_data, augmented_labels, augmented_ids

if __name__ == "__main__":
    dataset_path = "dataset/"
    data, masks, labels, ids = load_dataset(dataset_path)
    (train_data, train_labels, train_ids), (test_data, test_labels, test_ids) = split_dataset(data, labels, ids, test_size=0.2, seed=42)
    train_data, train_labels, train_ids = enrichissement_dataset(train_data, train_labels, train_ids, enrichissement=3)
    print(f"Loaded {len(train_data)} training images and {len(test_data)} test images with masks and labels {labels.unique().tolist()}")
