import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def enrichissement_image(   image, mask):
    # Appliquer des transformations aléatoires pour l'enrichissement de données
    if np.random.rand() > 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)
    if np.random.rand() > 0.5:
        image = transforms.functional.vflip(image)
        mask = transforms.functional.vflip(mask)
    angle = int(np.random.choice([0, 90, 180, 270]))
    image = transforms.functional.rotate(image, angle)
    mask = transforms.functional.rotate(mask, angle)
    return image, mask

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
        if filename.endswith(".png") and not filename.endswith("_m.png"):
            # Extraire le numéro de l'échantillon et la classe à partir du nom du fichier
            parts = filename.split('_')
            sample_id = int(parts[0][3:])
            class_label = int(parts[1][0])

            # Charger l'image et appliquer les transformations
            img_path = os.path.join(dataset_path, filename)
            image = Image.open(img_path)
            image = transform(image)

            # Charger le masque correspondant
            mask_filename = f"img{sample_id}_m.png"
            mask_path = os.path.join(dataset_path, mask_filename)
            mask = Image.open(mask_path)
            mask = transform(mask)

            data.append(image)
            masks.append(mask)
            labels.append(class_label)
            ids.append(sample_id)  # Ajouter l'id à la liste

    data = torch.stack(data)
    masks = torch.stack(masks)
    labels = torch.tensor(labels, dtype=torch.long)
    ids = torch.tensor(ids, dtype=torch.long)  # Convertir ids en tenseur
    return data, masks, labels, ids

def split_dataset(data, masks, labels, ids, test_size=0.2, seed=42):
    np.random.seed(seed)
    num_samples = len(data)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_idx = int((1 - test_size) * num_samples)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_data = data[train_indices]
    train_masks = masks[train_indices]
    train_labels = labels[train_indices]
    train_ids = ids[train_indices]

    test_data = data[test_indices]
    test_masks = masks[test_indices]
    test_labels = labels[test_indices]
    test_ids = ids[test_indices]

    return (train_data, train_masks, train_labels, train_ids), (test_data, test_masks, test_labels, test_ids)

def enrichissement_dataset(data, masks, labels, ids, enrichissement=3):
    augmented_data = []
    augmented_masks = []
    augmented_labels = []
    augmented_ids = []

    for i in range(len(data)):
        image = data[i]
        mask = masks[i]
        class_label = labels[i]
        sample_id = ids[i]

        augmented_data.append(image)
        augmented_masks.append(mask)
        augmented_labels.append(class_label)
        augmented_ids.append(sample_id)

        for _ in range(enrichissement):
            aug_image, aug_mask = enrichissement_image(image, mask)
            augmented_data.append(aug_image)
            augmented_masks.append(aug_mask)
            augmented_labels.append(class_label)
            augmented_ids.append(sample_id)

    augmented_data = torch.stack(augmented_data)
    augmented_masks = torch.stack(augmented_masks)
    augmented_labels = torch.tensor(augmented_labels, dtype=torch.long)
    augmented_ids = torch.tensor(augmented_ids, dtype=torch.long)

    return augmented_data, augmented_masks, augmented_labels, augmented_ids

if __name__ == "__main__":
    dataset_path = "dataset/"
    data, masks, labels, ids = load_dataset(dataset_path)
    (train_data, train_masks, train_labels, train_ids), (test_data, test_masks, test_labels, test_ids) = split_dataset(data, masks, labels, ids, test_size=0.2, seed=42)
    train_data, train_masks, train_labels, train_ids = enrichissement_dataset(train_data, train_masks, train_labels, train_ids, enrichissement=3)
    print(f"Loaded {len(train_data)} training images and {len(test_data)} test images with masks and labels {labels.unique().tolist()}")
