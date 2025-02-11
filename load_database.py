import os
import torch
from torchvision import transforms
from PIL import Image

def load_dataset(dataset_path):
    data = []
    masks = []
    labels = []
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    for filename in os.listdir(dataset_path):
        if filename.endswith(".png") and not filename.endswith("_m.png"):
            # Extraire le numéro de l'échantillon et la classe à partir du nom du fichier
            parts = filename.split('_')
            sample_num = int(parts[0][3:])
            class_label = int(parts[1][0])

            # Charger l'image et appliquer les transformations
            img_path = os.path.join(dataset_path, filename)
            image = Image.open(img_path)
            image = transform(image)

            # Charger le masque correspondant
            mask_filename = f"img{sample_num}_m.png"
            mask_path = os.path.join(dataset_path, mask_filename)
            mask = Image.open(mask_path)
            mask = transform(mask)

            data.append(image)
            masks.append(mask)
            labels.append(class_label)

    data = torch.stack(data)
    masks = torch.stack(masks)
    labels = torch.tensor(labels, dtype=torch.long)
    return data, masks, labels

if __name__ == "__main__":
    dataset_path = "dataset/"
    data, masks, labels = load_dataset(dataset_path)
    print(f"Loaded {len(data)} images with masks and labels {labels.unique().tolist()}")
