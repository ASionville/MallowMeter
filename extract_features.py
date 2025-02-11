import torch
import torch.nn.functional as F

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

def co_occurrence_matrix(channel, distances=[1], angles=[0]):
    # Calculer la matrice de co-occurrence pour un canal donné
    max_val = int(channel.max().item() + 1)
    comat = torch.zeros((max_val, max_val), dtype=torch.float32, device=channel.device)

    for d in distances:
        for angle in angles:
            dx = int(d * torch.cos(torch.tensor(angle)))
            dy = int(d * torch.sin(torch.tensor(angle)))
            shifted = F.pad(channel, (dx, dx, dy, dy), mode='constant', value=0)
            for i in range(channel.size(0)):
                for j in range(channel.size(1)):
                    comat[channel[i, j], shifted[i + dy, j + dx]] += 1

    return comat

def extract_features(images):
    features = []
    for image in images:
        lab_image = rgb_to_lab(image)
        l, a, b = lab_image[0], lab_image[1], lab_image[2]

        # Étendue, min/max, moments (skewness)
        l_range = l.max() - l.min()
        a_range = a.max() - a.min()
        b_range = b.max() - b.min()

        l_min, l_max = l.min(), l.max()
        a_min, a_max = a.min(), a.max()
        b_min, b_max = b.min(), b.max()

        l_skewness = ((l - l.mean()) ** 3).mean() / (l.std() ** 3)
        a_skewness = ((a - a.mean()) ** 3).mean() / (a.std() ** 3)
        b_skewness = ((b - b.mean()) ** 3).mean() / (b.std() ** 3)

        # Moyenne et écart-type des composantes L et b
        l_mean, l_std = l.mean(), l.std()
        b_mean, b_std = b.mean(), b.std()

        # Matrices de co-occurrence
        l_comat = co_occurrence_matrix(l)
        b_comat = co_occurrence_matrix(b)

        # Contraste, homogénéité, entropie, énergie
        l_contrast = (l_comat * (torch.arange(l_comat.size(0), device=l.device).view(-1, 1) - torch.arange(l_comat.size(1), device=l.device).view(1, -1)) ** 2).sum()
        b_contrast = (b_comat * (torch.arange(b_comat.size(0), device=b.device).view(-1, 1) - torch.arange(b_comat.size(1), device=b.device).view(1, -1)) ** 2).sum()

        l_homogeneity = (l_comat / (1 + (torch.arange(l_comat.size(0), device=l.device).view(-1, 1) - torch.arange(l_comat.size(1), device=l.device).view(1, -1)) ** 2)).sum()
        b_homogeneity = (b_comat / (1 + (torch.arange(b_comat.size(0), device=b.device).view(-1, 1) - torch.arange(b_comat.size(1), device=b.device).view(1, -1)) ** 2)).sum()

        l_entropy = -(l_comat * torch.log(l_comat + 1e-10)).sum()
        b_entropy = -(b_comat * torch.log(b_comat + 1e-10)).sum()

        l_energy = (l_comat ** 2).sum()
        b_energy = (b_comat ** 2).sum()

        features.append(torch.tensor([
            l_range, a_range, b_range,
            l_min, l_max, a_min, a_max, b_min, b_max,
            l_skewness, a_skewness, b_skewness,
            l_mean, l_std, b_mean, b_std,
            l_contrast, b_contrast,
            l_homogeneity, b_homogeneity,
            l_entropy, b_entropy,
            l_energy, b_energy
        ]))

    return torch.stack(features)

if __name__ == "__main__":
    from load_database import load_dataset
    dataset_path = "dataset/"
    data, labels = load_dataset(dataset_path)
    features = extract_features(data)
    print(f"Extracted features for {len(features)} images")
