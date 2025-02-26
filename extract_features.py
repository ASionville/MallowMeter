import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Calculer la matrice de co-occurrence pour un canal donné
def co_occurrence_matrix(channel, distance=1, direction="horizontal"): # Direction : "horizontal" ou "vertical"
    # Boucle sur chaque pixel de l'image en fonction de la distance et de l'angle
    comat = torch.zeros(256, 256, device=channel.device)
    if direction == "horizontal":
        for i in range(channel.size(0)):
            for j in range(channel.size(1) - distance):
                comat[channel[i, j].long(), channel[i, j + distance].long()] += 1
    elif direction == "vertical":
        for i in range(channel.size(0) - distance):
            for j in range(channel.size(1)):
                comat[channel[i, j].long(), channel[i + distance, j].long()] += 1
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")
    return comat / comat.sum()  

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

def extract_features(images):
    features = []
    for image in tqdm(images, desc="Extraction des features"):
        lab_image = rgb_to_lab(image)
        l, b = lab_image[0], lab_image[2]

        # Étendue, min/max, moments (skewness)
        b_skewness = ((b - b.mean()) ** 3).mean() / (b.std() ** 3)

        # Moyenne et écart-type des composantes L et b
        l_mean, l_std = l.mean(), l.std()
        b_mean, b_std = b.mean(), b.std()

        # Matrices de co-occurrence
        l_comat_1h = co_occurrence_matrix(l, distance=1, direction="horizontal")

        # Contraste, homogénéité, entropie, énergie (Tamura)
        l_contrast = (l_comat_1h * (torch.arange(l_comat_1h.size(0)) ** 2).float()).sum()

        l_homogeneity = (l_comat_1h / (1 + (torch.arange(l_comat_1h.size(0)) ** 2).float())).sum()

        l_entropy = -(l_comat_1h * torch.log(l_comat_1h + 1e-10)).sum()

        features.append(torch.tensor([
            # l_range,
            # b_range,
            # l_min,
            # l_max,
            # b_min,
            # b_max,
            # l_skewness,
            b_skewness,
            l_mean,
            l_std,
            b_mean,
            b_std,
            l_contrast,
            # b_contrast,
            l_homogeneity,
            # b_homogeneity,
            l_entropy,
            # l_energy,
            # b_energy
        ], device=image.device))

    return torch.stack(features)
