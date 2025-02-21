import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from load_database import load_dataset
from extract_features import extract_features

def verif_bdd():
    dataset_path = "dataset/"
    data, masks, labels, ids = load_dataset(dataset_path)
    print("Données chargées avec succès")
    features = extract_features(data)
    print("Caractéristiques extraites avec succès")

    # Convertir les données en DataFrame pour seaborn
    l_mean = features[:, 8].numpy()
    b_mean = features[:, 10].numpy()
    l_std = features[:, 9].numpy()
    b_std = features[:, 11].numpy()
    labels = labels.numpy()

    df = pd.DataFrame({
        'Classe': labels,
        'L_mean': l_mean,
        'B_mean': b_mean,
        'L_std': l_std,
        'B_std': b_std
    })

    # Boîtes à moustaches pour la moyenne des composantes L et b
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Classe', y='L_mean', data=df)
    plt.title('Moyenne de la composante L par classe')

    plt.subplot(1, 2, 2)
    sns.boxplot(x='Classe', y='B_mean', data=df)
    plt.title('Moyenne de la composante B par classe')

    plt.tight_layout()
    plt.show()

    # Boîtes à moustaches pour l'écart-type des composantes L et b
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Classe', y='L_std', data=df)
    plt.title('Écart-type de la composante L par classe')

    plt.subplot(1, 2, 2)
    sns.boxplot(x='Classe', y='B_std', data=df)
    plt.title('Écart-type de la composante B par classe')

    plt.tight_layout()
    plt.show()

    # Proportion de pixels blancs dans les masques
    white_pixel_proportion = masks.view(masks.size(0), -1).mean(dim=1).numpy()
    df_masks = pd.DataFrame({
        'Classe': labels,
        'Proportion_pixels_blancs': white_pixel_proportion
    })

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Classe', y='Proportion_pixels_blancs', data=df_masks)
    plt.title('Proportion de pixels blancs dans les masques par classe')
    plt.show()

    # Afficher le nombre de représentants de chaque classe
    class_counts = df['Classe'].value_counts().sort_index()
    print("Nombre de représentants de chaque classe:")
    print(class_counts)

if __name__ == "__main__":
    verif_bdd()
