import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from load_database import load_dataset
from extract_features import extract_features

def verif_bdd():
    dataset_path = "dataset/"
    data, labels, ids = load_dataset(dataset_path)
    print("Données chargées avec succès")
    features = extract_features(data)
    print("Caractéristiques extraites avec succès")

    feature_params = {
        0: ('l_range', 'Etendue L'),
        1: ('b_range', 'Etendue b'),
        2: ('l_min', 'Min L'),
        3: ('l_max', 'Max L'),
        4: ('b_min', 'Min b'),
        5: ('b_max', 'Max b'),
        6: ('l_skewness', 'Skewness L'),
        7: ('b_skewness', 'Skewness b'),
        8: ('l_mean', 'Moyenne L'),
        9: ('l_std', 'Ecart-type L'),
        10: ('b_mean', 'Moyenne b'),
        11: ('b_std', 'Ecart-type b'),
        12: ('l_contrast', 'Contraste L'),
        13: ('b_contraste', 'Contraste b'),
        14: ('l_homogeneity', 'Homogénéité L'),
        15: ('b_homogeneity', 'Homogénéité b'),
        16: ('l_entropy', 'Entropie L'),
        17: ('l_energy', 'Energie L'),
        18: ('b_energy', 'Energie b')
    }

    for index, (feature_name, title) in feature_params.items():
        plot_feature_boxplot(features, labels, index, feature_name, title)

def plot_feature_boxplot(features, labels, feature_index, feature_name, title):
    df = pd.DataFrame({
        f'{title}': features[:, feature_index].cpu().numpy(),
        'Classe': labels.cpu().numpy()
    })
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    palette = {0: '#f0ead2', 1: '#f4d35e', 2: '#f77f00', 3: '#d62828'}
    sns.boxplot(x='Classe', y=f'{title}', data=df, hue='Classe', palette=palette, legend=False)
    plt.title(f'Boites à moustaches : {title}')
    plt.savefig(f'boites_a_moustaches/{feature_name}.png')
    plt.close()

if __name__ == "__main__":
    verif_bdd()
