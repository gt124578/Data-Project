import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
import numpy as np

filename='C:/Users/Charl/Downloads/dataset.csv'
df = pd.read_csv(filename, encoding="iso-8859-1")


# Mood (ambiance) à partir de valence
def mood_category(valence):
    if valence >= 0.6:
        return 'happy'
    elif valence <= 0.4:
        return 'sad'
    else:
        return 'neutral'

df['mood_category'] = df['valence'].apply(mood_category)

# Tempo category
def tempo_category(tempo):
    if tempo < 90:
        return 'slow'
    elif tempo <= 140:
        return 'medium'
    else:
        return 'fast'

df['tempo_category'] = df['tempo'].apply(tempo_category)

# Energy level category
def energy_level(energy):
    if energy < 0.4:
        return 'low'
    elif energy <= 0.7:
        return 'medium'
    else:
        return 'high'

df['energy_level'] = df['energy'].apply(energy_level)

# Danceability category
def danceability_level(danceability):
    if danceability < 0.4:
        return 'low'
    elif danceability <= 0.7:
        return 'medium'
    else:
        return 'high'

df['danceability_level'] = df['danceability'].apply(danceability_level)

# Loudness category
def loudness_level(loudness):
    if loudness < -20:
        return 'quiet'
    elif loudness < -10:
        return 'medium'
    else:
        return 'loud'

df['loudness_level'] = df['loudness'].apply(loudness_level)


features_cat = ['mood_category', 'tempo_category', 'energy_level', 'danceability_level', 'loudness_level']


# Ordres personnalisés pour affichage
custom_orders = {
    'mood_category': ['sad', 'neutral', 'happy'],
    'tempo_category': ['slow', 'medium', 'fast'],
    'energy_level': ['low', 'medium', 'high'],
    'danceability_level': ['low', 'medium', 'high'],
    'loudness_level': ['quiet', 'medium', 'loud']
}

####### Diagramme en bâtons #######
# Créer une figure
plt.figure(figsize=(16, 10))

# Parcours de chaque feature
for i, feature in enumerate(features_cat, 1):
    plt.figure()

    # Compte des occurrences (distribution)
    dist_counts = df[feature].value_counts(normalize=True).reindex(custom_orders[feature])
    # Moyenne de popularité
    pop_means = df.groupby(feature)['popularity'].mean().reindex(custom_orders[feature])

    # Combine dans un seul DataFrame pour faciliter la visualisation
    combined_df = pd.DataFrame({
        'Distribution (%)': dist_counts * 100,
        'Popularité moyenne': pop_means
    })

    # Positions pour les barres côte à côte
    x = np.arange(len(combined_df))
    width = 0.4

    # Tracer les barres
    plt.bar(x - width/2, combined_df['Distribution (%)'], width=width, color='lightpink', label='Distribution (%)')
    plt.bar(x + width/2, combined_df['Popularité moyenne'], width=width, color='skyblue', label='Popularité moyenne')


    plt.title(f'{feature}')
    plt.xticks(x, custom_orders[feature], rotation=45)
    plt.title(f'Comparaison Distribution/Popularité')
    plt.xlabel(f'{feature}')
    plt.ylabel("Valeurs")
    plt.legend()

plt.tight_layout()
plt.show()
