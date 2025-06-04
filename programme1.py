# A installer, possiblement !
# pip install pandas numpy matplotlib seaborn scikit-learn -q !

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from matplotlib.colors import Normalize



filename = "../data/spotify.csv"
df = pd.read_csv(filename)


try:
    print("--- 1. Chargement des données réussi ---")
    print(f"La base de données contient {df.shape[0]} lignes et {df.shape[1]} colonnes.\n")
except FileNotFoundError:
    print(f"Erreur : Le fichier à l'emplacement '{upload}' n'a pas été trouvé.")
    exit()
except Exception as e:
    print(f"Une erreur est survenue lors du chargement du fichier : {e}")
    exit()


#------- Comptage des tracks en fonction de la popularité ------
count_popular=0
for i in range(0,10):
  count_popular += df[df['popularity'] == i].shape[0]
print(f"Le nombre de morceaux populaires est : {count_popular}")



# --- 2. Exploration Initiale des Données ---
print("--- 2. Exploration Initiale des Données ---")
print("\n--- Informations sur les colonnes (types, valeurs non nulles) ---")
df.info()

print("\n--- Statistiques descriptives pour les colonnes numériques ---")
print(df.describe(include='all'))

print("\n--- Nombre de valeurs manquantes par colonne ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("Aucune valeur manquante détectée.")

# --- 3. Nettoyage et Prétraitement des Données ---
print("\n--- 3. Nettoyage et Prétraitement ---")

df_processed = df.copy()

numerical_cols_with_nan = df_processed.select_dtypes(include=np.number).isnull().any()
cols_to_impute_mean = numerical_cols_with_nan[numerical_cols_with_nan].index
for col in cols_to_impute_mean:
    if df_processed[col].isnull().sum() > 0:
        print(f"Imputation de la moyenne pour les NaN dans la colonne numérique '{col}'")
        df_processed[col].fillna(df_processed[col].mean(), inplace=True)

categorical_cols_with_nan = df_processed.select_dtypes(include='object').isnull().any()
cols_to_impute_mode = categorical_cols_with_nan[categorical_cols_with_nan].index
for col in cols_to_impute_mode:
    if df_processed[col].isnull().sum() > 0:
        print(f"Imputation du mode pour les NaN dans la colonne catégorielle '{col}'")
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)


initial_rows = len(df_processed)
df_processed.drop_duplicates(inplace=True)
if initial_rows > len(df_processed):
    print(f"{initial_rows - len(df_processed)} lignes dupliquées ont été supprimées.")
else:
    print("Aucune ligne dupliquée trouvée.")

print("Le nettoyage et prétraitement de base est terminé.")


# --- 4. Analyse Exploratoire des Données (EDA) avec Visualisations ---
print("\n--- 4. Analyse Exploratoire des Données (Visualisations) ---")

numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_processed.select_dtypes(include='object').columns.tolist()

print(f"Colonnes numériques identifiées : {numerical_cols}")
print(f"Colonnes catégorielles identifiées : {categorical_cols}")

if numerical_cols:
    print("\n--- Distribution des variables numériques ---")
    for col in numerical_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df_processed[col].dropna(), kde=True, bins=30)
        plt.title(f'Distribution de {col}')
        plt.xlabel(col)
        plt.ylabel('Fréquence')
        plt.show()
else:
    print("Aucune colonne numérique trouvée pour les histogrammes.")

if len(numerical_cols) > 1:
    print("\n--- Matrice de corrélation des variables numériques ---")
    correlation_matrix = df_processed[numerical_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Matrice de Corrélation')
    plt.show()
else:
    print("Pas assez de colonnes numériques (>1) pour calculer une matrice de corrélation.")






# --- 5. La popularité et les autres variables ---

# Nous allons calculer la moyenne de popularité par genre et éventuellement la visualiser.

if 'track_genre' in df_processed.columns and 'popularity' in df_processed.columns:
    print("\n--- Analyse de la Popularité par Genre ---")

    # Calculer la moyenne de popularité par genre
    popularity_by_genre = df_processed.groupby('track_genre')['popularity'].mean().sort_values(ascending=False)

    print("\nMoyenne de popularité par genre (Top 10):")
    print(popularity_by_genre.head(10))

    print("\nMoyenne de popularité par genre (Bottom 10):")
    print(popularity_by_genre.tail(10))

    # Visualisation de la popularité moyenne par genre (Top N)
    top_n_genres = 20 # Nombre de genres à afficher
    plt.figure(figsize=(14, 8))
    popularity_by_genre.head(top_n_genres).plot(kind='bar')
    plt.title(f'Popularité Moyenne par Genre (Top {top_n_genres})')
    plt.xlabel('Genre Musical')
    plt.ylabel('Popularité Moyenne')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Vous pouvez également visualiser la popularité par genre en utilisant des boxplots
    # pour voir la distribution, mais cela peut être très dense avec beaucoup de genres.
    # Il est souvent préférable de sélectionner un sous-ensemble de genres.

    # Exemple de boxplot pour un sous-ensemble de genres (les plus populaires par exemple)
    genres_to_boxplot = popularity_by_genre.head(10).index.tolist()
    df_subset_for_boxplot = df_processed[df_processed['track_genre'].isin(genres_to_boxplot)]

    if not df_subset_for_boxplot.empty:
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='track_genre', y='popularity', data=df_subset_for_boxplot)
        plt.title('Distribution de la Popularité pour les Top 10 Genres')
        plt.xlabel('Genre Musical')
        plt.ylabel('Popularité')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("Le sous-ensemble de données pour le boxplot est vide.")


    # Pour une "matrice de corrélation" entre une variable catégorielle (genre) et une numérique (popularity),
    # la méthode standard est de calculer la moyenne/médiane de la variable numérique par catégorie.
    # On ne calcule pas de coefficient de corrélation comme pour deux variables numériques.
    # La visualisation de la popularité moyenne par genre (comme fait ci-dessus) est la manière standard de représenter cette relation.

else:
    print("Les colonnes 'track_genre' et/ou 'popularity' ne sont pas présentes dans le DataFrame.")




# --- 6. Régression linéaire (Matrice de Graphiques) ---


target_variables = numerical_cols
feature_variables = numerical_cols
n = len(numerical_cols)

fig, ax = plt.subplots(n, n, figsize=(10*n, 10*n))

if n == 1:
    ax = np.array([[ax]])

cmap = plt.get_cmap('coolwarm')
norm = Normalize(vmin=-1, vmax=1)
correlation_matrix = df_processed[numerical_cols].corr()


for j, target_variable in enumerate(target_variables):
    for i, feature_var in enumerate(feature_variables):
        if target_variable == feature_var:

            ax[j, i].text(0.5, 0.5, 'NA', ha='center', va='center', fontsize=12)
            ax[j, i].set_axis_off()
            continue
        corr = correlation_matrix.loc[target_variable, feature_var]
        color = cmap(norm(corr))
        ax[j, i].set_facecolor(color)
        ax[j, i].text(0.05, 0.95, f"ρ={corr:.2f}", transform=ax[j, i].transAxes,
                      fontsize=14, color="black", va="top", ha="left", bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        df_regression_subset = df_processed[[target_variable, feature_var]].copy().dropna()
        X = df_regression_subset[[feature_var]]
        y = df_regression_subset[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_sklearn = LinearRegression()
        model_sklearn.fit(X_train, y_train)

        ax[j, i].scatter(X_test, y_test, color='blue', label='Données réelles', alpha=0.5)
        X_test_sorted = X_test.sort_values(by=feature_var)
        y_pred_line = model_sklearn.predict(X_test_sorted)
        ax[j, i].plot(X_test_sorted, y_pred_line, color='red', linewidth=2, label='Ligne de régression')

        if j == n-1:
            ax[j, i].set_xlabel(feature_var)
        else:
            ax[j, i].set_xlabel("")
        if i == 0:
            ax[j, i].set_ylabel(target_variable)
        else:
            ax[j, i].set_ylabel("")
        ax[j, i].set_title(f"{target_variable} vs {feature_var}", fontsize=8)
        ax[j, i].legend(fontsize=6)
        ax[j, i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/Python/dataset.csv')
plt.show()
