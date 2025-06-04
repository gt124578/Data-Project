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
