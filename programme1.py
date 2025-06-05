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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures



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
#print("\n--- Aperçu des 5 premières lignes ---")
#print(df.head())

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
del numerical_cols[0]


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



# --- 7. Comparaison de l'orientation entre 2 variables ---


variable1 = 'energy'  # Remplacez par le nom d'une de vos colonnes numériques
variable2 = 'danceability' # Remplacez par le nom d'une autre de vos colonnes numériques

print(f"Comparaison de l'orientation pour la régression polynomiale entre '{variable1}' et '{variable2}'")


df_subset = df_processed[[variable1, variable2]].copy().dropna()

if df_subset.empty:
    print(f"Pas de données valides pour la comparaison entre '{variable1}' et '{variable2}'.")
else:
    # Définir le degré du polynôme
    polynomial_degree = 2 # Vous pouvez expérimenter avec différents degrés

    # --- Modèle 1: Prédire Variable1 (Y) à partir de Variable2 (X) ---
    print(f"\n--- Modèle 1: Prédire {variable1} à partir de {variable2} ---")
    X1 = df_subset[[variable2]]
    y1 = df_subset[variable1]


    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

    # Appliquer la transformation polynomiale et entraîner le modèle
    poly_features1 = PolynomialFeatures(degree=polynomial_degree)
    X1_train_poly = poly_features1.fit_transform(X1_train)
    X1_test_poly = poly_features1.transform(X1_test)

    model1 = LinearRegression()
    model1.fit(X1_train_poly, y1_train)

    # Évaluer le modèle
    y1_pred = model1.predict(X1_test_poly)
    r2_model1 = r2_score(y1_test, y1_pred)
    print(f"  R² (Prédire {variable1} par {variable2}) : {r2_model1:.4f}")

    # --- Modèle 2: Prédire Variable2 (Y') à partir de Variable1 (X') ---
    print(f"\n--- Modèle 2: Prédire {variable2} à partir de {variable1} ---")
    X2 = df_subset[[variable1]] 
    y2 = df_subset[variable2] 

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

    poly_features2 = PolynomialFeatures(degree=polynomial_degree)
    X2_train_poly = poly_features2.fit_transform(X2_train)
    X2_test_poly = poly_features2.transform(X2_test)

    model2 = LinearRegression()
    model2.fit(X2_train_poly, y2_train)


    y2_pred = model2.predict(X2_test_poly)
    r2_model2 = r2_score(y2_test, y2_pred)
    print(f"  R² (Prédire {variable2} par {variable1}) : {r2_model2:.4f}")


    print("\n--- Conclusion ---")
    if r2_model1 > r2_model2:
        print(f"La meilleure orientation pour la régression polynomiale est de prédire '{variable1}' à partir de '{variable2}' (R² = {r2_model1:.4f}).")
        best_x_var = variable2
        best_y_var = variable1
        best_model = model1
        best_poly_features = poly_features1
    else:
        print(f"La meilleure orientation pour la régression polynomiale est de prédire '{variable2}' à partir de '{variable1}' (R² = {r2_model2:.4f}).")
        best_x_var = variable1
        best_y_var = variable2
        best_model = model2
        best_poly_features = poly_features2

    # Visualiser le modèle avec la meilleure orientation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=best_x_var, y=best_y_var, data=df_subset, alpha=0.5, label='Données')

    X_plot_values = np.linspace(df_subset[best_x_var].min(), df_subset[best_x_var].max(), 100).reshape(-1, 1)
    X_plot_poly = best_poly_features.transform(X_plot_values)
    y_plot_pred = best_model.predict(X_plot_poly)

    plt.plot(X_plot_values, y_plot_pred, color='red', linestyle='-', linewidth=2, label=f'Régression Polynomiale (Degré {polynomial_degree}, R²={max(r2_model1, r2_model2):.2f})')

    plt.xlabel(best_x_var)
    plt.ylabel(best_y_var)
    plt.title(f'Régression Polynomiale ({best_y_var} vs {best_x_var}) - Meilleure Orientation')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# --- 8. Création des martices triangulaires supérieures des R², p-values et d'orientations ---



if 'numerical_cols' not in globals() or not numerical_cols:
    print("Erreur : La liste 'numerical_cols' n'est pas définie ou est vide.")
    print("Veuillez définir 'numerical_cols' avec les noms des colonnes numériques à analyser.")

else:
    print(f"Analyse de régression polynomiale pour les paires de variables dans : {numerical_cols}")

    n = len(numerical_cols)
    # Initialiser les matrices pour stocker les R², les orientations et les p-values
    r2_matrix = np.full((n, n), np.nan)
    orientation_matrix = np.full((n, n), np.nan, dtype=object)

    min_p_value_matrix = np.full((n, n), np.nan)

    polynomial_degree = 2 

    print("\nDébut de l'analyse de toutes les paires de variables...")

    # Boucler sur toutes les paires de variables (matrice triangulaire supérieure)
    for i in range(n):
        for j in range(i + 1, n): 
            variable1 = numerical_cols[i]
            variable2 = numerical_cols[j]

            print(f"\nAnalyse de la paire : '{variable1}' vs '{variable2}'")

 
            df_subset = df_processed[[variable1, variable2]].copy().dropna()

            if df_subset.empty:
                print(f"  Pas de données valides pour la paire '{variable1}' vs '{variable2}'.")
                continue

            # --- Modèle 1: Prédire Variable1 (Y) à partir de Variable2 (X) ---
            X1 = df_subset[[variable2]]
            y1 = df_subset[variable1]

            X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

            # Transformation polynomiale
            poly_features1 = PolynomialFeatures(degree=polynomial_degree)
            X1_train_poly = poly_features1.fit_transform(X1_train)
            X1_test_poly = poly_features1.transform(X1_test)

            # Modèle scikit-learn pour R² 
            model1_sklearn = LinearRegression()
            model1_sklearn.fit(X1_train_poly, y1_train)
            y1_pred = model1_sklearn.predict(X1_test_poly)
            r2_model1 = r2_score(y1_test, y1_pred)

            # Modèle statsmodels pour les p-values 
            X1_train_sm = sm.add_constant(X1_train_poly) 
            model1_sm = sm.OLS(y1_train, X1_train_sm).fit()
            p_values1 = model1_sm.pvalues 

            # --- Modèle 2: Prédire Variable2 (Y') à partir de Variable1 (X') ---
            X2 = df_subset[[variable1]] 
            y2 = df_subset[variable2] 


            X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

            # Transformation polynomiale
            poly_features2 = PolynomialFeatures(degree=polynomial_degree)
            X2_train_poly = poly_features2.fit_transform(X2_train)
            X2_test_poly = poly_features2.transform(X2_test)

            # Modèle scikit-learn pour R² (sur test set)
            model2_sklearn = LinearRegression()
            model2_sklearn.fit(X2_train_poly, y2_train)
            y2_pred = model2_sklearn.predict(X2_test_poly)
            r2_model2 = r2_score(y2_test, y2_pred)

            # Modèle statsmodels pour les p-values (sur train set)
            X2_train_sm = sm.add_constant(X2_train_poly) 
            model2_sm = sm.OLS(y2_train, X2_train_sm).fit()
            p_values2 = model2_sm.pvalues # Obtenir les p-values

            # --- Stocker les résultats ---
            max_r2 = max(r2_model1, r2_model2)
            r2_matrix[i, j] = max_r2

            if r2_model1 >= r2_model2:
                orientation_matrix[i, j] = f"Y={variable1}, X={variable2}"
                best_p_values = p_values1
            else:
                orientation_matrix[i, j] = f"Y={variable2}, X={variable1}"
                best_p_values = p_values2

            # Trouver la p-value minimale parmi les coefficients
            if len(best_p_values) > 1:
                min_p_value = best_p_values[1:].min()
                min_p_value_matrix[i, j] = min_p_value
            else: 
                 min_p_value_matrix[i, j] = np.nan 

            print(f"  R² (Prédire {variable1} par {variable2}) : {r2_model1:.4f}")
            print(f"  R² (Prédire {variable2} par {variable1}) : {r2_model2:.4f}")
            print(f"  Meilleur R² : {max_r2:.4f}")
            print(f"  Meilleure orientation : {orientation_matrix[i, j]}")
            print(f"  P-values pour la meilleure orientation : {best_p_values.tolist()}")
            print(f"  Min P-value des termes (hors intercept) : {min_p_value:.4f}")


    print("\nAnalyse terminée.")

    # Afficher les matrices résultantes
    r2_df = pd.DataFrame(r2_matrix, index=numerical_cols, columns=numerical_cols)
    orientation_df = pd.DataFrame(orientation_matrix, index=numerical_cols, columns=numerical_cols)
    min_p_value_df = pd.DataFrame(min_p_value_matrix, index=numerical_cols, columns=numerical_cols)


    print("\n--- Matrice des meilleurs R² (Triangulaire Supérieure) ---")
    print(r2_df.to_string(na_rep=''))

    print("\n--- Matrice des Meilleures Orientations (Triangulaire Supérieure) ---")
    print(orientation_df.to_string(na_rep=''))

    print("\n--- Matrice des P-values Minimales des Termes (hors intercept) pour la Meilleure Orientation (Triangulaire Supérieure) ---")
    print(min_p_value_df.to_string(na_rep='', float_format='{:.4f}'.format))


# --- 8. Heatmap de la matrice des R² et des p-values ---

if 'r2_df' not in globals():
    print("Erreur : Le DataFrame 'r2_df' n'est pas défini.")
    print("Veuillez exécuter la cellule précédente qui calcule la matrice des R².")
else:
    print("\n--- Génération de la Heatmap des meilleurs R² ---")

    plt.figure(figsize=(10, 8)) # Ajustez la taille de la figure si nécessaire

    # Créer un masque pour n'afficher que la partie triangulaire supérieure (sans la diagonale)
    # np.triu retourne la partie supérieure d'une matrice, en incluant la diagonale par défaut.
    # k=1 exclut la diagonale.
    mask = np.triu(np.ones_like(r2_df, dtype=bool), k=1)

    sns.heatmap(r2_df,cmap='Reds',annot=True,fmt=".2f")       # Appliquer le masque pour n'afficher que la partie supérieure

    plt.title('Heatmap des Meilleurs R² (Régression Polynomiale)')
    plt.xlabel('Variable (Axe des X potentiel)')
    plt.ylabel('Variable (Axe des Y potentiel)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show() # Affiche la première heatmap

    print("\n--- Génération de la Heatmap des P-values Minimales ---")

    plt.figure(figsize=(10, 8)) # Utilisez la même taille ou ajustez si nécessaire

    sns.heatmap(min_p_value_df,cmap='Reds',annot=True,fmt=".2f")

    plt.title('Heatmap des P-values Minimales (Termes hors Intercept)')
    plt.xlabel('Variable (Axe des X potentiel)')
    plt.ylabel('Variable (Axe des Y potentiel)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show() # Affiche la deuxième heatmap
