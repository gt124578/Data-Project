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

UPLOADED_FILE_NAME = '/content/drive/MyDrive/Python/dataset.csv'
df=pd.read_csv(UPLOADED_FILE_NAME)

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


numerical_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_processed.select_dtypes(include='object').columns.tolist()
del numerical_cols[0]


for target_variable, feature_var in target_feature_variables:
    cols_for_regression = [target_variable, feature_var]
    df_regression_subset = df_processed[cols_for_regression].copy().dropna()

    if df_regression_subset.empty:
        print(f"Skipping pair '{target_variable}' vs '{feature_var}' due to no valid data after dropna.")
        continue

    X = df_regression_subset[[feature_var]]
    y = df_regression_subset[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_sklearn = LinearRegression()
    model_sklearn.fit(X_train, y_train)

    df_test_sample = X_test.copy()
    df_test_sample['target'] = y_test

    if sample_percentage <= 1:
        df_test_sample = df_test_sample.sample(frac=sample_percentage, random_state=42)
    else:
        df_test_sample = df_test_sample.sample(n=int(sample_percentage), random_state=42)

    X_test_sample = df_test_sample[[feature_var]]
    y_test_sample = df_test_sample['target']

    fig, current_ax = plt.subplots(figsize=(6, 6))
    current_ax.scatter(X_test_sample, y_test_sample, color='blue', label='Données réelles', alpha=0.5)
    x_line = np.linspace(X_test[feature_var].min(), X_test[feature_var].max(), 100).reshape(-1, 1)
    y_pred_line = model_sklearn.predict(x_line)
    current_ax.plot(x_line, y_pred_line, color='red', linewidth=2, label='Ligne de régression')
    current_ax.set_xlabel(feature_var, alpha=0.5)
    current_ax.set_ylabel(target_variable, alpha=0.5)
    current_ax.set_title(f'Régression Linéaire Simple: {target_variable} vs {feature_var}')
    current_ax.legend()
    current_ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Afficher le plot au lieu de l'enregistrer
    plt.show()
    plt.close(fig)
