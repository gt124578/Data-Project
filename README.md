
## Prérequis

*   Python 3.7+
*   Bibliothèques Python listées dans `requirements.txt` (voir section Installation).



1.  **Installez les dépendances :**
    Créez un fichier `requirements.txt` avec le contenu suivant :
    ```
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    chardet
    ```
    Puis installez-les :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

1.  **Placez votre fichier de données** (par exemple, au format CSV) dans le dossier `data/`.
2.  **Configurez le script `scripts/data_processing.py` :**
    *   Modifiez la variable `FILE_PATH` pour qu'elle pointe vers votre fichier de données (e.g., `data/votre_base_de_donnees.csv`).
    *   Modifiez la variable `TARGET_VARIABLE` avec le nom exact de la colonne que vous souhaitez prédire.
    *   Optionnellement, spécifiez une liste de `INITIAL_FEATURES` si vous avez déjà une idée des variables explicatives à utiliser. Sinon, laissez `None` pour utiliser toutes les variables numériques disponibles (sauf la cible).
3.  **Exécutez le script principal :**
    ```bash
    python scripts/data_processing.py
    ```
    Les résultats de l'analyse, les statistiques et les graphiques (sauvegardés par défaut dans le répertoire racine du projet, ou dans `images/` si vous modifiez les chemins de sauvegarde) seront affichés et/ou générés.

## Étapes du Traitement

Le script `data_processing.py` suit les étapes suivantes :

1.  **Chargement des Données :** Lecture du fichier CSV.
2.  **Exploration Initiale :** Affichage des premières lignes, informations sur les types de données, statistiques descriptives, et décompte des valeurs manquantes.
3.  **Nettoyage des Données :**
    *   Imputation des valeurs manquantes pour les colonnes numériques (par la médiane par défaut).
    *   Imputation des valeurs manquantes pour les colonnes catégorielles (par le mode par défaut).
    *   *D'autres étapes de nettoyage peuvent être ajoutées ici (e.g., suppression de doublons, correction de types).*
4.  **Analyse Exploratoire des Données (EDA) :**
    *   Visualisation de la distribution de la variable cible.
    *   Calcul et visualisation de la matrice de corrélation entre les variables numériques.
    *   Scatter plots entre les variables numériques et la variable cible.
    *   Box plots entre les variables catégorielles et la variable cible (si numérique).
5.  **Préparation pour la Régression :**
    *   Sélection des variables (features) et de la variable cible (target).
    *   *Encodage des variables catégorielles (e.g., One-Hot Encoding) si nécessaire (non implémenté par défaut pour la régression linéaire simple, qui attend des entrées numériques).*
    *   Division des données en ensembles d'entraînement et de test.
6.  **Modélisation (Régression Linéaire) :**
    *   Entraînement d'un modèle de régression linéaire.
    *   Prédiction sur les ensembles d'entraînement et de test.
7.  **Évaluation du Modèle :**
    *   Calcul du RMSE (Root Mean Squared Error) et du R² (coefficient de détermination).
    *   Affichage des coefficients du modèle.
    *   Visualisation des prédictions par rapport aux valeurs réelles.

## Pistes d'Amélioration et Prochaines Étapes

*   **Nettoyage avancé :** Détection et traitement des outliers, gestion plus fine des données manquantes.
*   **Feature Engineering :** Création de nouvelles variables pertinentes à partir des variables existantes (e.g., interactions, transformations polynomiales, agrégations).
*   **Encodage des variables catégorielles :** Utiliser One-Hot Encoding ou Label Encoding pour inclure les variables catégorielles dans les modèles de régression.
*   **Sélection de features :** Utiliser des techniques plus avancées (e.g., Recursive Feature Elimination - RFE, sélection basée sur les tests statistiques).
*   **Validation croisée :** Pour une évaluation plus robuste du modèle.
*   **Tester d'autres modèles :** Régression Ridge, Lasso, ElasticNet, ou des modèles non linéaires si les relations ne sont pas linéaires.
*   **Analyse des résidus :** Vérifier les hypothèses de la régression linéaire (normalité, homoscédasticité, indépendance des erreurs).

## Licence

Ce projet est sous licence [Choisissez une licence, e.g., MIT]. Voir le fichier `LICENSE` pour plus de détails (vous devrez créer ce fichier).
