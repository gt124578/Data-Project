# Data-Project

Traitement de Données avec notre dataset

Ce projet présente les bases du traitement de données à l’aide du fichier `dataset.csv`. Vous trouverez ci-dessous les principales étapes et notions abordées.

---

## 1. Présentation du Dataset

Le fichier `dataset.csv` contient les données sur lesquelles nous allons appliquer différentes techniques de traitement de données.  
➡️ Assurez-vous d’avoir ce fichier à la racine du projet.

---

## 2. Chargement et Exploration des Données

Pour travailler avec les données, nous utilisons généralement la bibliothèque `pandas`.

```python
import pandas as pd

# Charger le dataset
df = pd.read_csv('dataset.csv')

# Aperçu des premières lignes
print(df.head())
