# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:34:04 2024

@author: moon
"""
git config --global user.name "johannalagoas"
git config --global user.email "johanna.lagoas@agroparistech.fr"

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


## 1- Lecture des fichier 

# Lire le fichier tsv
df_clinique = pd.read_csv(r'C:\Users\messa\Documents\Johanna_Lagoas_Stage\CRUKPAP\clinical_no_Inel.tsv', delimiter='\t')

# Lire le fichier txt et en faire un tableau exploitable
with open(r'C:\Users\messa\Documents\Johanna_Lagoas_Stage\CRUKPAP\dat_vst_NS_no_Inel_selected.txt', 'r') as file:
    lines = file.readlines()

# Extraire les noms des colonnes de la première ligne
original_column_names = lines[0].strip().split(' ')
column_names = ['patientid'] + original_column_names

# Créer une liste pour stocker les données
data = []

# Parcourir les lignes restantes et diviser les éléments
for line in lines[1:]:
    elements = line.strip().split(' ')
    # Insérer l'identifiant du patient en tant que premier élément
    patientid = elements[0]
    row_data = [patientid] + elements[1:]  # Ignorer le premier élément dans les colonnes originales
    data.append(row_data)

# Créer le DataFrame avec les données
df_genes = pd.DataFrame(data, columns=column_names)

df_genes['patientid'] = df_genes['patientid'].str.strip('"')
print(df_genes.head())
print(df_genes.shape)

# Nettoyer les colonnes 'patientid' et 'sampleid' pour supprimer les espaces ou les caractères non souhaités
df_clinique['sampleid'] = df_clinique['sampleid'].str.strip()
df_genes['patientid'] = df_genes['patientid'].str.strip()

# Sélectionner les colonnes souhaitées de df_clinique
clinique_cols = ['sampleid', 'sex', 'age', 'smokingstatusnum', 'packsyearnum']
df_clinique_filtered = df_clinique[clinique_cols]
print(df_clinique_filtered.head())
print(df_clinique_filtered.shape)


# Fusionner les DataFrames sur les colonnes correspondantes ('patientid' et 'sampleid')
df_combined = pd.merge(df_clinique_filtered, df_genes, left_on='sampleid', right_on='patientid')

# Remplacer les valeurs dans la colonne 'sex'
df_combined['sex'] = df_combined['sex'].replace({'male': 1, 'female': 2})

# Remplacer les valeurs dans la colonne 'age'
df_combined['age'] = df_combined['age'].replace({'(24.9,41.5]': 1, '(41.5,58]': 2, '(58,74.5]': 3, '(74.5,91.1]':4})

# Vérifier les premières lignes du DataFrame combiné
print(df_combined.head())
print(df_combined.shape)

##Donc, nos 2 tableaux de base sont df_clinique et df_genes

## 2- Sparation des données 


# Séparer les données cliniques en deux groupes selon la classe `cancer`
df_cancer_clinique = df_clinique.loc[df_clinique["cancer"] == 'cancer'].copy()
df_no_cancer_clinique = df_clinique.loc[df_clinique["cancer"]== 'no cancer'].copy()

# Extraire les identifiants des patients avec cancer
patients_cancer = df_cancer_clinique['sampleid']
patients_sains = df_no_cancer_clinique['sampleid']

# Supprimer les guillemets autour des valeurs dans df_genes
df_genes['patientid'] = df_genes['patientid'].str.strip('"')


# Filtrer les données génétiques pour inclure uniquement les patients atteints de cancer
df_cancer_genes = df_genes[df_genes['patientid'].isin(patients_cancer)]
df_no_cancer_genes = df_genes[df_genes['patientid'].isin(patients_sains)]


## 3- tp3


# Extraire les descripteurs et l'étiquette

features = df_combined.columns.drop(['patientid', 'sampleid'])
X = df_combined[features]
y = df_clinique['cancer']

n_folds = 10
kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Définir la grille des hyperparamètres à tester
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

## Initialiser le modèle logistique 
logistique = LogisticRegression(penalty='l1', solver='liblinear', C = 0.1, max_iter=1000)
'''
# Initialiser la recherche par validation croisée
grid_search = GridSearchCV(estimator=logistique, param_grid=param_grid, cv=5, scoring='accuracy')

# Effectuer la recherche par validation croisée sur les données
grid_search.fit(X, y)

# Afficher les meilleurs paramètres et la meilleure performance
print(f"Meilleurs paramètres : {grid_search.best_params_}")
print(f"Meilleure performance : {grid_search.best_score_:.2f}")
'''

# Effectuer la validation croisée
accuracies = [] ##contient les accuracies des différents tests
auc_score = [] ## contient l'auc de chaque test
for train_index, test_index in kf.split(X): ##kf.split(X_scaled) coupe X en 5 portions et contient 5 couples (x_train, x_test) dans lesquels il y a les indices des groupes qu'on utilise pour train et de celui qu'on teste
## ex : ([1, 2, 3, 4], [5]), ([1, 2, 3, 5], [4])... et on les test tous (5)
    X_train_list = []
    y_train_list = []
    ## on va etraire les echantillons pour le train dans X et y
    for k in train_index : 
        X_train_list.append(X.iloc[k, :])
        y_train_list.append(y.iloc[k])
    X_train = pd.DataFrame(X_train_list)
    y_train = pd.Series(y_train_list)
    
    X_test_list = []
    y_test_list = []
    ## on va extraire les echantillons pour le test dans X et y
    for k in test_index : 
        X_test_list.append(X.iloc[k, :])
        y_test_list.append(y.iloc[k])
    X_test = pd.DataFrame(X_test_list)
    y_test = pd.Series(y_test_list)
    
    logistique.fit(X_train, y_train)
    y_pred = logistique.predict(X_test)
    y_prob = logistique.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    auc = roc_auc_score(y_test, y_prob[:, 1])
    print(f'AUC: {auc:.2f}')
    auc_score.append(auc)

# Calculer la performance moyenne
mean_accuracy = np.mean(accuracies)
print(f'Mean Accuracy: {mean_accuracy:.2f}')

## Calculer l'auc moyenne
mean_auc = np.mean(auc_score)
print(f'Mean Auc: {mean_auc:.2f}')
