# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:02:46 2024

@author: moon
"""

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
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve, auc


## 1- Lecture des fichier 

# Lire le fichier txt CRUKPAP et en faire un tableau exploitable
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
df_genes_CRUKPAP = pd.DataFrame(data, columns=column_names)

# Lire le fichier tsv CRUKPAP
df_clinique_CRUKPAP = pd.read_csv(r'C:\Users\messa\Documents\Johanna_Lagoas_Stage\CRUKPAP\clinical_no_Inel.tsv', delimiter='\t')


# Lire le fichier tsv AEGIS
df_clinique_AEGIS = pd.read_csv(r'C:\Users\messa\Documents\Johanna_Lagoas_Stage\AEGIS\clindat_w_scores.tsv', delimiter='\t')
df_genes_mauvais = pd.read_csv(r'C:\Users\messa\Documents\Johanna_Lagoas_Stage\AEGIS\genes_w_scores.tsv', delimiter='\t')
df_genes_AEGIS = df_genes_mauvais.transpose()
# Renommer l'axe des colonnes
df_genes_AEGIS = df_genes_AEGIS.rename_axis('patientid', axis=1)

##Enlever les patients pour lesquels on a des NaN AEGIS
col_name = 'packyears'
# Trouver les lignes où la colonne 'packyears' a des NaN
rows_with_nan = df_clinique_AEGIS.loc[df_clinique_AEGIS[col_name].isna()]
# Supprimer les lignes spécifiées (il y en avait 4 donc c'est ok)
for row in rows_with_nan.index: 
    df_clinique_AEGIS = df_clinique_AEGIS.drop(row)
    df_genes_AEGIS = df_genes_AEGIS.drop(row)

# Enlever les genes de CRUKPAP qu'il n'y a pas dans AEGIS (là il faut garder la colonne patientid qu'on utilise après pour la fusion)

# Supprimer les guillemets autour des valeurs dans df_genes_CRUKPAP
df_genes_CRUKPAP.columns = df_genes_CRUKPAP.columns.str.strip('"')

# Obtenir les noms de colonnes de df_genes_AEGIS
colonnes_AEGIS = df_genes_AEGIS.columns

# Conserver la première colonne (patientid) de df_genes_CRUKPAP
patientid_column_CRUKPAP = df_genes_CRUKPAP['patientid']

# Filtrer les colonnes de df_genes_CRUKPAP pour ne garder que celles présentes dans df_genes_AEGIS
df_genes_CRUKPAP = df_genes_CRUKPAP.loc[:, df_genes_CRUKPAP.columns.isin(colonnes_AEGIS)]

# Réintégrer la colonne patientid dans le DataFrame filtré
df_genes_CRUKPAP = pd.concat([patientid_column_CRUKPAP, df_genes_CRUKPAP], axis=1)

# Sélectionner les colonnes souhaitées de df_clinique_CRUKPAP
clinique_cols = ['sex', 'age', 'smokingstatusnum', 'packsyearnum']
df_clinique_filtered = df_clinique_CRUKPAP[clinique_cols].copy()

# Remplacer les valeurs dans la colonne 'sex'
df_clinique_filtered['sex'] = df_clinique_filtered['sex'].replace({'male': 1, 'female': 2})

# Remplacer les valeurs dans la colonne 'age'
df_clinique_filtered['age'] = df_clinique_filtered['age'].replace({'(24.9,41.5]': 1, '(41.5,58]': 2, '(58,74.5]': 3, '(74.5,91.1]':4})



## 3- tp3

# Changer la colonne cancer par des 0 ou 1
print(df_clinique_CRUKPAP['cancer'])
df_clinique_CRUKPAP['cancer'] = df_clinique_CRUKPAP['cancer'].replace({'cancer':1, 'no cancer':0})
print(df_clinique_CRUKPAP['cancer'])

# Extraire les descripteurs et l'étiquette
X = df_clinique_filtered
y = df_clinique_CRUKPAP['cancer']

n_folds = 10
kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=42)
'''
# Définir la fonction pour calculer l'intervalle de confiance par bootstrap
def bootstrap_auc(model, X, y, n_bootstraps=1000, alpha=0.95):
    bootstrapped_aucs = []
    for _ in range(n_bootstraps):
        # Échantillonner avec remplacement
        X_resampled, y_resampled = resample(X, y)
        if len(np.unique(y_resampled)) < 2:
            # Skip this round if the resampled dataset has only one class
            continue

        y_pred_prob = model.predict_proba(X_resampled)[:, 1]
        auc = roc_auc_score(y_resampled, y_pred_prob)
        bootstrapped_aucs.append(auc)

    sorted_scores = np.array(bootstrapped_aucs)
    sorted_scores.sort()

    # Calculer les percentiles
    lower_bound = np.percentile(sorted_scores, ((1.0 - alpha) / 2.0) * 100)
    upper_bound = np.percentile(sorted_scores, (alpha + ((1.0 - alpha) / 2.0)) * 100)
    return lower_bound, upper_bound, bootstrapped_aucs

'''
def bootstrap_confidence_interval(data, num_bootstrap_samples=1000, alpha=0.05):
    # Liste pour stocker les moyennes des échantillons bootstrap
    bootstrap_samples = []
    
    # Générer des échantillons bootstrap et calculer la moyenne de chaque échantillon
    for _ in range(num_bootstrap_samples):
        # Générer un échantillon bootstrap en rééchantillonnant avec remplacement
        sample = resample(data)
        # Calculer la moyenne de l'échantillon bootstrap
        bootstrap_samples.append(np.mean(sample))
    
    # Calculer les percentiles pour déterminer l'intervalle de confiance
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - (alpha / 2)) * 100
    lower_bound = np.percentile(bootstrap_samples, lower_percentile)
    upper_bound = np.percentile(bootstrap_samples, upper_percentile)
    
    return lower_bound, upper_bound


## Initialiser le modèle logistique 
logistique = LogisticRegression(max_iter=1000)

# Effectuer la validation croisée
accuracies = [] ##contient les accuracies des différents tests
auc_score = [] ## contient l'auc de chaque test
auc_pr_score = [] ##contient l'auc pr de chaque test

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
    auc2 = roc_auc_score(y_test, y_prob[:, 1])
    print(f'AUC: {auc2:.2f}')
    auc_score.append(auc2)
    print(y_test)
    
    # Calculer les valeurs de précision, rappel et seuils
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob[:,1])
    # Calculer l'AUC-PR
    auc_pr = auc(recall, precision)
    print(f'AUCPR: {auc_pr:.2f}')
    # Ajouter à la boucle
    auc_pr_score.append(auc_pr)

# Calculer la performance moyenne
mean_accuracy = np.mean(accuracies)
print(f'Mean Accuracy: {mean_accuracy:.2f}')

## Calculer l'auc moyenne
mean_auc = np.mean(auc_score)
print(f'Mean Auc: {mean_auc:.2f}')

# Calculer la variance de l'AUC
auc_variance = np.var(auc_score, ddof=1)
print(f'AUC Variance: {auc_variance:.5f}')

## Calculer l'auc_pr moyenne
mean_auc_pr = np.mean(auc_pr_score)
print(f'Mean AUCPR : {mean_auc_pr:.2f}')

# Calculer l'intervalle de confiance pour l'AUC par bootstrap
lower_bound, upper_bound = bootstrap_confidence_interval(auc_score)
print(f"95% Confidence Interval AUC: [{lower_bound:.3f}, {upper_bound:.3f}]")

# Calculer l'intervalle de confiance pour l'AUCPR par bootstrap
lower_bound, upper_bound = bootstrap_confidence_interval(auc_pr_score)
print(f"95% Confidence Interval AUCPR: [{lower_bound:.3f}, {upper_bound:.3f}]")

'''
# Tracer l'histogramme des AUC bootstrap
plt.figure(figsize=(10, 5))
plt.hist(bootstrapped_aucs, bins=30, alpha=0.7, color='g')
plt.xlabel('AUC')
plt.ylabel('Frequency')
plt.title('Bootstrap AUC Distribution')
plt.grid(True)
plt.show()
'''