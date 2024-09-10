# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:32:26 2024

@author: moon
"""

import matplotlib.pyplot as plt
from math import sqrt

# Données
modeles = ['Clinique', 'ARN', 'Clinique+ARN']
moyennes = [0.79, 0.91, 0.93]
ecart_types = [sqrt(0.00824), sqrt(0.00299), sqrt(0.00168)]  # écart-type = sqrt(variance)
ci_bas = [0.69, 0.84, 0.82]
ci_haut = [0.95, 1.00, 0.99]

# Calculer les erreurs pour les barres d'erreur
erreurs = [[moy - bas for moy, bas in zip(moyennes, ci_bas)], 
           [haut - moy for haut, moy in zip(ci_haut, moyennes)]]

# Créer le diagramme
plt.figure(figsize=(10, 6))
plt.errorbar(modeles, moyennes, yerr=erreurs, fmt='o', capsize=5, capthick=2, marker='s', markersize=8)

# Ajouter les labels et le titre
plt.xlabel('Modèles')
plt.ylabel('Moyenne de l\'AUC')
plt.title('Comparaison des AUC moyens avec intervalles de confiance à 95%')
plt.ylim(0, 1.0)

# Afficher le diagramme
plt.grid(True)
plt.show()