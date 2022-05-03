import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import os

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/credit.pkl'
final_path = os.path.join(dir_path, file_path)

import pickle
with open(final_path, 'rb') as f:
    x_credit_training, x_credit_test, y_credit_training, y_credit_test = pickle.load(f)

x_credit = np.concatenate((x_credit_training, x_credit_test), axis = 0)
y_credit = np.concatenate((y_credit_training, y_credit_test), axis = 0)

print('\nNOVA FORMA:')
print(x_credit.shape, y_credit.shape)
print('\n\n')

# A técnica de GridSearch consiste em buscar os melhores parâmetros para serem utilizados em determinado algoritmo de machine learning e em determinada base de dados

# Decision Tree:
decision_tree_parameters = {
'criterion': ['gini', 'entropy'],
'splitter': ['best', 'random'],
'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 5, 10]
}
decision_tree_grid_search = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = decision_tree_parameters)  # Recebe o algoritmo de machine learning e os parâmetros que se deseja encontrar a melhor combinação
decision_tree_grid_search.fit(x_credit, y_credit)
decision_tree_best_parameters = decision_tree_grid_search.best_params_  # Informa a melhor combinação de parâmetros encontrados
decision_tree_best_score = decision_tree_grid_search.best_score_  # Informa o melhor resultado encontrado

print('ÁRVORE DE DECISÃO:')
print('Melhores Parâmetros Encontrados:')
print(decision_tree_best_parameters)
print('Melhor Resultado Encontrado:')
print(decision_tree_best_score)  # 0.983
print('\n\n')

# Random Forest
random_forest_parameters = {
'criterion': ['gini', 'entropy'],
'n_estimators': [10, 40, 100, 150],
'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 5, 10]
}
random_forest_grid_search = GridSearchCV(estimator = RandomForestClassifier(), param_grid = random_forest_parameters)
random_forest_grid_search.fit(x_credit, y_credit)
random_forest_best_parameters = random_forest_grid_search.best_params_
random_forest_best_score = random_forest_grid_search.best_score_

print('RANDOM FOREST:')
print('Melhores Parâmetros Encontrados:')
print(random_forest_best_parameters)
print('Melhor Resultado Encontrado:')
print(random_forest_best_score)  # 0.9875
print('\n\n')