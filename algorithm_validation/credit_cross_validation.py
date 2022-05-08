import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
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

from sklearn.model_selection import cross_val_score, KFold
decision_tree_results = []
random_forest_results = []
knn_results = []
logistic_regression_results = []
svm_results = []
neural_network_results = []

# O valor 30 para testes é bem aceito na comunidade científica
# O recomendado é que se use 10 como número de splits
for i in range(30):
    # Divide a base de dados entre treinamento e teste de 10 maneiras diferentes
    kfold = KFold(n_splits = 10, shuffle = True, random_state = i)  # O valor do random state precisa ser referente a itereção de i

    # Usando os parâmetros optidos a partir do Grind Search
    decision_tree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter = 'best')
    # Obtem 10 resultados diferentes de predições devido as diferentes divisões da base de dados entre treinamento e teste (devido ao KFold)
    decision_tree_scores = cross_val_score(decision_tree, x_credit, y_credit, cv = kfold)  
    decision_tree_results.append(decision_tree_scores.mean())  # Adiciona a média dos 10 valores obtidos pelo cross_val_score na lista de resultados da árvore de decisão

    # Repetirei o mesmo padrão para os demais algoritmos de machine learning:
    random_forest = RandomForestClassifier(criterion = 'entropy', min_samples_leaf = 1, min_samples_split=5, n_estimators = 10)
    random_forest_scores = cross_val_score(random_forest, x_credit, y_credit, cv = kfold)
    random_forest_results.append(random_forest_scores.mean())

    knn = KNeighborsClassifier()
    knn_scores = cross_val_score(knn, x_credit, y_credit, cv = kfold)
    knn_results.append(knn_scores.mean())

    logistic_regression = LogisticRegression(C = 1.0, solver = 'lbfgs', tol = 0.0001)
    logistic_regression_scores = cross_val_score(logistic_regression, x_credit, y_credit, cv = kfold)
    logistic_regression_results.append(logistic_regression_scores.mean())

    svm = SVC(kernel = 'rbf', C = 2.0)
    svm_scores = cross_val_score(svm, x_credit, y_credit, cv = kfold)
    svm_results.append(svm_scores.mean())

    neural_network = MLPClassifier(activation = 'relu', batch_size = 56, solver = 'adam')
    neural_network_scores = cross_val_score(neural_network, x_credit, y_credit, cv = kfold)
    neural_network.append(neural_network_scores.mean())

    print(decision_tree_results)
    print(random_forest_results)
    print(knn_results)
    print(logistic_regression_results)
    print(svm_results)
    print(neural_network_results)
