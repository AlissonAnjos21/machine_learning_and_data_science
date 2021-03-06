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
file_path = 'credit_and_census_data_base/censo.pkl'
final_path = os.path.join(dir_path, file_path)

import pickle
with open(final_path, 'rb') as f:
    x_censo_training, x_censo_test, y_censo_training, y_censo_test = pickle.load(f)

x_censo = np.concatenate((x_censo_training, x_censo_test), axis = 0)
y_censo = np.concatenate((y_censo_training, y_censo_test), axis = 0)

from sklearn.model_selection import cross_val_score, KFold
decision_tree_results = []
random_forest_results = []
knn_results = []
logistic_regression_results = []
svm_results = []
neural_network_results = []

for i in range(30):
    kfold = KFold(n_splits = 10, shuffle = True, random_state = i)

    decision_tree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter = 'best')
    decision_tree_scores = cross_val_score(decision_tree, x_censo, y_censo, cv = kfold)  
    decision_tree_results.append(decision_tree_scores.mean())  

    random_forest = RandomForestClassifier(criterion = 'entropy', min_samples_leaf = 1, min_samples_split=5, n_estimators = 10)
    random_forest_scores = cross_val_score(random_forest, x_censo, y_censo, cv = kfold)
    random_forest_results.append(random_forest_scores.mean())

    knn = KNeighborsClassifier()
    knn_scores = cross_val_score(knn, x_censo, y_censo, cv = kfold)
    knn_results.append(knn_scores.mean())

    logistic_regression = LogisticRegression(C = 1.0, solver = 'lbfgs', tol = 0.0001)
    logistic_regression_scores = cross_val_score(logistic_regression, x_censo, y_censo, cv = kfold)
    logistic_regression_results.append(logistic_regression_scores.mean())

    svm = SVC(kernel = 'rbf', C = 2.0)
    svm_scores = cross_val_score(svm, x_censo, y_censo, cv = kfold)
    svm_results.append(svm_scores.mean())

    neural_network = MLPClassifier(activation = 'relu', batch_size = 56, solver = 'adam')
    neural_network_scores = cross_val_score(neural_network, x_censo, y_censo, cv = kfold)
    neural_network_results.append(neural_network_scores.mean())

results = pd.DataFrame({
    'DECISION TREE': decision_tree_results, 
    'RANDOM FOREST': random_forest_results,
    'KNN': knn_results,
    'LOGISTIC REGRESSION': logistic_regression_results,
    'SVM': svm_results,
    'NEURAL NETWORK': neural_network_results
    })

print(results.describe())

results_file = 'credit_and_census_data_base/censo_results.pkl'
final_results_path = os.path.join(dir_path, results_file)
with open(final_results_path, 'wb') as f:
    pickle.dump([decision_tree_results, random_forest_results, knn_results, logistic_regression_results, svm_results, neural_network_results], f)
