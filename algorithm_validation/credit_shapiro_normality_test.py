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
file_path = 'credit_and_census_data_base/credit_results.pkl'
final_path = os.path.join(dir_path, file_path)

import pickle
with open(final_path, 'rb') as f:
    decision_tree_results, random_forest_results, knn_results, logistic_regression_results, svm_results, neural_network_results = pickle.load(f)

from scipy.stats import shapiro

# Lembrando que, apenas a segunda coluna que contém o valor de p
print('\nDECISION TREE:')
print(shapiro(decision_tree_results))
sns.displot(decision_tree_results, kind='kde')
plt.show()

print('\nRANDOM FOREST:')
print(shapiro(random_forest_results))
sns.displot(random_forest_results, kind='kde')
plt.show()

print('\nKNN:')
print(shapiro(knn_results))
sns.displot(knn_results, kind='kde')
plt.show()

print('\nLOGISTIC REGRESSION:')
print(shapiro(logistic_regression_results))
sns.displot(logistic_regression_results, kind='kde')
plt.show()

print('\nSVM:')
print(shapiro(svm_results))
sns.displot(svm_results, kind='kde')
plt.show()

print('\nNEURAL NETWORK:')
print(shapiro(neural_network_results))
sns.displot(neural_network_results, kind='kde')
plt.show()
