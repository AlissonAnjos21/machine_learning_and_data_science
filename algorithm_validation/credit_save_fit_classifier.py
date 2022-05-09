import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/credit.pkl'
final_path = os.path.join(dir_path, file_path)

import pickle
with open(final_path, 'rb') as f:
    x_credit_training, x_credit_test, y_credit_training, y_credit_test = pickle.load(f)

x_credit = np.concatenate((x_credit_training, x_credit_test), axis = 0)
y_credit = np.concatenate((y_credit_training, y_credit_test), axis = 0)

# Agora que eu já sei qual é o melhor algoritmo para ser utilizado, eu usarei toda a base de dados para criar um treinador completo. Assim, agora eu posso realizar a predição de valores externos à base de dados
# Serão salvos os três algoritmos mais eficientes, ou seja, o Neural Network, Decision Tree e o SVM
# Não esquecer de usar os melhores parâmetros que foram obtidos a partir do grind_search

from sklearn.neural_network import MLPClassifier
final_neural_network_classifier = MLPClassifier(activation='relu', batch_size = 56, solver='adam')
final_neural_network_classifier.fit(x_credit, y_credit)

from sklearn.tree import DecisionTreeClassifier
final_decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
final_decision_tree_classifier.fit(x_credit, y_credit)

from sklearn.svm import SVC
final_svm_classifier = SVC(C = 2.0, kernel='rbf', probability=True)
final_svm_classifier.fit(x_credit, y_credit)

# Caminhos
neural_network_file_path = 'credit_and_census_data_base/credit_final_neural_network_classifier.sav'
neural_network_final_file_path = os.path.join(dir_path, neural_network_file_path)

decision_tree_file_path = 'credit_and_census_data_base/credit_final_decision_tree_classifier.sav'
decision_tree_final_file_path = os.path.join(dir_path, decision_tree_file_path)

svm_file_path = 'credit_and_census_data_base/credit_final_svm_classifier.sav'
svm_final_file_path = os.path.join(dir_path, svm_file_path)

# Salvando:
pickle.dump(final_neural_network_classifier, open(neural_network_final_file_path, 'wb'))
pickle.dump(final_decision_tree_classifier, open(decision_tree_final_file_path, 'wb'))
pickle.dump(final_svm_classifier, open(svm_final_file_path, 'wb'))

print('Success!')
