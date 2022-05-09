import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import os

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/credit.pkl'
final_path = os.path.join(dir_path, file_path)

import pickle
with open(final_path, 'rb') as f:
    x_credit_training, x_credit_test, y_credit_training, y_credit_test = pickle.load(f)

x_credit = np.concatenate((x_credit_training, x_credit_test), axis = 0)
y_credit = np.concatenate((y_credit_training, y_credit_test), axis = 0)

# Simulando um novo valor para ser predito (Em cenários reais nunca use um valor da base de treinamento)
data_test = x_credit[0]
data_test = data_test.reshape(1, -1)  # -1 Representa um valor não específico, ou seja, ele cumpre a condição do outro parâmetro (nesse caso o 1) e assim ele arruma o outro campo (seu próprio campo) com os dados que sobrarem

neural_network_path = 'credit_and_census_data_base/credit_final_neural_network_classifier.sav'
decision_tree_path = 'credit_and_census_data_base/credit_final_decision_tree_classifier.sav'
svm_path = 'credit_and_census_data_base/credit_final_svm_classifier.sav'

final_neural_network_path = os.path.join(dir_path, neural_network_path)
final_decision_tree_path = os.path.join(dir_path, decision_tree_path)
final_svm_path = os.path.join(dir_path, svm_path)

neural_network = pickle.load(open(final_neural_network_path, 'rb'))
decision_tree = pickle.load(open(final_decision_tree_path, 'rb'))
svm = pickle.load(open(final_svm_path, 'rb'))

# Resposta Real: 0
print('1° NEURAL NETWORK:')
print(neural_network.predict(data_test)) # Resposta Recebida: 0
print('2° DECISION TREE:')
print(decision_tree.predict(data_test)) # Resposta Recebida: 0
print('3° SVM:')
print(svm.predict(data_test)) # Resposta Recebida: 0
