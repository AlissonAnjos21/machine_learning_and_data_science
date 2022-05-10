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
file_path = 'credit_data_base/credit.pkl'
final_path = os.path.join(dir_path, file_path)

import pickle
with open(final_path, 'rb') as f:
    x_credit_training, x_credit_test, y_credit_training, y_credit_test = pickle.load(f)

x_credit = np.concatenate((x_credit_training, x_credit_test), axis = 0)
y_credit = np.concatenate((y_credit_training, y_credit_test), axis = 0)

data_test = x_credit[0]
data_test = data_test.reshape(1, -1)

neural_network_path = 'credit_data_base/credit_final_neural_network_classifier.sav'
decision_tree_path = 'credit_data_base/credit_final_decision_tree_classifier.sav'
svm_path = 'credit_data_base/credit_final_svm_classifier.sav'

final_neural_network_path = os.path.join(dir_path, neural_network_path)
final_decision_tree_path = os.path.join(dir_path, decision_tree_path)
final_svm_path = os.path.join(dir_path, svm_path)

neural_network = pickle.load(open(final_neural_network_path, 'rb'))
decision_tree = pickle.load(open(final_decision_tree_path, 'rb'))
svm = pickle.load(open(final_svm_path, 'rb'))

neural_network_result = neural_network.predict(data_test)
decision_tree_result = decision_tree.predict(data_test)
svm_result = svm.predict(data_test)

# Recebendo a probabilidade que cada uma das classes tem para ser escolhida
neural_network_probability = neural_network.predict_proba(data_test)
confidence_neural_network = neural_network_probability.max()

decision_tree_probability = decision_tree.predict_proba(data_test)
confidence_decision_tree = decision_tree_probability.max()

svm_probability = svm.predict_proba(data_test)
confidence_svm = svm_probability.max()  # Dentre a probabilidade das duas classes, escolhe aquela que é maior

min_confidence = 0.999999
algorithms = 0
yes = 0
no = 0

if confidence_neural_network >= min_confidence:
    if neural_network_result[0] == 0:
        yes += 1
    else:
        no += 1
    algorithms += 1

if confidence_decision_tree >= min_confidence:
    if decision_tree_result[0] == 0:
        yes += 1
    else:
        no += 1
    algorithms += 1

if confidence_svm >= min_confidence:
    if svm_result[0] == 0:
        yes += 1
    else:
        no += 1
    algorithms += 1

if no > yes:
    print(f'No. Based in {algorithms}')
elif no == yes:
    print(f'Tie. Based in {algorithms}')
else:
    print(f'Yes. Based in {algorithms}')
