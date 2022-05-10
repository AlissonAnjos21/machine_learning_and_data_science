import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/censo.pkl'
final_path = os.path.join(dir_path, file_path)

import pickle
with open(final_path, 'rb') as f:
    x_censo_training, x_censo_test, y_censo_training, y_censo_test = pickle.load(f)

print(x_censo_training.shape, y_censo_training.shape)
print(x_censo_test.shape, y_censo_test.shape)

from sklearn.neural_network import MLPClassifier
censo_neural_network = MLPClassifier(tol = 0.000010, max_iter = 1000, hidden_layer_sizes = (55, 55), verbose = True, random_state = 0)
censo_neural_network.fit(x_censo_training, y_censo_training)
prediction = censo_neural_network.predict(x_censo_test)

from sklearn.metrics import accuracy_score, classification_report
print('PERCENTUAL DE ACERTO:')
print(accuracy_score(y_censo_test, prediction))  # 0.8198 = 81.98%
print('INFORMAÇÕES DA PREDIÇÃO:')
print(classification_report(y_censo_test, prediction))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(censo_neural_network)
cm.fit(x_censo_training, y_censo_training)
cm.score(x_censo_test, y_censo_test)
plt.show()
