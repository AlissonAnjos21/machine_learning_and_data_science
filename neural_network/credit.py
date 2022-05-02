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

from sklearn.neural_network import MLPClassifier
# Para se saber a quantidade de neurônios que deve se utilizar, a convensão é que se use a fórumula: (número de entradas + número de saídas) / 2. Porém, não é obrigatório usar da mesma
credit_neural_network = MLPClassifier(verbose = True, tol = 0.0000100, max_iter = 1500, hidden_layer_sizes = (20, 20), activation = 'relu', solver = 'adam')  # Cria um objeto do tipo rede neural multicamada, que realiza 1500 iterações e possui 2 camadas e 20 neurônios em cada camada, possuindo tolerância de 0.0000100
credit_neural_network.fit(x_credit_training, y_credit_training)
prediction = credit_neural_network.predict(x_credit_test)

print('\n\n')

from sklearn.metrics import accuracy_score, classification_report
print('PERCENTUAL DE ACERTO:') # 0.998 = 99.80%
print(accuracy_score(y_credit_test, prediction))
print('INFORMAÇÕES DE PREDIÇÃO:')
print(classification_report(y_credit_test, prediction))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(credit_neural_network)
cm.fit(x_credit_training, y_credit_training)
cm.score(x_credit_test, y_credit_test)
plt.show()
