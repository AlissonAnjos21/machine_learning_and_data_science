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

print(x_credit_training.shape, y_credit_training.shape)
print(x_credit_test.shape, y_credit_test.shape)

from sklearn.neighbors import KNeighborsClassifier
# É válido lembrar que para que o algoritmo kNN atinja maior eficiência, é necessário realizar a padronização dos dados. Assim como foi realizado durante os primeiros exemplos da exploração de dados.

knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)  # O n_neighbors diz respeito a quantidade de vizinhos que ele ultilizará, nesse caso 5 vizinhos. O p = 2 informa que o cálculo de aproximação que será usado é o euclidiano.
knn_credit.fit(x_credit_training, y_credit_training)
prediction = knn_credit.predict(x_credit_test)

from sklearn.metrics import accuracy_score, classification_report
print('\nPERCENTUAL DE ACERTO:')
print(accuracy_score(y_credit_test, prediction))  # 0.986 = 98.60%

print('\n\nDADOS DA PREDIÇÃO:')
print(classification_report(y_credit_test, prediction))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(knn_credit)
cm.fit(x_credit_training, y_credit_training)
cm.score(x_credit_test, y_credit_test)
plt.show()