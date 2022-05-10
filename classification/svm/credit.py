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

print('\nVALORES TREINAMENTO:')
print(x_credit_training.shape, y_credit_training.shape)
print('VALORES TESTE:')
print(x_credit_test.shape, y_credit_test.shape)

from sklearn.svm import SVC
svm_credit = SVC(kernel='rbf', random_state=0, C=2)  # O kernel é um importante parâmetro que converte medidas não-lineares para medidas lineares, existem vários algoritmos de kernel, nesse caso o usado é o rbf. Já o C é um parâmetro referente ao valor da punição, quanto maior a punição mais eficaz é o algoritmo, porém mais recursos de hardware são requeridos, além de que chega em um ponto que não é possível melhorar mais e qualquer incrimentação no valor é apenas um desperdicio de eficácia de processamento.
svm_credit.fit(x_credit_training, y_credit_training)
prediction = svm_credit.predict(x_credit_test)

from sklearn.metrics import accuracy_score, classification_report
print('\nPERCENTUAL DE ACERTO:')
print(accuracy_score(y_credit_test, prediction))  # 0.988 = 98.80%
print('\nINFORMAÇÕES DA PREDIÇÃO:')
print(classification_report(y_credit_test, prediction))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(svm_credit)
cm.fit(x_credit_training, y_credit_training)
cm.score(x_credit_test, y_credit_test)
plt.show()
