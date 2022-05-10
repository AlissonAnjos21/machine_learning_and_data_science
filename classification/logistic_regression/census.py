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

print('\nVALORES TREINAMENTO:')
print(x_censo_training.shape, y_censo_training.shape)
print('VALORES TESTE:')
print(x_censo_test.shape, y_censo_test.shape)

from sklearn.linear_model import LogisticRegression
logistic_regression_censo = LogisticRegression(random_state=0)
logistic_regression_censo.fit(x_censo_training, y_censo_training)

print('\nVALOR DE B0:')
print(logistic_regression_censo.intercept_)
print('\nVALORES DOS COEFICIENTES (Bs):')
print(logistic_regression_censo.coef_)

prediction = logistic_regression_censo.predict(x_censo_test)

from sklearn.metrics import accuracy_score, classification_report
print('\nPERCENTUAL DE ACERTO:')
print(accuracy_score(y_censo_test, prediction))  # 0.8495 = 84.95%
print('\nINFORMAÇÕES SOBRE A PREDIÇÃO:')
print(classification_report(y_censo_test, prediction))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(logistic_regression_censo)
cm.fit(x_censo_training, y_censo_training)
cm.score(x_censo_test, y_censo_test)
plt.show()
