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

from sklearn.linear_model import LogisticRegression
logistic_regression_credit = LogisticRegression(random_state=0)
logistic_regression_credit.fit(x_credit_training, y_credit_training)

print('\nVALOR DE B0:')
print(logistic_regression_credit.intercept_)
print('\nVALORES DOS COEFICIENTES (Bs):')
print(logistic_regression_credit.coef_)

prediction = logistic_regression_credit.predict(x_credit_test)

from sklearn.metrics import accuracy_score, classification_report
print('\nPERCENTUAL DE ACERTO:')
print(accuracy_score(y_credit_test, prediction))  # 0.946 = 94.60%
print('\nINFORMAÇÕES DA PREDIÇÃO:')
print(classification_report(y_credit_test, prediction))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(logistic_regression_credit)
cm.fit(x_credit_training, y_credit_training)
cm.score(x_credit_test, y_credit_test)
plt.show()
