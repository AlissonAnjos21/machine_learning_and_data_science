import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/loan_risk.pkl'
final_path = os.path.join(dir_path, file_path)

import pickle
with open(final_path, 'rb') as f:
    x_loan_risk, y_loan_risk = pickle.load(f)

print('QUANTIDADE DE REGISTROS')
print(x_loan_risk.shape, y_loan_risk.shape)

# Para ficar mais fácil, serão utlizadas apenas duas classes, o risco alto e o risco baixo, assim, apagaremos os registros com a classe moderada
print(y_loan_risk)  # Os registros de índices 2, 7 e 11 possuem o valor "moderado", logo, eles serão deletados

x_loan_risk = np.delete(x_loan_risk, [2, 7, 11], axis=0)
y_loan_risk = np.delete(y_loan_risk, [2, 7, 11], axis=0)

print('\nNOVOS VALORE X:')
print(x_loan_risk)
print('\nNOVOS VALORE Y:')
print(y_loan_risk)
print('\n\n')

from sklearn.linear_model import LogisticRegression
logistic_regression_loan_risk = LogisticRegression(random_state = 0)  # Também pode receber o parâmetro "max_iter" que diz quantas vezes o algoritmo vai rodar para achar os valores de B, por padrão ele é 100.
logistic_regression_loan_risk.fit(x_loan_risk, y_loan_risk)

print(logistic_regression_loan_risk.intercept_)  # Informa o valor de B0
print(logistic_regression_loan_risk.coef_)  # Informa os valores dos B (B1, B2, B3 ... BN)
print('\n\n')

# O resultado precisa ser, respectivamente: Baixo e Alto
prediction = logistic_regression_loan_risk.predict([[0,0,1,2], [2,0,0,0]])
print(prediction)  # ['baixo' 'alto'] (Funcionou!!!
