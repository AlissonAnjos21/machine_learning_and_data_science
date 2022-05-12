from tkinter import Y
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(__file__)
file_path = 'data_bases/plano_saude.csv'
final_path = os.path.join(dir_path, file_path)

health_insurance_base = pd.read_csv(final_path)
print(health_insurance_base)

x_health_insurance = health_insurance_base.iloc[:, 0].values
y_health_insurance = health_insurance_base.iloc[:, 1].values

print('X values:')
print(x_health_insurance)
print()
print('Y values:')
print(y_health_insurance)

print()
print()
print()

print(np.corrcoef(x_health_insurance, y_health_insurance))  # Retorna uma matriz mostrando a correlação entre os parâmetros informados

# Os valores de x estão em formato vetorial, ou seja, estão em apenas uma dimensão. Para realizar as manipulações é necessário transformá-lo em uma matriz

print('\nX Shape:')
print(x_health_insurance.shape)
print('\nY Shape:')
print(y_health_insurance.shape)

x_health_insurance = x_health_insurance.reshape(-1, 1)  # -1: Ele ajusta com o que sobrar contanto que a condição do outro parâmetro seja cumprida

print('\nReshaped X:')
print(x_health_insurance)
print(x_health_insurance.shape)

from sklearn.linear_model import LinearRegression
linear_regressor_health_insurance = LinearRegression()
linear_regressor_health_insurance.fit(x_health_insurance, y_health_insurance)


