import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(__file__)
file_path = 'data_bases/plano_saude2.csv'
final_path = os.path.join(dir_path, file_path)

health_insurance_base = pd.read_csv(final_path)

x_health_insurance = health_insurance_base.iloc[:, 0:1].values
y_health_insurance = health_insurance_base.iloc[:, 1].values

from sklearn.preprocessing import PolynomialFeatures

polynomial = PolynomialFeatures(degree=4)  # O atributo degree representa o número de vezes que o número será elevado a potência
# Para cada resultado da elevação de uma potência ele gera uma nova coluna para os valores de x, a primeira coluna sempre será 1, a segunda sempre será o próprio valor e as conseguintes serão os resultados das potências
# Ex: degree = 3 | num = 2 | Resultado: [1, 2, 4, 8]

x_health_insurance_polynomial = polynomial.fit_transform(x_health_insurance)

from sklearn.linear_model import LinearRegression
# A regressão polinomial é basicamente a regressão linear só que com mais dados
health_insurance_polynomial_regression = LinearRegression()
health_insurance_polynomial_regression.fit(x_health_insurance_polynomial, y_health_insurance)

new_person = [[21]]
new_person = polynomial.transform(new_person)  # Tranforma para o mesmo formato dos itens de x

print('Predicting a new value:')
print(health_insurance_polynomial_regression.predict(new_person))  # 341.7272769

predictions = health_insurance_polynomial_regression.predict(x_health_insurance_polynomial)

# Gráfico
new_graph = px.scatter(x = x_health_insurance[:, 0], y = y_health_insurance)
new_graph.add_scatter(x = x_health_insurance[:, 0], y = predictions, name = 'Polynomial Regression')
new_graph.show()
