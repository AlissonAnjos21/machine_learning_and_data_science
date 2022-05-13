import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(__file__)
file_path = 'data_bases/house_prices.csv'
final_path = os.path.join(dir_path, file_path)

house_price_base = pd.read_csv(final_path)

# Objetivo final: preço da casa através da metragem quadrada

# Explorando a base de dados
print(house_price_base.describe())

print(house_price_base.isnull().sum())  # 0
# Felizmente não existe nenhum valor faltante

# Criando um mapa de calor da correlação:
resize_figure = plt.figure(figsize = (20, 20))
sns.heatmap(house_price_base.corr(), annot = True)
plt.show()

print()
print()
print()

# Matriz de correlação sem mapa de calor:
print(house_price_base.corr())

print()
print()
print()

# Base de dados completa:
print(house_price_base)

print()
print()
print()

# Predição:
# X = metragem quadrada (a partir do que eu vou prever)
x_house_price = house_price_base.iloc[:, 5:6].values  # Ao invés de passar diretamente 5 como valor referente a coluna, eu posso passar 5:6, assim o valor virá em formato de matriz. Isso faz com que eu não precise utilizar a função reshape
print(x_house_price)
print()

# Y = preço (o que eu vou prever)
y_house_price = house_price_base.iloc[:, 2].values
print(y_house_price)
print()

from sklearn.model_selection import train_test_split
x_house_price_training, x_house_price_test, y_house_price_training, y_house_price_test = train_test_split(x_house_price, y_house_price, test_size = 0.3, random_state = 0)

print('\nTraining:')
print(x_house_price_training.shape, y_house_price_training.shape)

print('\nTest:')
print(x_house_price_test.shape, y_house_price_test.shape)

from sklearn.linear_model import LinearRegression
house_price_linear_regression = LinearRegression()
house_price_linear_regression.fit(x_house_price_training, y_house_price_training)

print()

# Percentual de eficácia do algoritmo se comparado com os valores de treinamento:
print(house_price_linear_regression.score(x_house_price_training, y_house_price_training))  # 0.494

# Percentual de eficácia do algoritmo se comparado com os valores de teste
print(house_price_linear_regression.score(x_house_price_test, y_house_price_test))  # 0.488


