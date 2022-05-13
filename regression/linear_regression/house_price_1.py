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
