import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
caminho_arquivo = os.path.dirname(__file__)
caminho_novo = 'census_data_base\census.csv'
caminho_csv = os.path.join(caminho_arquivo, caminho_novo)


base_censo = pd.read_csv(caminho_csv)

# Observando os dados por meio de representações gráficas responsivas e associativas a partir de Categorias paralelas:
grafico_paralelo_1 = px.parallel_categories(base_censo, dimensions=['occupation', 'relationship'])
grafico_paralelo_1.show()

grafico_paralelo_2 = px.parallel_categories(base_censo, dimensions=['workclass', 'occupation', 'income'])
grafico_paralelo_2.show()

grafico_paralelo_3 = px.parallel_categories(base_censo, dimensions=['education', 'income'])
grafico_paralelo_3.show()

grafico_paralelo_4 = px.parallel_categories(base_censo, dimensions=['occupation', 'sex', 'income'])
grafico_paralelo_4.show()

grafico_paralelo_5 = px.parallel_categories(base_censo, dimensions=['occupation', 'race'])
grafico_paralelo_5.show()
