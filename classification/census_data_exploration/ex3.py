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

# Observando os dados por meio de representações gráficas responsivas e associativas a partir de Treemaps:
grafico_de_arvore_1 = px.treemap(base_censo, path=['age', 'income'])
grafico_de_arvore_1.show()

grafico_de_arvore_2 = px.treemap(base_censo, path=['workclass', 'income'])
grafico_de_arvore_2.show()

grafico_de_arvore_3 = px.treemap(base_censo, path=['occupation', 'relationship', 'age'])
grafico_de_arvore_3.show()

grafico_de_arvore_4 = px.treemap(base_censo, path=['workclass', 'age'])
grafico_de_arvore_4.show()

grafico_de_arvore_5 = px.treemap(base_censo, path=['occupation', 'sex'])
grafico_de_arvore_5.show()
