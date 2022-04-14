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

print(base_censo)

print('\n\n\n')

print(base_censo.describe())

print('\n\n\n')

print(base_censo.isnull().sum())  # Soma quantos valores nulos existem em cada coluna e retorna (Nesse caso, aparentemente não existe nenhum valor faltante)
