import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
caminho_arquivo = os.path.dirname(__file__)
caminho_novo = 'credit_data_base\credit_data.csv'
caminho_csv = os.path.join(caminho_arquivo, caminho_novo)


base_credit = pd.read_csv(caminho_csv)
# Assim como diz o nome, ele retorna descrições dos dados, como algumas médias, desvios, etc
# count - qtd respostas / mean - média / std - desvio padrão
print(base_credit.describe())

print('\n#############################################################################\n')

# Selecionando a pessoa com a maior renda
# Eu rodo o describe e pego o valor da pessoa que ganha mais
print(base_credit[base_credit['income'] >= 69995.685578])

print('\n#############################################################################\n')

# Selecionando a pessoa com a menor dívida
# Faço a mesma coisa do exemplo anterior
print(base_credit[base_credit['loan'] <= 1.377630])
