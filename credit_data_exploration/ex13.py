from re import X
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

# Tratando dados com os conteúdos aprendidos, para depois padronizá-los

# Substitui os valores de idade negativa pelo valor médio das idades
base_credit.loc[base_credit['age'] < 0, 'age'] = base_credit['age'][base_credit['age'] > 0].mean()
# Preenche os campos de idade nula com a média das idades
base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)

print('\n\nIdades menores que 0: ', base_credit['age'][base_credit['age'] < 0], '\n\n')  # Vazio

X_base = base_credit.iloc[:, 1:4].values
Y_base = base_credit.iloc[:, 4].values

print(X_base)
print(Y_base)

print('\n\n\n\n\n')

print(X_base)  # Percebe-se que existem intervalos muito grandes entre os valores da base de dados

# Para ver tal diferença com mais detalhes, veremos quais são os menores valores das colunas
print('\nMENORES VALORES (ANTES):')
print(X_base[:, 0].min())  # Pega o valor minimo referente à todas as linhas da coluna 0  (renda)
print(X_base[:, 1].min())  # Pega o valor minimo referente à todas as linhas da coluna 1  (idade)
print(X_base[:, 2].min())  # Pega o valor minimo referente à todas as linhas da coluna 2  (dívida)

# Agora veremos os maiores
print('\nMAIORES VALORES (ANTES):')
print(X_base[:, 0].max())
print(X_base[:, 1].max())
print(X_base[:, 2].max())

# O algoritmo de tratamento de dados pode considerar que um número de maior valor é mais importante, por isso, usam-se meios para aproximar os valores (Padronização e Normalização)

# A padronização é recomendada quando se tem outliers na base de dados, ou seja, valores que estão muito fora do padrão (Como por exemplo aqueles valores negativos que foram tratados no começo desse código)

# Embora aqueles valores já tenham sido tratados, ainda existe um grande abismo de diferença entre os valores, por isso, aplicarei aqui a Padronização

from sklearn.preprocessing import StandardScaler
data_scaler = StandardScaler()
X_base = data_scaler.fit_transform(X_base)  # Realiza a padronização dos dados

# Agora todos eles estão na mesma escala
print('\n\nDepois da Padronização:')
print(X_base)

print('\n\n\n')

print('\nMENORES VALORES (DEPOIS):')
print(X_base[:, 0].min())
print(X_base[:, 1].min())
print(X_base[:, 2].min())

print('\nMAIORES VALORES (DEPOIS):')
print(X_base[:, 0].max())
print(X_base[:, 1].max())
print(X_base[:, 2].max())
