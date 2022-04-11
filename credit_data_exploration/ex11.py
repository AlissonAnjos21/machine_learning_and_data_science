from email.mime import base
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

# Informa os valores de base_credit que são nulos, ou seja, aqueles que não foram preenchidos
print(base_credit.isnull(), '\n\n')

# Informa a soma de valores que são nulos em cada diferente coluna
print(base_credit.isnull().sum(), '\n\n')  # É visível que a coluna idade é a única que possui valores nulos, sendo 3 valores

# Localiza os valores de idade nulos em base_credit
print(base_credit.loc[pd.isnull(base_credit['age'])], '\n\n')
print(base_credit.loc[base_credit['age'].isnull()], '\n\n')  # Equivalente ao exemplo acima

# Preenche os valores nulos de base_credit com a média das idades
base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)  # O inplace=True serve para que ele realmente altere o valor da variável para o novo valor. Caso fosse False, ele apenas alteraria esse valor na memória

print(base_credit.loc[pd.isnull(base_credit['age'])], '\n\n')  # Vazio, ou seja, não existem mais valores nulos

# Localiza aqueles que possuírem um desses ids
print(base_credit.loc[(base_credit['clientid'] == 29) | (base_credit['clientid'] == 31) | (base_credit['clientid'] == 32)], '\n\n')

print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])], '\n\n')  # Equivalente ao código acima, retornando os valores com esses ids
