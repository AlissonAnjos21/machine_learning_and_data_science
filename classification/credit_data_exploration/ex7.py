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

# Tratando dados inconsistentes, forma 01:
# Apagar toda a coluna que possui esses dados, junto com todos os registros (inclusive os coerentes):
base_credit_2 = base_credit.drop('age', axis=1)  # axis = 0 - Apaga as linhas / axis = 1 - Apaga as colunas
# print(base_credit_2[base_credit_2['age'] < 0])  # Erro!!! A coluna "age" foi apagada, não é mais possível ver esses registros

print(base_credit_2.head())  # Só possui as colunas clientid, income, loan e default
