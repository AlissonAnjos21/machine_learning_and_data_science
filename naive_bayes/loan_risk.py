import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

caminho_rota = os.path.dirname(__file__)
caminho_arquivo = 'credit_and_census_data_base/loan_risk.pkl'
caminho_final = os.path.join(caminho_rota, caminho_arquivo)

import pickle

with open(caminho_final, 'rb') as f:
    x_loan_risk, y_loan_risk = pickle.load(f)

print(x_loan_risk.shape)  # (14, 4)
print(y_loan_risk.shape)  # (14,)

print(x_loan_risk)
#               História |       Divída |               Garantias |             Renda
# Boa              0     | Alta    0    | Nenhuma           1     | 0_15          0
# Desconhecida     1     | Baixa   1    | Adequada          0     | 15_35         1
# Ruim             2     |              |                         | acima_35      2

print(y_loan_risk)  # Alto, Moderado, Baixo

# Momento de realizar a predição:

from sklearn.naive_bayes import GaussianNB

naive_loan_risk = GaussianNB()
naive_loan_risk.fit(x_loan_risk, y_loan_risk)  # Recebe os previsores e a classe. Após isso, ele os relaciona

predicao = naive_loan_risk.predict([[0, 0, 1, 2], [2, 0, 0, 0]])  # Valores pegos da tabela acima

# É suposto que o resultado seja Baixo para o primeiro e Moderado para o segundo

print('\n\n\n')

print(predicao)  # Sucesso!!! O primeiro caso foi classificado como risco Baixo, enquanto o segundo como Moderado
