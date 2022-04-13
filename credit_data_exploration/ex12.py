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

# Para predições, normalmente usá-se dois tipos de variáveis, as X que são dos previsores e as Y que são das classes/respostas

# O iloc seleciona linhas e colunas do data frame
X_base = base_credit.iloc[:, 1:4].values  # Primeiro parâmetro é referente a quais linhas pegar (nesse caso todas), já o segundo parâmetro, é relativo a quais colunas pegar (nesse caso, da 1 à 3 (4 não é incluído))
Y_base = base_credit.iloc[:, 4].values  # Colocando a coluna 4 em uma única variável

print(X_base)  # Funcionou!
print(Y_base)  # Funcionou!
