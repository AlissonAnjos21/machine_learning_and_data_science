# Buscando valores incosistentes
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

# Informa as linhas daqueles que possuem idade negativa
print(base_credit.loc[base_credit['age'] < 0])

print('\n################################################################\n')

# Faz a mesma coisa só que de outra forma
print(base_credit[base_credit['age'] < 0])
