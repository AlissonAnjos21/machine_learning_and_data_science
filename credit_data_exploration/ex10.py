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

# Tratando dados inconsistentes, forma 03:
# Preencher esses valores com a média desse tipo de valor
print(base_credit['age'].mean())  # Imprime a média das idades, porém essa média consta com os valores inconsistentes

print('\n')

print(base_credit['age'][base_credit['age'] > 0].mean())  # Imprime a média das idades, média sem as inconsistências

base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92  # Nos lugares onde age for menor que 0, ela se tornará 40.92

print(base_credit.loc[base_credit['age'] < 0])  # Vazio

print(base_credit.head(27))
