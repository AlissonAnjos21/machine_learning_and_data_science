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

print('\n\n', base_credit[base_credit['age'] < 0].index, '\n\n')  # Int64Index([15, 21, 26], dtype='int64') 

# Tratando dados inconsistentes, forma 02:
# Apagar os resistros com valores inconsistentes
base_credit_2 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
print(base_credit_2.head(27))  # Perceba que os elementos com os índices 15, 21 e 26 foram apagados por possuírem idade incosistente 

print('\n\n\n')

# Forma alternativa: print(base_credit_2.loc[base_credit_2['age'] < 0])
print(base_credit_2[base_credit_2['age'] < 0])  # Vazio
