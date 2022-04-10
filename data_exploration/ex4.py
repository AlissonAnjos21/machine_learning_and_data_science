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

# Gera um gráfico de barras mostrando a quantidade de cada uma das possibilidades de default. Nesse caso, a quantidade de 0 e 1 
sns.countplot(x = base_credit['default'])
# Como, por padrão, o gráfico não aparece na tela, eu preciso chamá-lo
plt.show()

# Faz a separação das idades por um gráfico de barras com intervalos
plt.hist(x = base_credit['age'])
plt.show()

plt.hist(x = base_credit['income'])
plt.show()

plt.hist(x = base_credit['loan'])
plt.show()
