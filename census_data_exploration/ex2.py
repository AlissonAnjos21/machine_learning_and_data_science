import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
caminho_arquivo = os.path.dirname(__file__)
caminho_novo = 'census_data_base\census.csv'
caminho_csv = os.path.join(caminho_arquivo, caminho_novo)


base_censo = pd.read_csv(caminho_csv)

# Observando os dados por meio de representações gráficas:
# Vendo as opções de renda e quantas pessoas responderam cada uma das opções
print(np.unique(base_censo['income'], return_counts=True))

print('\n\n\n\n\n')

# Representando o dado anterior com gráficos:
sns.countplot(base_censo['income'])  # Renda
plt.show()

plt.hist(x = base_censo['age'])  # Idade
plt.show()

plt.hist(x = base_censo['hour-per-week'])  # Horas de trabalho por semana
plt.show()

plt.hist(x = base_censo['education-num'])  # Quantidade de anos estudadando
plt.show()
