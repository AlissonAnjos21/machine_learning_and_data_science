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

# Retorna os diferentes valores existentes na coluna default
print(np.unique(base_credit['default']))  # 0 e 1

# Porém, supunhamos que eu queira saber quantas vezes esses diferentes valores foram colocados
print(np.unique(base_credit['default'], return_counts=True))
