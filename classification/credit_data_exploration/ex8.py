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

# Informa todos os index existentes
print(base_credit.index)

# Informa os index dos elementos de base_credit que possuem "age" menor que 0
print(base_credit[base_credit['age'] < 0].index)
