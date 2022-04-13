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

X_base = base_credit.iloc[:, 1:4].values
Y_base = base_credit.iloc[:, 4].values

print(X_base)
print(Y_base)
