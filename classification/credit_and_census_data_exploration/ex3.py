import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

# Fazendo o pre-processamento da tabela risco crédito, será usado tudo que foi aprendido (com exceção do OneHotEncoder e do StandardScaler, mas isso é devido apenas ao pequeno porte da tabela, não é como se não desse para tratar, porém ao que se deseja utilizar ela, não é viável)

caminho_arquivo = os.path.dirname(__file__)
caminho_risk = 'credit_and_census_data_base/risco_credito.csv'
caminho_csv_risk = os.path.join(caminho_arquivo, caminho_risk)
base_risk = pd.read_csv(caminho_csv_risk)

x_base_risk = base_risk.iloc[:, 0:4].values
y_base_risk = base_risk.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder

label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

x_base_risk[:, 0] = label_encoder_historia.fit_transform(x_base_risk[:, 0])
x_base_risk[:, 1] = label_encoder_divida.fit_transform(x_base_risk[:, 1])
x_base_risk[:, 2] = label_encoder_garantia.fit_transform(x_base_risk[:, 2])
x_base_risk[:, 3] = label_encoder_renda.fit_transform(x_base_risk[:, 3])

import pickle

caminho_loan_risk = 'credit_and_census_data_base/loan_risk.pkl'
caminho_pkl_loan_risk= os.path.join(caminho_arquivo, caminho_loan_risk)

with open(caminho_pkl_loan_risk, 'wb') as f:
    pickle.dump([x_base_risk, y_base_risk], f)
