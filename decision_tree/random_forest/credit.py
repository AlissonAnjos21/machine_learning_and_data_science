import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

caminho_rota = os.path.dirname(__file__)
caminho_arquivo = '../credit_and_census_data_base/credit.pkl'
caminho_final = os.path.join(caminho_rota, caminho_arquivo)

import pickle
with open(caminho_final, 'rb') as f:
    x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f)

print(x_credit_treinamento.shape, y_credit_treinamento.shape)
print(x_credit_teste.shape, y_credit_teste.shape)
