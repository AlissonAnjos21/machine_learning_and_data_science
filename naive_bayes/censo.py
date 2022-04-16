import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

caminho_rota = os.path.dirname(__file__)
caminho_arquivo = 'credit_and_census_data_base/censo.pkl'
caminho_final = os.path.join(caminho_rota, caminho_arquivo)

import pickle

with open(caminho_final, 'rb') as f:
    x_censo_treinamento, x_censo_teste, y_censo_treinamento, y_censo_teste = pickle.load(f)

print(x_censo_treinamento.shape, x_censo_teste.shape)  # Ok
print(y_censo_treinamento.shape, y_censo_teste.shape)  # Ok

from sklearn.naive_bayes import GaussianNB
