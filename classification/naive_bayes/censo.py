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
naive_censo = GaussianNB()
naive_censo.fit(x_censo_treinamento, y_censo_treinamento)
predicao = naive_censo.predict(x_censo_teste)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_censo_teste, predicao))  # 0.4767 = 47.67% de acerto (APENAS)

from sklearn.metrics import classification_report
# Informações interessantes sobre os percentuais de erros e acertos
print(classification_report(y_censo_teste, predicao))

# Vendo graficamente a representação dos erros e acertos obtidos
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(naive_censo)
cm.fit(x_censo_treinamento, y_censo_treinamento)
cm.score(x_censo_teste, y_censo_teste)
plt.show()
