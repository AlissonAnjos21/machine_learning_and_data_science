import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

caminho_rota = os.path.dirname(__file__)
caminho_arquivo = '../credit_and_census_data_base/censo.pkl'
caminho_final = os.path.join(caminho_rota, caminho_arquivo)

import pickle
with open(caminho_final, 'rb') as f:
    x_censo_treinamento, x_censo_teste, y_censo_treinamento, y_censo_teste = pickle.load(f)

print(x_censo_treinamento.shape, y_censo_treinamento.shape)
print(x_censo_teste.shape, y_censo_teste.shape)

from sklearn.ensemble import RandomForestClassifier
random_forest_censo = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state = 0)
random_forest_censo.fit(x_censo_treinamento, y_censo_treinamento)
prediction = random_forest_censo.predict(x_censo_teste)

from sklearn.metrics import accuracy_score, classification_report
print('\nPERCENTUAL DE ACERTO:')
print(accuracy_score(y_censo_teste, prediction))  # 0.8507 = 85.07%
print('\nINFORMAÇÕES DA PREDIÇÃO:')
print(classification_report(y_censo_teste, prediction))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(random_forest_censo)
cm.fit(x_censo_treinamento, y_censo_treinamento)
cm.score(x_censo_teste, y_censo_teste)
plt.show()
