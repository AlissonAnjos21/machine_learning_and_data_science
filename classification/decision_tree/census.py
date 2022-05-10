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

print(x_censo_treinamento.shape, y_censo_treinamento.shape)
print(x_censo_teste.shape, y_censo_teste.shape)
print('\n')

from sklearn.tree import DecisionTreeClassifier
tree_censo = DecisionTreeClassifier(criterion='entropy', random_state=0)
tree_censo.fit(x_censo_treinamento, y_censo_treinamento)
prediction = tree_censo.predict(x_censo_teste)

from sklearn.metrics import accuracy_score, classification_report
print('PERCENTUAL DE ACERTO:')
print(accuracy_score(y_censo_teste, prediction))  # 0.8104 = 81.04%
print('\nINFORMAÇÕS SOBRE AS PREDIÇÕES:')
print(classification_report(y_censo_teste, prediction))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(tree_censo)
cm.fit(x_censo_treinamento, y_censo_treinamento)
cm.score(x_censo_teste, y_censo_teste)
plt.show()
