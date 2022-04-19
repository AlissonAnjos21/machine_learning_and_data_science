import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

caminho_rota = os.path.dirname(__file__)
caminho_arquivo = 'credit_and_census_data_base/credit.pkl'
caminho_final = os.path.join(caminho_rota, caminho_arquivo)

import pickle
with open(caminho_final, 'rb') as f:
    x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f)

print(x_credit_treinamento.shape, y_credit_treinamento.shape)  # Ok
print(x_credit_teste.shape, y_credit_teste.shape)  # Ok
print('\n')

from sklearn.tree import DecisionTreeClassifier
tree_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)
tree_credit.fit(x_credit_treinamento, y_credit_treinamento)
prediction = tree_credit.predict(x_credit_teste)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_credit_teste, prediction))  # 0.982 = 98.2% de acerto

from sklearn.metrics import classification_report
print(classification_report(y_credit_teste, prediction))  # Dados interessantes sobre os resultados

# Analisando graficamente os resultados:
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(tree_credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)
plt.show()

# Representação gráfica da árvore de decisão:
from sklearn import tree
predictors = ['income', 'age', 'loan']
figure, axes = plt.subplots(nrows=1, ncols=1, figsize= (12,12))
tree.plot_tree(tree_credit, feature_names=predictors, class_names=['0','1'], filled=True)
plt.show()
final_tree_path = os.path.join(caminho_rota, 'credit_and_census_images/credit_tree.png')
figure.savefig(final_tree_path)
