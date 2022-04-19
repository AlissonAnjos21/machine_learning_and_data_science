import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

caminho_rota = os.path.dirname(__file__)
caminho_arquivo = 'credit_and_census_data_base/loan_risk.pkl'
caminho_final = os.path.join(caminho_rota, caminho_arquivo)

import pickle
with open(caminho_final, 'rb') as f:
    x_loan_risk, y_loan_risk = pickle.load(f)

print('\nPREVISORES:')
print(x_loan_risk)
print('\n')
print('CLASSES:')
print(y_loan_risk)
print('\n\n\n')

#               História |       Divída |               Garantias |             Renda
# Boa              0     | Alta    0    | Nenhuma           1     | 0_15          0
# Desconhecida     1     | Baixa   1    | Adequada          0     | 15_35         1
# Ruim             2     |              |                         | acima_35      2

from sklearn.tree import DecisionTreeClassifier
tree_loan_risk = DecisionTreeClassifier(criterion='entropy')
tree_loan_risk.fit(x_loan_risk, y_loan_risk)
predict = tree_loan_risk.predict([[0,0,1,2], [2,0,0,0]])
print('PREDIÇÃO:')
print(predict)  # Baixo, Alto

print('\n')

print('GANHO DE INFORMAÇÃO:')
print(tree_loan_risk.feature_importances_)  # Informa o ganho de informação por entropia de cada atributo
print('\nCLASSES EXISTENTES:')
print(tree_loan_risk.classes_)

print('\n\n')

from sklearn import tree
predictors = ['historia', 'dividas', 'garantias', 'renda']
figure, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
# print(tree.plot_tree(tree_loan_risk, feature_names=predictors, class_names=tree_loan_risk.classes_, filled=True)) Imprime Informações úteis sobre o processo de cálculo
tree.plot_tree(tree_loan_risk, feature_names=predictors, class_names=tree_loan_risk.classes_, filled=True)
plt.show()
