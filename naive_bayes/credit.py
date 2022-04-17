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

print(x_credit_treinamento.shape, x_credit_teste.shape)  # Ok
print(y_credit_treinamento.shape, y_credit_teste.shape)  # Ok

from sklearn.naive_bayes import GaussianNB

naive_credit = GaussianNB()  # Instancio um objeto GaussianNB
naive_credit.fit(x_credit_treinamento, y_credit_treinamento)  # Informo os dados que serão utilizados para treinamento

predicao = naive_credit.predict(x_credit_teste)  # Tento predizer qual será o resultado, a partir dos dados de teste

print('DADOS REAIS:')
print(y_credit_teste)
print('DADOS OBTIDOS POR PREDIÇÃO:')
print(predicao)
print('\n\n')

# Após uma rápida observação dos dados já é possível saber que não foi possível acertar todos os resultados
# Para ver de forma mais clara o percentual de acerto, usaremos isto:

from sklearn.metrics import accuracy_score

# Informo os resultados de teste, ou seja, os dados que são os resultados reais, e os dados obtidos por predição.
print(accuracy_score(y_credit_teste, predicao))  # 0.938 = 93.8% de acerto
print('\n')

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_credit_teste, predicao))  # Retorna uma matriz contendo quais dos valores que eram pra ser ele apontou que eram (acertou) e que não eram (errou). Além dos valores que não eram para ser que ele apontou que eram (errou) e que não eram (acertou).
print('\n\n')

# Uma maneira gráfica que apresenta melhor os resultados e a explicação anterior:

from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(naive_credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)  # Funciona de maneira semelhante ao que foi explicado anteriormente
cm.score(x_credit_teste, y_credit_teste)  # Após receber o x_teste ele realiza os cálculos da técnica Naive Bayes e após isso compara com o resultado oficial, ou seja, com o y_teste.
# As interseções onde os números laterais são iguais, são os casos que o programa acertou
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_credit_teste, predicao))  # Informa alguns dados interessantes sobre os acertos e os erros da predição
