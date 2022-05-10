import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
caminho_arquivo = os.path.dirname(__file__)
caminho_novo = 'census_data_base\census.csv'
caminho_csv = os.path.join(caminho_arquivo, caminho_novo)


base_censo = pd.read_csv(caminho_csv)
X_censo = base_censo.iloc[:, 0:14].values
Y_censo = base_censo.iloc[:, 14].values

# Tratando atributos categóricos:

# Os algoritmos de machine learning utilizam vários cálculos matemáticos, e não é possível realizar esses cálculos com dados do tipo Categórico (str)
# Por isso, esses dados do tipo Categórico (str), serão convertidos para o tipo Numérico
# Para isso, será usada uma técnica chamada LabelEncoder

from sklearn.preprocessing import LabelEncoder

print('\n\n\nANTES:')
print(X_censo[0])  # Vendo todos os dados da primeira linha da lista

# A partir disso é possível saber quais desses elementos são categóricos
# Logo, agora basta apenas converter esses valores para o tipo Numérico e extrapolar isso para todas as outras linhas

# Crio uma variável label encoder com o nome da coluna e após isso instancio um objeto da classe LabelEncoder
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_contry = LabelEncoder()

# Agora, eu obtenho os dados convertidos para o tipo Numérico a partir da definição dos valores dos campos referentes a essas colunas como o próprio valor, só que agora convertido para o tipo Numérico após passar o valor original para o método de classe "fit_transform()"
# Resumo: o novo valor da coluna, é igual ao valor antigo da coluna após ter passado pelo método fit_transform da classe LabelEncoder

# Faço isso para cada coluna Categórica
X_censo[:, 1] = label_encoder_workclass.fit_transform(X_censo[:, 1])
X_censo[:, 3] = label_encoder_education.fit_transform(X_censo[:, 3])
X_censo[:, 5] = label_encoder_marital.fit_transform(X_censo[:, 5])
X_censo[:, 6] = label_encoder_occupation.fit_transform(X_censo[:, 6])
X_censo[:, 7] = label_encoder_relationship.fit_transform(X_censo[:, 7])
X_censo[:, 8] = label_encoder_race.fit_transform(X_censo[:, 8])
X_censo[:, 9] = label_encoder_sex.fit_transform(X_censo[:, 9])
X_censo[:, 13] = label_encoder_contry.fit_transform(X_censo[:, 13])

print('\n\n')

print('DEPOIS:')
print(X_censo[0])  # Vendo todos os dados da primeira linha da lista após a modificação

print('\n\nTODOS OS VALORES:')
print(X_censo)  # Tudo convertido com sucesso para o tipo Numérico
