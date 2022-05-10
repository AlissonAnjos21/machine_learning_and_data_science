import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

caminho_arquivo = os.path.dirname(__file__)

# Credit:
caminho_novo_credit = 'census_and_credit_data_base\credit_data.csv'
caminho_csv_credit = os.path.join(caminho_arquivo, caminho_novo_credit)
base_credit = pd.read_csv(caminho_csv_credit)
base_credit.loc[base_credit['age'] < 0, 'age'] = base_credit['age'][base_credit['age'] > 0].mean()
base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)
X_credit = base_credit.iloc[:, 1:4].values
Y_credit = base_credit.iloc[:, 4].values
data_scaler = StandardScaler()
X_credit = data_scaler.fit_transform(X_credit)

# Censo:
caminho_novo_censo = 'census_and_credit_data_base\census.csv'
caminho_csv_censo = os.path.join(caminho_arquivo, caminho_novo_censo)
base_censo = pd.read_csv(caminho_csv_censo)
X_censo = base_censo.iloc[:, 0:14].values
Y_censo = base_censo.iloc[:, 14].values
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_contry = LabelEncoder()
X_censo[:, 1] = label_encoder_workclass.fit_transform(X_censo[:, 1])
X_censo[:, 3] = label_encoder_education.fit_transform(X_censo[:, 3])
X_censo[:, 5] = label_encoder_marital.fit_transform(X_censo[:, 5])
X_censo[:, 6] = label_encoder_occupation.fit_transform(X_censo[:, 6])
X_censo[:, 7] = label_encoder_relationship.fit_transform(X_censo[:, 7])
X_censo[:, 8] = label_encoder_race.fit_transform(X_censo[:, 8])
X_censo[:, 9] = label_encoder_sex.fit_transform(X_censo[:, 9])
X_censo[:, 13] = label_encoder_contry.fit_transform(X_censo[:, 13])
onehotencoder_censo = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
X_censo = onehotencoder_censo.fit_transform(X_censo).toarray()
scaler_censo = StandardScaler()
X_censo = scaler_censo.fit_transform(X_censo)

# Dividindo as bases de treinamento e as bases de teste:

from sklearn.model_selection import train_test_split

# Creio as variáveis de treinamento e de teste da base de dados do credit
X_credit_treinamento, X_credit_teste, Y_credit_treinamento, Y_credit_teste = train_test_split(X_credit, Y_credit, test_size = 0.25, random_state = 0)  # test_size diz respeito a quantos da base de dados total será utilizado para as variáveis de teste (nesse caso 0.25, ou seja, 25%). Já o random_state é que, por padrão, a cada execução ele geraria valores diferentes, já com o random_state definido, sempre que executado o resultado será o mesmo. Isso ajuda em momentos de teste.

# Creio as variáveis de treinamento e de teste da base de dados do censo
X_censo_treinamento, X_censo_teste, Y_censo_treinamento, Y_censo_teste = train_test_split(X_censo, Y_censo, test_size = 0.15, random_state = 0)

print('\nPREVISORES (X):')
print(X_credit.shape, X_censo.shape)
print(X_credit_treinamento.shape, X_censo_treinamento.shape)
print(X_credit_teste.shape, X_censo_teste.shape, '\n\n\n')

print('CLASSES (Y):')
print(Y_credit.shape, Y_censo.shape)
print(Y_credit_treinamento.shape, Y_censo_treinamento.shape)
print(Y_credit_teste.shape, Y_censo_teste.shape, '\n')
