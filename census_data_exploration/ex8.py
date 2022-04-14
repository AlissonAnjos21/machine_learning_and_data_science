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
from sklearn.preprocessing import LabelEncoder
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencoder_censo = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
X_censo = onehotencoder_censo.fit_transform(X_censo).toarray()

# Usando a Padronização para escalonar (padronizar/deixar na mesma escala) os valores Numéricos padrão e os valores Númericos que era Categóricos, pois ainda existem valores Numéricos padrão com o valor muito alto 

# Escalonamento:
from sklearn.preprocessing import StandardScaler
scaler_censo = StandardScaler()

X_censo = scaler_censo.fit_transform(X_censo)  # Padronização concluída

print('\n\n\nBASE DE DADOS COMPLETA:\n', X_censo, '\n\n\n')

print('\n\n\nPRIMEIRA LINHA DA BASE DE DADOS:\n', X_censo[0], '\n\n\n')
