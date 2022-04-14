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

# Como foi já mencionado anteriormente, os algoritmos de machine learning realizam suas ações baseados em vários tipos de cálculos matemáticos
# Porém, acontece que devido a isso, eles possuem a tendência de atribuir que os números maiores possuem maior significância do que os números menores, embora em muitos casos isso realmente funcione dessa forma, existem aqueles casos onde um número maior não significa nada relativo a esse tipo de assunto
# Um exemplo disso é a quantidade de colunas que uma base de dados possui. Não é porquê o número do índice é 10 que ele é mais importante que o índice 1. São apenas índices diferentes
# Para corrigir isso, existe a técnica chamada OneHotEncoder

# 
