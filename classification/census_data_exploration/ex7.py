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
# Porém, acontece que devido a isso, eles possuem a tendência de atribuir que os números maiores possuem maior significatividade do que os números menores, embora em muitos casos realmente funcione dessa forma, existem aqueles casos onde um número maior não significa nada relativo a esse tipo de assunto
# Um exemplo disso é o número do dia da semana. Não é porquê o número do dia da semana é 10 que ele é mais importante que o dia da semana 1. São apenas dias da semana diferentes
# Para corrigir isso, existe a técnica chamada OneHotEncoder

# Como funciona?
# Pense em um dado, ao jogá-lo ele pode fornecer 6 diferentes resultados:
#
# Dado: (1, 2, 3, 4, 5, 6)
#
# Ao invés de usar os números dessa maneira, faz-se assim, imagine que ao jogar o dado eu obtive o valor 4, depois o valor 6 e depois o valor 1:
# Dado:
# 1 2 3 4 5 6
# ___________
# 0 0 0 1 0 0
# 0 0 0 0 0 1
# 1 0 0 0 0 0

print('\n\n\n')

print(len(np.unique(base_censo['relationship'])))  # 6 (Ou seja, possibilidades de respostas diferentes)

print('\n\n\n')

# Cada uma das diferentes opções de resposta dessa coluna será considerada uma sub-coluna
# Quando um dado dentre os possíveis dados não corresponder ao da determinada sub-coluna, ele receberá o valor 0, quando ele corresponder receberá o valor 1. Não é possível ele possuir mais de uma sub-coluna com o valor 1

# Técnica OneHotEncoder:

# Antes de tudo, veremos a quantidade de linhas e colunas da base de dados (Isso será interessante de se saber para ocasiões futuras)
print('ANTES:')
print(X_censo.shape)  # (32561, 14)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder_censo = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')

X_censo = onehotencoder_censo.fit_transform(X_censo).toarray()

print('\n\n\n')

print(X_censo)  # Resultado com todas as colunas Numéricas padrão e as Categóricas que passaram pelo processo de LabelEncoder, anteriormente, e agora pelo processo de OneHotEncoder

print('\n\n\n')

print(X_censo[0])  # Apenas a primeira linha

print('\n\n\n')

# A quantidade de colunas aumentou bastante, é possível visualizar essa mudanção ao usarmos o .shape, novamente, só que agora, após a alteração
print('DEPOIS:')
print(X_censo.shape)  # (32561, 108)
