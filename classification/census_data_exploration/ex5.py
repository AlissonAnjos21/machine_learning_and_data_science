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

# Dividindo os previsores e a classe

# Relembrando: 
# Classe é o que eu quero prever. Ex: quero prever se algo vai ser x ou y
# Previsores é o que eu vou utilizar para conseguir prever, é a partir das tendencias desses dados que será possível afirmar com algum grau de certeza que uma coisa será de determinada forma 

X_censo = base_censo.iloc[:, 0:14].values  # Nunca esquecer do ".values", caso não colocá-lo a variável ainda será do tipo pandas. É o ".values" que o converte para o tipo numpy

Y_censo = base_censo.iloc[:, 14].values

print('\nPREVISORES:')
print(X_censo)

print('\n\n\n')

print('CLASSE:')
print(Y_censo)
