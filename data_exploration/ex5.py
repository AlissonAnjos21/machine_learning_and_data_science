import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
caminho_arquivo = os.path.dirname(__file__)
caminho_novo = 'credit_data_base\credit_data.csv'
caminho_csv = os.path.join(caminho_arquivo, caminho_novo)


base_credit = pd.read_csv(caminho_csv)

# Primeiro passa-se o arquivo, depois passa-se os parâmetros do arquivo
# Color é opcional. Para usá-lo, é preciso fornecer um atributo do arquivo lido, o mesmo ajudará a informar quais daqueles fazem alguma coisa 
grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')  # Usando o color='default' para saber quem não pagou a dívida
grafico.show()

# No exemplo acima, ficou evidente que existem dados bastante incomuns como, por exemplo, idades negativas. Esse tipo de dado é bastante prejudicial para a análise dos dados e, por isso, devem ser tratados

# Outro exemplo:
grafico = px.scatter_matrix(base_credit, dimensions=['income', 'loan'], color='age')  # Usando o color='age' 
grafico.show()
