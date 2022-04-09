import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Obtendo o caminho ao qual esta o arquivo csv
import os
caminho_arquivo = os.path.dirname(__file__)
caminho_novo = 'credit_data_base\credit_data.csv'
caminho_csv = os.path.join(caminho_arquivo, caminho_novo)

# Leio o arquivo csv com o pandas e atribuo o seu valor a uma variável
base_credit = pd.read_csv(caminho_csv)

# Explicando e classificando as colunas da tabela:
# clientid: id do cliente, é uma variável do tipo categórica e nominal. Pois, apesar do id ser um número, o id equivale ao nome do cliente, além disso, ele é nominal pois um cliente não é melhor que outro de maior número
# income: Renda anual, variável numérica contínua
# age: idade, variável numérica contínua, pois nesse caso, a idade está exibida abrangendo o conjunto dos números reais
# loan: valor da dívida, variável numérica contínua
# default: quem pagou ou não o empréstimo (0 - Pagou / 1 - Não pagou), variável numérica discreta
