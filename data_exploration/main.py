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
