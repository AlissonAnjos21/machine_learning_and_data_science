import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/loan_risk.pkl'
final_path = os.path.join(dir_path, file_path)

import pickle
with open(final_path, 'rb') as f:
    x_loan_risk, y_loan_risk = pickle.load(f)

print('QUANTIDADE DE REGISTROS')
print(x_loan_risk.shape, y_loan_risk.shape)

# Para ficar mais fácil, serão utlizadas apenas duas classes, o risco alto e o risco baixo, assim, apagaremos os registros com a classe moderada
print(y_loan_risk)  # Os registros de índices 2, 7 e 11 possuem o valor "moderado", logo, eles serão deletados

