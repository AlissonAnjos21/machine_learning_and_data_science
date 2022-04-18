import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

caminho_rota = os.path.dirname(__file__)
caminho_arquivo = 'credit_and_census_data_base/loan_risk.pkl'
caminho_final = os.path.join(caminho_rota, caminho_arquivo)

import pickle

with open(caminho_final, 'rb') as f:
    x_loan_risk, y_loan_risk = pickle.load(f)


