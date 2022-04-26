import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/credit.pkl'
final_path = os.path.join(dir_path, file_path)

import pickle
with open(final_path, 'rb') as f:
    x_credit_training, x_credit_test, y_credit_training, y_credit_test = pickle.load(f)

print('\nVALORES TREINAMENTO:')
print(x_credit_training.shape, y_credit_training.shape)
print('VALORES TESTE:')
print(x_credit_test.shape, y_credit_test.shape)


