import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(__file__)
file_path = 'data_bases/plano_saude.csv'
final_path = os.path.join(dir_path, file_path)

health_insurance_base = pd.read_csv(final_path)
print(health_insurance_base)

x_health_insurance = health_insurance_base.iloc[:, 0].values
y_health_insurance = health_insurance_base.iloc[:, 1].values

print('X values:')
print(x_health_insurance)
print()
print('Y values:')
print(y_health_insurance)
