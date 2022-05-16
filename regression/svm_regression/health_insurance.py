import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(__file__)
file_path = 'data_bases/plano_saude2.csv'
final_path = os.path.join(dir_path, file_path)

health_insurance_base = pd.read_csv(final_path)

x_health_insurance = health_insurance_base.iloc[:, 0:1].values
y_health_insurance = health_insurance_base.iloc[:, 1].values

# Tipos de kernel: linear, polynomial, rbf
from sklearn.svm import SVR
health_insurance_regression = SVR(kernel = 'linear')
health_insurance_regression.fit(x_health_insurance, y_health_insurance)

predictions = health_insurance_regression.predict(x_health_insurance)

# Gráfico:
graph = px.scatter(x = x_health_insurance.ravel(), y = y_health_insurance)
graph.add_scatter(x = x_health_insurance.ravel(), y = predictions, name = 'SVM  Linear Regression')
graph.show()
