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

# Para uma maior eficácia no uso das redes neurais, também é necessário escalonar os dados
from sklearn.preprocessing import StandardScaler
x_standard_scaler = StandardScaler()
x_health_insurance_scaled = x_standard_scaler.fit_transform(x_health_insurance)
y_standard_scaler = StandardScaler()
y_health_insurance_scaled = y_standard_scaler.fit_transform(y_health_insurance.reshape(-1, 1))

from sklearn.neural_network import MLPRegressor
health_insurance_neural_network = MLPRegressor(max_iter = 1000)
health_insurance_neural_network.fit(x_health_insurance_scaled, y_health_insurance_scaled.ravel())

print('Score:')
print(health_insurance_neural_network.score(x_health_insurance_scaled, y_health_insurance_scaled))  # 0.962

# Gráfico
predictions = health_insurance_neural_network.predict(x_health_insurance_scaled)
neural_network_graph = px.scatter(x = x_health_insurance_scaled.ravel(), y = y_health_insurance_scaled.ravel())
neural_network_graph.add_scatter(x = x_health_insurance_scaled.ravel(), y = predictions, name = 'Neural Network Regression')
neural_network_graph.show()
