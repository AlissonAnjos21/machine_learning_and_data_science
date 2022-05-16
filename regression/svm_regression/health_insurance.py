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

# Para que o SVM fucione em sua melhor potência, é necessário normalizar os dados, pois, ao contrário de alguns outros algoritmos, ele não possui essa função pré-imbutida
from sklearn.preprocessing import StandardScaler
x_standard_scaler = StandardScaler()
x_health_insurance_scaled = x_standard_scaler.fit_transform(x_health_insurance)
y_standard_scaler = StandardScaler()
y_health_insurance_scaled = y_standard_scaler.fit_transform(y_health_insurance.reshape(-1, 1))

from sklearn.svm import SVR
health_insurance_svm_linear_regression = SVR(kernel = 'linear')
health_insurance_svm_linear_regression.fit(x_health_insurance, y_health_insurance)
linear_predictions = health_insurance_svm_linear_regression.predict(x_health_insurance)

health_insurance_svm_polynomial_regression = SVR(kernel = 'poly', degree = 3)
health_insurance_svm_polynomial_regression.fit(x_health_insurance, y_health_insurance)
polynomial_predictions = health_insurance_svm_polynomial_regression.predict(x_health_insurance)

health_insurance_svm_rbf_regression = SVR(kernel = 'rbf')
health_insurance_svm_rbf_regression.fit(x_health_insurance_scaled, y_health_insurance_scaled)
rbf_predictions = health_insurance_svm_rbf_regression.predict(x_health_insurance_scaled)

# Gráficos:
linear_graph = px.scatter(x = x_health_insurance.ravel(), y = y_health_insurance.ravel())
linear_graph.add_scatter(x = x_health_insurance.ravel(), y = linear_predictions, name = 'SVM  Linear Regression')
linear_graph.show()

polynomial_graph = px.scatter(x = x_health_insurance.ravel(), y = y_health_insurance.ravel())
polynomial_graph.add_scatter(x = x_health_insurance.ravel(), y = polynomial_predictions, name = 'SVM  Polynomial Regression')
polynomial_graph.show()

rbf_graph = px.scatter(x = x_health_insurance_scaled.ravel(), y = y_health_insurance_scaled.ravel())
rbf_graph.add_scatter(x = x_health_insurance_scaled.ravel(), y = rbf_predictions, name = 'SVM  RBF Regression')
rbf_graph.show()
