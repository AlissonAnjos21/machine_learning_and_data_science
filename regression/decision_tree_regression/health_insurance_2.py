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

from sklearn.tree import DecisionTreeRegressor
decision_tree_regressor = DecisionTreeRegressor()
decision_tree_regressor.fit(x_health_insurance, y_health_insurance)

x_health_insurance_test = np.arange(min(x_health_insurance), max(x_health_insurance), 0.1)
x_health_insurance_test = x_health_insurance_test.reshape(-1, 1)

predictions = decision_tree_regressor.predict(x_health_insurance_test)

# Gráfico
new_graph = px.scatter(x = x_health_insurance.ravel(), y = y_health_insurance)
new_graph.add_scatter(x = x_health_insurance_test.ravel(), y = predictions, name = 'Decision Tree Regression')
new_graph.show()  # É possível ver os splits de forma mais detalhada
