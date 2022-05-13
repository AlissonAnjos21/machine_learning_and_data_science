import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(__file__)
file_path = 'data_bases/house_prices.csv'
final_path = os.path.join(dir_path, file_path)

house_price_base = pd.read_csv(final_path)

x_house_price = house_price_base.iloc[:, 5:6].values 
y_house_price = house_price_base.iloc[:, 2].values
from sklearn.model_selection import train_test_split
x_house_price_training, x_house_price_test, y_house_price_training, y_house_price_test = train_test_split(x_house_price, y_house_price, test_size = 0.3, random_state = 0)
from sklearn.linear_model import LinearRegression
house_price_linear_regression = LinearRegression()
house_price_linear_regression.fit(x_house_price_training, y_house_price_training)
predictions = house_price_linear_regression.predict(x_house_price_training)

# Criando gráficos
graph = px.scatter(x = x_house_price_training.ravel(), y = predictions)
graph.show()

real_result_graph = px.scatter(x = x_house_price_training.ravel(), y = y_house_price_training)
predictions_result_graph = px.scatter(x = x_house_price_training.ravel(), y = predictions)
predictions_result_graph.data[0].line.color = 'black'
concatenate_graph = go.Figure(data = real_result_graph.data + predictions_result_graph.data)
concatenate_graph.show()
