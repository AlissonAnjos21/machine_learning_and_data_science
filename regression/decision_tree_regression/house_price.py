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

x_house_price = house_price_base.iloc[:, 3:19].values
y_house_price = house_price_base.iloc[:, 2].values

from sklearn.model_selection import train_test_split
x_house_price_training, x_house_price_test, y_house_price_training, y_house_price_test = train_test_split(x_house_price, y_house_price, test_size = 0.3, random_state = 0)

from sklearn.tree import DecisionTreeRegressor
decision_tree_regressor = DecisionTreeRegressor()
decision_tree_regressor.fit(x_house_price_training, y_house_price_training)

from sklearn.metrics import mean_absolute_error
predictions = decision_tree_regressor.predict(x_house_price_test)
print('\nMean Absolute Error')
print(mean_absolute_error(y_house_price_test, predictions))  # 99673.769

print('\nDecision Tree Training Database Score:')
print(decision_tree_regressor.score(x_house_price_training, y_house_price_training))  # 0.999

print('\nDecision Tree Test Database Score:')
print(decision_tree_regressor.score(x_house_price_test, y_house_price_test))  # 0.718
