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

from sklearn.preprocessing import StandardScaler
x_standard_scaler = StandardScaler()
x_house_price_training_standard_scaler = x_standard_scaler.fit_transform(x_house_price_training)
x_house_price_test_standard_scaler = x_standard_scaler.transform(x_house_price_test)
y_standard_scaler = StandardScaler()
y_house_price_training_standard_scaler = y_standard_scaler.fit_transform(y_house_price_training.reshape(-1, 1))
y_house_price_test_standard_scaler = y_standard_scaler.transform(y_house_price_test.reshape(-1, 1))

from sklearn.svm import SVR
house_price_svm_regressor = SVR(kernel = 'rbf')
house_price_svm_regressor.fit(x_house_price_training_standard_scaler, y_house_price_training_standard_scaler.ravel())

print('\nScore with training base:')
print(house_price_svm_regressor.score(x_house_price_training_standard_scaler, y_house_price_training_standard_scaler.ravel()))
print('\nScore with test base:')
print(house_price_svm_regressor.score(x_house_price_test_standard_scaler, y_house_price_test_standard_scaler.ravel()))
