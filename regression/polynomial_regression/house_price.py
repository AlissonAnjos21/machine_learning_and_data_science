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

from sklearn.preprocessing import PolynomialFeatures
polynomial = PolynomialFeatures(degree = 2)

x_house_price_training_polynomial = polynomial.fit_transform(x_house_price_training)  # É necessário usar o "fit" apenas na primeira vez, nas próximas podemos usar o "transform" diretamente
x_house_price_test_polynomial = polynomial.transform(x_house_price_test)

from sklearn.linear_model import LinearRegression
house_price_polynomial_regression = LinearRegression()
house_price_polynomial_regression.fit(x_house_price_training_polynomial, y_house_price_training)

print('\nScore with training base:')
print(house_price_polynomial_regression.score(x_house_price_training_polynomial, y_house_price_training))

print('\nScore with test base:')
print(house_price_polynomial_regression.score(x_house_price_test_polynomial, y_house_price_test))

predictions = house_price_polynomial_regression.predict(x_house_price_test_polynomial)

from sklearn.metrics import mean_absolute_error

print('\nMean Absolute Error:')
print(mean_absolute_error(y_house_price_test, predictions))
