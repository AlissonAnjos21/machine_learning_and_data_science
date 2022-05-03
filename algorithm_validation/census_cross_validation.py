import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import os

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/censo.pkl'
final_path = os.path.join(dir_path, file_path)

import pickle
with open(final_path, 'rb') as f:
    x_censo_training, x_censo_test, y_censo_training, y_censo_test = pickle.load(f)

x_censo = np.concatenate((x_censo_training, x_censo_test), axis = 0)
y_censo = np.concatenate((y_censo_training, y_censo_test), axis = 0)
