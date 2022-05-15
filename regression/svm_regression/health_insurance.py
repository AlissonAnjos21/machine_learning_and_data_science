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