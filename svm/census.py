import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/censo.pkl'
final_path = os.path.join(dir_path, file_path)

import pickle
with open(final_path, 'rb') as f:
    x_censo_training, x_censo_test, y_censo_training, y_censo_test = pickle.load(f)

print('\nVALORES TREINAMENTO:')
print(x_censo_training.shape, y_censo_training.shape)
print('VALORES TESTE:')
print(x_censo_test.shape, y_censo_test.shape)

from sklearn.svm import SVC
svm_censo = SVC(kernel='linear', random_state=0)
svm_censo.fit(x_censo_training, y_censo_training)
prediction = svm_censo.predict(x_censo_test)

from sklearn.metrics import accuracy_score, classification_report
print('\nPERCENTUAL DE ACERTO:')
print(accuracy_score(y_censo_test, prediction))  # 0.8507 = 85.07%
print('\nINFORMAÇÕES DA PREDIÇÃO:')
print(classification_report(y_censo_test, prediction))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(svm_censo)
cm.fit(x_censo_training, y_censo_training)
cm.score(x_censo_test, y_censo_test)
plt.show()
