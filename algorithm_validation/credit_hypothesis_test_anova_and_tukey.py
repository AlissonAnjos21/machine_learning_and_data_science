import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import os

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/credit_results.pkl'
final_path = os.path.join(dir_path, file_path)

import pickle
with open(final_path, 'rb') as f:
    decision_tree_results, random_forest_results, knn_results, logistic_regression_results, svm_results, neural_network_results = pickle.load(f)

from scipy.stats import f_oneway
# Teste ANOVA:
# Buscando saber se existe uma diferênça significativa entre os resultados (Sim ou Não)
_, p = f_oneway(decision_tree_results, random_forest_results, knn_results, logistic_regression_results, svm_results, neural_network_results)  # O primeiro valor não é conveniente, então será usado apenas o segundo valor, que é referente ao valor de p

print('\nVALOR DE p:')
print(p)

# Valor padrão que indica que a confiança do teste é de 95%
alpha = 0.05

if p > alpha:
    print('A hipótese alternativa foi rejeitada, ou seja, os resultados são muito iguais. Logo, não importa qual algoritmo se use, o resultado será bem próximo')
else:
    print('A hipótese nula foi rejeitada, ou seja, os dados são diferentes. Logo, vale a pena tentar descobrir qual dos algoritmos é o melhor para ser utilizado')

# Teste de Tukey
# Busca qual dos algorimos é o mais eficaz
# Eu crio um dicionário contendo duas chaves, uma delas é a accuracy que contem a concatenação de todos os resultados de cada algoritmo, a outra chave é a algorithm que contém como valor uma lista com 30 valores contendo o nome de cada algoritmo (o 30 é porque cada variável de resultado dos algoritmos contém 30 valores de resultado)
algorithms_results = {
    'accuracy': np.concatenate([decision_tree_results, random_forest_results, knn_results, logistic_regression_results, svm_results, neural_network_results]),
    'algorithm': [
        'decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree', 'decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree','decision_tree', 
        'random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest', 
        'knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn', 
        'logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression','logistic_regression',
        'svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm',
        'neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network','neural_network'
        ]
}

results_data_frame = pd.DataFrame(algorithms_results)

from statsmodels.stats.multicomp import MultiComparison
algorithms_compare = MultiComparison(results_data_frame['accuracy'], results_data_frame['algorithm'])
algorithms_test = algorithms_compare.tukeyhsd()

print(algorithms_test)  # Naqueles campos iguais a True significa que existe uma diferença significativa entre os algoritmos. Logo, entre os dois eu, normalmente, escolherei aquele que possui uma maior média em seu percentual (mean). Se não tiver diferença significativa independente de qual dos algoritmos eu escolha, o resultado obtido não será considerado significante
print()

results = pd.DataFrame({
    'DECISION TREE': decision_tree_results, 
    'RANDOM FOREST': random_forest_results,
    'KNN': knn_results,
    'LOGISTIC REGRESSION': logistic_regression_results,
    'SVM': svm_results,
    'NEURAL NETWORK': neural_network_results
    })

print(results.mean())

algorithms_test.plot_simultaneous()
plt.show()
