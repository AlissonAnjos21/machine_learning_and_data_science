import os
import Orange

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/census_regras.csv'
final_path = os.path.join(dir_path, file_path)

censo_base = Orange.data.Table(final_path)

training_and_test_censo_data_base = Orange.evaluation.testing.sample(censo_base, n = 0.25)

training_censo_data_base = training_and_test_censo_data_base[1]
test_censo_data_base = training_and_test_censo_data_base[0]

print('TAMANHO BASE TREINAMENTO:')
print(len(training_censo_data_base))
print('TAMANHO BASE TESTE:')
print(len(test_censo_data_base))

print('\n')

cn2 = Orange.classification.rules.CN2Learner()
censo_rules = cn2(training_censo_data_base)  # Cria as regras

# Não é recomendável printar as regras de uma base de dados muito grande
# for rule in censo_rules.rule_list:
#     print(rule)

prediction = Orange.evaluation.testing.TestOnTestData(training_censo_data_base, test_censo_data_base, [lambda testdata: censo_rules])
print('PERCENTUAL DE ACERTO DAS PREDIÇÕES:')
print(Orange.evaluation.CA(prediction))  # 0.8228 = 82.28%
