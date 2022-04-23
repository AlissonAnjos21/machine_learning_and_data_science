import os
import Orange

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/credit_data_regras.csv'
final_path = os.path.join(dir_path, file_path)

credit_data_base = Orange.data.Table(final_path)
print('PEVISORES E CLASSE:')
print(credit_data_base.domain)  # [income, age, loan | default]

print('\n\n\n\n')

# Dividindo a base de dados em base de treinamento e base de testes:
training_and_test_credit_data_base = Orange.evaluation.testing.sample(credit_data_base, n = 0.25)
# Em bases de dados muito grandes não é recomendável printar os dados
print('BASE TESTE:')
print(training_and_test_credit_data_base[0])
print('\nBASE TREINAMENTO:')
print(training_and_test_credit_data_base[1])

training_credit_data_base = training_and_test_credit_data_base[1]  # Base de treinamento
test_credit_data_base = training_and_test_credit_data_base[0]  # Base de teste

print('\n\n')

# Para confirmar se deu certo, basta ver o tamanho de ambos:
print('TAMANHO BASE TREINAMENTO:')
print(len(training_credit_data_base))
print('TAMANHO BASE TESTE:')
print(len(test_credit_data_base))

cn2 = Orange.classification.rules.CN2Learner()
credit_rules = cn2(training_credit_data_base)  # Cria as regras com base nos dados de treinamento

print('\n')

print('AS REGRAS DEFINIDAS FORAM:')
for rule in credit_rules.rule_list:
    print(rule)

print('\n\n')

prediction = Orange.evaluation.testing.TestOnTestData(training_credit_data_base, test_credit_data_base, [lambda testdata: credit_rules])  # Realiza as predições com a base de teste

print('PERCENTUAL DE ACERTO:')
print(Orange.evaluation.CA(prediction))  # 0.9740 = 97.40%
