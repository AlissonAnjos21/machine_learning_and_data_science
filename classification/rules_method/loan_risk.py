import os
import Orange

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/risco_credito_regras.csv'
final_path = os.path.join(dir_path, file_path)

loan_risk_base = Orange.data.Table(final_path)
print(loan_risk_base)

print('\n\n')

print('NOMES DAS COLUNAS DA TABELA:')
print(loan_risk_base.domain)

print('\n\n')

cn2 = Orange.classification.rules.CN2Learner()
loan_risk_rules = cn2(loan_risk_base)

# Informa quais regras o algoritmo estabeleceu
for rule in loan_risk_rules.rule_list:
    print(rule)

print('\n\n')

# Criando predições:
prediction = loan_risk_rules([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']])
print('RESULTADO DA PREDIÇÃO:')
print(prediction)

print('\n')

print('CLASSES:')
print(loan_risk_base.domain.class_var.values)  # Alto = índice 0 / Baixo = índice 1 / Moderado = índice 2

print('\n')

# Como o valor retornado pelo predictor é referente ao índice da classe, é só usá-lo para ver a que classe ele pertence
for i in prediction:
    print('Pertence à classe: ' + loan_risk_base.domain.class_var.values[i])  # Imprimirá a classe que pertence a respectiva predição
