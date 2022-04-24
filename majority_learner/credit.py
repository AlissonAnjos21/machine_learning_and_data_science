import os
import Orange

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/credit_data_regras'
final_path = os.path.join(dir_path, file_path)

credit_data_base = Orange.data.Table(final_path)

print(credit_data_base.domain)

print('\n\n')

from collections import Counter
print(Counter(str(item.get_class()) for item in credit_data_base))  # Retorna um dicionário onde cada chave é um dos valores que aparecem e seu valor é quantas vezes o valor aparece

print('\n\n')

# O Majority Learner serve basicamente para você ver se vale apena usar um algoritmo de predição. Pois, ele classifica novos registros com base na maioria de registros existentes
# Exemplo: eu tenho um registro, e peço para o majority learner classificá-lo, esse registro receberá a classe presente na maioria dos registros da base de dados.
# Como a parte de testar algoritmos entra nisso? É fácil, se um algoritmo que define novos registros com base na maioria, for mais eficiente que um que realiza predições, então vale mais apena registrar com base na maioria

majority_learner = Orange.classification.MajorityLearner()
not_prediction = Orange.evaluation.testing.TestOnTestData(credit_data_base, credit_data_base, [majority_learner])
print(Orange.evaluation.CA(not_prediction))  # 0.8585 = 85.85%