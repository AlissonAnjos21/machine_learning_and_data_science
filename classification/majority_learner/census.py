import os
import Orange

dir_path = os.path.dirname(__file__)
file_path = 'credit_and_census_data_base/census_regras'
final_path = os.path.join(dir_path, file_path)

censo_data_base = Orange.data.Table(final_path)

print('\n')
print(censo_data_base.domain)
print('\n\n')

from collections import Counter
print(Counter(str(item.get_class()) for item in censo_data_base))

print('\n\n')

majority_learner = Orange.classification.MajorityLearner()
not_prediction = Orange.evaluation.TestOnTestData(censo_data_base, censo_data_base, [majority_learner])
print('PERCENTUAL DE APARIÇÃO DA MAIORIA:')
print(Orange.evaluation.CA(not_prediction))  # 0.7591 = 75.91%
