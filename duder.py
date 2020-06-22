import os
import pandas as pd
import seaborn as sns
from fseval.analysis.lib.metrics import compute_interpolated_metrics

# rocdata = []
# for _, dataset in pd.read_csv('./example/descriptor.csv').iterrows():
#     path = os.path.join('./results/test3/roc', dataset.path + '.csv')
#     data = pd.read_csv(path)
#     rocdata.append(data)
# rocdata = pd.concat(rocdata)

# metricdata = rocdata\
#     .groupby(['description', 'ranking_method', 'replica_no'])\
#     .apply(compute_interpolated_metrics)\
#     .reset_index()
# import matplotlib.pyplot as plt
# sns.lineplot(data=metricdata, x='fpr', y='tpr', hue='ranking_method', err_style=None)
# plt.show()
# print('end')


rocdata = []
for _, dataset in pd.read_csv('./example/descriptor.csv').iterrows():
    path = os.path.join('./results/test6/validation', dataset.path + '.csv')
    data = pd.read_csv(path)
    rocdata.append(data)
rocdata = pd.concat(rocdata)

import matplotlib.pyplot as plt
sns.lineplot(data=rocdata, x='n_features', y='score', hue='ranking_method')
plt.show()
print('end')


# from fseval.analysis.lib.stability import getVarianceofStability
