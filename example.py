import pandas as pd

from gbdt import Dataset
from gbdt import gbdt_train


print('Load data...')
df_train = pd.read_csv('./data/regression.train', header=None, sep='\t')
df_test = pd.read_csv('./data/regression.test', header=None, sep='\t')

y_train = df_train[0].values
y_test = df_test[0].values
X_train = df_train.drop(0, axis=1).values
X_test = df_test.drop(0, axis=1).values

train_data = Dataset(X_train, y_train)
eval_data = Dataset(X_test, y_test)

params = {}

print('Start training...')
gbm = gbdt_train(params,
                 train_data,
                 num_boost_round=20,
                 valid_sets=eval_data,
                 early_stopping_rounds=5)

print('Done')
