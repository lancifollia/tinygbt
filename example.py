import pandas as pd
from sklearn.metrics import mean_squared_error

from tinygbt import Dataset, GBT


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
gbt = GBT()
gbt.train(params,
          train_data,
          num_boost_round=20,
          valid_set=eval_data,
          early_stopping_rounds=5)

print('Start predicting...')
y_pred = []
for x in X_test:
    y_pred.append(gbt.predict(x, num_iteration=gbt.best_iteration))

print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
