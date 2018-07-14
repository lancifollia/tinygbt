from itertools import izip

import numpy as np


class Dataset(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y


class GBDT(object):
    def __init__(self):
        self.params = {}

    def _calc_training_data_scores(self, train_set, models):
        if len(models) == 0:
            return None
        X = train_set.X
        scores = np.zeros(len(X))
        for i in range(len(X)):
            scores[i] = self.predict(X[i], models=models)
        return scores

    def _calc_gradient(self, train_set, scores):
        ''' L2 loss only '''
        labels = train_set.y
        hessian = np.full(len(labels), 2)  # 2: hessian of L2
        if scores is None:
            grad = np.random.uniform(size=len(labels))
        else:
            grad = np.array([2 * (labels[i] - scores[i]) for i in range(len(labels))])
        return grad, hessian

    def _build_learner(self, grad, hessian):
        pass

    def _calc_rmse(self, models, data_set):
        errors = []
        for x, y in izip(data_set.X, data_set.y):
            errors.append(y - self.predict(x, models))
        return np.square(np.mean(np.square(errors)))

    def train(self, params, train_set, num_boost_round=20, valid_set=None, early_stopping_rounds=5):
        self.params.update(params)
        models = []

        for iter_cnt in range(num_boost_round):
            scores = self._calc_training_data_scores(train_set, models)
            grad, hessian = self._calc_gradient(train_set, scores)
            learner = self._build_learner(grad, hessian)
            self.models.append(learner)
            train_error = self._calc_rmse(models, train_set)
            val_error = self._calc_rmse(models, valid_set) if valid_set else None
            print('iter {}, train error: {}, val_error: {}'.format(iter_cnt, train_error, val_error))

        self.models = models

    def predict(self, x, models=None):
        if models is None:
            models = self.models
        assert models is not None

        # TODO: predict

        return 0.0


def gbdt_train(params, train_set, num_boost_round=20, valid_set=None, early_stopping_rounds=5):
    gbdt = GBDT()
    gbdt.train(params, train_set, num_boost_round, valid_set, early_stopping_rounds)
    return gbdt
