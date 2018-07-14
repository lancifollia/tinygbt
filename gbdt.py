from itertools import izip

import numpy as np


class Dataset(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y


class TreeNode(object):
    def __init__(self):
        self.is_leaf = False
        self.left_child = None
        self.right_child = None
        self.split_feature_id = None
        self.split_val = None
        self.weight = None

    def _calc_split_gain(self, G, H, G_l, H_l, G_r, H_r, lambd):
        def calc_term(g, h):
            return np.square(g) / (h + lambd)
        return calc_term(G_l, H_l) + calc_term(G_r, H_r) - calc_term(G, H)

    def _calc_leaf_weight(self, grad, hessian, lambd):
        return np.sum(grad) / (np.sum(hessian) + lambd)

    def build(self, instances, grad, hessian, depth, param):
        if depth > param['max_depth']:
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grad, hessian, param['lambda'])
            return
        G = np.sum(grad)
        H = np.sum(hessian)
        best_gain = 0.
        best_fid = None
        best_val = 0.
        best_left_instance_ids = None
        best_right_instance_ids = None
        for fid in instances.columns:
            G_l, H_l = 0., 0.
            sorted_instance_ids = instances.sort_values(by=fid).index.tolist()
            for j in range(sorted_instance_ids):
                G_l += grad[j]
                H_l += hessian[j]
                G_r = G - G_l
                H_r = H - H_l
                current_gain = self._calc_split_gain(G, H, G_l, H_l, G_r, H_r, param['lambda'])
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_fid = fid
                    best_val = instances.iloc[sorted_instance_ids[j]][fid]
                    best_left_instance_ids = sorted_instance_ids[:j+1]
                    best_right_instance_ids = sorted_instance_ids[j+1:]
        if best_gain < param['min_split_gain']:
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grad, hessian, param['lambda'])
        else:
            self.split_feature_id = best_fid
            self.split_val = best_val

            self.left_child = TreeNode()
            self.left_child.build(instances.iloc[best_left_instance_ids],
                                  grad[best_left_instance_ids],
                                  hessian[best_left_instance_ids],
                                  depth+1, param)

            self.right_child = TreeNode()
            self.right_child.build(instances.iloc[best_right_instance_ids],
                                   grad[best_right_instance_ids],
                                   hessian[best_right_instance_ids],
                                   depth+1, param)

    def predict(self, x):
        if self.is_leaf:
            return self.weight
        else:
            if x[self.split_feature_id] <= self.split_val:
                return self.left_child.predict(x)
            else:
                return self.right_child.predict(x)


class Tree(object):
    ''' Classification and regression tree for tree ensemble '''
    def __init__(self):
        self.root = None

    def build(self, instances, grad, hessian, param):
        assert len(X) == len(grad) == len(hessian)
        self.root = TreeNode()
        current_depth = 0
        self.root.build(instances, grad, hessian, current_depth, param)

    def predict(self, x):
        current_node = self.node
        return self.node.predict(x)


class GBDT(object):
    def __init__(self):
        self.params = {'gamma': 0.,
                       'lambda': 1.,
                       'min_split_gain': 0.5,
                       'max_depth': 5,
                       }

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
        hessian = np.full(len(labels), 2)
        if scores is None:
            grad = np.random.uniform(size=len(labels))
        else:
            grad = np.array([2 * (labels[i] - scores[i]) for i in range(len(labels))])
        return grad, hessian

    def _build_learner(self, train_set, grad, hessian):
        learner = Tree()
        learner.build(train_set.X, grad, hessian, self.param)
        return learner

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
            learner = self._build_learner(train_set, grad, hessian)
            self.models.append(learner)
            train_error = self._calc_rmse(models, train_set)
            val_error = self._calc_rmse(models, valid_set) if valid_set else None
            print('iter {}, train error: {}, val_error: {}'.format(iter_cnt, train_error, val_error))

        self.models = models

    def predict(self, x, models=None):
        if models is None:
            models = self.models
        assert models is not None
        return np.sum(m.predict(x) for m in models)


def gbdt_train(params, train_set, num_boost_round=20, valid_set=None, early_stopping_rounds=5):
    gbdt = GBDT()
    gbdt.train(params, train_set, num_boost_round, valid_set, early_stopping_rounds)
    return gbdt
