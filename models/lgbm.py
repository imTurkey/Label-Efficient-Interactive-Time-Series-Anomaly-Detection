import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split

class LightGBM():

    def __init__(self, **params):
        self.param = {
            'num_leaves': 200, 
            'objective': 'binary',
            # 'metric':'average_precision',
            'verbosity': -1,
            'is_unbalance': True,
            # 'device': 'gpu'
        }
        self.param.update(params)
 
    def train(self, data, label, weight=None):
        data = np.array(data)
        label = np.array(label)

        X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size = 0.2, random_state = 0)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test)

        train_data = lgb.Dataset(data, label=label, weight=weight)
        # print(self.param)
        self.model = lgb.train(
            self.param, 
            train_data, 
            valid_sets=[valid_data],
            early_stopping_rounds=50,
            verbose_eval=False
        )

    def predict(self, data):
        data = np.array(data)
        pred = self.model.predict(data)

        return pred

        