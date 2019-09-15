from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import urllib.error
import urllib.request
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import re
import japanize_matplotlib
import gensim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from sklearn.pipeline import Pipeline
import warnings
import lightgbm as lgb

warnings.filterwarnings('ignore')
sys.path.append('//anaconda3/lib/python3.7/site-packages')
sys.path.append('/usr/local/lib/python3.7/site-packages')

pd.set_option('display.max_rows', 100)

tqdm.pandas(ncols=100, position=0)


class KfoldlgbPredictor(object):
    """
    numpy.arrayを渡すと、Stratified_kfoldでValidationして
    学習してくれる。

    [WIP] 二値分類、回帰分類への対応


    input:
        train_X,train_y:numpy.array(2dim) ,numpy_array(1dim)
        test_X:numpy.array(2dim)
        step:sklearn.Pipelineの引数。
        EX) steps = [
       ('pca', PCA()),
        ('rf', RandomForestClassifier())]

    """

    def __init__(self, n_split, train_X, train_y, test_X=None, steps=None):
        self.skf = StratifiedKFold(n_split=n_split)
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.class_num = len(np.unique(self.train_y))
        self.lgbm_params = {
            'num_iterations': 500,
            'learning_rate': 0.04,
            'objective': 'multiclass',
            'num_class': self.class_num,
            'metric': {'multi_logloss', 'multi_error'},
            'early_stopping_rounds': 30,
            'verbose': 1
        }
        self.pipeline = Pipeline(steps=steps)

    def input_params(self, params):
        self.lgbm_params = params

    def predict(self, argmax=True):
        oof = np.zeros((len(self.train_X), self.class_num))
        preds = np.zeros((len(self.test_X), self.class_num))

        for train_idx, val_idx in self.skf.split(self.train_X, self.train_y):
            X_train = self.pipeline.fit_transform(self.train_X[train_idx]) if self.pipeline else self.train_X[train_idx]
            X_val = self.pipeline.transform(self.train_X[val_idx]) if self.pipeline else self.train_X[val_idx]
            X_test = self.pipeline.transform(self.test_X) if self.pipeline else self.test_X

            y_train = self.train_y[train_idx]
            y_val = self.train_y[val_idx]

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
            model = lgb.train(self.lgbm_params, lgb_train, valid_sets=lgb_eval)

            oof[val_idx, :] = model.predict(X_val, num_iteration=model.best_iteration)
            preds += model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits

        if argmax:
            oof = oof.argmax(axis=1)
            preds = preds.argmax(axis=1)

        return oof, preds
