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
import seaborn as sns

warnings.filterwarnings('ignore')
sys.path.append('//anaconda3/lib/python3.7/site-packages')
sys.path.append('/usr/local/lib/python3.7/site-packages')

pd.set_option('display.max_rows', 100)

tqdm.pandas(ncols=100, position=0)


class LGBMKfoldlgbPredictor(object):
    """
    numpy.arrayを渡すと、Stratified_kfoldでValidationして
    学習してくれる。

    [WIP] 回帰分類への対応


    input:
        train_X,train_y:numpy.array(2dim) ,numpy_array(1dim)
        test_X:numpy.array(2dim)
        step:sklearn.Pipelineの引数。
        EX) steps = [
       ('pca', PCA()),
        ('rf', RandomForestClassifier())]

    """

    def __init__(self, n_split, train_X, train_y, test_X=None, steps=None, params=None):
        self.skf = StratifiedKFold(n_splits=n_split)
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.class_num = len(np.unique(self.train_y))
        self.lgbm_params = params
        self.pipeline = Pipeline(steps=steps) if steps else None

    def input_params(self, params):
        self.lgbm_params = params

    def predict(self, argmax=True):
        oof = np.zeros((len(self.train_X), self.class_num)) if self.class_num > 2 else np.zeros((len(self.train_X),))
        preds = np.zeros((len(self.test_X), self.class_num)) if self.class_num > 2 else np.zeros((len(self.test_X),))
        FIs = np.zeros(self.train_X.shape[1])
        for train_idx, val_idx in self.skf.split(self.train_X, self.train_y):
            X_train = self.pipeline.fit_transform(self.train_X[train_idx]) if self.pipeline else self.train_X[train_idx]
            X_val = self.pipeline.transform(self.train_X[val_idx]) if self.pipeline else self.train_X[val_idx]
            X_test = self.pipeline.transform(self.test_X) if self.pipeline else self.test_X

            y_train = self.train_y[train_idx]
            y_val = self.train_y[val_idx]

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
            model = lgb.train(self.lgbm_params, lgb_train, valid_sets=lgb_eval)
            if self.class_num == 2:
                oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            else:
                oof[val_idx, :] = model.predict(X_val, num_iteration=model.best_iteration)

            preds += model.predict(X_test, num_iteration=model.best_iteration) / self.skf.n_splits
            FIs += model.feature_importance()

        if argmax:
            oof = oof.argmax(axis=1)
            preds = preds.argmax(axis=1)

        return oof, preds, FIs


def get_oof(clf, x_train, y_train, x_test):
    
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def get_FI_plot(FIs, columns, max_row=50):
    df = pd.DataFrame({'FI': FIs, 'col': columns})
    df = df.sort_values('FI', ascending=False).reset_index(drop=True).iloc[:max_row, :]
    sns.barplot(x='FI', y='col', data=df)
    plt.show()

'''
親ディレクトリのモジュールをインポートする方法
import sys
import pathlib
# base.pyのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path('./').resolve().parent
# モジュールのあるパスを追加
sys.path.append(str(current_dir) + '/models')

from models.get_lgbm_params import get_lgbm_params
from models.predict import LGBMKfoldlgbPredictor
'''
