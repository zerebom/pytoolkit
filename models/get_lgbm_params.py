# src:https://qiita.com/R1ck29/items/50ba7fa5afa49e334a8f


def get_lgbm_params(objective, class_num=10):
    '''
    name:Default:description
    learning_rate:0.1:
    num_iterations:100:木の数
    num_leaves:31:木にある分岐の個数
    max_depth: -1:データが少ないときに過学習を防ぐために設定。デフォはなし


    '''
    if objective == 'binary':
        lgbm_params = {
            'num_iterations': 500,
            'learning_rate': 0.08,
            'objective': 'binary',
            'metric': {'binary_logloss', 'binary_error'},
            'early_stopping_rounds': 30,
            'verbose': 1
        }
    elif objective == 'multiclass':
        lgbm_params = {
            'num_iterations': 500,
            'learning_rate': 0.08,
            'objective': 'multiclass',
            'num_class': class_num,
            'metric': {'multi_logloss', 'multi_error'},
            'early_stopping_rounds': 30,
            'verbose': 1
        }
    else:
        lgbm_params = {
            'num_iterations': 500,
            'learning_rate': 0.08,
            'objective': 'multiclass',
            'num_class': class_num,
            'metric': {'multi_logloss', 'multi_error'},
            'early_stopping_rounds': 30,
            'verbose': 1
        }
    return lgbm_params
