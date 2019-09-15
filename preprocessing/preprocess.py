import json
import pandas as pd

def json2dict(json_path):
    """複雑なJsonファイルは一度Dictにして、その後処理する"""

    with open(json_path, mode='r') as f:
        return json.load(f)


def json2df(json_path, col):
    """key,valueが一対一対応している"""
    with open(json_path, mode='r') as f:
        title_dict = json.load(f)
    return pd.DataFrame.from_dict(title_dict, orient='index', columns=[col]).reset_index()
