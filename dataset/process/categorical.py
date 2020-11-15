import json
import os
from typing import Dict, List, Union

from pandas import DataFrame, read_csv, Series
from tqdm import tqdm

CountDict = Dict[Union[int, str], int]
CountDicts = Dict[str, CountDict]


def to_dict(s: Series) -> CountDict:
    keys = s.index
    values = (int(i) for i in s.values)
    return dict(zip(keys, values))


def arrange_key(k: str) -> Union[int, str]:
    if k.isnumeric():
        return int(k)
    else:
        return k


def get_count_dict(json_path: str, categorical_cols: List[str] = None, data: DataFrame = None) -> CountDicts:
    if os.path.isfile(json_path):
        with open(json_path, 'r') as fp:
            count_dicts = json.load(fp)
        count_dicts = {c: {arrange_key(k): v for k, v in d.items()} for c, d in count_dicts.items()}
    elif data and categorical_cols:
        count_dicts = {c: to_dict(data[c].value_counts(sort=True)) for c in tqdm(list(data)) if c in categorical_cols}
        with open(json_path, 'w') as fp:
            json.dump(count_dicts, fp, indent=2)
    else:
        raise ValueError
    return count_dicts


def transform():
    pass


if __name__ == '__main__':
    data_dir = '../data/avazu'
    _data_path = os.path.join(data_dir, 'train')

    _categorical_cols = [
        'hour',
        'C1',
        'banner_pos',
        'site_id',
        'site_domain',
        'site_category',
        'app_id',
        'app_domain',
        'app_category',
        'device_id',
        'device_ip',
        'device_model',
        'device_type',
        'device_conn_type',
        'C14',
        'C15',
        'C16',
        'C17',
        'C18',
        'C19',
        'C20',
        'C21',
    ]
    _json_path = os.path.join(data_dir, 'train_count.json')
    counts = get_count_dict(json_path=_json_path, categorical_cols=_categorical_cols)
    print(counts)
