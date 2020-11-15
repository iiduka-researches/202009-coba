import json
import os.path
import pickle
from multiprocessing import Pool
from typing import Dict, Sequence

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

TRAIN_FILES = (
    'dataframe-0',
    'dataframe-1',
    'dataframe-2',
    'dataframe-3',
    'dataframe-4',
    'dataframe-5',
    'dataframe-6',
)
TEST_FILES = (
    'dataframe-7',
    'dataframe-8',
    'dataframe-9',
)
TARGET_COL = 'click'
FEATURE_COLS = [
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


def _load_pickle(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def _load(paths: Sequence[str], processes) -> pd.DataFrame:
    with Pool(processes) as p:
        data_imap = p.imap(_load_pickle, paths)
        data = pd.concat(tuple(tqdm(data_imap, total=len(paths), desc='Loading batches')))
    return data


def _load_size_dict(path: str) -> Dict[str, int]:
    with open(path, 'r') as f:
        return json.load(f)


class AvazuDataset(Dataset):
    features = FEATURE_COLS
    size_dict = None

    def __init__(self, train=True, data_dir='dataset/data/avazu/processed/', train_files=TRAIN_FILES,
                 test_files=TEST_FILES, processes=None, size_dict_path='dataset/data/avazu/size.json') -> None:
        if train:
            paths = tuple(os.path.join(data_dir, f) for f in train_files)
        else:
            paths = tuple(os.path.join(data_dir, f) for f in test_files)
        data = _load(paths, processes=processes)
        self.data = data[FEATURE_COLS].values
        self.target = data[TARGET_COL].values.reshape((-1, 1))
        AvazuDataset.size_dict = _load_size_dict(size_dict_path)

        super(Dataset, self).__init__()

    def __getitem__(self, index):
        data, target = self.data[index], self.target[index]
        return data, target

    def __len__(self):
        return len(self.data)
