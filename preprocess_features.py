import os
import argparse
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from features import get_features
from utils import load_anomaly, merge_all_arrays, mean_std_normalize

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yahoo', type=str, help='yahoo/kensho/kpi')
args = parser.parse_known_args()[0]
dataset = args.dataset

if not os.path.exists(f"data/data_preprocessed/{dataset}"):
    os.mkdir(f"data/data_preprocessed/{dataset}")

train_data, train_labels, train_timestamps, test_data, test_labels, test_timestamps = load_anomaly(dataset)


def prepare_features(data, timestamps, split=''):
    res = {}
    # Join uad score as features
    with open(f'data/data_preprocessed/{dataset}/{dataset}_uad_{split}_score.pkl', 'rb') as f:
        uads = pickle.load(f)

    for column in uads.columns:
        uads[column] = mean_std_normalize(uads[column])

    for k in tqdm(data):
        names, features = get_features(timestamps[k], data[k], with_sr=False)
        features = np.array(features).transpose()
        features = pd.DataFrame(features, columns=names)
    
        res[k] = features

    res = merge_all_arrays(res)
    res = pd.concat([res, uads], axis=1)
    res = res.fillna(0)
    with open(f"data/data_preprocessed/{dataset}/{dataset}_{split}_features.pkl", 'wb') as f:
        pickle.dump(res, f)
        print("dnn features saved to", f.name)

prepare_features(train_data, train_timestamps, split='train')
prepare_features(test_data, test_timestamps, split='test')
