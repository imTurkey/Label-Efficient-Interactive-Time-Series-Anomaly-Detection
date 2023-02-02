import argparse
import sys
import os

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle

cwd = str(Path.cwd())
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yahoo', type=str, help='yahoo/kensho/kpi')
args = parser.parse_known_args()[0]
dataset = args.dataset
if not os.path.exists(f"data/data_preprocessed"):
    os.mkdir("data/data_preprocessed")

print(f'generating {dataset} dataset...')
if dataset == 'yahoo':
    data_dir = os.path.join(cwd,"data/data_raw/Yahoo")
    train_list = np.loadtxt('data/data_split/yahoo_train.txt', dtype=str).tolist()
    test_list = np.loadtxt('data/data_split/yahoo_test.txt', dtype=str).tolist()
elif dataset == 'kpi':
    data_dir = os.path.join(cwd,"data/data_raw/kpi")
    train_list = os.listdir(data_dir + '/Train')
    train_list = ['Train/' + i for i in train_list]
    test_list = os.listdir(data_dir + '/Test')
    test_list = ['Test/' + i for i in test_list]

artifact_dir = os.path.join(cwd,f"data/data_preprocessed/{dataset}/{dataset}.pkl")
if not os.path.exists(f"data/data_preprocessed/{dataset}"):
    os.mkdir(f"data/data_preprocessed/{dataset}")

train_data = {}
train_labels = {}
train_timestamps = {}
test_data = {}
test_labels = {}
test_timestamps = {}

for filename in tqdm(train_list):
    ts = pd.read_csv(os.path.join(data_dir,filename))
    data = np.array(ts['value']).astype(np.float64)
    labels = np.array(ts['label']).astype(int)
    timestamps = np.array(ts['timestamp'])

    train_data[filename] = data
    train_labels[filename] = labels
    train_timestamps[filename] = timestamps

for filename in tqdm(test_list):
    ts = pd.read_csv(os.path.join(data_dir,filename))
    data = np.array(ts['value']).astype(np.float64)
    labels = np.array(ts['label']).astype(int)
    timestamps = np.array(ts['timestamp'])

    test_data[filename] = data
    test_labels[filename] = labels
    test_timestamps[filename] = timestamps

with open(artifact_dir, 'wb') as f:
    pickle.dump({
        'train_data': train_data,
        'train_labels': train_labels,
        'train_timestamps': train_timestamps,
        'test_data': test_data,
        'test_labels': test_labels,
        'test_timestamps': test_timestamps,
    }, f)