import os
import argparse
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from realseries.models.iforest import IForest
from realseries.models.sr import SpectralResidual
from realseries.models.stl import STL
from realseries.models.lumino import Lumino
from realseries.models.rcforest import RCForest

from utils import load_anomaly
cwd = str(Path.cwd())

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yahoo', type=str, help='yahoo/kensho/kpi')
parser.add_argument('--output_label', action='store_true')

args = parser.parse_known_args()[0]
dataset = args.dataset
if args.output_label:
    output_type = 'label'
else:
    output_type = 'score'

if dataset == 'yahoo':
    train_data, train_labels, train_timestamps, test_data, test_labels, test_timestamps = load_anomaly("yahoo")
    tree_size=200
elif dataset == 'kensho':
    train_data, train_labels, train_timestamps, test_data, test_labels, test_timestamps = load_anomaly("kensho")
    tree_size=50
elif dataset == 'kpi_cut':
    train_data, train_labels, train_timestamps, test_data, test_labels, test_timestamps = load_anomaly("kpi_cut")
    tree_size=200
elif dataset == 'kpi':
    train_data, train_labels, train_timestamps, test_data, test_labels, test_timestamps = load_anomaly("kpi")
    tree_size=400
elif dataset == 'msai_all':
    train_data, train_labels, train_timestamps, test_data, test_labels, test_timestamps = load_anomaly("msai_all")
    tree_size=50
else:
    raise

artifact_dir = os.path.join(cwd,f"data/data_preprocessed/{dataset}/{dataset}_artifacts_uad")
if not os.path.exists(artifact_dir):
    os.mkdir(artifact_dir)

processed = os.listdir(artifact_dir)
all_file = set(train_data.keys()) | set(test_data.keys())
to_process = list(set(all_file).difference(set(processed)))

# init models
delay=7
IF = IForest(
    n_estimators=1000,
    max_samples='auto',
    contamination=0.01,
    random_state=0)

STL_model = STL()
lumino_model = Lumino()
rcforest_model = RCForest(
    shingle_size=1,
    num_trees=100,
    tree_size=tree_size
)

start = time.time()
for filename in tqdm(to_process):
    # try:
    print(filename)
    try:
        open(os.path.join(artifact_dir, filename))
        continue
    except:
        pass
    try:
        train_value = train_data[filename]
        train_label = train_labels[filename]
        train_timestamp = train_timestamps[filename]
    except:
        train_value = test_data[filename]
        train_label = test_data[filename]
        train_timestamp = test_data[filename]
    train_set = pd.concat([pd.DataFrame(train_timestamp),
                            pd.DataFrame(train_value),
                            pd.DataFrame(train_label)], axis=1)
    train_set.columns = ["timestamp", "value", "label"]
    train_set['timestamp'] = pd.to_datetime(train_set['timestamp'])
    train_set = train_set.set_index("timestamp")

    train_set_data, train_set_label = train_set.iloc[:, :-1], train_set.iloc[:, -1]

    # IForest

    IF.fit(train_set_data)
    if_score = IF.detect(train_set_data)
    thres = np.percentile(if_score, 99)
    pred_label_if = (if_score > thres)
    # pred_label_if = adjust_predicts(pred_label_if, train_label, delay)

    # Spectral Residual

    SR = SpectralResidual(
        series=np.array(train_set_data).squeeze(),
        threshold=0.7,
        mag_window=200,
        score_window=10)

    pred_label_sr = SR.detect()
    sr_label = np.array(pred_label_sr["isAnomaly"])
    sr_score = np.array(pred_label_sr["score"])
    # sr_label = adjust_predicts(sr_label, train_label, delay)

    # STL

    decomp = STL_model.fit(
        train_set_data,
        period=90,
        lo_frac=0.6,
        lo_delta=0.01
    )
    resid = np.array(decomp["resid"].squeeze())
    stl_score = abs(resid)
    stl_score_norm = (stl_score - np.min(stl_score)) / (np.max(stl_score) - np.min(stl_score))
    stl_label = (stl_score_norm > 0.7)
    # stl_label = adjust_predicts(stl_label, train_label, delay)

    # Lumino

    lumino_score = lumino_model.detect(np.array(train_set_data).squeeze())
    thres = lumino_score.min() + 0.7 * (lumino_score.max() - lumino_score.min())
    lumino_label = (lumino_score > thres)
    # lumino_label = adjust_predicts(lumino_label, train_label, delay)

    # RCForest
    try:
        rcforest_score = rcforest_model.detect(np.array(train_set_data).astype("float64").squeeze())
    except:
        rcforest_score = np.zeros_like(train_set_data)
    thres = rcforest_score.min() + 0.7 * (rcforest_score.max() - rcforest_score.min())
    rcforest_label = (rcforest_score > thres)
    # rcforest_label = adjust_predicts(rcforest_label, train_label, delay)

    train_set.insert(train_set.shape[1], 'iforest_label', pred_label_if.astype(int))
    train_set.insert(train_set.shape[1], 'iforest_score', if_score)
    train_set.insert(train_set.shape[1], 'sr_label', sr_label)
    train_set.insert(train_set.shape[1], 'sr_score', sr_score)
    train_set.insert(train_set.shape[1], 'stl_label', stl_label.astype(int))
    train_set.insert(train_set.shape[1], 'stl_score', stl_score_norm)
    train_set.insert(train_set.shape[1], 'lumino_label', lumino_label.astype(int))
    train_set.insert(train_set.shape[1], 'lumino_score', lumino_score)
    train_set.insert(train_set.shape[1], 'rcforest_label', rcforest_label.astype(int))
    train_set.insert(train_set.shape[1], 'rcforest_score', rcforest_score)


    train_set.to_csv(os.path.join(artifact_dir, filename))
    print("UAD result saved to", os.path.join(artifact_dir, filename))

end = time.time()

print("Time cost: %s minutes"%((end-start)/60))

def prepare_uad(data, split=''):
    # Prepare uad labels
    uad = {}
    for k in data.keys():
        uad[k] = pd.read_csv(os.path.join(artifact_dir, k))
    tmp = iter(uad.values())
    uads = pd.concat(tmp, axis=0, keys=uad.keys())

    if output_type == 'label':
        uad_res = uads.drop(
            columns=["timestamp", "value", "label", "iforest_score", "sr_score",
                    "stl_score", "lumino_score", "rcforest_score"],
            axis=1,
            index=None,
            inplace=False
        )
    else:
        uad_res = uads.drop(
            columns=["timestamp", "value", "label", "iforest_label", "sr_label",
                    "stl_label", "lumino_label", "rcforest_label"],
            axis=1,
            index=None,
            inplace=False
        )

    with open(f'data/data_preprocessed/{dataset}/{dataset}_uad_{split}_{output_type}.pkl', 'wb') as f:
        pickle.dump(uad_res, f)
        print("UAD labels saved to", f.name)

prepare_uad(train_labels, split='train')
prepare_uad(test_labels, split='test')
