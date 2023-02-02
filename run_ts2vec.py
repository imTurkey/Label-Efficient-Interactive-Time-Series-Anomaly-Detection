import os
import argparse
from tqdm import tqdm
import pickle

from ts2vec.ts2vec import TS2Vec
import ts2vec.datautils as datautils
from utils import merge_all_arrays, load_anomaly

def get_all_repr(data):
    all_ts_repr = {}
    for k in tqdm(data):
        fed_data = data[k]
        ts_repr_wom = model.encode(
            fed_data.reshape(1,-1,1),
            sliding_length=1,
            sliding_padding=sliding_padding,
            batch_size=256
        ).squeeze()

        ts_repr = model.encode(
            fed_data.reshape(1,-1,1),
            mask='mask_middle',
            sliding_length=1,
            sliding_padding=sliding_padding,
            batch_size=256
        ).squeeze()
        
        ts_repr = ts_repr - ts_repr_wom
        
        all_ts_repr[k] = ts_repr
    all_ts_repr = merge_all_arrays(all_ts_repr)
    return all_ts_repr

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yahoo', type=str, help='yahoo/kensho/kpi')
parser.add_argument('--train', action='store_true')
args = parser.parse_known_args()[0]
dataset = args.dataset

if not os.path.exists(f"data/ts2vec/{dataset}/"):
    os.makedirs(f"data/ts2vec/{dataset}/")

if dataset == 'yahoo':
    train_data, train_labels, train_timestamps, test_data, test_labels, test_timestamps = load_anomaly("yahoo")
    sliding_padding=200
elif dataset == 'kensho':
    train_data, train_labels, train_timestamps, test_data, test_labels, test_timestamps = load_anomaly("kensho")
    sliding_padding=50
elif dataset == 'kpi_cut':
    train_data, train_labels, train_timestamps, test_data, test_labels, test_timestamps = load_anomaly("kpi_cut")
    sliding_padding=200
elif dataset == 'kpi':
    train_data, train_labels, train_timestamps, test_data, test_labels, test_timestamps = load_anomaly("kpi")
    sliding_padding=400
elif dataset == 'msai_all':
    train_data, train_labels, train_timestamps, test_data, test_labels, test_timestamps = load_anomaly("msai_all")
    sliding_padding=50
else:
    raise


# Norm
for k in train_data:            
    mean, std = train_data[k].mean(), train_data[k].std()
    train_data[k] = (train_data[k] - mean) / std
for k in test_data:            
    mean, std = test_data[k].mean(), test_data[k].std()
    test_data[k] = (test_data[k] - mean) / std

t2v_train_data = datautils.gen_ano_train_data(train_data)
print('train_data:', t2v_train_data.shape)

model = TS2Vec(
        input_dims=t2v_train_data.shape[-1],
        batch_size=1,
        device='cpu',
    )

if args.train:
    print('training...')
    loss_log = model.fit(
            t2v_train_data,
            verbose=True,
            n_epochs=5
        )
    model.save(f"data/ts2vec/{dataset}/{dataset}_ts2vec_model.ckpt")
    print(f"ts2vec model saved to data/ts2vec/{dataset}/{dataset}_ts2vec_model.ckpt")
else:
    try:
        model.load(f"data/ts2vec/{dataset}/{dataset}_ts2vec_model.ckpt")
    except:
        raise('ts2vec model not fit yet!')


def prepare_repr(data, split=''):
    train_repr = get_all_repr(data)
    with open(f"data/ts2vec/{dataset}/{dataset}_{split}_repr.pkl", 'wb') as f:
        pickle.dump(train_repr, f)
        print("train repr saved to", f.name) 

prepare_repr(train_data, split='train')
prepare_repr(test_data, split='test')

