import numpy as np
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt

ZERO = 0.00000001

def merge_all_pd(pd_dict):
    tmp = iter(pd_dict.values())
    all_ = pd.concat(tmp, axis=0, keys=pd_dict.keys())
    return all_

def merge_all_arrays(array_dict):
    for k in array_dict:
        array_dict[k] = pd.DataFrame(array_dict[k])
    tmp = iter(array_dict.values())
    all_ = pd.concat(tmp, axis=0, keys=array_dict.keys())
    return all_
    
def load_anomaly(name):
    with open(f'data/data_preprocessed/{name}/{name}.pkl', 'rb') as f:
        res = pickle.load(f)
    return res['train_data'], res['train_labels'], res['train_timestamps'], res['test_data'], res['test_labels'], res['test_timestamps']

def min_max_normalize(data):
    min_ = data.min()
    max_ = data.max()
    data = (data - min_) / (max_ - min_ + ZERO)
    return data

def limit_min_max_normalize(data, limit):
    min_ = np.percentile(data, limit)
    max_ = np.percentile(data, 100-limit)
    data[data>max_] = max_
    data[data<min_] = min_
    data = (data - min_) / (max_ - min_ + ZERO)
    return data

def tn_fn_fp_tp(pred, target, thres: float=0.5, beta=1.0) -> float:
    pred = (pred<thres).cpu().numpy()
    target = target.bool().cpu().numpy()
    idx = pred * 2 + target
    tn = (idx == 0).sum()  # tn 0+0
    fn = (idx == 1).sum()  # fn 0+1
    fp = (idx == 2).sum()  # fp 2+0
    tp = (idx == 3).sum()  # tp 2+1

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = (1 + beta**2) * (precision * recall) / (
        beta**2 * precision + recall + 1e-7)
    return tn, fn, fp, tp

def f1_metrics(tn, fn, fp, tp, beta=1.0):
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = (1 + beta**2) * (precision * recall) / (
        beta**2 * precision + recall + 1e-7)
    return precision, recall, f1

def plot_precision_recall_curve(pred, target, name):
    display = PrecisionRecallDisplay.from_predictions(target, pred, name=name)
    _ = display.ax_.set_title(f"Precision-Recall curve")
    _ = display.ax_.legend(loc='upper right')
    # display.figure_.savefig(f'./pic/{name}.jpg')
    # print(f"pic saved to ./pic/{name}.jpg")
    ap = display.average_precision
    plt.close()
    return ap

def mean_std_normalize(p: 'DataFrame'):
    mean = p.mean()
    std = p.std()
    if std < ZERO:
        std = ZERO
    rs = (p - mean) / std
    return rs

def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def adjust_fake_label(fake_label, golden_labels):
    # modify with golden labels
    fake_label.loc[golden_labels[golden_labels.label==1].index]=1
    fake_label.loc[golden_labels[golden_labels.label==0].index]=0
    return fake_label

def get_anomaly(key, index):
    ts = all_data[key]
    mean, std = ts.mean(), ts.std()
    ts = (ts - mean) / std
    if (index - 200) >= 0 and (index + 200) <= len(ts):
        ts_value = ts[index - 200: index + 200]
    elif (index - 200) < 0:
        ts_value = np.pad(ts[0: index + 200], (200 - index, 0))
    elif (index + 200) > len(ts):
        ts_value = np.pad(ts[index - 200:], (0, index + 200 - len(ts)))
    ts_label = all_labels[key].iloc[index].item()
    return ts_value, ts_label  
    
def get_point_randomly(real_label, golden_labels):
    waiting_list = real_label.index.tolist()
    random.shuffle(waiting_list)
    for key, index in waiting_list:
        if golden_labels.loc[key].iloc[index].item() == -1:
            return key, index

def print_mean_std(score_list):
    score_list = np.array(score_list)
    for i in range(score_list.shape[1]):
        score = score_list[:,i,:]
        n_interact = score[0][0]
        mean = score[:,-1].mean()
        std = score[:,-1].std()
        print(n_interact, mean, '±', std)

def point_metrics(y_pred, y_true, beta=1.0):
    """Calculate precison recall f1 bny point to point comparison.

    Args:
        y_pred (ndarray): The predicted y.
        y_true (ndarray): The true y.
        beta (float): The balance for calculating `f score`.

    Returns:
        tuple: Tuple contains ``precision, recall, f1, tp, tn, fp, fn``.
    """
    idx = y_pred * 2 + y_true
    tn = (idx == 0).sum()  # tn 0+0
    fn = (idx == 1).sum()  # fn 0+1
    fp = (idx == 2).sum()  # fp 2+0
    tp = (idx == 3).sum()  # tp 2+1

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = (1 + beta**2) * (precision * recall) / (
        beta**2 * precision + recall + 1e-7)
    return precision, recall, f1, tp, tn, fp, fn