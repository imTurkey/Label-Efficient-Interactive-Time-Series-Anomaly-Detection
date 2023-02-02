import copy
import numpy as np
import pandas as pd
import random
import time

from utils import min_max_normalize, limit_min_max_normalize

class ActiveLearning():
    def __init__(self, anomaly_prob_init):
        self.index_list = anomaly_prob_init.index.tolist()
        # init uncerntainty
        self.uncertainty = pd.concat([anomaly_prob_init for _ in range(5)], axis=1)
        self.uncertainty[:] = 0
        self.uncertainty.columns = ["lf_dis", "lf_abs", "uncertainty_s", "diversity", "anomaly_prob_init"]

        # Anomaly probability of UADs
        self.uncertainty['anomaly_prob_init'] = anomaly_prob_init

        # Edit here if you want to remove some metrics [1, 0.5, 0.5, 1, 0.2]
        self.weight = pd.DataFrame(pd.Series([1, 0.5, 0.5, 1, 0.2], index=self.uncertainty.columns, name=0))

        print("Active learning model ready!")


    def cal_uncertainty(self, dnn_pred, LFs, ts2vec, golden_labels):
        # dnn_pred = limit_min_max_normalize(dnn_pred, limit=10)
        pred_1 = LFs.replace(-1,0).sum(axis=1)
        pred_0 = LFs.replace(1,-1).replace(0,1).replace(-1,0).sum(axis=1)
        pred_anomaly = pred_1/(pred_1+pred_0)
        pred_anomaly = pred_anomaly.fillna(0)
        # Agreement of labeling functions
        self.uncertainty["lf_dis"] = -np.array(pred_anomaly * np.log(pred_anomaly + 1e-20) + (1 - pred_anomaly) * np.log(1 - pred_anomaly + 1e-20))
        # Hit time with labeling functions
        self.uncertainty["lf_abs"] = np.log(-LFs.where(LFs == -1, other = 0).sum(axis=1) + 1)
        # Uncertainty of the supervised end model
        self.uncertainty["uncertainty_s"] = -np.array(dnn_pred * np.log(dnn_pred + 1e-20) + (1 - dnn_pred) * np.log(1 - dnn_pred + 1e-20))
        # Diversity
        self.uncertainty["diversity"] = ts2vec.cal_diversity(golden_labels)
    
    def get_next_sample(self, golden_labels, ts_num=1, len_padding=None):

        uncertainty_norm = min_max_normalize(self.uncertainty)
        uncertainty_weighted = uncertainty_norm.dot(self.weight)
  
        uncertainty_weighted_sort = uncertainty_weighted.loc[golden_labels.label == -1]
        uncertainty_weighted_sort = uncertainty_weighted_sort.sort_values(by=0, ascending=False)
        
        

        top_k = uncertainty_weighted_sort.index[:ts_num].tolist()
        # print(top_k)
        return top_k

    def get_next_sample_random(self, golden_labels, ts_num=1, len_padding=None, empty_keys = None):
        
        random.shuffle(self.index_list)
        top_k = self.index_list[:ts_num]
        return top_k

        
            
        

    
