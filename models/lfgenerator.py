import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ABSTAIN = -1

class LFGenerator():
    def __init__(self, dataset='yahoo', model='ts2vec'):

        # load all repr
        self.dataset = dataset
        self.model = model

        if model=='ts2vec':
            with open(f"data/ts2vec/{dataset}/{dataset}_train_repr.pkl", 'rb') as f:
                self.all_repr = pickle.load(f)
            l2 = np.sqrt((self.all_repr**2).sum(axis=1))
            self.all_repr_norm = self.all_repr.div(l2, axis=0)
            self.thr = 8
            print("Ts2Vec model ready!")

        elif model=='dnn_features':
            with open(f"data/data_preprocessed/{dataset}/{dataset}_train_features.pkl", 'rb') as f:
                self.all_repr = pickle.load(f)
            self.thr = 8
            print("Statistic model ready!")

    def predict(self, sample_key, sample_index):
        sample_repr = self.get_repr(sample_key, sample_index)
        
        if self.model=='dnn_features':
            # inner product
            distance = np.dot(self.all_repr, sample_repr)
            pred = self.gen_lf(distance)

        elif self.model=='ts2vec':
            # L1 Distance with norm
            sample_repr_norm = np.divide(sample_repr, np.sqrt((sample_repr**2).sum()))
            distance = np.abs(self.all_repr_norm - sample_repr).sum(axis=1)
            pred = self.gen_lf(distance)

        return pred

    def gen_lf(self, distance):
        if self.model=='dnn_features':
            thr = np.mean(distance) + self.thr * np.std(distance)
            pred = distance > thr

        elif self.model=='ts2vec':
            thr = np.mean(distance) - self.thr * np.std(distance)
            pred = distance < thr

        pred_int = np.zeros_like(pred).astype(int)
        pred_int[pred==True] = 1
        pred_int[pred==False] = -1  

        return pred_int

    def get_repr(self, key, index):
        return self.all_repr.loc[key].iloc[index]

    def cal_diversity(self, golden_labels):
        num_golden_labels = (golden_labels != ABSTAIN).sum().item()
        if num_golden_labels:
            golden_labels_sort = golden_labels.sort_values(by='label', ascending=False)
            index_list = golden_labels_sort[:num_golden_labels].index
            labeled_repr = self.all_repr.loc[index_list]

            # diversity calculation controled under 1 min
            if num_golden_labels > 1000:
                labeled_repr = labeled_repr.sample(1000)

            # use pytorch to accelerate
            all_repr = torch.from_numpy(np.array(self.all_repr)).to(torch.float)
            labeled_repr = torch.from_numpy(np.array(labeled_repr).transpose()).to(torch.float)
            diversity = 1-torch.mm(all_repr, labeled_repr).sum(axis=1)/num_golden_labels
            diversity = np.array(diversity)
        else:
            diversity = np.zeros_like(golden_labels)
        return diversity