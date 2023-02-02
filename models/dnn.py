import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from utils import *
from visualization import *


class MySampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        self.anomaly_indices = np.where(np.array(dataset[:][1]).squeeze() == 1)[0]
        self.anomaly_len = len(self.anomaly_indices)
        self.normal_indices = np.where(np.array(dataset[:][1]).squeeze() == 0)[0]

    def __iter__(self):
        a = self.anomaly_indices
        b = np.random.choice(self.normal_indices, 2 * self.anomaly_len)
        indices = np.concatenate((a, b), axis=0)
        np.random.shuffle(indices)
        return iter(torch.from_numpy(indices).tolist())

class DNN():
    def __init__(self, gpu=0):
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        # define dnn self.model
        num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 80, 2, 80, 100
        self.model = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens1),
            nn.ReLU(),
            nn.Linear(num_hiddens1, num_hiddens2),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(num_hiddens2, num_outputs),
            nn.Sigmoid(),
        )
        # init params
        for params in self.model.parameters():
            nn.init.normal_(params, mean=0, std=0.2)        

    def train(self, features, fake_label, num_epochs=50):
        # init dataset    
        dataset_fake = TensorDataset(
            torch.from_numpy(np.array(features)).to(torch.float32),
            torch.from_numpy(np.array(fake_label)).to(torch.float32)
        )

        train_size = int(len(dataset_fake)*0.8)
        valid_size = len(dataset_fake) - train_size
        train_dataset,valid_dataset = torch.utils.data.random_split(
            dataset_fake,[train_size, valid_size]
        )
        train_loader = DataLoader(dataset=train_dataset, batch_size=512, sampler=MySampler(train_dataset)
        )
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=512, shuffle=True)

        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        eval_result = dict()
        eval_result["f1_score"] = []
        eval_result["cross_loss"] = []
        eval_result["valid_loss"] = [np.inf]
        for epoch in range(num_epochs):
            tn_all, fn_all, fp_all, tp_all = 0, 0, 0, 0
            train_l_sum, train_acc_sum, n = .0, .0, 0
            for X, y in train_loader:
                X, y, self.model = X.to(self.device), y.to(self.device), self.model.to(self.device)
                # Forward
                y_hat = self.model(X)
                import pdb;pdb.set_trace()
                cross_loss = criterion(y_hat, y.squeeze(1).long())
                # cross_loss = criterion(y_hat, y)
                eval_result["cross_loss"].append(cross_loss.item())
                # L1 and L2 regularization
                l1_regularization = torch.tensor([0], dtype=torch.float32).to(self.device)
                l2_regularization = torch.tensor([0], dtype=torch.float32).to(self.device)
                for param in self.model.parameters():
                    l1_regularization += torch.norm(param, 1)
                    l2_regularization += torch.norm(param, 2)

                loss = cross_loss + l1_regularization.item() * 0.0001 + l2_regularization.item() * 0.0001

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                tn, fn, fp, tp = tn_fn_fp_tp(y_hat[..., 0], y.squeeze(), thres=0.5)
                tn_all += tn
                fn_all += fn
                fp_all += fp
                tp_all += tp
                train_l_sum += loss.item()
                n += y.shape[0]

            precision, recall, train_f1 = f1_metrics(tn_all, fn_all, fp_all, tp_all)
            # print("train:", tn_all, fn_all, fp_all, tp_all)

            eval_result["f1_score"].append(train_f1)

            valid_loss, valid_f1 = self.valid(self.model, valid_loader)

            if valid_loss > eval_result["valid_loss"][-1]:
                count += 1
                print("loss ascend {} times.".format(count))
            else:
                count = 0
                eval_result["valid_loss"].append(valid_loss)
            
            if count > 4:
                return

            if (epoch+1)%5 == 0:
                print("epoch {0:d}, train loss {1:.10f}, train f1 {2:.2f}, valid loss {3:.10f}, valid f1 {4:.2f}".format(epoch + 1, train_l_sum / n, train_f1, valid_loss, valid_f1))
    
    def valid(self, net, valid_loader):
        net.eval()
        tn_all, fn_all, fp_all, tp_all = 0, 0, 0, 0
        valid_loss_sum, valid_acc_sum, n = .0, .0, 0
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.MSELoss()
        for X, y in valid_loader:
            X, y, net = X.to(self.device), y.to(self.device), net.to(self.device)
            with torch.no_grad():
                y_hat = net(X)

                cross_loss = criterion(y_hat, y.squeeze(1).long())
                # cross_loss = criterion(y_hat, y)

                # L1 and L2 regularization
                l1_regularization = torch.tensor([0], dtype=torch.float32).to(self.device)
                l2_regularization = torch.tensor([0], dtype=torch.float32).to(self.device)
                for param in net.parameters():
                    l1_regularization += torch.norm(param,1)
                    l2_regularization += torch.norm(param,2)
                
                loss = cross_loss + l1_regularization.item()*0.0001 + l2_regularization.item()*0.0001
            
            tn, fn, fp, tp = tn_fn_fp_tp(y_hat[...,0], y.squeeze(), thres=0.5)
            tn_all += tn
            fn_all += fn
            fp_all += fp
            tp_all += tp
            valid_loss_sum += loss.item()
            n += y.shape[0]

        precision, recall, f1 = f1_metrics(tn_all, fn_all, fp_all, tp_all)
        # print("valid:", tn_all, fn_all, fp_all, tp_all)

        valid_loss = valid_loss_sum / n
        net.train()
        return valid_loss, f1

    def predict(self, features):
        dataset = TensorDataset(
            torch.from_numpy(np.array(features)).to(torch.float32),
        )
        test_loader = DataLoader(dataset=dataset, batch_size=1024, shuffle=False)
        pred = pd.DataFrame()

        for X in test_loader:
            X, self.model = X[0].to(self.device), self.model.to(self.device)
            with torch.no_grad():
                y_hat = self.model(X)
                y_hat = 1 - pd.DataFrame(y_hat[..., 0].cpu().numpy())
            pred = pd.concat([pred, y_hat], axis=0)     
            pred = pred.fillna(0)       
        return np.array(pred)