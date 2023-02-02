from models.lfgenerator import ABSTAIN
import numpy as np
import pandas as pd


class WeakSupervisionModel():
    def __init__(self, ws_type, length=None):
        self.ws_type = ws_type
        if ws_type == "snorkel":
            # Snorkel
            from snorkel.labeling.model import LabelModel
            self.model = LabelModel(cardinality=2, verbose=False)

        elif ws_type == "flyingsquid":
            ## Flyingsquid
            from flyingsquid.label_model import LabelModel
            self.model = LabelModel(length)

        elif ws_type == "majorityvoter":
            ## Majority vote
            from snorkel.labeling.model.baselines import MajorityLabelVoter
            self.model = MajorityLabelVoter()

    def fit(self, LFs, alpha=0.01, seed=123):
        LFs = np.array(LFs)
        if self.ws_type == "snorkel":
            # Snorkel
            self.model.fit(
                L_train=np.array(LFs),
                class_balance=np.array([1-alpha, alpha]),
                lr=0.001,
                n_epochs=200,
                seed=seed,
                optimizer = 'adam',
                progress_bar = False
            )

        elif self.ws_type == "flyingsquid":
            ## Flyingsquid
            self.model.fit(
                L_train=np.array(LFs),
                class_balance=np.array([0.99, 0.01])
            )

        elif self.ws_type == "majorityvoter":
            ## Majority vote
            pass
        
    def pred_proba(self, LFs):
        LFs_np = np.array(LFs)
        pred = self.model.predict_proba(LFs_np)
        proba = pred[:, 1]
        proba = np.array(pd.DataFrame(proba).fillna(0))
        return pd.DataFrame(proba, index=LFs.index, columns=['score'])

    def gen_fake_label(self, proba_pos, proba_neg, thres_pos, thres_neg):
        fake_label = proba_pos.copy()
        fake_label.columns = ['label']
        fake_label[:] = -1
        fake_label[proba_pos.score >= thres_pos] = 1
        fake_label[proba_neg.score <= thres_neg] = 0

        return fake_label

    def thres_adaptive_decend(self, num_LF, score):
        thres = 99.7
        return np.percentile(score, thres)