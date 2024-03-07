"""
Create on 8 March,2023

@author:NannanSun

Function: A predictor for classification
"""

import torch
import os
from train_cnn import config
from models import (
    cnn_module,
    CNN_RNN,
    dilated_cnn_module,
    cnn2d_transformer_module,
    transformer_module,
)

# import h5py
from train_cnn import eval
import pandas as pd

# import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    f1_score,
)

# import matplotlib.pyplot as plt
# from lime.lime_text import LimeTextExplainer
# from transformers import BertTokenizer, BertForSequenceClassification
# from torch import nn
# import random
from torch.utils.data import DataLoader
from dataloader import SpectraDataset, collate_fn
import numpy as np


class Predict(object):

    def __init__(self, model_path="./outputs/model.pt"):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:2")
        else:
            self.device = torch.device("cpu")

        self.model = transformer_module(config.enformer_params)
        self.model_path = model_path
        self.model.load_state_dict(torch.load(self.model_path))

    def pred(self, test_ls, test_labels_dict):
        torch.manual_seed(0)

        val_set = SpectraDataset(test_ls, test_labels_dict)
        test_loader = DataLoader(
            val_set,
            batch_size=8,
            shuffle=True,
            num_workers=8,
            prefetch_factor=2,
            collate_fn=collate_fn,
        )

        model = self.model.to(self.device)
        accuracy, pred, labels, filenames, Logits, Vectors = eval(test_loader, model, 0)
        # pred_proba_list = [line.detach().numpy() for line in Logits]

        print(accuracy)
        preds = [p.item() for p in pred]

        labels = [int(l.detach().numpy()) for l in labels]

        pred_scores = [round(float(line[-1]), 10) for line in Logits]
        # pred_scores_positive = []
        # for probability in pred_scores:
        #     if probability==0:
        #         pred_scores_positive.append(round(random.uniform(0.01, 0.1),4))
        #     elif probability==1.0:
        #         pred_scores_positive.append(round(random.uniform(0.9, 0.99),4))
        #     else:
        #         pred_scores_positive.append(probability)
        # pred_proba_negative = [(1- prob) for prob in pred_scores_positive]
        print("confusion-matrix", confusion_matrix(labels, preds))
        infos_dict = {"filenames": filenames, "preds": preds, "labels": labels}
        infos = pd.DataFrame(infos_dict)
        print(infos)

        print(classification_report(labels, preds))
        roc_auc = roc_auc_score(labels, pred_scores)
        print("auc", roc_auc)
        fpr, tpr, thread = roc_curve(labels, pred_scores)
        # roc_infos = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thread": thread})

        return infos_dict, labels, Vectors


if __name__ == "__main__":
    predictor = Predict()
    test_bin = "./mzml/parsed/IPX0000937000_resolution_10_sparse"
    test_path = "./mzml/labels/0_fold_IPX0000937000_test.xlsx"
    test_infos = pd.read_excel(test_path)
    test_labels_dict = dict(zip(test_infos["file_name"], test_infos["label"]))
    test_ls = [
        os.path.join(test_bin, file + ".pt") for file in test_infos["file_name"].values
    ]
    predictor.pred(test_ls, test_labels_dict)

    ####可解释性
    # torch.manual_seed(0)
    # load_Dataset = loadDataset(batch_size=1, max_rt_len=8500)
    # predictor = Predict()
    # test_bin = "./mzml/parsed/IPX0000937000_resolution_10_sparse"
    # test_path = "./mzml/labels/0_fold_IPX0000937000_test.xlsx"
    # test_infos = pd.read_excel(test_path)
    # test_labels_dict = dict(
    #         zip(test_infos["file_name"], test_infos["label"]))
    # test_ls = [
    #         os.path.join(test_bin, file + ".h5")
    #         for file in test_infos["file_name"].values
    #     ]
    # test_loader, _, = load_Dataset.load_train_val_test_data(
    #         test_ls, test_ls, test_ls, test_labels_dict, test_labels_dict,
    #         test_labels_dict)
    # for (batch_samples,batch_labels,batch_filemanes) in test_loader:
    #     batch_samples = torch.sum(batch_samples,axis=1)
    #     text = " ".join([str(i) for i in range(batch_samples.shape[-1])])
    #     #batch_samples_padding = torch.unsqueeze(batch_samples_padding,1)
    #     print(batch_samples.shape)
    #     explanation = explain_instance(batch_samples,text)
    # print(explanation.as_list())
