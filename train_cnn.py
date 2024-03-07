"""
Create on March 1,2023

@author:NannanSun

Function: train model for spectra
"""

import argparse
import os
from random import shuffle
from statistics import mode
import sys
from xml.etree.ElementPath import prepare_descendant
import torch
from models import (
    cnn_module,
    dilated_cnn_module,
    cnn_transformer_module,
    transformer_module,
)
import logging
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from torch.utils.tensorboard import SummaryWriter
import datetime
from torch import nn
from omegaconf import OmegaConf
from dataloader import SpectraDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

os.environ["WANDB_MODE"] = "offline"


config = OmegaConf.create(
    {
        "model_params": {
            "cnn_params": {
                "cnn_input_dim": 1,
                "cnn_output_dim": 1,
                "stride": [2, 2],
                "cnn_kernel": [5, 5],
                "dilation_size": [1, 4],
                "max_pooling_kernel": [3],
                "max_pooling_stride": [2],
            },
            "cnn_layers_num": 1,
            "linear_dim": 128,
            "label_num": 2,
            "rnn_params": {
                "rnn_hidden_dim": 256,
                "rnn_num_layers": 2,
                "rnn_bidirectional": True,
            },
            "rnn_name": "LSTM",
        },
        "enformer_params": {
            "cnn_params": {
                "input_dim": 1,
                "filter_dim": 1,
                "stride": [1, 2],
                "cnn_kernel": [5, 2],
            },
            "transfomer_params": {
                "input_dim": 192,
                "num_heads": 8,
                "value_dim": 24,
                "key_dim": 48,
                "scaling": True,
                "attention_dropout_rate": 0.05,
                "relative_position_symmetric": False,
                "num_relative_position_features": 24,
                "positional_dropout_rate": 0.01,
                "zero_initialize": True,
            },
            "dropout_rate": 0.2,
            "num_transformer_layers": 1,
            "channels": 192,
            "cnn_layers_num": 1,
            "linear_dim": 512,
            "label_num": 2,
        },
        "train_params": {
            "epoches": 200,
            "batch_size": 8,
            "lr": 0.001,
            "log_interval": 20,
            "early_stop": 1000,
        },
        "dataset_params": {
            "dataset_dir": "mzml/parsed/IPX0000937000_resolution_10_sparse",
            "max_rt_len": None,
            "train_labels_path": "mzml/labels/0_fold_IPX0000937000_train.xlsx",
            "val_labels_path": "mzml/labels/0_fold_IPX0000937000_test.xlsx",
        },
        "cuda": True,
        "model_dir": "outputs/models/model_resolution_10",
        "loger_dir": "outputs/logs/logs_resolution_10",
    }
)

wandb.init(
    entity="sano-drugai",
    project="spectra-hcc-classification",
    config={
        "learning_rate": 0.001,
        "architecture": "Transformer",
        "dataset": "IPX0000937000-resolution-10-sparse",
        "epochs": 200,
        "batch_size": 8,
        "early_stop": 1000,
    },
)

now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
if not os.path.exists(config.loger_dir):
    os.mkdir(config.loger_dir)
log = open(config.loger_dir + "/" + now_time + ".log", "w")
if torch.cuda.is_available():
    device = torch.device("cuda:2")
else:
    device = torch.device("cpu")


def train():
    train_infos = pd.read_excel(config.dataset_params.train_labels_path)
    train_labels_dict = dict(zip(train_infos["file_name"], train_infos["label"]))
    train_ls = [
        os.path.join(config.dataset_params.dataset_dir, file + ".pt")
        for file in train_infos["file_name"].values
    ]
    shuffle(train_ls)
    val_infos = pd.read_excel(config.dataset_params.val_labels_path)
    val_labels_dict = dict(zip(val_infos["file_name"], val_infos["label"]))
    val_ls = [
        os.path.join(config.dataset_params.dataset_dir, file + ".pt")
        for file in val_infos["file_name"].values
    ]
    shuffle(val_ls)
    train_set = SpectraDataset(train_ls, train_labels_dict)
    train_iter = DataLoader(
        train_set,
        batch_size=config.train_params.batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    val_set = SpectraDataset(val_ls, val_labels_dict)
    val_iter = DataLoader(
        val_set,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    print("Dataset download finished")

    model = transformer_module(config.enformer_params)

    if os.path.exists(config.model_dir):
        print("load pre-model")
        log.write("load pre-model\n")
        model_nums = [
            int(file.split(".")[0].strip("best"))
            for file in os.listdir(config.model_dir)
        ]
        max_num = max(model_nums)
        model_path = os.path.join(config.model_dir, "best" + str(max_num) + ".pt")
        model.load_state_dict(torch.load(model_path))
        steps = max_num
        print("pre-model loaded", "and the steps is:", steps)
        log.write("pre-model loaded and the steps is:%d\n" % steps)
    else:
        steps = 0
    # if config.cuda:
    #     model = model.cuda(1)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train_params.lr)
    # scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=60,T_mult=1)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    best_acc = 0
    last_step = 0
    loss = torch.nn.CrossEntropyLoss()
    # testacc = eval(test_loader,model,steps)
    # print("test-accuracy",testacc)
    for epoch in range(config.train_params.epoches):
        # 在每个epoch 结束时更新learn rate
        # scheduler.step()

        for batch_samples, batch_labels, _, _, _ in tqdm(train_iter):
            batch_samples = batch_samples.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_samples)
            loss_value = loss(logits, batch_labels)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            sqsum = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    sqsum += (param.grad**2).sum().item()

            optimizer.step()
            steps += 1
            if steps % config.train_params.log_interval == 0:
                corrects = (torch.max(logits, 1)[1] == batch_labels).sum()
                train_acc = 100.0 * corrects / config.train_params.batch_size
                print(
                    "epoch: {} -- loss:{:.4f} -- acc:{:.4f}%({}/{})".format(
                        epoch,
                        loss_value.item(),
                        train_acc,
                        corrects,
                        config.train_params.batch_size,
                    )
                )
                log.write(
                    "train_epoch:%d\t--loss:%.4f\t--acc:%.4f\t%d/%d\n"
                    % (
                        epoch,
                        loss_value.item(),
                        train_acc,
                        corrects,
                        config.train_params.batch_size,
                    )
                )
                wandb.log(
                    {"train loss": loss_value.item(), "train batch accuary": train_acc}
                )
            if steps % 20 == 0:
                dev_acc, _, _, _, _ = eval(val_iter, model, steps)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    print(
                        "Saving the best model, acc:{:.4f},steps:{}\n".format(
                            best_acc, str(steps)
                        )
                    )
                    save_model(model, "best", steps)
                    log.write(
                        "Saving the best model\tacc:%.4f\tsteps:%d\n"
                        % (best_acc, steps)
                    )
                else:
                    if steps - last_step >= config.train_params.early_stop:
                        print(
                            "nearly stop by {} steps, acc:{:.4f}%".format(
                                config.train_params.early_stop, best_acc
                            )
                        )
                        log.write(
                            "nearly stop by %d steps\t acc:%.4f\n"
                            % (config.train_params.early_stop, best_acc)
                        )
                        raise KeyboardInterrupt
            # if steps % 200 ==0:
            #     print("Saving the  model, accuracy:{}\n".format(best_acc))
            #     save_model(model,"steps",steps)
            #     log.write("Saving the common model")
            # scheduler.step()


def eval(val_loader, model, steps):
    model.eval()
    corrects, ave_loss = 0, 0
    loss = torch.nn.CrossEntropyLoss()
    # pred_infos = {"file":[],"pred_label":[],"true_label":[]}
    pred_label = []
    filenames = []
    Logits = []
    labels = []
    Vectors = []
    with torch.no_grad():
        for batch_samples, batch_labels, batch_filename, _, _ in val_loader:
            # if config.cuda:
            batch_samples = batch_samples.to(device)
            batch_labels = batch_labels.to(device)
            logits_1, xx = model(batch_samples)
            Vectors.extend(xx.cpu().detach().numpy())
            softmax = nn.Softmax(dim=1)
            logits = softmax(logits_1)
            Logits.extend(logits.cpu().detach().numpy())
            # print("logits,batch_labels",(logits,batch_labels))
            loss_value = loss(logits, batch_labels)
            ave_loss += loss_value.item()
            corrects += (torch.max(logits, 1)[1] == batch_labels).sum()
            pred_label.extend(torch.max(logits, 1)[1])
            filenames.extend(batch_filename)
            labels.extend(batch_labels.cpu())
    data_nums = len(val_loader.dataset)
    ave_loss /= data_nums
    accuracy = 100 * corrects / data_nums
    print(
        "\n Evalation - loss:{:.6f} acc:{:.4f} %{}/{}\n".format(
            ave_loss, accuracy, corrects, data_nums
        )
    )
    log.write(
        "Evalation - loss:%.4f\tacc:%.4f\t%d/%d\n"
        % (ave_loss, accuracy, corrects, data_nums)
    )
    wandb.log(
        {
            "val loss": ave_loss,
            "val accuracy": accuracy,
            "(corrects,data_nums)": (corrects, data_nums),
        }
    )
    return (accuracy, pred_label, labels, filenames, Logits, Vectors)


def save_model(model, prefix, steps):
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    saved_path = os.path.join(config.model_dir, prefix + str(steps)) + ".pt"
    torch.save(model.state_dict(), saved_path)


if __name__ == "__main__":
    train()
