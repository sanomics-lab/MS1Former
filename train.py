"""
Create on March 1,2023

@author:NannanSun

Function: train model for spectra
"""

import argparse
import os
from random import shuffle
from sched import scheduler
from statistics import mode
import sys
import torch
from models import CNN_RNN
from dataloader import loadDataset
import logging
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from torch.utils.tensorboard import SummaryWriter
import datetime
from omegaconf import OmegaConf

config = OmegaConf.create(
    {
        "model_params": {
            "cnn_params": {
                "cnn_input_dim": 1,
                "cnn_output_dim": 1,
                "stride": 5,
                "cnn_kernel": 10,
            },
            "cnn_layers_num": 1,
            "linear_dim": 512,
            "label_num": 2,
            "rnn_params": {
                "rnn_hidden_dim": 256,
                "rnn_num_layers": 2,
                "rnn_bidirectional": True,
            },
            "rnn_name": "LSTM",
        },
        "train_params": {
            "epoches": 200,
            "batch_size": 8,
            "lr": 0.0009,
            "log_interval": 10,
            "early_stop": 1000,
        },
        "dataset_params": {
            "dataset_dir": "mzml/parsed/IPX0000937000_resolution_10_sparse",
            "max_rt_len": None,
            "train_labels_path": "mzml/labels/0_fold_IPX0000937000_train.xlsx",
            "val_labels_path": "mzml/labels/0_fold_IPX0000937000_test.xlsx",
            "test_labels_path": "mzml/labels/0_fold_IPX0000937000_test.xlsx",
        },
        "cuda": True,
        "model_dir": "outputs/models/model_dir_cnn_rnn",
        "loger_dir": "outputs/logs/logs_cnn_rnn",
    }
)
writer = SummaryWriter("tensorboard/tensorboard_cnn_rnn")
now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log = open(config.loger_dir + "/" + now_time + ".log", "w")


def train():
    train_infos = pd.read_excel(config.dataset_params.train_labels_path)
    train_labels_dict = dict(zip(train_infos["file_name"], train_infos["label"]))
    train_ls = [
        os.path.join(config.dataset_params.dataset_dir, file + ".h5")
        for file in train_infos["file_name"].values
    ]
    shuffle(train_ls)
    val_infos = pd.read_excel(config.dataset_params.val_labels_path)
    val_labels_dict = dict(zip(val_infos["file_name"], val_infos["label"]))
    val_ls = [
        os.path.join(config.dataset_params.dataset_dir, file + ".h5")
        for file in val_infos["file_name"].values
    ]
    shuffle(val_ls)
    # test_infos = pd.read_excel(config.dataset_params.test_labels_path)
    # test_labels_dict = dict(zip(test_infos["file_name"],test_infos["label"]))
    # test_ls = [os.path.join(config.dataset_params.dataset_dir,file+".h5") for file in test_infos["file_name"].values]
    # shuffle(test_ls)
    load_Dataset = loadDataset(
        batch_size=config.train_params.batch_size,
        max_rt_len=config.dataset_params.max_rt_len,
    )
    train_loader, val_loader, test_loader = load_Dataset.load_train_val_test_data(
        train_ls, val_ls, val_ls, train_labels_dict, val_labels_dict, val_labels_dict
    )
    print("Dataset download finished")

    model = CNN_RNN(config.model_params)

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
    if config.cuda:
        model = model.cuda()
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
        for batch_samples, batch_labels in train_loader:
            if config.cuda:
                batch_samples = batch_samples.cuda()
                batch_labels = batch_labels.cuda()
            optimizer.zero_grad()
            logits = model(batch_samples)
            # import IPython
            # IPython.embed()
            loss_value = loss(logits, batch_labels)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            sqsum = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # writer.add_scalar("param_grad",np.array(param.grad.mean().item()),steps)
                    sqsum += (param.grad**2).sum().item()
            writer.add_scalar("grad_norm", np.sqrt(sqsum), steps)

            optimizer.step()
            steps += 1

            writer.add_scalar(
                "learn_rate", np.array(optimizer.param_groups[0]["lr"]), steps
            )
            writer.add_scalar("train_loss", np.array(loss_value.item()), steps)
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
            if steps % 10 == 0:
                dev_acc = eval(val_loader, model, steps)
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
    corrects, ave_loss = 0, 0
    loss = torch.nn.CrossEntropyLoss()
    # pred_infos = {"file":[],"pred_label":[],"true_label":[]}
    for batch_samples, batch_labels in val_loader:
        # import IPython
        # IPython.embed()
        if config.cuda:
            batch_samples = batch_samples.cuda()
            batch_labels = batch_labels.cuda()
        logits = model(batch_samples)
        loss_value = loss(logits, batch_labels)
        ave_loss += loss_value.item()
        corrects += (torch.max(logits, 1)[1] == batch_labels).sum().item()
    data_nums = len(val_loader.dataset)
    ave_loss /= data_nums
    accuracy = 100 * corrects / data_nums
    writer.add_scalar("val_loss", np.array(ave_loss), steps)
    writer.add_scalar("val_accuracy", np.array(accuracy), steps)

    print(
        "\n Evalation - loss:{:.6f} acc:{:.4f} %{}/{}\n".format(
            ave_loss, accuracy, corrects, data_nums
        )
    )
    log.write(
        "Evalation - loss:%.4f\tacc:%.4f\t%d/%d\n"
        % (ave_loss, accuracy, corrects, data_nums)
    )
    return accuracy


def save_model(model, prefix, steps):
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    saved_path = os.path.join(config.model_dir, prefix + str(steps)) + ".pt"
    torch.save(model.state_dict(), saved_path)


if __name__ == "__main__":
    train()
