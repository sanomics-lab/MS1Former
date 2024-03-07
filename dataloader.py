"""
Create on Feb 28

@author:NannanSun

Function: loader data
"""

from html.entities import html5
import os
import sys
import torch
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm


def calculate_differences(lst):
    differences = []
    for i in range(len(lst) - 1):
        diff = lst[i + 1] - lst[i]
        differences.append(diff)
    return differences


def padding_samples(samples, batch_first=False, max_rt_len=None, padding_value=0):
    """ """
    if max_rt_len == None:
        # import pdb;pdb.set_trace()
        max_len = max([sample.size()[0] for sample in samples])
    else:
        max_len = max_rt_len
    out_tensors = []
    for tensor_sample in samples:
        if tensor_sample.size(0) < max_len:
            tensor_sample = torch.cat(
                [
                    tensor_sample,
                    torch.tensor(
                        [[padding_value] * tensor_sample.size(-1)]
                        * (max_len - tensor_sample.size()[0])
                    ),
                ],
                dim=0,
            )
        else:
            tensor_sample = tensor_sample[:max_len]
        out_tensors.append(tensor_sample)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


def normalize_matrix(matrix):
    # 获取矩阵的最大值和最小值
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    # 对矩阵进行归一化处理
    normalized_matrix = (matrix - min_val) / (max_val - min_val)

    return normalized_matrix


def read_spectra(spectra_path, label):

    # for file_path in tqdm(spectra_path_list):
    if not spectra_path.endswith(".pt"):
        pass
    loaded_sparse_tensor = torch.load(spectra_path)
    dense_tensor_restored = loaded_sparse_tensor.to_dense()
    infos = dense_tensor_restored.numpy()
    # print("infos",infos.shape)
    # begin= int(infos.shape[0]*0.14)
    # end = int(infos.shape[0]*0.14*(-0.9))
    begin = int(infos.shape[0] * 0.14)
    end = int(infos.shape[0] * 0.14 * (-2.9))
    infos = infos[begin:end, :]
    # print("infos",infos.shape)
    infos = normalize_matrix(infos)
    # try:
    # # with h5py.File(spectra_path,"r")as f:
    # #     infos = f['data'][()]
    # loaded_sparse_tensor = torch.load(spectra_path)
    # dense_tensor_restored = loaded_sparse_tensor.to_dense()
    # infos = dense_tensor_restored.numpy()
    # # begin= int(infos.shape[0]*0.14)
    # # end = int(infos.shape[0]*0.14*(-0.9))
    # begin= int(infos.shape[0]*0.14)
    # end = int(infos.shape[0]*0.14*(-0.9))
    # infos = infos[begin:end, :]
    # infos = normalize_matrix(infos)
    # # tensor_ = torch.tensor(infos,dtype=torch.float32)
    # # sepctra_label = torch.tensor(label,dtype=torch.long)
    # except:
    #     print("The error file name is",spectra_path)
    return (torch.tensor(infos, dtype=torch.float32), label, begin, end)


class SpectraDataset(Dataset):

    def __init__(self, files_list, lable_dict):
        self.x_path = files_list
        self.labels = [
            lable_dict[filename.split("/")[-1].split(".")[0]] for filename in files_list
        ]
        self.file_names = [
            filename.split("/")[-1].split(".")[0] for filename in files_list
        ]

    def __getitem__(self, index):

        spectra_infos, sepctra_label, begin, end = read_spectra(
            self.x_path[index], self.labels[index]
        )
        filename = self.file_names[index]

        return (spectra_infos, sepctra_label, filename, begin, end)

    def __len__(self):

        return len(self.x_path)


def collate_fn(batch):

    # 将文本序列进行变长padding
    spectra, labels, filenames, begin, end = zip(*batch)
    padded_texts = padding_samples(spectra, batch_first=True)
    sepctra_label = torch.tensor(labels, dtype=torch.long)
    return (padded_texts, sepctra_label, filenames, begin, end)


class Datasetloader:
    def __init__(self, batch_size=2, max_rt_len=None):
        """
        Args:
        """
        self.batch_size = batch_size
        self.max_rt_len = max_rt_len

    def dataloader(self, train_set):
        # train_iter = DataLoader(train_set,batch_size=self.batch_size,shuffle=True,collate_fn=self.generate_batch)
        train_iter = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        return train_iter

    def generate_batch(self, data_batch):
        """
        A function is defined to process dataset in a batch, and it will be feed into DataLoader
        """
        batch_samples, batch_labels = [], []
        for sample, label in data_batch:
            batch_samples.append(sample)
            batch_labels.append(label)
        batch_samples_padding = padding_samples(
            batch_samples,
            padding_value=0,
            batch_first=True,
            max_rt_len=self.max_rt_len,
        )
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        return batch_samples_padding, batch_labels


if __name__ == "__main__":
    train_path = ""
    batch_size = 8
    dataloader = loadDataset(batch_size=batch_size, max_rt_len=None)
    train_loader, val_loader, test_loader = dataloader.load_train_val_test_data(
        train_path, train_path, train_path
    )
