from scipy.io import loadmat
import numpy as np
from scipy import signal
import logging
import math
from torch.utils.data import Dataset
import torch
import os
import argparse
import logging
import time
from termcolor import colored
import torch.nn as nn
import torch.nn.functional as F
import sklearn.preprocessing as preprocessing


def filter_power(x, fs):
    # 设计Butterworth带阻滤波器
    f0 = 50  # 工频频率
    Q = 30  # 质量因数
    b, a = signal.iirnotch(f0, Q, fs)
    x_filtered = np.zeros_like(x)
    # 使用滤波器滤除50Hz的工频干扰
    for ch in range(x.shape[0]):
        x_filtered[ch] = signal.filtfilt(b, a, x[ch])
    return x_filtered


def get_data(dataset_name = 'benchmack', subject_id = 1, used_slide_window =True, tw = 250, non_overlapping_rate=.15):
    if dataset_name == 'benchmack':
        totalsubject = 35  # # of subjects
        totalblock = 6  # # of blocks
        totalcharacter = 40  # # of characters
        sampling_rate = 250  # Sampling rate
        visual_latency = 0.14  # Average visual latency of subjects
        visual_cue = 0.5  # Length of visual cue used at collection of the dataset
        total_ch = 64  # # of channels used at collection of the dataset
        channels = [47, 53, 54, 55, 56, 57, 60, 61, 62]
        total_delay = visual_latency + visual_cue
        delay_sample_point = round(total_delay * sampling_rate)

        # read data from dataset path
        nameofdata = '/data/2016_Tsinghua_SSVEP_database/S' + str(subject_id) + '.mat'
        data = loadmat(nameofdata)['data']
        # Taking data from spesified channels, and signal interval
        subject_data = data[channels, delay_sample_point:delay_sample_point + 5 * sampling_rate, :, :]

        # for chr in range(totalcharacter):
        #     for blk in range(totalblock):
        #             tmp_raw = subject_data[:,:,chr,blk]
        #             processed_signal = filter_power(tmp_raw,sampling_rate)
        #             subject_data[:, :, chr, blk] = processed_signal


        if used_slide_window == True:
            x = np.array([], dtype=np.float32).reshape(0, len(channels), tw)  # data
            y = np.zeros([0], dtype=np.int32)  # true label
            step = int(math.ceil(tw * non_overlapping_rate))
            for run_idx in range(totalblock):
                for freq_idx in range(totalcharacter):
                    raw_data = subject_data[:, :, freq_idx, run_idx]
                    n_samples = int(math.floor((raw_data.shape[1] - tw) / step))
                    _x = np.zeros([n_samples, len(channels), tw], dtype=np.float32)
                    _y = np.ones([n_samples], dtype=np.int32) * freq_idx
                    for i in range(n_samples):
                        _x[i, :, :] = raw_data[:, i * step:i * step + tw]

                    x = np.append(x, _x, axis=0)  # [?,tw,ch], ?=runs*cl*samples
                    y = np.append(y, _y)
            subject_data = x
            subject_label = y

        else:
            subject_data = np.reshape(subject_data.transpose([3, 2, 0, 1]),
                                       [totalblock * totalcharacter, len(channels), -1])
            subject_label = np.tile(np.arange(40, dtype=np.int32), totalblock)

        N = 1250
        fft_data = np.fft.rfft(a=subject_data, n=N, axis=2)
        # 在python的计算方式中，fft结果的直接取模和真实信号的幅值不一样。
        # 对于非直流量的频率，直接取模幅值会扩大N/2倍， 所以需要除了N乘以2。
        # 对于直流量的频率(0Hz)，直接取模幅值会扩大N倍，所以需要除了N。
        fft_amp0 = np.array(np.abs(fft_data) / N * 2)  # 用于计算双边谱
        fft_amp0[0] = 0.5 * fft_amp0[0]

        # 线性归一化
        amp_norm = np.zeros_like(fft_amp0[:, :, 40:320])
        for k in range(fft_amp0.shape[0]):
            for i in range(9):
                amp_min = np.min(fft_amp0[k, i, 40:320])
                amp_max = np.max(fft_amp0[k, i, 40:320])
                amp_norm[k, i, :] = (fft_amp0[k, i, 40:320] - amp_min) / (amp_max - amp_min) * (255 - 0) + 0
                amp_norm[k, i, :] = amp_norm[k, i, :].astype(np.uint8)
        fft_phase = (np.angle(fft_data[:, :, 40:320]) + np.pi) % (2 *np.pi)
        fft_phase = np.angle(fft_data[:, :, 40:320]) * 180 / np.pi
        subject_fft_data = np.concatenate([np.expand_dims(amp_norm,1),np.expand_dims(fft_phase,1)],axis=1)
    return subject_fft_data, subject_label


class CustomDataset(Dataset):
    # initialization: data and label
    def __init__(self, Data1,  Label):
        self.Data1 = Data1
        self.Label = Label
    # get the size of data

    def __len__(self):
        return len(self.Data1)
    # get the data and label

    def __getitem__(self, index):
        data1 = torch.Tensor(self.Data1[index])
        label = torch.LongTensor(self.Label[index])
        return data1, label


class CustomDataset2(Dataset):
    # initialization: data and label
    def __init__(self, Data1,Data2, Label):
        self.Data1 = Data1
        self.Data2 = Data2
        self.Label = Label
    # get the size of data

    def __len__(self):
        return len(self.Data1)
    # get the data and label

    def __getitem__(self, index):
        data1 = torch.Tensor(self.Data1[index])
        data2 = torch.Tensor(self.Data2[index])
        label = torch.LongTensor(self.Label[index])
        return data1,data2, label


def filterbank(X, num_subbands):
    """
    Suggested filterbank function for benchmark dataset
    """
    srate = 250
    filterbank_X = np.zeros((num_subbands,X.shape[0], X.shape[1]))

    for k in range(1, num_subbands + 1, 1):
        Wp = [(8 * k) / (srate / 2), 90 / (srate / 2)]
        Ws = [(8 * k - 2) / (srate / 2), 100 / (srate / 2)]

        gstop = 40
        while gstop >= 20:
            try:
                N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
                bpB, bpA = signal.cheby1(N, 0.5, Wn, btype='bandpass')
                filterbank_X[k - 1, :, :] = signal.filtfilt(bpB, bpA, X, axis=1, padtype='odd',
                                                            padlen=3 * (max(len(bpB), len(bpA)) - 1))
                break
            except:
                gstop -= 1
    return filterbank_X


def create_logger(args):
    # create logger
    os.makedirs(args.output_log_dir, exist_ok=True)
    time_str = time.strftime('%m-%d-%H-%M')
    log_file = args.model_name + '_'+ args.dataset + '_data_length_' + str(args.data_length) + '_lr_' + str(args.lr) + \
               '_{}.log'.format(time_str)
    final_log_file = os.path.join(args.output_log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #
    fmt = '[%(asctime)s] %(message)s'
    color_fmt = colored('[%(asctime)s]', 'green') + ' %(message)s'

    file = logging.FileHandler(filename=final_log_file, mode='a')
    file.setLevel(logging.INFO)
    file.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console)

    return logger


class SharedAndSpecificLoss(nn.Module):
    def __init__(self, ):
        super(SharedAndSpecificLoss, self).__init__()
        self.classification_loss = LabelSmoothingLoss()
        self.weight_1 = torch.tensor(1/4, requires_grad=True)
        self.weight_2 = torch.tensor(1/4, requires_grad=True)
        self.weight_3 = torch.tensor(1/4, requires_grad=True)
        self.weight_4 = torch.tensor(1 / 4, requires_grad=True)
        self.weight_similar = torch.tensor(0.2, requires_grad=True)
        self.weight_orth = torch.tensor(0.2, requires_grad=True)

    # Should be orthogonal
    @staticmethod
    def orthogonal_loss(shared, specific):
        shared = torch.sigmoid(shared)
        specific = torch.sigmoid(specific)
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        correlation_matrix = torch.mul(shared, specific)
        cost = correlation_matrix.mean()
        return cost

    # should be big
    @staticmethod
    def dot_product_normalize(shared_1, shared_2):
        # num_of_samples = shared_1.size(0)
        # shared_1 = shared_1 - shared_1.mean()
        # shared_2 = shared_2 - shared_2.mean()
        # shared_1 = F.normalize(shared_1, p=2, dim=1)
        # shared_2 = F.normalize(shared_2, p=2, dim=1)
        # # Dot product
        # match_map = torch.bmm(shared_1.view(num_of_samples, 1, -1), shared_2.view(num_of_samples, -1, 1))
        # mean = match_map.mean()
        diff = shared_1 - shared_2
        # loss = torch.abs(diff ** 2)
        loss = torch.mean(torch.abs(diff))
        return loss

    def forward(self, output, target):
        # # Similarity Loss
        # classification_output, shared1_output,specific1_output,  shared2_output, specific2_output, att1, att2,  = output
        # similarity_loss = self.dot_product_normalize(att1[0], att2[0])+ self.dot_product_normalize(att1[1], att2[1])
        #
        #
        # # orthogonal restrict
        # orthogonal_loss1 = self.orthogonal_loss(shared1_output, specific1_output)
        # orthogonal_loss2 = self.orthogonal_loss(shared2_output, specific2_output)
        #
        # # Classification Loss
        # classification_loss = F.cross_entropy(classification_output, target)
        #
        # # loss = orthogonal_loss1 * 0.2 + orthogonal_loss2 * 0.2 + similarity_loss * 0.2 + classification_loss
        # loss = orthogonal_loss1 * 0.5 + orthogonal_loss2 * 0.5 + classification_loss + similarity_loss * 0.4

        classification_output, amp_out, phase_out, amp, phase, att1, att2, = output
        similarity_loss = self.dot_product_normalize(att1[0], att2[0]) + self.dot_product_normalize(att1[1], att2[1])
        # orthogonal_loss1 = self.orthogonal_loss(amp[0], amp[1])
        # orthogonal_loss2 = self.orthogonal_loss(phase[0], phase[1])
        # Classification Loss
        classification_loss_1 = F.cross_entropy(classification_output, target)
        classification_loss_2 = F.cross_entropy(amp_out, target) * self.weight_2
        classification_loss_3 = F.cross_entropy(phase_out, target) * self.weight_3
        # classification_loss_4 = F.cross_entropy(t_out, target) * self.weight_4
        loss = classification_loss_1
        # loss = classification_loss_1
        return loss




class LabelSmoothingLoss(nn.Module):
    "Implement label smoothing."

    def __init__(self, class_num=40, smoothing=0.01):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.class_num = class_num

    def forward(self, x, target):
        assert x.size(1) == self.class_num
        if self.smoothing == None:
            return nn.CrossEntropyLoss()(x, target)

        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.class_num - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        logprobs = F.log_softmax(x, dim=-1)
        mean_loss = -torch.sum(true_dist * logprobs) / x.size(-2)
        return mean_loss



if __name__ == '__main__':
    from scipy.io import loadmat
    from sklearn.metrics import classification_report
    # cm = loadmat('save_metric/ConvCA_beta_timelen_250_sub_1.mat')['test_confusion_store']
    # ture_label = loadmat('save_metric/CCNN_benchmark_timelen_50_sub_2.mat')['test_labels_store']
    # predict_label = loadmat('save_metric/CCNN_benchmark_timelen_50_sub_2.mat')['test_outputs_store']
    # print(classification_report(ture_label.reshape(-1), predict_label.reshape(-1)))
    accs = []
    for idx in range(1, 67):
        acc = loadmat('save_metric/ConvCA_beta_timelen_250_sub_'+ str(idx) +'.mat')['test_f1_store']
        accs.append(acc)
    accs = np.reshape(accs,[-1,1])
