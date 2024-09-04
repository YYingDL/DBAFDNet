# this file contain a DBACFNet method to decode SSVEP singles
# created by Yang Yi, University of Macau, 2023.09.05
# email: yc27932@umac.mo
from __future__ import division
from __future__ import print_function
import numpy as np
import torch.utils.data as Data
from scipy.io import loadmat
from utils import CustomDataset, filterbank, create_logger
import argparse
from train import train
import torch
import torch.nn as nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JFPAA_net parameters')
    parser.add_argument('--dataset', type=str, default='benchmark',
                        help='the dataset used for SSVEP, "benchmark" or "beta"')
    parser.add_argument('--batch_size', type=int, default=128, help='size for one batch, integer')
    parser.add_argument('--data_length', type=int, default=int(250 * 1.0), help='size for data_length, integer')
    parser.add_argument('--num_class', type=int, default=40, help='number for target, integer')
    parser.add_argument('--epochs', type=int, default=100, help='training epoch, integer')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
    parser.add_argument('--model_name', default='FBDBACFNet', type=str,
                        help='the current model name: [CCA, FBCCA, MSI, TRCA, EEGNet, ConvCA, GuneyDNN, CCNN, FBSSVEPFormer, DBACFNet,'
                             'FBDBACFNet]')
    parser.add_argument('--output_log_dir', default='./train_log', type=str,
                        help='output path, subdir under output_root')
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='learning rate')
    args = parser.parse_args()
    logger = create_logger(args)
    logger.info(args)
    if args.dataset == 'benchmark':
        path1 = '/home/user_yy/Dataset/Benchmark Dataset/'
        path2 = '.mat'
        channels = [47, 53, 54, 55, 56, 57, 60, 61, 62]
        name = ['S' + str(i) for i in range(1, 36)]
        t_delay = int((0.5 + 0.12) * 250)
        block = 6
        subband = 3
        frequencySet = np.arange(8.0, 16.0, step=0.2)
        phaseSet = np.array(
            [0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1,
             1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5]) * np.pi
    elif args.dataset == 'beta':
        path1 = '/home/user_yy/Dataset/BETA Dataset/'
        path2 = '.mat'
        channels = [47, 53, 54, 55, 56, 57, 60, 61, 62]
        name = ['S' + str(i) for i in range(1, 71)]
        t_delay = int((0.5 + 0.13) * 250)
        block = 4
        subband = 3
        frequencySet = np.array(
            [8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0,
             12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4, 13.6, 13.8, 14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.2,
             15.4, 15.6, 15.8, 8.0, 8.2, 8.4])
    sub_accs, sub_f1s, sub_precision, sub_recall_score = [], [], [], []
    index_class = range(0, 40)
    for i in range(len(name)):
        path = path1 + name[i] + path2
        mat = loadmat(path)
        if args.dataset == 'beta':
            data1c = mat['data']['EEG'][0][0]
            data1c = data1c[channels, t_delay:t_delay + args.data_length, :, :args.num_class]
            data1c = data1c.transpose(3, 2, 0, 1)
        else:
            data1c = mat['data']
            data1c = data1c[channels, t_delay:t_delay + args.data_length, :args.num_class, :]
            data1c = data1c.transpose(2, 3, 0, 1)  # label*block*C*T
        data_tmp = np.zeros((block * args.num_class, data1c.shape[2], data1c.shape[3]))
        label_tmp = np.zeros(block * args.num_class)
        for j in range(args.num_class):
            data_tmp[block * j:block * j + block] = data1c[j]
            label_tmp[block * j:block * j + block] = np.ones(block) * j

        if (i == 0):
            train_datac = data_tmp
            train_labelc = label_tmp
        else:
            train_datac = np.append(train_datac, data_tmp, axis=0)
            train_labelc = np.append(train_labelc, label_tmp)

    filterBank_data = np.zeros([train_datac.shape[0], subband, len(channels), train_datac.shape[2]])
    for idx in range(train_datac.shape[0]):
        tmp_raw = train_datac[idx, :, :]
        processed_signal = filterbank(tmp_raw, num_subbands=subband)
        filterBank_data[idx, :, :, :] = processed_signal
    filterBank_datas = filterBank_data

    for j in range(len(name)):
        logger.info('----------------------------------------------------------------------------------')
        logger.info('The current subject is sub %s, the data lenght is %s' % (j + 1, args.data_length))
        train_idx = [i for i in range(filterBank_datas.shape[0])]
        del train_idx[j * block * args.num_class: (j + 1) * block * args.num_class]
        test_idx = [i for i in range(j * block * args.num_class, (j + 1) * block * args.num_class)]
        criterion = nn.CrossEntropyLoss()
        if args.model_name == 'CCA':
            from models.CCA import CCA_Base

            cca = CCA_Base(args=args, sub=j)
            acc, f1, precision, recall_score = cca.cca_classify(frequencySet, train_datac[test_idx],
                                                                train_labelc[test_idx], template=False)
        elif args.model_name == 'FBCCA':
            from models.FBCCA import FBCCA

            fbcca = FBCCA(args=args, sub=j)
            acc, f1, precision, recall_score = fbcca.fbcca_classify(frequencySet, filterBank_datas[test_idx],
                                                                    train_labelc[test_idx], template=False)
        elif args.model_name == 'MSI':
            from models.MSI import MSI_Base

            msi = MSI_Base(args=args, sub=j)
            acc, f1, precision, recall_score = msi.msi_classify(frequencySet, train_datac[test_idx],
                                                                train_labelc[test_idx])
        elif args.model_name == 'TRCA':
            from models.TRCA import TRCA

            trca = TRCA(args=args, sub=j, train_dataset=(filterBank_datas[train_idx], train_labelc[train_idx]),
                        test_dataset=(filterBank_datas[test_idx], train_labelc[test_idx]))
            acc, f1, precision, recall_score = trca.fit()
        elif args.model_name == 'GuneyDNN':
            from models.GuneyDNN import GuneyDNN

            model = GuneyDNN(sample_length=args.data_length)

        elif args.model_name == 'ConvCA':
            from models.ConvCA import ConvCA

            train_idx = [i for i in range(filterBank_datas.shape[0])]
            del train_idx[j * block * args.num_class: (j + 1) * block * args.num_class]
            reference = filterBank_datas[train_idx, 0].reshape(len(name) - 1, args.num_class, block, len(channels),
                                                               args.data_length)
            reference = np.transpose(reference, (3, 1, 4, 2, 0)).reshape(len(channels), args.num_class,
                                                                         args.data_length, -1)
            reference = np.mean(reference, axis=-1)
            reference = np.expand_dims(reference, axis=0)
            reference = torch.tensor(reference, dtype=torch.float32)
            model = ConvCA(reference_signals=reference, sample_length=args.data_length)

        elif args.model_name == 'EEGNet':
            from models.EEGNet import EEGNet

            model = EEGNet(sample_length=args.data_length)

        elif args.model_name == 'FBSSVEPFormer':
            from models.FBSSVEPFormer import FBSSVEPFormer

            model = FBSSVEPFormer()

        elif args.model_name == 'CCNN':
            from models.CCNN import CCNN

            model = CCNN()

        elif args.model_name == 'DBACFNet':
            from models.DBACFNet import DBACFNet
            from models.DBACFNet import SharedAndSpecificLoss

            model = DBACFNet()
            criterion = SharedAndSpecificLoss()

        elif args.model_name == 'FBDBACFNet':
            from models.FBDBACFNet import FBDBACFNet
            from models.FBDBACFNet import SharedAndSpecificLoss

            model = FBDBACFNet()
            criterion = SharedAndSpecificLoss()
        if args.model_name in ['EEGNet', 'ConvCA', 'GuneyDNN', 'CCNN', 'FBSSVEPFormer', 'DBACFNet', 'FBDBACFNet']:
            np.random.shuffle(train_idx)
            # train_ratio = 0.9
            # num_train = int(train_ratio * len(train_idx))
            num_train = len(train_idx)
            train_fft_data = filterBank_datas[train_idx[:num_train]]
            train_label = train_labelc[train_idx[:num_train]]

            # valid_fft_data = Preprocessed_train[train_idx[num_train:]]
            # valid_label = train_labelc[train_idx[num_train:]]

            valid_fft_data = filterBank_datas[test_idx, :, :, :args.data_length]
            valid_label = train_labelc[test_idx]

            test_fft_data = filterBank_datas[test_idx, :, :, :args.data_length]
            test_label = train_labelc[test_idx]

            train_loader = torch.utils.data.DataLoader(
                dataset=CustomDataset(train_fft_data, np.reshape(train_label, [-1, 1])),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=4)

            valid_loader = torch.utils.data.DataLoader(
                dataset=CustomDataset(valid_fft_data, np.reshape(valid_label, [-1, 1])),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=4)

            test_loader = torch.utils.data.DataLoader(
                dataset=CustomDataset(test_fft_data, np.reshape(test_label, [-1, 1])),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=4)
            acc, f1, precision, recall_score = train(args, model, criterion, train_loader, valid_loader, test_loader,
                                                     timeLen=args.data_length, sub=j + 1)
        sub_accs.append(acc)
        sub_f1s.append(f1)
        sub_precision.append(precision)
        sub_recall_score.append(recall_score)
    sub_accs = np.reshape(np.array(sub_accs), [-1, 1])
    sub_f1s = np.reshape(np.array(sub_f1s), [-1, 1])
    sub_precision = np.reshape(np.array(sub_precision), [-1, 1])
    sub_recall_score = np.reshape(np.array(sub_recall_score), [-1, 1])
    logger.info('-----------------------------------------')
    logger.info('The mean accuracy of dataset: %.4f, The F1_macros of dataset: %.4f, The precision of dataset: %.4f,'
                'The recall_score of dataset: %.4f,' % (
                np.mean(sub_accs), np.mean(sub_f1s), np.mean(sub_precision), np.mean(sub_recall_score)))
    np.save(args.model_name + args.dataset + '_timeline_' + str(args.data_length) + '_acc.npy', np.array(sub_accs))
