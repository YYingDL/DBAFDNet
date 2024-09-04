# this file contain a TRCA method to decode SSVEP singles
# created by Yang Yi, University of Macau, 2023.09.05
# email: yc27932@umac.mo
import scipy
from scipy import signal
import numpy as np
import math
import logging
import timeit
import datetime


class TRCA():
    def __init__(self, args, train_dataset, test_dataset, Fs=250, Nf=40, channel_num=9, sub=1, subband=3):
        self.Fs = Fs
        self.Nf = Nf
        self.Nc = channel_num
        self.sample_length = args.data_length
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.sub = sub
        # default=0, help="TRCA or eTRCA"
        self.is_ensemble = 0
        self.Nm = subband
        if args.dataset == 'beta':
            block =4
        elif args.dataset == 'benchmark':
            block = 6
        self.train_data = train_dataset[0][:, :self.Nm].reshape(-1, self.Nf, block, self.Nm, self.Nc, self.sample_length)  # (Nh, Nm, Nc, T) -> (Nsub, Nf, Nb,Nm, Nc, T)
        self.train_label = train_dataset[1].reshape(-1, self.Nf, block)  # (Nh, N) -> (Nf, Nb, 1)
        self.test_data = test_dataset[0][:, :self.Nm].reshape(-1, self.Nf, block, self.Nm, self.Nc, self.sample_length)
        self.test_label = test_dataset[1].reshape(self.Nf, block)
        self.train_data = np.transpose(self.train_data, (1, 3, 4, 5, 2, 0)).reshape(self.Nf, self.Nm, self.Nc, self.sample_length, -1)
        self.test_data = np.transpose(self.test_data, (1, 3, 4, 5, 2, 0)).reshape(self.Nf, self.Nm, self.Nc, self.sample_length, -1)

    def load_data(self):
        self.train_data = self.filter_bank(self.train_data)
        self.test_data = self.filter_bank(self.test_data)
        # print("train_data.shape:", self.train_data.shape)
        # print("test_data.shape:", self.test_data.shape)

    def filter_bank(self, eeg):
        result = np.zeros((self.Nf, self.Nm, self.Nc, self.T, eeg.shape[1]))

        nyq = self.Fs / 2
        passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
        stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
        highcut_pass, highcut_stop = 80, 90

        gpass, gstop, Rp = 3, 40, 0.5
        for i in range(self.Nm):
            Wp = [passband[i] / nyq, highcut_pass / nyq]
            Ws = [stopband[i] / nyq, highcut_stop / nyq]
            [N, Wn] = signal.cheb1ord(Wp, Ws, gpass, gstop)
            [B, A] = signal.cheby1(N, Rp, Wn, 'bandpass')
            data = signal.filtfilt(B, A, eeg, padlen=3 * (max(len(B), len(A)) - 1)).copy()
            result[:, i, :, :, :] = np.transpose(data, (0, 2, 3, 1))

        return result

    def train_trca(self, eeg):
        [num_targs, _, num_chans, num_smpls, _] = eeg.shape
        trains = np.zeros((num_targs, self.Nm, num_chans, num_smpls))
        W = np.zeros((self.Nm, num_targs, num_chans))
        for targ_i in range(num_targs):
            eeg_tmp = eeg[targ_i, :, :, :, :]
            for fb_i in range(self.Nm):
                traindata = eeg_tmp[fb_i, :, :, :]
                trains[targ_i, fb_i, :, :] = np.mean(traindata, 2)
                w_tmp = self.trca(traindata)
                W[fb_i, targ_i, :] = np.real(w_tmp[:, 0])

        return trains, W

    def trca(self, eeg):
        [num_chans, num_smpls, num_trials] = eeg.shape
        S = np.zeros((num_chans, num_chans))
        for trial_i in range(num_trials - 1):
            x1 = eeg[:, :, trial_i]
            x1 = x1 - np.expand_dims(np.mean(x1, 1), 1).repeat(x1.shape[1], 1)
            for trial_j in range(trial_i + 1, num_trials):
                x2 = eeg[:, :, trial_j]
                x2 = x2 - np.expand_dims(np.mean(x2, 1), 1).repeat(x2.shape[1], 1)
                S = S + np.matmul(x1, x2.T) + np.matmul(x2, x1.T)

        UX = eeg.reshape(num_chans, num_smpls * num_trials)
        UX = UX - np.expand_dims(np.mean(UX, 1), 1).repeat(UX.shape[1], 1)
        Q = np.matmul(UX, UX.T)
        W, V = scipy.sparse.linalg.eigs(S, 6, Q)
        return V

    def test_trca(self, eeg, trains, W, is_ensemble):
        num_trials = eeg.shape[4]
        if self.Nm == 1:
            fb_coefs = [i for i in range(1, self.Nm + 1)]
        else:
            fb_coefs = [math.pow(i, -1.25) + 0.25 for i in range(1, self.Nm + 1)]
        fb_coefs = np.array(fb_coefs)
        results = np.zeros((self.Nf, num_trials))
        rho_list = np.zeros((self.Nf, self.Nf))

        for targ_i in range(self.Nf):
            test_tmp = eeg[targ_i, :, :, :, :]
            r = np.zeros((self.Nm, self.Nf, num_trials))

            for fb_i in range(self.Nm):
                testdata = test_tmp[fb_i, :, :, :]

                for class_i in range(self.Nf):
                    traindata = trains[class_i, fb_i, :, :]
                    if not is_ensemble:
                        w = W[fb_i, class_i, :]
                    else:
                        w = W[fb_i, :, :].T
                    for trial_i in range(num_trials):
                        testdata_w = np.matmul(testdata[:, :, trial_i].T, w)
                        traindata_w = np.matmul(traindata[:, :].T, w)
                        r_tmp = np.corrcoef(testdata_w.flatten(), traindata_w.flatten())
                        r[fb_i, class_i, trial_i] = r_tmp[0, 1]

            rho = np.einsum('j, jkl -> kl', fb_coefs, r)  # (num_targs, num_trials)

            tau = np.argmax(rho, axis=0)
            results[targ_i, :] = tau

        return results

    def cal_itr(self, n, p, t):
        if p < 0 or 1 < p:
            print('Accuracy need to be between 0 and 1.')
            exit()
        elif p < 1 / n:
            print('The ITR might be incorrect because the accuracy < chance level.')
            itr = 0
        elif p == 1:
            itr = math.log2(n) * 60 / t
        else:
            itr = (math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))) * 60 / t
        return itr

    def fit(self):
        t1_total = timeit.default_timer()
        # Training stage
        # print(traindata.shape)
        trains, W = self.train_trca(self.train_data)

        # Test stage
        # print(testdata.shape)
        predicted_class = self.test_trca(self.test_data, trains, W, self.is_ensemble)

        # Evaluation


        t2_total = timeit.default_timer()
        acc, f1, precision, recall_score = self.save_mat(self.test_label.reshape(-1), predicted_class.reshape(-1))
        logging.info(
            'Save model and data, times: %s, the acc is : %s, the f1 is: %s, the precision is: %s, the recall_score is: %s' %
            (datetime.timedelta(seconds=t2_total - t1_total), acc, f1, precision, recall_score))
        return acc, f1, precision, recall_score

    def save_mat(self, labels, predicted_labels):
        import scipy.io as sio
        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
        store_var = {}
        store_var['test_acc_store'] = accuracy_score(labels, predicted_labels)
        store_var['test_outputs_store'] = predicted_labels
        store_var['test_labels_store'] = labels
        store_var['test_confusion_store'] = confusion_matrix(labels, predicted_labels)
        store_var['test_f1_store'] = f1_score(labels, predicted_labels, average='macro')
        store_var['test_precision_store'] = precision_score(labels, predicted_labels, average='macro')
        store_var['test_recall_store'] = recall_score(labels, predicted_labels, average='macro')
        sio.savemat(
            'save_metric/' + str(self.model_name) + '_' + self.dataset + '_timelen_' + str(
                self.sample_length) + '_sub_' + str(self.sub)
            + '.mat', store_var)

        return store_var['test_acc_store'], store_var['test_f1_store'], store_var['test_precision_store'], store_var[
            'test_recall_store']


