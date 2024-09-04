# this file contain a FBCCA method to decode SSVEP singles
# created by Yang Yi, University of Macau, 2023.09.05
# email: yc27932@umac.mo
import numpy as np
import math
from scipy import signal
from sklearn.cross_decomposition import CCA
import timeit
import datetime
import logging
from scipy import linalg
from numpy import linalg as nplinalg


class FBCCA():
    def __init__(self, args, Fs=250, Nf=40, channel_num=9, sub=1, subband=3):
        super(FBCCA, self).__init__()
        self.Fs = Fs
        self.Nf = Nf
        self.Nc = channel_num
        self.sample_length = args.data_length
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.sub = sub
        self.Nm = subband

    def get_Reference_Signal(self, num_harmonics, targets):
        reference_signals = []
        t = np.arange(0, (self.sample_length / self.Fs), step=1.0 / self.Fs)
        for f in targets:
            reference_f = []
            for h in range(1, num_harmonics + 1):
                reference_f.append(np.sin(2 * np.pi * h * f * t)[0:self.sample_length])
                reference_f.append(np.cos(2 * np.pi * h * f * t)[0:self.sample_length])
            reference_signals.append(reference_f)
        reference_signals = np.asarray(reference_signals)
        return reference_signals

    def get_Template_Signal(self, X, targets):
        reference_signals = []
        num_per_cls = X.shape[0] // self.Nf
        for cls_num in range(len(targets)):
            reference_f = X[cls_num * num_per_cls:(cls_num + 1) * num_per_cls]
            reference_f = np.mean(reference_f, axis=0)
            reference_signals.append(reference_f)
        reference_signals = np.asarray(reference_signals)
        return reference_signals

    def find_correlation(self, n_components, X, Y):
        cca = CCA(n_components)
        corr = np.zeros(n_components)
        num_freq = Y.shape[0]
        result = np.zeros(num_freq)
        for freq_idx in range(0, num_freq):
            matched_X = X

            cca.fit(matched_X.T, Y[freq_idx].T)
            # cca.fit(X.T, Y[freq_idx].T)
            x_a, y_b = cca.transform(matched_X.T, Y[freq_idx].T)
            for i in range(0, n_components):
                corr[i] = np.corrcoef(x_a[:, i], y_b[:, i])[0, 1]
                result[freq_idx] = np.max(corr)

        return result

    def filter_bank(self, eeg):
        result = np.zeros((eeg.shape[0], self.Nm, eeg.shape[-2], self.T))

        nyq = self.Fs / 2
        if self.dataset == 'Direction':
            passband = [4, 10, 16, 22, 28, 34, 40]
            stopband = [2, 6, 10, 16, 22, 28, 34]
            highcut_pass, highcut_stop = 40, 50

        else:
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
            result[:, i, :, :] = data

        return result

    def fbcca_classify(self, targets, test_data, test_labels, num_harmonics=5, train_data=None, template=False):
        t1_total = timeit.default_timer()
        if template:
            train_data = self.filter_bank(train_data)
            reference_signals = np.zeros((self.Nf, self.Nm, self.Nc, self.T))
            for fb_i in range(0, self.Nm):
                reference_signals[:, fb_i] = self.get_Template_Signal(train_data[:, fb_i], targets)
        else:
            reference_signals = self.get_Reference_Signal(num_harmonics, targets)

        # test_data = self.filter_bank(test_data)

        predicted_class = []
        num_segments = test_data.shape[0]

        fb_coefs = [math.pow(i, -1.25) + 0.25 for i in range(1, self.Nm + 1)]  # w(n) = n^(-0.5) + 1.25
        for segment in range(0, num_segments):
            result = np.zeros(self.Nf)
            # result = ¦² w(n) * (¦Ñ(k))^2
            for fb_i in range(0, self.Nm):
                x = test_data[segment, fb_i]
                y = reference_signals[:, fb_i] if template else reference_signals
                w = fb_coefs[fb_i]

                # result += (w * (self.find_correlation(1, x, y) ** 2))
                # this will have fast caculate speed
                result += (w * (self.ssvep_cca_qr(x, y) ** 2))
            predicted_class.append(np.argmax(result))

        predicted_class = np.array(predicted_class)
        t2_total = timeit.default_timer()
        acc, f1, precision, recall_score = self.save_mat(test_labels, predicted_class)
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

    def ssvep_cca_qr(self, X, Y):
        # 对 X 进行处理
        num_freq = Y.shape[0]
        result = np.zeros(num_freq)
        for freq_idx in range(0, num_freq):
            matched_X = X.T
            matched_Y = Y[freq_idx].T
            matched_X = matched_X - matched_X.mean(axis=0)  #
            [Q1a, R1a] = linalg.qr(matched_X, mode='economic')
            # 对 Y 进行处理
            [Q2a, R2a] = linalg.qr(matched_Y, mode='economic')
            # 进行 SVD 分解
            [svdU, svdD, svdV] = nplinalg.svd(np.dot(Q1a.T, Q2a))
            result[freq_idx] = svdD[0]

        return result
