# this file contain a MSI method to decode SSVEP singles
# created by Yang Yi, University of Macau, 2023.09.05
# email: yc27932@umac.mo
import numpy as np
import logging
import timeit
import datetime

class MSI_Base():
    def __init__(self, args, Fs=250, Nf=40, channel_num=9, sub=1):
        super(MSI_Base, self).__init__()
        self.Fs = Fs
        self.Nf = Nf
        self.Nc = channel_num
        self.sample_length = args.data_length
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.sub = sub

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

    def find_Synchronization_Index(self, X, Y):
        num_freq = Y.shape[0]
        num_harm = Y.shape[1]
        result = np.zeros(num_freq)
        for freq_idx in range(0, num_freq):
            y = Y[freq_idx]
            X = X[:] - np.mean(X).repeat(self.sample_length * self.Nc).reshape(self.Nc, self.sample_length)
            X = X[:] / np.std(X).repeat(self.sample_length * self.Nc).reshape(self.Nc, self.sample_length)

            y = y[:] - np.mean(y).repeat(self.sample_length * num_harm).reshape(num_harm, self.sample_length)
            y = y[:] / np.std(y).repeat(self.sample_length * num_harm).reshape(num_harm, self.sample_length)

            c11 = (1 / self.sample_length) * (np.dot(X, X.T))
            c22 = (1 / self.sample_length) * (np.dot(y, y.T))
            c12 = (1 / self.sample_length) * (np.dot(X, y.T))
            c21 = c12.T

            C_up = np.column_stack([c11, c12])
            C_down = np.column_stack([c21, c22])
            C = np.row_stack([C_up, C_down])

            # print("c11:", c11)
            # print("c22:", c22)

            v1, Q1 = np.linalg.eig(c11)
            v2, Q2 = np.linalg.eig(c22)
            V1 = np.diag(v1 ** (-0.5))
            V2 = np.diag(v2 ** (-0.5))

            C11 = np.dot(np.dot(Q1, V1.T), np.linalg.inv(Q1))
            C22 = np.dot(np.dot(Q2, V2.T), np.linalg.inv(Q2))

            # print("Q1 * Q1^(-1):", np.dot(Q1, np.linalg.inv(Q1)))
            # print("Q2 * Q2^(-1):", np.dot(Q2, np.linalg.inv(Q2)))

            U_up = np.column_stack([C11, np.zeros((self.Nc, num_harm))])
            U_down = np.column_stack([np.zeros((y.shape[0], self.Nc)), C22])
            U = np.row_stack([U_up, U_down])
            R = np.dot(np.dot(U, C), U.T)

            eig_val, _ = np.linalg.eig(R)
            # print("eig_val:", eig_val, eig_val.shape)
            E = eig_val / np.sum(eig_val)
            S = 1 + np.sum(E * np.log(E)) / np.log(self.Nc + num_harm)
            result[freq_idx] = S

        return result

    def msi_classify(self, targets, test_data, test_labels, num_harmonics=5):
        t1_total = timeit.default_timer()
        reference_signals = self.get_Reference_Signal(num_harmonics, targets)


        predicted_class = []
        num_segments = test_data.shape[0]

        for segment in range(0, num_segments):
            result = self.find_Synchronization_Index(test_data[segment], reference_signals)
            predicted_class.append(np.argmax(result))

        predicted_class = np.array(predicted_class)
        t2_total = timeit.default_timer()
        acc, f1, precision, recall_score = self.save_mat(test_labels, predicted_class)
        logging.info(
            'Save model and data, times: %s, the acc is : %s, the f1 is: %s, the precision is: %s, the recall_score is: %s' %
            (datetime.timedelta(seconds=t2_total - t1_total), acc, f1, precision, recall_score))
        return self.save_mat(test_labels, predicted_class)

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
