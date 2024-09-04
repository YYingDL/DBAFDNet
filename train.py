import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timeit
import datetime
import scipy.io
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import logging
from torch.optim import lr_scheduler

def train(args, model, criterion, train_dataloader, valid_dataloader, test_dataloader, timeLen, sub):
    #  set metrics for saving
    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc_store = []
    valid_acc_store = []
    train_epoch_loss = []
    valid_epoch_loss = []
    best_acc = 0
    # 定义模型、损失函数和优化器
    model = model.to(args.device)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    t1_total = timeit.default_timer()
    # train
    # if not os.path.exists('save_model/' + str(args.model_name) + '_' +args.dataset + '_timelen_' + str(timeLen) + '_sub_' + str(sub) + '_model.pt'):
    for epoch in range(args.epochs):
        # save a time to show the time of run this model
        t1 = timeit.default_timer()
        model.train()
        corr_trial_num = 0
        total_trial_num = 0
        # =====================train============================
        for idx, (data_x, data_y) in enumerate(train_dataloader):
            data_x = data_x.to(args.device)
            data_y = data_y.squeeze().to(args.device)
            outputs = model(data_x)
            optimizer.zero_grad()
            loss = criterion(outputs, data_y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_epoch_loss.append(loss.item())
            if args.model_name == 'DBACFNet' or args.model_name == 'FBDBACFNet':
                corr_trial_num += sum(outputs[0].max(axis=1)[1] == data_y).cpu()
            else:
                corr_trial_num += sum(outputs.max(axis=1)[1] == data_y).cpu()
            total_trial_num += data_y.size()[0]

        train_acc_store.append(corr_trial_num / total_trial_num)
        train_epochs_loss.append(np.average(train_epoch_loss))

        # =====================valid============================
        with torch.no_grad():
            model.eval()
            corr_trial_num = 0
            total_trial_num = 0
            for idx, (data_x, data_y) in enumerate(valid_dataloader):
                data_x = data_x.to(args.device)
                data_y = data_y.squeeze().to(args.device)
                outputs = model(data_x)
                loss = criterion(outputs, data_y)
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())
                if args.model_name == 'DBACFNet' or args.model_name == 'FBDBACFNet':
                    corr_trial_num += sum(outputs[0].max(axis=1)[1] == data_y).cpu()
                else:
                    corr_trial_num += sum(outputs.max(axis=1)[1] == data_y).cpu()
                total_trial_num += data_y.size()[0]
            valid_acc_store.append(corr_trial_num / total_trial_num)
            valid_epochs_loss.append(np.average(valid_epoch_loss))
            if corr_trial_num / total_trial_num > best_acc:
                best_acc = corr_trial_num / total_trial_num
                torch.save(model, 'save_model/' + str(args.model_name) + '_' +args.dataset + '_timelen_' + str(timeLen) + '_sub_'
                           + str(sub) + '_model.pt')
        t2 = timeit.default_timer()
        # logger.info a loss and acc in each 10 epoches
        scheduler.step()
        if epoch % 10 == 0:
            logging.info('----------------------------------------------------------------------------------')
            logging.info('final loss: %s, time: %.4f' % (np.average(train_epoch_loss), t2 - t1))
            logging.info('train mean acc: %.2f%%, time: %.4f' % (train_acc_store[epoch] * 100, t2 - t1))
            logging.info('test mean acc : %.2f%%, time: %.4f, loss: %.4f' % (
            valid_acc_store[epoch] * 100, t2 - t1, np.average(valid_epochs_loss)))

    t2_total = timeit.default_timer()
    model = torch.load('save_model/' + str(args.model_name) + '_' + args.dataset +'_timelen_' + str(timeLen) + '_sub_' + str(sub) + '_model.pt')
    test_outputs = []
    test_labels = []
    corr_trial_num = 0
    total_trial_num = 0
    with torch.no_grad():
        model.eval()
        test_epoch_loss = []
        for idx, (data_x, data_y) in enumerate(test_dataloader):
            data_x = data_x.to(args.device)
            data_y = data_y.squeeze().to(args.device)
            outputs = model(data_x)
            loss = criterion(outputs, data_y)
            test_epoch_loss.append(loss.item())
            if args.model_name == 'DBACFNet' or args.model_name == 'FBDBACFNet':
                corr_trial_num += sum(outputs[0].max(axis=1)[1] == data_y).cpu()
                test_outputs += outputs[0].max(axis=1)[1].cpu().tolist()
            else:
                corr_trial_num += sum(outputs.max(axis=1)[1] == data_y).cpu()
                test_outputs += outputs.max(axis=1)[1].cpu().tolist()
            test_labels +=data_y.cpu().tolist()
            total_trial_num += data_y.size()[0]
        test_acc = corr_trial_num / total_trial_num
    logging.info('Save model and data, times: %s, the best acc in valid data: %s the acc in test data: %s' %
          (datetime.timedelta(seconds=t2_total - t1_total), best_acc, test_acc))

    store_var = {}
    store_var['train_acc_store'] = train_acc_store
    store_var['valid_acc_store'] = valid_acc_store
    store_var['test_acc_store'] = test_acc
    store_var['test_outputs_store'] = test_outputs
    store_var['test_labels_store'] = test_labels
    store_var['test_confusion_store'] = confusion_matrix(test_labels, test_outputs)
    store_var['test_f1_store'] = f1_score(test_labels, test_outputs, average='macro')
    store_var['test_precision_store'] = precision_score(test_labels, test_outputs, average='macro')
    store_var['test_recall_store'] = recall_score(test_labels, test_outputs, average='macro')
    store_var['train_loss_history'] = train_epoch_loss
    store_var['valid_loss_history'] = valid_epochs_loss
    scipy.io.savemat('save_metric/' + str(args.model_name) + '_' +args.dataset +'_timelen_' + str(timeLen) + '_sub_' + str(sub)
                     + '.mat', store_var)
    return test_acc.item(), precision_score(test_labels,test_outputs,average='macro'), \
        f1_score(test_labels,test_outputs,average='macro'), recall_score(test_labels,test_outputs,average='macro')
