import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
import random

def print_once(*message):
    if not hasattr(print_once, "has_run"):
        if message is not None:
            print(message)
        else:
            print("Function is running")
        print_once.has_run = True

def weighted_mse_loss(X, Y, attention_indices=[], weight_factor=3):
    """
    计算带权重的均方误差（Weighted MSE）

    参数:
    - X: 预测值的张量
    - Y: 目标值的张量
    - attention_indices: 关注的下标列表
    - weight_factor: 关注下标对应项的权重放大因子

    返回:
    - 带权重的均方误差
    """
    mse_loss = torch.nn.functional.mse_loss(X, Y, reduction='none')  # 计算未加权的均方误差
    # print("mse_loss:",mse_loss.shape)
    
    # exit()
    # 根据关注的下标列表将对应项的误差放大
    flag = True
    for index in attention_indices:
        mse_loss[:, index,:] *= weight_factor

    weighted_mse = mse_loss.mean()  # 计算带权重的均方误差
    # exit()
    return weighted_mse


def random_reset(input,reset_index):
    
    res = torch.clone(input)
    res[reset_index,:] = 0
def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        # self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        # self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = val_loss
        # score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            # self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta :
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.criterion = weighted_mse_loss

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            # for u in range(len(prior)):
            #     series_loss += (torch.mean(my_kl_loss(series[u], (
            #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                    self.win_size)).detach())) + torch.mean(
            #         my_kl_loss(
            #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                     self.win_size)).detach(),
            #             series[u])))
            #     prior_loss += (torch.mean(
            #         my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                            self.win_size)),
            #                    series[u].detach())) + torch.mean(
            #         my_kl_loss(series[u].detach(),
            #                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                            self.win_size)))))
            # series_loss = series_loss / len(prior)
            # prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output,input,[],10)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        flag = False
        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                reset_input = torch.clone(input)
                l = input.shape[1]
                reset_l = int(l*0.7)
                reset_index = random.sample(range(l), reset_l)
                reset_input[:,reset_index,:] = 0
                # print(reset_input[0,reset_index,:])
                # exit()
                output, series, prior, _ = self.model(reset_input)
                # if flag == False:
                #     print(output.shape,series[0].shape,prior[0].shape)
                #     flag = True
                print('end.',i)
                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                # for u in range(len(prior)):
                #     temp = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                #     series_loss += (torch.mean(my_kl_loss(series[u], temp.detach())) + torch.mean(my_kl_loss(temp.detach(),series[u])))
                #     # prior_loss += (torch.mean(my_kl_loss(temp,series[u].detach())) + torch.mean(my_kl_loss(series[u].detach(), temp)))
                # series_loss = series_loss / len(prior)
                # # prior_loss = prior_loss / len(prior)
                # prior_loss = series_loss.clone()
                # rec_loss = self.criterion(output, input)
                rec_loss = self.criterion(output,input,reset_index,3)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * series_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                # loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            # for u in range(len(prior)):
            #     if u == 0:
            #         series_loss = my_kl_loss(series[u], (
            #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                        self.win_size)).detach()) * temperature
            #         prior_loss = my_kl_loss(
            #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                     self.win_size)),
            #             series[u].detach()) * temperature
            #     else:
            #         series_loss += my_kl_loss(series[u], (
            #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                        self.win_size)).detach()) * temperature
            #         prior_loss += my_kl_loss(
            #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                     self.win_size)),
            #             series[u].detach()) * temperature

            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            metric = 1
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            # for u in range(len(prior)):
            #     if u == 0:
            #         series_loss = my_kl_loss(series[u], (
            #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                        self.win_size)).detach()) * temperature
            #         prior_loss = my_kl_loss(
            #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                     self.win_size)),
            #             series[u].detach()) * temperature
            #     else:
            #         series_loss += my_kl_loss(series[u], (
            #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                        self.win_size)).detach()) * temperature
            #         prior_loss += my_kl_loss(
            #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                     self.win_size)),
            #             series[u].detach()) * temperature
            # # Metric
            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            metric = 1
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            # for u in range(len(prior)):
            #     if u == 0:
            #         series_loss = my_kl_loss(series[u], (
            #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                        self.win_size)).detach()) * temperature
            #         prior_loss = my_kl_loss(
            #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                     self.win_size)),
            #             series[u].detach()) * temperature
            #     else:
            #         series_loss += my_kl_loss(series[u], (
            #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                        self.win_size)).detach()) * temperature
            #         prior_loss += my_kl_loss(
            #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                     self.win_size)),
            #             series[u].detach()) * temperature
            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = 1 * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

        return accuracy, precision, recall, f_score
