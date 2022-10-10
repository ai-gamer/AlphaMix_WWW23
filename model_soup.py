from asyncore import socket_map
import numpy as np
import torch
import pandas as pd
import statistics
from test import test_base_by_date, max_drawdown, save_res, get_test_setting, test_be_by_date, get_valid_setting
from nn import *
import os

if __name__ == '__main__':
    dataset = 'sz_50'
    
    if dataset == 'sz_50':
        soup_lst = ['BE_batch256_hidden128_lr0.0005_clf_3','BE_batch256_hidden128_lr0.0005_clf_0','BE_batch256_hidden128_lr0.0005_clf_2','BE_batch256_hidden128_lr0.0005_clf_4','BE_batch256_hidden128_lr0.0005_clf_5','BE_batch256_hidden128_lr0.0005_clf_6','BE_batch256_hidden128_lr0.0005_clf_7','BE_batch256_hidden128_lr0.0005_clf_8','BE_batch256_hidden128_lr0.0005_clf_9','BE_batch256_hidden128_lr0.0005_clf_1',
        'BE_batch256_hidden128_lr0.0002_clf_0','BE_batch256_hidden128_lr0.0002_clf_1','BE_batch256_hidden128_lr0.0002_clf_2','BE_batch256_hidden128_lr0.0002_clf_3','BE_batch256_hidden128_lr0.0002_clf_4','BE_batch256_hidden128_lr0.0002_clf_5','BE_batch256_hidden128_lr0.0002_clf_6','BE_batch256_hidden128_lr0.0002_clf_7','BE_batch256_hidden128_lr0.0002_clf_8','BE_batch256_hidden128_lr0.0002_clf_9']
        model = 'BE_batch{}_hidden{}_lr{}_clf'.format(128,128,str(1e-4))
        seed = 0
        net, stock_lst, feature_lst, target, valid_date, criterion = get_valid_setting(dataset, model, seed)
        best_valid_profit = -1    
        ret, sr, vol, dd, mdd, cr, sor = [], [], [], [], [], [], []
        sd_lst = []
        for soup_item in soup_lst:
            print(soup_item)
            net = torch.load('./model/{}/{}/{}.pth'.format(dataset, soup_item[:-2], soup_item))
            net.eval()
            sd = net.state_dict()
            sd_lst.append(sd)
            for key in sd_lst[0]:
                key_sum = sum(item[key] for item in sd_lst)/len(sd_lst)
                sd_lst[0][key] = key_sum
            device = torch.device('cpu')
            net_soup = BE_clf(150, 128).to(device)
            net_soup.load_state_dict(sd_lst[0])

            _, ret_lst, _,_ = test_be_by_date(model, net_soup, valid_date, dataset, stock_lst, feature_lst, target,
                                                      25, 4, 'clf', criterion)
            ret.append(sum(ret_lst))
            if sum(ret_lst)> best_valid_profit:
                best_valid_profit = sum(ret_lst)
                best_soup = net_soup
            else:
                soup_lst = soup_lst[:-1]
        print(soup_lst)
        acc, loss, ece, ret, sr, vol, dd, mdd, cr, sor = [], [], [], [], [], [], [], [], [], []
        net, stock_lst, feature_lst, target, test_date, criterion = get_test_setting(dataset, model, seed)
        acc_lst, ret_lst, loss_lst, err = test_be_by_date(model, best_soup, test_date, dataset, stock_lst, feature_lst, target,
                                                      25, 4, 'clf', criterion)
        print('test return:{}'.format(sum(ret_lst)))
        acc.append(np.mean(acc_lst))
        loss.append(np.mean(loss_lst))
        ece.append(err)
        ret.append(sum(ret_lst))
        sr.append(sum(ret_lst) / np.std(ret_lst) / np.sqrt(len(ret_lst)))
        vol.append(np.std(ret_lst))
        mdd.append(max_drawdown(ret_lst))
        cr.append(sum(ret_lst) / max_drawdown(ret_lst))
        neg_ret_lst = []
        for day_ret in ret_lst:
            if day_ret < 0:
                neg_ret_lst.append(day_ret)
        dd.append(np.std(neg_ret_lst))
        sor.append(sum(ret_lst) / np.std(neg_ret_lst) / np.sqrt(len(ret_lst)))
        save_res(acc, loss, ece, ret, sr, vol, dd, mdd, cr, sor, dataset, 'model_soup')

# if __name__ == '__main__':
#     dataset = 'acl18'
    
#     if dataset == 'acl18':
#         soup_lst = ['BE_batch128_hidden128_lr0.0002_clf_4','BE_batch128_hidden128_lr0.0002_clf_5','BE_batch128_hidden128_lr0.0002_clf_6','BE_batch128_hidden128_lr0.0002_clf_7','BE_batch128_hidden128_lr0.0002_clf_8','BE_batch128_hidden128_lr0.0002_clf_9','BE_batch128_hidden128_lr0.0002_clf_3',
#         'BE_batch32_hidden128_lr0.0005_clf_0','BE_batch32_hidden128_lr0.0005_clf_1','BE_batch32_hidden128_lr0.0005_clf_2','BE_batch32_hidden128_lr0.0005_clf_3','BE_batch32_hidden128_lr0.0005_clf_4','BE_batch32_hidden128_lr0.0005_clf_5','BE_batch32_hidden128_lr0.0005_clf_6','BE_batch32_hidden128_lr0.0005_clf_7',
#         'BE_batch32_hidden128_lr0.0005_clf_8','BE_batch32_hidden128_lr0.0005_clf_9']
#         model = 'BE_batch{}_hidden{}_lr{}_clf'.format(128,128,str(1e-4))
#         seed = 0
#         net, stock_lst, feature_lst, target, valid_date, criterion = get_valid_setting(dataset, model, seed)
#         best_valid_profit = -1    
#         ret, sr, vol, dd, mdd, cr, sor = [], [], [], [], [], [], []
#         sd_lst = []
#         for soup_item in soup_lst:
#             print(soup_item)
#             net = torch.load('./model/{}/{}/{}.pth'.format(dataset, soup_item[:-2], soup_item))
#             net.eval()
#             sd = net.state_dict()
#             sd_lst.append(sd)
#             for key in sd_lst[0]:
#                 key_sum = sum(item[key] for item in sd_lst)/len(sd_lst)
#                 sd_lst[0][key] = key_sum
#             device = torch.device('cpu')
#             net_soup = BE_clf(110, 128).to(device)
#             net_soup.load_state_dict(sd_lst[0])

#             _, ret_lst, _,_ = test_be_by_date(model, net_soup, valid_date, dataset, stock_lst, feature_lst, target,
#                                                       10, 4, 'clf', criterion)
#             ret.append(sum(ret_lst))
#             if sum(ret_lst)> best_valid_profit:
#                 best_valid_profit = sum(ret_lst)
#                 best_soup = net_soup
#             else:
#                 soup_lst = soup_lst[:-1]
#         print(soup_lst)
#         acc, loss, ece, ret, sr, vol, dd, mdd, cr, sor = [], [], [], [], [], [], [], [], [], []
#         net, stock_lst, feature_lst, target, test_date, criterion = get_test_setting(dataset, model, seed)
#         acc_lst, ret_lst, loss_lst, err = test_be_by_date(model, best_soup, test_date, dataset, stock_lst, feature_lst, target,
#                                                       10, 4, 'clf', criterion)
#         print('test return:{}'.format(sum(ret_lst)))
#         acc.append(np.mean(acc_lst))
#         loss.append(np.mean(loss_lst))
#         ece.append(err)
#         ret.append(sum(ret_lst))
#         sr.append(sum(ret_lst) / np.std(ret_lst) / np.sqrt(len(ret_lst)))
#         vol.append(np.std(ret_lst))
#         mdd.append(max_drawdown(ret_lst))
#         cr.append(sum(ret_lst) / max_drawdown(ret_lst))
#         neg_ret_lst = []
#         for day_ret in ret_lst:
#             if day_ret < 0:
#                 neg_ret_lst.append(day_ret)
#         dd.append(np.std(neg_ret_lst))
#         sor.append(sum(ret_lst) / np.std(neg_ret_lst) / np.sqrt(len(ret_lst)))
#         save_res(acc, loss, ece, ret, sr, vol, dd, mdd, cr, sor, dataset, 'model_soup')


   