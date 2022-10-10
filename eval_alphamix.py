import numpy as np
import torch
import pandas as pd
import statistics
from test import test_base_by_date, max_drawdown, save_res, get_test_setting, test_be_by_date


if __name__ == '__main__':
    # for dataset in ['acl18', 'sz_50']:
    # for dataset in ['acl18']:
    #     for batch_size in [32]:
    #         for hidden in [128]:
    #             for lr in [5e-4]:
    #                 model = 'BE_batch{}_hidden{}_lr{}_clf'.format(batch_size,hidden,str(lr))
    #                 acc, loss, ece, ret, sr, vol, dd, mdd, cr, sor = [], [], [], [], [], [], [], [], [], []
    #                 for seed in range(10):
    #                     net, stock_lst, feature_lst, target, test_date, criterion = get_test_setting(dataset, model, seed)

    #                     if dataset == 'acl18':
    #                         acc_lst, ret_lst, loss_lst, err = test_be_by_date(model, net, test_date, dataset, stock_lst, feature_lst, target,
    #                                                   10, 4, 'clf', criterion)
    #                     elif dataset == 'sz_50':
    #                         acc_lst, ret_lst, loss_lst, err = test_be_by_date(model, net, test_date, dataset, stock_lst, feature_lst, target,
    #                                                   25, 4, 'clf', criterion)
    #                     acc.append(np.mean(acc_lst))
    #                     loss.append(np.mean(loss_lst))
    #                     ece.append(err)
    #                     ret.append(sum(ret_lst))
    #                     sr.append(sum(ret_lst) / np.std(ret_lst) / np.sqrt(len(ret_lst)))
    #                     vol.append(np.std(ret_lst))
    #                     mdd.append(max_drawdown(ret_lst))
    #                     cr.append(sum(ret_lst) / max_drawdown(ret_lst))
    #                     neg_ret_lst = []
    #                     for day_ret in ret_lst:
    #                         if day_ret < 0:
    #                             neg_ret_lst.append(day_ret)
    #                     dd.append(np.std(neg_ret_lst))
    #                     sor.append(sum(ret_lst) / np.std(neg_ret_lst) / np.sqrt(len(ret_lst)))
    #                 save_res(acc, loss, ece, ret, sr, vol, dd, mdd, cr, sor, dataset, model)
    for dataset in ['sz_50']:
    # for dataset in ['acl18']:
        # for batch_size in [32]:
        for batch_size in [256]:
        # for batch_size in [32, 64, 128, 256,512]:
            for hidden in [128]:
                for lr in [5e-4]:
                # for lr in [5e-5, 1e-4, 2e-4, 5e-4]:
                    print(batch_size,lr)
                    for ens in [2,8,16,32]:
                        model = 'BE_batch{}_hidden{}_lr{}_clf'.format(batch_size,hidden,str(lr))
                        acc, loss, ece, ret, sr, vol, dd, mdd, cr, sor = [], [], [], [], [], [], [], [], [], []
                        ret_seq_lst = []
                        for seed in range(10):
                            net, stock_lst, feature_lst, target, test_date, criterion = get_test_setting(dataset, model, seed)

                            if dataset == 'acl18':
                                acc_lst, ret_lst, loss_lst, err = test_be_by_date(model, net, test_date, dataset, stock_lst, feature_lst, target,
                                                      10, 4, 'clf', criterion,4)
                            elif dataset == 'sz_50':
                                acc_lst, ret_lst, loss_lst, err = test_be_by_date(model, net, test_date, dataset, stock_lst, feature_lst, target,
                                                      25, 4, 'clf', criterion,4)
                            ret_seq_lst.append(ret_lst)
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
                        save_res(acc, loss, ece, ret, sr, vol, dd, mdd, cr, sor, dataset, model)

