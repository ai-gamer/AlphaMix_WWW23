from nn import *
from train import train_router, set_random

# set_random(0)
# data_dir = './data/sz_50/'
# seq_length = 25
# feature_num = 6
# x_train = np.loadtxt('./data/sz_50_data/x_train_60.txt').reshape(-1, 25, 6)
# x_val = np.loadtxt('./data/sz_50_data/x_val_60.txt').reshape(-1, 25, 6)
# x_test = np.loadtxt('./data/sz_50_data/x_test_60.txt').reshape(-1, 25, 6)
# y_train = np.loadtxt('./data/sz_50_data/y_train_60.txt')
# y_train_label = [1 if item >= 0 else 0 for item in y_train]
# y_val = np.loadtxt('./data/sz_50_data/y_val_60.txt')
# y_val_label = [1 if item >= 0 else 0 for item in y_val]
# y_test = np.loadtxt('./data/sz_50_data/y_test_60.txt')
# y_test_label = [1 if item >= 0 else 0 for item in y_test]
# data = {'X_train': x_train, 'X_val': x_val, 'X_test': x_test, 'y_train': {'target': y_train, 'label': y_train_label},
#         'y_val': {'target': y_val, 'label': y_val_label}, 'y_test': {'target': y_test, 'label': y_test_label}}
# dataset = 'sz_50'

# model_lst = ['BE_batch256_hidden128_lr0.0005_clf_3','BE_batch256_hidden128_lr0.0005_clf_5','BE_batch256_hidden128_lr0.0005_clf_0']
# net_lst = []
# for model in model_lst:
#     net = torch.load('./model/{}/{}/{}.pth'.format(dataset, model[:-2], model))
#     net.eval()
#     net_lst.append(net)
# batch_size = 128
# device = torch.device('cpu')
# for bce_weight in [1.5]:
# # for bce_weight in [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]:
#     router_net = expert_router(12, 32).to(device)
#     router_net.train()
#     optimizer = torch.optim.Adam(router_net.parameters(), lr=1e-3)
#     print(bce_weight)
#     train_router(data, net_lst, batch_size,dataset, 'router', device, 10, optimizer, router_net,bce_weight)



set_random(0)
data_dir = './data/acl18/split/'
seq_length = 10
feature_num = 11
dataset = 'acl18'
x_train = np.loadtxt(data_dir + 'train_x.txt').reshape(-1, seq_length, feature_num)
x_val = np.loadtxt(data_dir + 'val_x.txt').reshape(-1, seq_length, feature_num)
x_test = np.loadtxt(data_dir + 'test_x.txt').reshape(-1, seq_length, feature_num)
y_train = np.loadtxt(data_dir + 'train_y.txt')
y_train_label = [1 if item >= 0 else 0 for item in y_train]
y_val = np.loadtxt(data_dir + 'val_y.txt')
y_val_label = [1 if item >= 0 else 0 for item in y_val]
y_test = np.loadtxt(data_dir + 'test_y.txt')
y_test_label = [1 if item >= 0 else 0 for item in y_test]
data = {'X_train': x_train, 'X_val': x_val, 'X_test': x_test, 'y_train': {'target': y_train, 'label': y_train_label},
        'y_val': {'target': y_val, 'label': y_val_label}, 'y_test': {'target': y_test, 'label': y_test_label}}

model_lst = ['BE_batch32_hidden128_lr0.0005_clf_3','BE_batch32_hidden128_lr0.0005_clf_5','BE_batch32_hidden128_lr0.0005_clf_6']
net_lst = []
for model in model_lst:
    net = torch.load('./model/{}/{}/{}.pth'.format(dataset, model[:-2], model))
    net.eval()
    net_lst.append(net)
batch_size = 128
device = torch.device('cpu')
for bce_weight in [1.5]:
# for bce_weight in [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]:
    router_net = expert_router(12, 32).to(device)
    router_net.train()
    optimizer = torch.optim.Adam(router_net.parameters(), lr=1e-3)
    print(bce_weight)
    train_router(data, net_lst, batch_size,dataset, 'router', device, 10, optimizer, router_net,bce_weight)