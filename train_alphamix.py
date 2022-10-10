from nn import *
from train import fit_base_model, set_random, get_dataset


if __name__ == '__main__':
    '''train LSTM,GRU,MLP,ALSTM clf on acl18'''
    # set_random(0)
    # data_dir = './data/acl18/split/'
    # seq_length = 10
    # feature_num = 11
    # dataset = 'acl18'
    # mode = 'clf'
    # data = get_dataset(dataset, data_dir, seq_length, feature_num, mode)
    # device = torch.device('cpu')
    # criterion = nn.CrossEntropyLoss()
    # # batch_size = 128
    # # train_episodes = 4
    # # hidden = 16
    # # lr = 1e-4
    # train_episodes = 8
    # for batch_size in [32]:
    #     for hidden in [128]:
    #     # for hidden in [32,64,128]:
    #         for lr in [ 5e-4]:
    #             for expert_num in [8,16,32]:
    #                 net = BE_clf(seq_length*feature_num, hidden,expert_num).to(device)
    #                 optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #                 for seed in range(10):
    #                     fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'BE_batch{}_hidden{}_lr{}_ens{}_clf'.format(batch_size,hidden,str(lr),expert_num), seed,
    #                    'clf', device)
    set_random(0)
    data_dir = './data/sz_50_data/'
    seq_length = 25
    feature_num = 6
    dataset = 'sz_50'
    mode = 'clf'
    data = get_dataset(dataset, data_dir, seq_length, feature_num, mode)
    device = torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    # batch_size = 128
    # train_episodes = 4
    # hidden = 16
    # lr = 1e-4
    train_episodes = 8
    for batch_size in [256]:
        for hidden in [128]:
        # for hidden in [32,64,128]:
            for lr in [5e-4]:
                for expert_num in [2,8,16,32]:
                    net = BE_clf(seq_length*feature_num, hidden,expert_num).to(device)
                    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                    for seed in range(10):
                        fit_base_model(data, net, criterion, optimizer, batch_size, train_episodes, dataset, 'BE_batch{}_hidden{}_lr{}_ens{}_clf'.format(batch_size,hidden,str(lr),expert_num), seed,
                            'clf', device)

   

    