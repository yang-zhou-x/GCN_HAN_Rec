# -*- coding: utf-8 -*-
'''
@File    : main.py
@Time    : 2020
@Author  : ZHOU, YANG
@Contact : yzhou.x@icloud.com
'''

import torch as t
import models as m
import utils as u
import argparse
from datetime import datetime
import os
import random


def main(args):
    # load dataset
    g_homo, g_list, pairs, labels, train_mask, test_mask = u.load_data(
        args['name'], args['train_size'])

    # transfer
    pairs = t.from_numpy(pairs).to(args['device'])
    labels = t.from_numpy(labels).to(args['device'])
    train_mask = t.from_numpy(train_mask).to(args['device'])
    test_mask = t.from_numpy(test_mask).to(args['device'])
    feat1 = t.randn(g_homo.number_of_nodes(),
                    args['in_feats']).to(args['device'])
    feat2 = t.randn(g_list[0].number_of_nodes(),
                    args['in_feats']).to(args['device'])
    labels = labels.view(-1, 1).to(dtype=t.float32)

    # model
    if args['model'] == 'SRG':
        model = m.SRG(rgcn_in_feats=args['in_feats'],
                      rgcn_out_feats=args['embedding_size'],
                      rgcn_num_blocks=args['num_b'],
                      rgcn_dropout=0.,
                      han_num_meta_path=args['num_meta_path'],
                      han_in_feats=args['in_feats'],
                      han_hidden_feats=args['embedding_size'],
                      han_head_list=args['head_list'],
                      han_dropout=args['drop_out'],
                      fc_hidden_feats=args['fc_units']
                      ).to(args['device'])
    elif args['model'] == 'SRG_GAT':
        model = m.SRG_GAT(rgcn_in_feats=args['in_feats'],
                          rgcn_out_feats=args['embedding_size'],
                          rgcn_num_blocks=args['num_b'],
                          rgcn_dropout=args['drop_out'],
                          han_num_meta_path=args['num_meta_path'],
                          han_in_feats=args['in_feats'],
                          han_hidden_feats=args['embedding_size'],
                          han_head_list=args['head_list'],
                          han_dropout=args['drop_out'],
                          fc_hidden_feats=args['fc_units']
                          ).to(args['device'])
    elif args['model'] == 'SRG_no_GRU':
        model = m.SRG_no_GRU(gcn_in_feats=args['in_feats'],
                             gcn_out_feats=args['embedding_size'],
                             gcn_num_layers=args['num_l'],
                             han_num_meta_path=args['num_meta_path'],
                             han_in_feats=args['in_feats'],
                             han_hidden_feats=args['embedding_size'],
                             han_head_list=args['head_list'],
                             han_dropout=args['drop_out'],
                             fc_hidden_feats=args['fc_units']
                             ).to(args['device'])
    elif args['model'] == 'SRG_Res':
        model = m.SRG_Res(gcn_in_feats=args['in_feats'],
                          gcn_out_feats=args['embedding_size'],
                          gcn_num_layers=args['num_l'],
                          han_num_meta_path=args['num_meta_path'],
                          han_in_feats=args['in_feats'],
                          han_hidden_feats=args['embedding_size'],
                          han_head_list=args['head_list'],
                          han_dropout=args['drop_out'],
                          fc_hidden_feats=args['fc_units']
                          ).to(args['device'])
    elif args['model'] == 'SRG_no_GCN':
        model = m.SRG_no_GCN(han_num_meta_path=args['num_meta_path'],
                             han_in_feats=args['in_feats'],
                             han_hidden_feats=args['embedding_size'],
                             han_head_list=args['head_list'],
                             han_dropout=args['drop_out'],
                             fc_hidden_feats=args['fc_units']
                             ).to(args['device'])
    else:
        raise ValueError('wrong name of the model')

    dt = datetime.now()
    model_path = args['name'] + \
        f'_{dt.date()}_{dt.hour}-{dt.minute}-{dt.second}.pth'
    model_path = os.path.join(os.getcwd(), 'saved_models', model_path)
    early_stop = u.EarlyStopping(model_path, patience=args['patience'])

    # log
    log_path = args['name'] + f'_{dt.date()}_{dt.hour}-{dt.minute}-{dt.second}'
    log_path = os.path.join(os.getcwd(), 'log', log_path)
    log = [str(args) + '\n']
    log.append('epoch train_MAE train_RMSE test_MAE test_RMSE elapse\n')

    # loss, optimizer
    loss_func = t.nn.MSELoss()
    optimizer = t.optim.Adam(
        model.parameters(), lr=args['lr'], weight_decay=args['decay'])
    # scheduler = t.optim.lr_scheduler.StepLR(optimizer, 100, .1)

    # train
    for epoch in range(args['epochs']):
        dt = datetime.now()

        model.train()
        y_pred = model(g_homo, feat1, g_list, feat2, pairs)
        loss = loss_func(y_pred[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()

        train_mae, train_rmse = u.metrics(
            y_pred[train_mask].detach(), labels[train_mask])
        test_mae, test_rmse = u.evaluate(
            model, g_homo, feat1, g_list, feat2, pairs, labels, test_mask)
        stop = early_stop.step(test_rmse, test_mae, model)

        elapse = str(datetime.now() - dt)[:10] + '\n'
        log.append(' '.join(str(x) for x in (epoch, train_mae,
                                             train_rmse, test_mae, test_rmse, elapse)))
        print(f'epoch={epoch} | train_MAE={train_mae} | train_RMSE={train_rmse} | test_MAE={test_mae} | test_RMSE={test_rmse} | elapse={elapse}')

        if stop:
            break

    early_stop.load_checkpoint(model)
    test_mae, test_rmse = u.evaluate(
        model, g_homo, feat1, g_list, feat2, pairs, labels, test_mask)
    print(f'test_MAE={test_mae} | test_RMSE={test_rmse}')

    # save log
    with open(log_path, 'w') as f:
        f.writelines(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SRG model')
    parser.add_argument('-n', '--name', type=str,
                        default='ciao', help='dataset name, ciao or epinions')
    parser.add_argument('-s', '--seed', type=int,
                        default=2020, help='random seed')
    parser.add_argument('-ts', '--train_size', type=float,
                        default=0.8, help='train set size')
    parser.add_argument('-es', '--embedding_size', type=int,
                        default=64, help='embedding size of user/item')
    parser.add_argument('-ifs', '--in_feats', type=int,
                        default=128, help='input feature size')
    parser.add_argument('-nb', '--num_b', type=int,
                        default=3, help='# of RecGCN blocks')
    parser.add_argument('-do', '--drop_out', type=float,
                        default=0.5, help='dropout rate')
    parser.add_argument('-e', '--epochs', type=int,
                        default=800, help='# of epochs')
    parser.add_argument('-m', '--model', type=str,
                        default='SRG', help='SRG, SRG_GAT, SRG_no_GRU, SRG_no_GCN or SRG_Res')
    parser.add_argument('-nl', '--num_l', type=int,
                        default=2, help='# of GCN layers for SRG_no_GRU/SRG_Res')
    parser.add_argument('-fcu', '--fc_units', type=int,
                        default=32, help='# of units for fully connected layer')
    command_line_args = parser.parse_args().__dict__

    args = {
        'lr': .005,  # learning rate
        'decay': .001,  # weight decay
        'epochs': 800,
        'patience': 100,
        'device': 'cuda' if t.cuda.is_available() else 'cpu',
        'num_meta_path': 5,
        'head_list': [8],
        'fc_units': 32,
        'name': 'ciao',  # dataset name, ciao or epinions
        'seed': 2020,
        'train_size': .8,  # train set size
        'embedding_size': 64,  # embedding size of user/item
        'in_feats': 128,  # input feature size
        'num_b': 3,  # the number of RecGCN/RecGAT blocks
        'drop_out': .5,  # drop out rate
        'num_l': 2,  # tthe number of GCN layers for SRG_no_GRU/SRG_Res
        'model': 'SRG_GAT'  # SRG, SRG_GAT, SRG_no_GRU, SRG_no_GCN or SRG_Res
    }
    args.update(command_line_args)

    random.seed(args['seed'])
    t.manual_seed(args['seed'])
    if args['device'] == 'cuda':
        t.cuda.manual_seed(args['seed'])

    main(args)
