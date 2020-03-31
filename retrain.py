# -*- coding: utf-8 -*-
'''
@File    : retrain.py
@Time    : 2020/03
@Author  : ZHOU, YANG
@Contact : yzhou.x@icloud.com
'''

import torch as t
import utils as u
import models as m
from datetime import datetime
import os
import random
import argparse
import pandas as pd


def retrain(args):
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

    model.load_state_dict(t.load(args['model_path']))

    # log
    log = []
    df = pd.read_csv(args['log_path'], sep=' ', header=1,
                     usecols=['test_MAE', 'test_RMSE'])
    mae, rmse = df.min()
    model_path = args['model_path'][:-4] + '(2).pth'
    early_stop = u.EarlyStopping(
        model_path, patience=args['patience'], rmse=rmse, mae=mae)

    # loss, optimizer
    loss_func = t.nn.MSELoss()
    optimizer = t.optim.Adam(
        model.parameters(), lr=args['lr'], weight_decay=args['decay'])

    # train
    for epoch in range(args['epochs']):
        dt = datetime.now()

        model.train()
        y_pred = model(g_homo, feat1, g_list, feat2, pairs)
        loss = loss_func(y_pred[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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

    # save log
    with open(args['log_path'], 'a') as f:
        f.writelines(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retrain model')
    parser.add_argument('-l', '--log', type=str,
                        help='log name, e.g. ciao_2020-03-30_21-12-32')
    parser.add_argument('-e', '--epochs', type=int,
                        default=300, help='# of epochs for retraining')
    command_line_args = parser.parse_args().__dict__

    path = os.path.join(os.getcwd(), 'log', command_line_args['log'])
    command_line_args['log_path'] = path
    path = os.path.join(os.getcwd(), 'saved_models',
                        command_line_args['log'] + '.pth')
    command_line_args['model_path'] = path

    with open(command_line_args['log_path'], 'r') as f:
        args = f.readline().strip()
        args = eval(args)

    args.update(command_line_args)

    random.seed(args['seed'])
    t.manual_seed(args['seed'])
    if args['device'] == 'cuda':
        t.cuda.manual_seed(args['seed'])

    retrain(args)
