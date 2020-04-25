# -*- coding: utf-8 -*-
'''
@File    : utils.py
@Time    : 2020/02
@Author  : ZHOU, YANG
@Contact : yzhou.x@icloud.com
'''

import os
import random
import numpy as np
import pandas as pd
import torch as t
from dgl import DGLGraph


def get_binary_mask(size, train_indices):
    '''get binary mask for train/val/test dataset.  
    |val set|:|test set| == 1:1

    Parameter
    ---------
        size: int, total size
        train_indices: list
    Return
    ------
        train_mask: np.array[bool]
        val_mask: np.array[bool]
        test_mask: np.array[bool]
    '''
    train_mask = np.zeros(size, dtype=np.bool)
    train_mask[train_indices] = True

    val_test_size = size - len(train_indices)
    val_size = int(val_test_size / 2)
    test_size = val_test_size - val_size

    val_test_mask = (1 - train_mask).astype(bool)
    val_test_indices = np.arange(size)[val_test_mask].tolist()
    val_indices = random.sample(val_test_indices, val_size)
    test_indices = list(set(val_test_indices).difference(set(val_indices)))

    val_mask = np.zeros(size, dtype=np.bool)
    val_mask[val_indices] = True
    test_mask = np.zeros(size, dtype=np.bool)
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask


def _load_data_epinions(train_size):
    '''loading dataset Epinions.'''
    # social relation data
    init_path = os.getcwd()
    trust_path = os.path.join(init_path, 'datasets/Epinions/trust_data.txt')
    trust = pd.read_csv(trust_path, sep=' ', header=None,
                        names=['1', 'source_user_id', 'target_user_id', 'trust_statement_value'])
    trust.drop(columns='1', inplace=True)

    # user-item data
    rating_path = os.path.join(
        init_path, 'datasets/Epinions/ratings_data.txt')
    rating = pd.read_csv(rating_path, sep=' ', header=None,
                         names=['user_id', 'item_id', 'rating_value'])

    # social graph
    num_users = max(trust.iloc[:, 0].max(),
                    trust.iloc[:, 1].max(), rating.iloc[:, 0].max())
    g_homo = DGLGraph()
    g_homo.add_nodes(num_users)  # number of users
    edge_list = trust.iloc[:, :2].to_numpy()
    edge_list -= 1  # counting from 0
    src, dst = tuple(zip(*edge_list.tolist()))
    g_homo.add_edges(src, dst)
    g_homo.add_edges(dst, src)
    g_homo.add_edges(g_homo.nodes(), g_homo.nodes())  # add self-loop

    # user-item graph
    num_items = rating.iloc[:, 1].max()  # number of items
    rating.iloc[:, :2] -= 1  # counting from 0
    rating.iloc[:, 1] += num_users  # for item indices

    pairs = rating.iloc[:, :2].to_numpy()
    labels = rating.iloc[:, 2].to_numpy()

    train_size = int(rating.shape[0] * train_size)
    train_idx = random.sample(range(rating.shape[0]), train_size)
    train_mask, val_mask, test_mask = get_binary_mask(
        rating.shape[0], train_idx)

    g_list = []
    for i in range(1, 6):
        sub = rating[train_mask].query(f'rating_value == {i}').iloc[:, :2]\
            .to_numpy().tolist()
        tmp_graph = DGLGraph()
        tmp_graph.add_nodes(num_items + num_users)
        src, dst = tuple(zip(*sub))
        tmp_graph.add_edges(src, dst)
        tmp_graph.add_edges(dst, src)
        tmp_graph.add_edges(tmp_graph.nodes(),
                            tmp_graph.nodes())  # add self-loop
        g_list.append(tmp_graph)

    return g_homo, g_list, pairs, labels, train_mask, val_mask, test_mask


def _load_data_ciao(train_size):
    '''loading dataset Ciao.'''
    # social relation data
    init_path = os.getcwd()
    trust_path = os.path.join(init_path, 'datasets/Ciao/trusts.txt')
    trust = pd.read_csv(trust_path, sep=',', header=None,
                        names=['trustorID', 'trusteeID', 'trustValue'])

    # user-item data
    rating_path = os.path.join(
        init_path, 'datasets/Ciao/movie-ratings.txt')
    rating = pd.read_csv(rating_path, sep=',', header=None,
                         names=['userId', 'movieId', 'movieRating'], usecols=[0, 1, 4])

    # social graph
    num_users = max(trust.iloc[:, 0].max(),
                    trust.iloc[:, 1].max(), rating.iloc[:, 0].max())
    g_homo = DGLGraph()
    g_homo.add_nodes(num_users)  # number of users
    edge_list = trust.iloc[:, :2].to_numpy()
    edge_list -= 1  # counting from 0
    src, dst = tuple(zip(*edge_list.tolist()))
    g_homo.add_edges(src, dst)
    g_homo.add_edges(dst, src)
    g_homo.add_edges(g_homo.nodes(), g_homo.nodes())  # add self-loop

    # user-item graph
    num_items = rating.iloc[:, 1].max()  # number of items
    rating.iloc[:, :2] -= 1  # counting from 0
    rating.iloc[:, 1] += num_users  # for item indices

    pairs = rating.iloc[:, :2].to_numpy()
    labels = rating.iloc[:, 2].to_numpy()

    train_size = int(rating.shape[0] * train_size)
    train_idx = random.sample(range(rating.shape[0]), train_size)
    train_mask, val_mask, test_mask = get_binary_mask(
        rating.shape[0], train_idx)

    g_list = []
    for i in range(1, 6):
        sub = rating[train_mask].query(f'movieRating == {i}').iloc[:, :2]\
            .to_numpy().tolist()
        tmp_graph = DGLGraph()
        tmp_graph.add_nodes(num_items + num_users)
        src, dst = tuple(zip(*sub))
        tmp_graph.add_edges(src, dst)
        tmp_graph.add_edges(dst, src)
        tmp_graph.add_edges(tmp_graph.nodes(),
                            tmp_graph.nodes())  # add self-loop
        g_list.append(tmp_graph)

    return g_homo, g_list, pairs, labels, train_mask, val_mask, test_mask


def load_data(name, train_size=0.8):
    '''loading dataset.
    |val set|:|test set| == 1:1

    Parameter
    ---------
        name: str, 'epinions' or 'ciao'
        train_size: float
    Return
    ---------
        g_homo: dgl.DGLGraph, social graph
        g_list: List[dgl.DGLGraph], user-item meta graphs
        pairs: np.array, user-item pairs
        labels: np.array, rating labels
        train_mask: np.array(bool)
        val_mask: np.array(bool)
        test_mask: np.array(bool)
    '''
    if name == 'epinions':
        return _load_data_epinions(train_size)
    elif name == 'ciao':
        return _load_data_ciao(train_size)
    else:
        raise ValueError('wrong name of dataset')


def mean_absolute_error(y_pred, y_true):
    ''' MAE.

    Parameter
    ---------
        y_pred: np.array
        y_ture: np.array
    Return
    ------
        MAE: float
    '''
    if y_pred.shape == y_true.shape:
        return np.sum(np.abs(y_pred - y_true)) / y_pred.shape[0]
    else:
        raise ValueError('the shape of inputs is wrong.')


def root_mean_squared_error(y_pred, y_true):
    ''' RMSE.

    Parameter
    ---------
        y_pred: np.array
        y_ture: np.array
    Return
    ------
        RMSE: float
    '''
    if y_pred.shape == y_true.shape:
        return np.sqrt(np.sum(np.square(y_pred - y_true)) / y_pred.shape[0])
    else:
        raise ValueError('the shape of inputs is wrong.')


def metrics(y_pred, y_true, decimal=4):
    '''metrics for evaluating.

    Parameters
    ----------
        y_pred: torch.tensor, w/o gradient
        y_true: torch.tensor, w/o gradient
        decimal: int, decimal places
    Returns
    -------
        mae: float, mean absolute error
        rmse: float, root mean squared error
    '''
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    mae = mean_absolute_error(y_pred, y_true)
    rmse = root_mean_squared_error(y_pred, y_true)
    return round(mae, decimal), round(rmse, decimal)


def evaluate(model, g_homo, feat1, g_list, feat2, pairs, labels, mask):
    '''model evaluation.

    Parameter
    ---------
        model: torch model
        g_homo: dgl.DGLGraph, for Recurrent GCNs
        feat1: (#nodes, rgcn_in_feats), features of user nodes
        g_list: List[dgl.DGLGraph]
        feat2: (#nodes, han_in_feats), , features of user/item nodes
        pairs: [[user_id, item_id],...]
        labels: tensor, w/o gradient
        mask: tensor, w/o gradient
    Return
    ------
        mae, float
        rmse, float
    '''
    model.eval()
    with t.no_grad():
        y_pred = model(g_homo, feat1, g_list, feat2, pairs)
    mae, rmse = metrics(y_pred[mask].detach(), labels[mask])
    return mae, rmse


class EarlyStopping:
    def __init__(self, file_path, patience=30, rmse=None, mae=None):
        self.file_path = file_path
        self.patience = patience
        self.counter = 0
        self.curr_rmse = rmse
        self.curr_mae = mae
        self.early_stop = False

    def step(self, rmse, mae, model):
        if self.curr_rmse is None:
            self.curr_rmse = rmse
            self.curr_mae = mae
            self.save_checkpoint(model)
        elif rmse >= self.curr_rmse and mae >= self.curr_mae:
            self.counter += 1
            print(f'Early Stopping Counter: {self.counter} / {self.patience}.')
            if self.counter == self.patience:
                self.early_stop = True
        else:
            if rmse <= self.curr_rmse and mae <= self.curr_mae:
                self.save_checkpoint(model)
                self.curr_rmse = rmse
                self.curr_mae = mae
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        t.save(model.state_dict(), self.file_path)

    def load_checkpoint(self, model):
        model.load_state_dict(t.load(self.file_path))
