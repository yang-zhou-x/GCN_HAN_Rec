# -*- coding: utf-8 -*-
'''
@File    : model.py
@Time    : 2020
@Author  : ZHOU, YANG
@Contact : yzhou.x@icloud.com
'''

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv
from dgl import metapath_reachable_graph


class RecGCNBlock(nn.Module):
    '''Recurrent GCN block with GRU.

    Parameter
    ---------
        in_feats: int, input feature size
        out_feats: int, output feature size
    Input
    -----
        graph: dgl.DGLGraph
        feat: (#nodes, in_feats)
    Output
    ------
        feat: (#nodes, out_feats)
    '''

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.gcn_layer = GraphConv(in_feats, out_feats)
        self.gru_cell = nn.GRUCell(out_feats, out_feats)

    def forward(self, graph, feat):
        feat = self.gcn_layer(graph, feat)
        feat = self.gru_cell(feat)
        return F.relu(feat)


class RecGCN(nn.Module):
    '''Recurrent GCN with GRU.

    Parameter
    ----------
        in_feats: int, the number of features in the inputs
        out_feats: int, the number of features in the outputs
        num_blocks: int, default 4
    Input
    -----
        graph: dgl.DGLGraph
        feat: (#nodes, in_feats), features of nodes
    Output
    ------
        feat: (#nodes, out_feats), output features
    '''

    def __init__(self, in_feats, out_feats, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [RecGCNBlock(in_feats, in_feats) for _ in range(num_blocks - 1)])
        self.blocks.append(RecGCNBlock(in_feats, out_feats))

    def forward(self, graph, feat):
        for block in self.blocks:
            feat = block(graph, feat)
        return feat


class SemanticAttention(nn.Module):
    '''semantic attention for HAN'''

    def __init__(self, in_size, hidden_size=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.layers(z)
        beta = F.softmax(w, dim=1)
        return (beta * z).sum(1)


class HANLayer(nn.Module):
    """HAN layer.
    from https://github.com/dmlc/dgl/tree/master/examples/pytorch/han 

    Arguments
    ---------
        num_meta_paths : number of homogeneous graphs generated from the metapaths
        in_size : input feature dimension
        out_size : output feature dimension
        layer_num_heads : number of attention heads
        dropout : Dropout probability
    Inputs
    ------
        g : list[DGLGraph]
            List of graphs
        h : tensor
            Input features
    Outputs
    -------
        tensor
            The output feature
    """

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for _ in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []
        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = t.stack(
            semantic_embeddings, dim=1)  # (N, M, D * K)
        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    '''Heterogeneous Graph Attention Network.

    Parameter
    ---------
        num_meta_paths: number of homogeneous graphs generated from the metapaths
        in_feats: input feature size
        hidden_feats: hidden feature size
        head_list: attention heads for each layer
        dropout: dropout probability
    Input
    -----
        graph_list: List[dgl.DGLGraph]
        feat: (#nodes, in_feats), original features
    Output
    ------
        feat: (#nodes, hidden_feats * head_list[-1]), output features
    '''

    def __init__(self, num_meta_paths, in_feats, hidden_feats, head_list, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [HANLayer(num_meta_paths, in_feats,
                      hidden_feats, head_list[0], dropout)]
        )
        for l in range(1, len(head_list)):
            self.layers.append(HANLayer(num_meta_paths, hidden_feats * head_list[l-1],
                                        hidden_feats, head_list[l], dropout))

    def forward(self, graph_list, feat):
        for layer in self.layers:
            feat = layer(graph_list, feat)
        return feat


class SRG(nn.Module):
    '''Social Recommendation model based on GCN-GRU(SRG).

    Parameter
    ---------
        rgcn_in_feats: int, input features size for RecGCN
        rgcn_out_feats: int, output features size for RecGCN
        rgcn_num_blocks: int
        han_num_meta_path: int
        han_in_feats: int
        han_hidden_feats: int
        han_head_list: List[int]
        han_dropout: int
        fc_hidden_feats: int, hidden feature size for fully connected layer
    Input
    -----
        g_homo: dgl.DGLGraph, for RecGCN
        feat1: (#nodes, rgcn_in_feats), features of user nodes
        g_list: List[dgl.DGLGraph]
        feat2: (#nodes, han_in_feats), , features of user/item nodes
        pairs: [[user_id, item_id],...]
    Output
    ------
        score
    '''

    def __init__(self,
                 rgcn_in_feats,
                 rgcn_out_feats,
                 rgcn_num_blocks,
                 han_num_meta_path,
                 han_in_feats,
                 han_hidden_feats,
                 han_head_list,
                 han_dropout,
                 fc_hidden_feats
                 ):
        super().__init__()
        self.recurrent_gcn = RecGCN(
            rgcn_in_feats, rgcn_out_feats, rgcn_num_blocks)
        self.han = HAN(num_meta_paths=han_num_meta_path,
                       in_feats=han_in_feats,
                       hidden_feats=han_hidden_feats,
                       head_list=han_head_list,
                       dropout=han_dropout)
        self.fc = nn.Sequential(
            nn.Linear(rgcn_out_feats + han_hidden_feats *
                      han_head_list[-1] * 2, fc_hidden_feats),
            nn.Linear(fc_hidden_feats, 1)
        )

    def forward(self, g_homo, feat1, g_list, feat2, pairs):
        user_feat = self.recurrent_gcn(g_homo, feat1)
        user_item_feat = self.han(g_list, feat2)
        feat = t.cat((
            user_feat[pairs[:, 0]],
            user_item_feat[pairs[:, 0]],
            user_item_feat[pairs[:, 1]]
        ), dim=1)
        return self.fc(feat)
