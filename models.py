# -*- coding: utf-8 -*-
'''
@File    : model.py
@Time    : 2020
@Author  : ZHOU, YANG
@Contact : yzhou.x@icloud.com
'''

import torch as t
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv


class RecGCNblock(nn.Module):
    '''Recurrent GCN block with GRU.

    Parameter
    ---------
        in_feats: int, input feature size
        out_feats: int, output feature size
    Input
    -----
        graph: dgl.DGLGraph
        h: (#nodes, in_feats), input features
    Output
    ------
        h: (#nodes, out_feats), output features
    '''

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.gcn_layer = GraphConv(in_feats, out_feats)
        self.gru_cell = nn.GRUCell(out_feats, out_feats)

    def forward(self, graph, h, first_cell=False):
        x = self.gcn_layer(graph, h)
        if first_cell:
            h = self.gru_cell(x)
        else:
            h = self.gru_cell(x, h)
        return F.elu(h)


class RecGCN(nn.Module):
    '''Recurrent GCN with GRU.

    Parameter
    ----------
        in_feats: int, # of features in the inputs
        out_feats: int, # of features in the outputs
        num_blocks: int, # of blocks
        dropout: float, dropout rate for input features in each block
    Input
    -----
        graph: dgl.DGLGraph
        h: (#nodes, in_feats), features of nodes
    Output
    ------
        h: (#nodes, out_feats), output features
    '''

    def __init__(self, in_feats, out_feats, num_blocks, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([RecGCNblock(in_feats, out_feats)])
        self.blocks.extend(
            [RecGCNblock(out_feats, out_feats) for _ in range(num_blocks - 1)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, h):
        h = self.dropout(h)
        h = self.blocks[0](graph, h, first_cell=True)
        for block in self.blocks[1:]:
            h = self.dropout(h)
            h = block(graph, h)
        return h


class RecGATblock(nn.Module):
    '''Recurrent GAT block with GRU.

    Parameter
    ---------
        in_feats: int, input feature size
        out_feats: int, output feature size
        num_heads: int, # of attention head
        dropout: float, dropout rate for GAT layer
    Input
    -----
        graph: dgl.DGLGraph
        h: (#nodes, in_feats), input features
    Output
    ------
        h: (#nodes, out_feats), output features
    '''

    def __init__(self, in_feats, out_feats, num_heads=8, dropout=0.):
        super().__init__()
        self.gat_layer = GATConv(
            in_feats, out_feats, num_heads, dropout, dropout)
        self.gru_cell = nn.GRUCell(out_feats * num_heads, out_feats)

    def forward(self, graph, h, first_cell=False):
        x = self.gat_layer(graph, h)
        x = x.view(h.size()[0], -1)
        if first_cell:
            h = self.gru_cell(x)
        else:
            h = self.gru_cell(x, h)
        return F.elu(h)


class RecGAT(nn.Module):
    '''Recurrent GAT with GRU.

    Parameter
    ----------
        in_feats: int, the number of features in the inputs
        out_feats: int, the number of features in the outputs
        num_blocks: int, the number of blocks
        dropout: float, dropout rate for GAT layer
    Input
    -----
        graph: dgl.DGLGraph
        h: (#nodes, in_feats), features of nodes
    Output
    ------
        h: (#nodes, out_feats), output features
    '''

    def __init__(self, in_feats, out_feats, num_blocks, dropout):
        super().__init__()
        self.blocks = nn.ModuleList(
            [RecGATblock(in_feats, out_feats, dropout=dropout)]
        )
        self.blocks.extend(
            [RecGATblock(out_feats, out_feats, dropout=dropout)
             for _ in range(num_blocks - 1)]
        )

    def forward(self, graph, h):
        h = self.blocks[0](graph, h, first_cell=True)
        for block in self.blocks[1:]:
            h = block(graph, h)
        return h


class SemanticAttention(nn.Module):
    '''semantic attention for HAN.
    from https://github.com/dmlc/dgl/tree/master/examples/pytorch/han
    '''

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
    '''Social Recommendation model based on Recurrent GCNs.

    Parameter
    ---------
        rgcn_in_feats: int, input features size for Recurrent GCNs
        rgcn_out_feats: int, output features size for Recurrent GCNs
        rgcn_num_blocks: int, # of blocks for Recurrent GCNs
        rgcn_dropout: float, dropout rate for input features in each block
        han_num_meta_path: int, # of meta-path for HAN
        han_in_feats: int, input features size for HAN
        han_hidden_feats: int, hidden features size for HAN
        han_head_list: List[int], # of heads for each HAN layer
        han_dropout: float, dropout rate for GAT layer in HAN
        fc_hidden_feats: int, hidden feature size for fully connected layer
    Input
    -----
        g_homo: dgl.DGLGraph, for Recurrent GCNs
        feat1: (#nodes, rgcn_in_feats), features of user nodes
        g_list: List[dgl.DGLGraph]
        feat2: (#nodes, han_in_feats), , features of user/item nodes
        pairs: [[user_id, item_id],...]
    Output
    ------
        ratings, tensor with gradient
    '''

    def __init__(self,
                 rgcn_in_feats,
                 rgcn_out_feats,
                 rgcn_num_blocks,
                 rgcn_dropout,
                 han_num_meta_path,
                 han_in_feats,
                 han_hidden_feats,
                 han_head_list,
                 han_dropout,
                 fc_hidden_feats
                 ):
        super().__init__()
        self.recurrent_gcn = RecGCN(
            rgcn_in_feats, rgcn_out_feats, rgcn_num_blocks, rgcn_dropout)
        self.han = HAN(num_meta_paths=han_num_meta_path,
                       in_feats=han_in_feats,
                       hidden_feats=han_hidden_feats,
                       head_list=han_head_list,
                       dropout=han_dropout)
        self.fc = nn.Sequential(
            nn.Linear(rgcn_out_feats + han_hidden_feats *
                      han_head_list[-1] * 2, fc_hidden_feats),
            nn.ELU(),
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


class SRG_GAT(nn.Module):
    '''Social Recommendation model based on Recurrent GCNs.

    Parameter
    ---------
        rgcn_in_feats: int, input features size for Recurrent GCNs
        rgcn_out_feats: int, output features size for Recurrent GCNs
        rgcn_num_blocks: int, # of blocks for Recurrent GCNs
        rgcn_dropout: float, dropout rate for GAT layer
        han_num_meta_path: int, # of meta-path for HAN
        han_in_feats: int, input features size for HAN
        han_hidden_feats: int, hidden features size for HAN
        han_head_list: List[int], # of heads for each HAN layer
        han_dropout: float, dropout rate for GAT layer in HAN
        fc_hidden_feats: int, hidden feature size for fully connected layer
    Input
    -----
        g_homo: dgl.DGLGraph, for Recurrent GCNs
        feat1: (#nodes, rgcn_in_feats), features of user nodes
        g_list: List[dgl.DGLGraph]
        feat2: (#nodes, han_in_feats), , features of user/item nodes
        pairs: [[user_id, item_id],...]
    Output
    ------
        ratings, tensor with gradient
    '''

    def __init__(self,
                 rgcn_in_feats,
                 rgcn_out_feats,
                 rgcn_num_blocks,
                 rgcn_dropout,
                 han_num_meta_path,
                 han_in_feats,
                 han_hidden_feats,
                 han_head_list,
                 han_dropout,
                 fc_hidden_feats
                 ):
        super().__init__()
        self.recurrent_gat = RecGAT(
            rgcn_in_feats, rgcn_out_feats, rgcn_num_blocks, rgcn_dropout)
        self.han = HAN(num_meta_paths=han_num_meta_path,
                       in_feats=han_in_feats,
                       hidden_feats=han_hidden_feats,
                       head_list=han_head_list,
                       dropout=han_dropout)
        self.fc = nn.Sequential(
            nn.Linear(rgcn_out_feats + han_hidden_feats *
                      han_head_list[-1] * 2, fc_hidden_feats),
            nn.ELU(),
            nn.Linear(fc_hidden_feats, 1)
        )

    def forward(self, g_homo, feat1, g_list, feat2, pairs):
        user_feat = self.recurrent_gat(g_homo, feat1)
        user_item_feat = self.han(g_list, feat2)
        feat = t.cat((
            user_feat[pairs[:, 0]],
            user_item_feat[pairs[:, 0]],
            user_item_feat[pairs[:, 1]]
        ), dim=1)
        return self.fc(feat)


class SRG_no_GRU(nn.Module):
    '''SRG without GRU.

    Parameter
    ---------
        gcn_in_feats: int, input features size for GCN
        gcn_out_feats: int, output features size for GCN
        gcn_num_layers: int, the number of blocks for GCN
        han_num_meta_path: int, the number of meta-path for HAN
        han_in_feats: int, input features size for HAN
        han_hidden_feats: int, hidden features size for HAN
        han_head_list: List[int], the number of heads for each layer
        han_dropout: int, drop out rate
        fc_hidden_feats: int, hidden feature size for fully connected layer
    Input
    -----
        g_homo: dgl.DGLGraph, for GCN
        feat1: (#nodes, gcn_in_feats), features of user nodes
        g_list: List[dgl.DGLGraph]
        feat2: (#nodes, han_in_feats), , features of user/item nodes
        pairs: [[user_id, item_id],...]
    Output
    ------
        ratings, tensor with gradient
    '''

    def __init__(self,
                 gcn_in_feats,
                 gcn_out_feats,
                 gcn_num_layers,
                 han_num_meta_path,
                 han_in_feats,
                 han_hidden_feats,
                 han_head_list,
                 han_dropout,
                 fc_hidden_feats
                 ):
        super().__init__()
        self.gcns = nn.ModuleList(
            [GraphConv(gcn_in_feats, gcn_out_feats, activation=F.relu)])
        self.gcns.extend(
            [GraphConv(gcn_out_feats, gcn_out_feats, activation=F.relu)
             for _ in range(gcn_num_layers - 1)]
        )
        self.han = HAN(num_meta_paths=han_num_meta_path,
                       in_feats=han_in_feats,
                       hidden_feats=han_hidden_feats,
                       head_list=han_head_list,
                       dropout=han_dropout)
        self.fc = nn.Sequential(
            nn.Linear(gcn_out_feats + han_hidden_feats *
                      han_head_list[-1] * 2, fc_hidden_feats),
            nn.ELU(),
            nn.Linear(fc_hidden_feats, 1)
        )

    def forward(self, g_homo, feat1, g_list, feat2, pairs):
        for gcn in self.gcns:
            feat1 = gcn(g_homo, feat1)
        feat2 = self.han(g_list, feat2)
        feat = t.cat((
            feat1[pairs[:, 0]],
            feat2[pairs[:, 0]],
            feat2[pairs[:, 1]]
        ), dim=1)
        return self.fc(feat)


class SRG_Res(nn.Module):
    '''SRG without GRU, but residual.

    Parameter
    ---------
        gcn_in_feats: int, input features size for GCN
        gcn_out_feats: int, output features size for GCN
        gcn_num_layers: int, the number of blocks for GCN
        han_num_meta_path: int, the number of meta-path for HAN
        han_in_feats: int, input features size for HAN
        han_hidden_feats: int, hidden features size for HAN
        han_head_list: List[int], the number of heads for each layer
        han_dropout: int, drop out rate
        fc_hidden_feats: int, hidden feature size for fully connected layer
    Input
    -----
        g_homo: dgl.DGLGraph, for GCN
        feat1: (#nodes, gcn_in_feats), features of user nodes
        g_list: List[dgl.DGLGraph]
        feat2: (#nodes, han_in_feats), , features of user/item nodes
        pairs: [[user_id, item_id],...]
    Output
    ------
        ratings, tensor with gradient
    '''

    def __init__(self,
                 gcn_in_feats,
                 gcn_out_feats,
                 gcn_num_layers,
                 han_num_meta_path,
                 han_in_feats,
                 han_hidden_feats,
                 han_head_list,
                 han_dropout,
                 fc_hidden_feats
                 ):
        super().__init__()
        self.gcns = nn.ModuleList(
            [GraphConv(gcn_in_feats, gcn_out_feats, activation=F.relu)])
        self.gcns.extend(
            [GraphConv(gcn_out_feats, gcn_out_feats, activation=F.relu)
             for _ in range(gcn_num_layers - 1)]
        )
        self.han = HAN(num_meta_paths=han_num_meta_path,
                       in_feats=han_in_feats,
                       hidden_feats=han_hidden_feats,
                       head_list=han_head_list,
                       dropout=han_dropout)
        self.fc = nn.Sequential(
            nn.Linear(gcn_out_feats + han_hidden_feats *
                      han_head_list[-1] * 2, fc_hidden_feats),
            nn.ELU(),
            nn.Linear(fc_hidden_feats, 1)
        )

    def forward(self, g_homo, feat1, g_list, feat2, pairs):
        feat1 = self.gcns[0](g_homo, feat1)
        for gcn in self.gcns[1:]:
            feat1 = gcn(g_homo, feat1) + feat1
        feat2 = self.han(g_list, feat2)
        feat = t.cat((
            feat1[pairs[:, 0]],
            feat2[pairs[:, 0]],
            feat2[pairs[:, 1]]
        ), dim=1)
        return self.fc(feat)


class SRG_no_GCN(nn.Module):
    '''SRG without recurrent GCNs.

    Parameter
    ---------
        han_num_meta_path: int, the number of meta-path for HAN
        han_in_feats: int, input features size for HAN
        han_hidden_feats: int, hidden features size for HAN
        han_head_list: List[int], the number of heads for each layer
        han_dropout: int, drop out rate
        fc_hidden_feats: int, hidden feature size for fully connected layer
    Input
    -----
        g_homo: placeholder
        feat1: placeholder
        g_list: List[dgl.DGLGraph]
        feat2: (#nodes, han_in_feats), , features of user/item nodes
        pairs: [[user_id, item_id],...]
    Output
    ------
        ratings, tensor with gradient
    '''

    def __init__(self,
                 han_num_meta_path,
                 han_in_feats,
                 han_hidden_feats,
                 han_head_list,
                 han_dropout,
                 fc_hidden_feats
                 ):
        super().__init__()
        self.han = HAN(num_meta_paths=han_num_meta_path,
                       in_feats=han_in_feats,
                       hidden_feats=han_hidden_feats,
                       head_list=han_head_list,
                       dropout=han_dropout)
        self.fc = nn.Sequential(
            nn.Linear(han_hidden_feats*han_head_list[-1] * 2, fc_hidden_feats),
            nn.ELU(),
            nn.Linear(fc_hidden_feats, 1)
        )

    def forward(self, g_homo, feat1, g_list, feat2, pairs):
        feat2 = self.han(g_list, feat2)
        feat2 = t.cat((feat2[pairs[:, 0]], feat2[pairs[:, 1]]), dim=1)
        return self.fc(feat2)
