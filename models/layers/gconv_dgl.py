import math

import torch
import torch.nn as nn

from utils.tensor import dot
import dgl
import dgl.nn as dgl_nn


class GConv(nn.Module):
    """DGL Implementation of GCN
    """

    def __init__(self, in_features, out_features, dgl_g, bias=True):
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.dgl_g = dgl_g
        #self.gconv = dgl_nn.conv.GraphConv(in_features, out_features, bias=False)
        self.gconv = dgl_nn.conv.ChebConv(in_features, out_features, 1, bias=False)
        self.loop_weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.gconv.reset_parameters()
        nn.init.xavier_uniform_(self.loop_weight.data)

    def forward(self, inputs):
        support_loop = torch.matmul(inputs, self.loop_weight)
        batch_size = inputs.shape[0]
        fea_dim = inputs.shape[-1]
        batch_dgl_g = dgl.batch([self.dgl_g for k in range(batch_size)])
        gcn_res = self.gconv(batch_dgl_g, inputs.view(-1, fea_dim)).reshape(support_loop.shape)
        output = gcn_res + support_loop
        if self.bias is not None:
            ret = output + self.bias
        else:
            ret = output
        return ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
