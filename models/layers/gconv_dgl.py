import math

import torch
import torch.nn as nn
from torch.nn import init

from dgl.base import DGLError
import dgl.function as fn

from utils.tensor import dot
import dgl
import dgl.nn as dgl_nn


# pylint: disable=W0235
class GraphConv_Customized(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None):
        super(GraphConv_Customized, self).__init__()
        if norm not in ('none', 'both', 'left', 'right', 'chebmat'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        # NOTE: We reset param in GConv for matching the pytorch implementation init sequence
        #self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat, weight=None):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: "math:`(\text{in_feats}, \text{out_feats})`.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        graph = graph.local_var()

        if self._norm == 'both':
            degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat = feat * norm
        
        if self._norm == 'left':
            degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
            norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat = norm * feat

        if weight is not None:
            if self.weight is not None:
                raise DGLError('External weight is provided while at the same time the'
                               ' module has defined its own weight parameter. Please'
                               ' create the module with flag weight=False.')
        else:
            weight = self.weight

        if weight is not None:
            mult_rst = torch.matmul(feat, weight)

        # aggregate first then mult W
        graph.srcdata['h'] = mult_rst
        
        if self._norm == 'chebmat':
            #graph.edata['chebmat'].to(feat.device)
            graph.to(feat.device)
            graph.update_all(fn.u_mul_e('h', 'chebmat', 'm'),
                            fn.sum(msg='m', out='h'))
        else:
            graph.update_all(fn.copy_src(src='h', out='m'),
                            fn.sum(msg='m', out='h'))

        rst = graph.dstdata['h']

        if self._norm in ['both', 'right']:
            degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
            if self._norm == 'both':
                norm = torch.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst


    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)



class GConv(nn.Module):
    """DGL Implementation of GCN
    """

    def __init__(self, in_features, out_features, dgl_g, bias=True):
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.dgl_g = dgl_g
        #self.gconv = dgl_nn.conv.GraphConv(in_features, out_features, bias=False)
        #self.gconv = dgl_nn.conv.ChebConv(in_features, out_features, 1, bias=False)
        self.gconv = GraphConv_Customized(in_features, out_features, norm='chebmat', bias=False)
        self.loop_weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.gconv.reset_parameters()
        #nn.init.xavier_uniform_(self.gconv.weight.data)
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
