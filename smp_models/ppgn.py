# Code taken from https://github.com/cvignac/SMP
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_scatter import scatter


class PowerfulLayer(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, num_layers: int, activation=nn.LeakyReLU()):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat= out_feat
        self.m1 =  nn.Sequential(*[nn.Linear(in_feat if i ==0 else out_feat, out_feat) if i % 2 == 0 else activation for i in range(num_layers*2-1)])
        self.m2 =  nn.Sequential(*[nn.Linear(in_feat if i ==0 else out_feat, out_feat) if i % 2 == 0 else activation for i in range(num_layers*2-1)])
        self.m4 = nn.Sequential(nn.Linear(in_feat + out_feat, out_feat, bias=True))

    def forward(self, x, mask):
        """ x: batch x N x N x in_feat""" 
        norm = mask[:,0].squeeze(-1).float().sum(-1).sqrt().view(mask.size(0), 1, 1, 1)
        mask = mask.unsqueeze(1).squeeze(-1)
        out1 = self.m1(x).permute(0, 3, 1, 2) * mask           # batch, out_feat, N, N
        out2 = self.m2(x).permute(0, 3, 1, 2) * mask       # batch, out_feat, N, N
        out = out1 @ out2                                                   # batch, out_feat, N, N
        del out1, out2
        out =  out / norm                                # Fix from SPECTRE - Normalize to make std~1 (to preserve gradient norm, similar to self attention normalization)
        out = torch.cat((out.permute(0, 2, 3, 1), x), dim=3)                # batch, N, N, out_feat
        del x
        out = self.m4(out)
        return out

class FeatureExtractor(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=nn.LeakyReLU()):
        super().__init__()
        self.lin1 = nn.Sequential(nn.Linear(in_features, out_features, bias=True))
        self.lin2 = nn.Sequential(nn.Linear(in_features, out_features, bias=False))
        self.lin3 = nn.Sequential(nn.Linear(out_features, out_features, bias=False))
        self.activation = activation

    def forward(self, u, mask):
        """ u: (batch_size, num_nodes, num_nodes, in_features)
            output: (batch_size, out_features). """
        u = u * mask
        n = mask[:,0].sum(1)
        diag = u.diagonal(dim1=1, dim2=2)       # batch_size, channels, num_nodes
        trace = torch.sum(diag, dim=2)
        del diag
        out1 = self.lin1.forward(trace / n)

        s = (torch.sum(u, dim=[1, 2]) - trace) / (n * (n-1))
        del trace
        out2 = self.lin2.forward(s)  # bs, out_feat
        del s
        out = out1 + out2
        out = out + self.lin3.forward(self.activation(out))
        return out


class Powerful(nn.Module):
    def __init__(self, num_classes: int, num_input_features: int,  num_layers: int, hidden: int, hidden_final: int, dropout_prob: float,
                 simplified: bool, layers_per_conv: int=1, activation_function='leaky_relu', negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
        self.activation_function = activation_function
        if self.activation_function == 'gelu':
            activation = nn.GELU()
        elif self.activation_function == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.LeakyReLU(self.negative_slope)
        # input_features should be node features + edge features
        input_features = num_input_features + 1 # + adj
        self.layer_after_conv = not simplified
        self.dropout_prob = dropout_prob
        self.no_prop = FeatureExtractor(input_features, hidden_final, activation)
        initial_conv = PowerfulLayer(input_features, hidden, layers_per_conv, activation)
        self.convs = nn.ModuleList([initial_conv])
        self.bns = nn.ModuleList([])
        for i in range(1, num_layers):
            self.convs.append(PowerfulLayer(hidden, hidden, layers_per_conv))

        self.feature_extractors = torch.nn.ModuleList([])
        for i in range(num_layers):
            self.bns.append(nn.BatchNorm2d(hidden))
            self.feature_extractors.append(FeatureExtractor(hidden, hidden_final))
        if self.layer_after_conv:
            self.after_conv = nn.Linear(hidden_final, hidden_final)
        self.final_lin = nn.Linear(hidden_final, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_function == 'gelu':
                    nn.init.xavier_uniform_(m.weight)
                elif self.activation_function == 'relu':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                else:
                    nn.init.kaiming_uniform_(m.weight, a=self.negative_slope, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
            
    def forward(self, x, edge_index, batch, edge_feat=None):
        # Convert to dense representation
        x, mask = to_dense_batch(x, batch)
        # Make mask B x N x N x 1
        mask =  mask.unsqueeze(-1).repeat(1,1,mask.size(1))
        mask = (mask * mask.transpose(-2,-1)).float().unsqueeze(-1)
        if edge_feat is not None:
            u, edge_feat = to_dense_adj(edge_index, batch, edge_feat)
        else:
            u = to_dense_adj(edge_index, batch)
            u = u[..., None] # batch, N, N, 1

        u = torch.cat([u, torch.diag_embed(x.transpose(-2,-1), dim1=1, dim2=2)], dim=-1)
        u = u * mask

        out = self.no_prop.forward(u, mask)
        for conv, extractor, bn in zip(self.convs, self.feature_extractors, self.bns):
            u = conv(u, mask)
            u = bn(u.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            u = u * mask
            out = out + extractor.forward(u, mask)
        out = F.relu(out) / len(self.convs)
        if self.layer_after_conv:
            out = out + F.relu(self.after_conv(out))
        out = F.dropout(out, p=self.dropout_prob, training=self.training)
        out = self.final_lin(out)
        return F.log_softmax(out, dim=-1), 0
