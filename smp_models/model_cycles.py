# Code taken from https://github.com/cvignac/SMP
import torch
import torch.nn.functional as F
import torch.nn as nn
from smp_models.smp_layers import SimplifiedFastSMPLayer, FastSMPLayer, SMPLayer
from smp_models.utils.layers import GraphExtractor, EdgeCounter, BatchNorm
from smp_models.utils.misc import create_batch_info, map_x_to_u


class SMP(torch.nn.Module):
    def __init__(self, num_input_features: int, num_classes: int, num_layers: int, hidden: int, layer_type: str,
                 hidden_final: int, dropout_prob: float, use_batch_norm: bool, use_x: bool, map_x_to_u: bool,
                 num_towers: int, simplified: bool):
        """ num_input_features: number of node features
            layer_type: 'SMP', 'FastSMP' or 'SimplifiedFastSMP'
            hidden_final: size of the feature map after pooling
            use_x: for ablation study, run a MPNN instead of SMP
            map_x_to_u: map the node features to the local context
            num_towers: inside each SMP layers, use towers to reduce the number of parameters
            simplified: less layers in the feature extractor.
        """
        super().__init__()
        self.map_x_to_u, self.use_x = map_x_to_u, use_x
        self.dropout_prob = dropout_prob
        self.use_batch_norm = use_batch_norm
        self.edge_counter = EdgeCounter()
        self.num_classes = num_classes

        self.no_prop = GraphExtractor(in_features=num_input_features, out_features=hidden_final, use_x=use_x)
        self.initial_lin = nn.Linear(num_input_features, hidden)

        layer_type_dict = {'SMP': SMPLayer, 'FastSMP': FastSMPLayer, 'SimplifiedFastSMP': SimplifiedFastSMPLayer}
        conv_layer = layer_type_dict[layer_type]

        self.convs = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        self.feature_extractors = torch.nn.ModuleList([])
        for i in range(0, num_layers):
            self.convs.append(conv_layer(in_features=hidden, num_towers=num_towers, out_features=hidden, use_x=use_x))
            self.batch_norm_list.append(BatchNorm(hidden, use_x))
            self.feature_extractors.append(GraphExtractor(in_features=hidden, out_features=hidden_final, use_x=use_x,
                                                          simplified=simplified))

        # Last layers
        self.simplified = simplified
        self.after_conv = nn.Linear(hidden_final, hidden_final)
        self.final_lin = nn.Linear(hidden_final, num_classes)

    def forward(self, data):
        """ data.x: (num_nodes, num_features)"""
        x, edge_index = data.x, data.edge_index
        batch_info = create_batch_info(data, self.edge_counter)

        # Create the context matrix
        if self.use_x:
            assert x is not None
            u = x
        elif self.map_x_to_u:
            u = map_x_to_u(data, batch_info)
        else:
            u = data.x.new_zeros((data.num_nodes, batch_info['n_colors']))
            u.scatter_(1, data.coloring, 1)
            u = u[..., None]

        # Forward pass
        out = self.no_prop(u, batch_info)
        u = self.initial_lin(u)
        for i, (conv, bn, extractor) in enumerate(zip(self.convs, self.batch_norm_list, self.feature_extractors)):
            if self.use_batch_norm and i > 0:
                u = bn(u)
            u = conv(u, edge_index, batch_info)
            global_features = extractor.forward(u, batch_info)
            out += global_features / len(self.convs)

        # Two layer MLP with dropout and residual connections:
        if not self.simplified:
            out = torch.relu(self.after_conv(out)) + out
        out = F.dropout(out, p=self.dropout_prob, training=self.training)
        out = self.final_lin(out)
        if self.num_classes > 1:
            # Classification
            return F.log_softmax(out, dim=-1)
        else:
            # Regression
            assert out.shape[1] == 1
            return out[:, 0]

    def reset_parameters(self):
        for layer in [self.no_prop, self.initial_lin, *self.convs, *self.batch_norm_list, *self.feature_extractors,
                      self.after_conv, self.final_lin]:
            layer.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__
