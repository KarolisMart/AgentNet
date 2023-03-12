# This implementation is based on https://github.com/weihua916/powerful-gnns and https://github.com/chrsmrrs/k-gnn/tree/master/examples
# Datasets are implemented based on the description in the corresonding papers (see the paper for references)
import numpy as np
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GINConv, GINEConv, global_add_pool
from sage.all import graphs as sage_graphs
from test_tube.hpc import SlurmCluster

from util import cos_anneal, get_cosine_schedule_with_warmup
from model import AgentNet, add_model_args
from smp_models.model_cycles import SMP
from smp_models.ppgn import Powerful

torch.set_printoptions(profile="full")
# Synthetic datasets

class SymmetrySet:
    def __init__(self):
        self.hidden_units = 0
        self.num_classes = 0
        self.num_features = 0
        self.num_nodes = 0

    def addports(self, data):
        data.ports = torch.zeros(data.num_edges, 1)
        degs = degree(data.edge_index[0], data.num_nodes, dtype=torch.long) # out degree of all nodes
        for n in range(data.num_nodes):
            deg = degs[n]
            ports = np.random.permutation(int(deg))
            for i, neighbor in enumerate(data.edge_index[1][data.edge_index[0]==n]):
                nb = int(neighbor)
                data.ports[torch.logical_and(data.edge_index[0]==n, data.edge_index[1]==nb), 0] = float(ports[i])
        return data

    def makefeatures(self, data):
        data.x = torch.ones((data.num_nodes, 1))
        data.id = torch.tensor(np.random.permutation(np.arange(data.num_nodes))).unsqueeze(1)
        return data

    def makedata(self):
        pass

class FourCycles(SymmetrySet):
    def __init__(self, batch_size):
        super().__init__()
        self.p = 4
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 4 * self.p
        self.graph_class = True
        self.batch_size = batch_size

    def gen_graph(self, p):
        edge_index = None
        for i in range(p):
            e = torch.tensor([[i, p + i, 2 * p + i, 3 * p + i], [2 * p + i, 3 * p + i, i, p + i]], dtype=torch.long)
            if edge_index is None:
                edge_index = e
            else:
                edge_index = torch.cat([edge_index, e], dim=-1)
        top = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            top[i * p + t] = 1
        bottom = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            bottom[i * p + t] = 1
        for i, bit in enumerate(top):
            if bit:
                e = torch.tensor([[i // p, p + i % p], [p + i % p, i // p]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
        for i, bit in enumerate(bottom):
            if bit:
                e = torch.tensor([[2 * p + i // p, 3 * p + i % p], [3 * p + i % p, 2 * p + i // p]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
        return Data(edge_index=edge_index, num_nodes=self.num_nodes), int(any(np.logical_and(top, bottom)))

    def makedata(self, num_graphs=-1):
        if num_graphs < 1:
            size = self.batch_size // 2
        else:
            size = num_graphs // 2
        p = self.p
        trues = []
        falses = []
        while len(trues) < size or len(falses) < size:
            data, label = self.gen_graph(p)
            data = self.makefeatures(data)
            data = self.addports(data)
            data.y = label
            if label and len(trues) < size:
                trues.append(data)
            elif not label and len(falses) < size:
                falses.append(data)
        return trues + falses

class CSL(SymmetrySet):
    def __init__(self, batch_size):
        super().__init__()
        self.hidden_units = 32
        self.num_classes = 10 # num skips
        self.num_features = 1
        self.num_nodes = 41
        self.graph_class = True
        self.batch_size = batch_size

    def makedata(self, num_graphs=-1):
        
        size=self.num_nodes
        skips = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]
        graphs = []
        if num_graphs < 1:
            mult = max(1, self.batch_size // len(skips)) # if batch size > 10 increase the dataset size
        else:
            mult = max(1, num_graphs // len(skips)) # if batch size > 10 increase the dataset size
        
        for j in range(mult):
            for s, skip in enumerate(skips):
                edge_index = torch.tensor([[0, size-1], [size-1, 0]], dtype=torch.long)
                for i in range(size - 1):
                    e = torch.tensor([[i, i+1], [i+1, i]], dtype=torch.long)
                    edge_index = torch.cat([edge_index, e], dim=-1)
                for i in range(size):
                    e = torch.tensor([[i, i], [(i - skip) % size, (i + skip) % size]], dtype=torch.long)
                    edge_index = torch.cat([edge_index, e], dim=-1)
                data = Data(edge_index=edge_index, num_nodes=self.num_nodes)
                data = self.makefeatures(data)
                data = self.addports(data)
                data.y = torch.tensor(s)
                graphs.append(data)

        return graphs        

class WL3(SymmetrySet):
    def __init__(self, batch_size):
        super().__init__()
        self.p = 4
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 16
        self.graph_class = True
        self.batch_size = batch_size

    def makedata(self, num_graphs=-1):
        if num_graphs < 1:
            size = self.batch_size
        else:
            size = num_graphs
        p = self.p
        graphs = []
        for i in range(size):
            if i % 2 == 0:
                G = sage_graphs.ShrikhandeGraph()
                G = G.networkx_graph()
                data = from_networkx(G)

                label = 0
            else:
                G = sage_graphs.RookGraph([4, 4])
                G = G.networkx_graph()
                data = from_networkx(G)
                label = 1
            data = self.makefeatures(data)
            data = self.addports(data)
            data.y = label
            graphs.append(data)
        return graphs

class Ladder(SymmetrySet):
    def __init__(self, batch_size, num_nodes, fraction):
        super().__init__()
        self.p = 4
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = num_nodes * 2 # should be power of 2 and the graph has 2*num_nodes nodes
        self.graph_class = True
        self.batch_size = batch_size
        self.fraction = fraction

    def gen_graph(self, crossed, num_nodes, fraction):
        if num_nodes < 0:
            num_nodes = self.num_nodes
        if fraction < 0:
            fraction = self.fraction
        n = num_nodes // 2
        G = nx.circular_ladder_graph(n)
        if crossed:
            # Change fraction number of ladder cells into crossed cells (| | edges become X edges)
            frac = fraction * 0.5 # we only have 1/2 the number of cells for n-lenght ladder
            n_cells = int(n * frac)
            mult = int(1/frac)

            G.remove_edges_from(
                        (i*mult + j, i*mult + j + n) for i in range(n_cells) for j in range(2) # cell is two concecutive nodes on the cycle/path
                    )
            G.add_edges_from(
                        (i*mult + 0, i*mult + 1 + n) for i in range(n_cells) # do first diagonal of the cross
                    )
            G.add_edges_from(
                        (i*mult + 1, i*mult + 0 + n) for i in range(n_cells) # do second diagonal of the cross
                    )
        
        data = from_networkx(G)
        return Data(edge_index=data.edge_index, num_nodes=num_nodes), int(crossed)

    def makedata(self, num_nodes=-1, fraction=-1.0, num_graphs=-1):
        if num_graphs < 1:
            size = self.batch_size
        else:
            size = num_graphs
        graphs = []
        for i in range(size):
            data, label = self.gen_graph((i % 2) == 0, num_nodes, fraction)   
            data = self.makefeatures(data)
            data = self.addports(data)
            data.y = label
            graphs.append(data)
        return graphs

def main(args, cluster=None):
    print(args, flush=True)

    if args.dataset == "CSL":
        dataset = CSL(args.batch_size)
    elif args.dataset == "fourcycles":
        dataset = FourCycles(args.batch_size)
    elif args.dataset == "WL3":
        dataset = WL3(args.batch_size)
    elif args.dataset == "ladder":
        dataset = Ladder(args.batch_size, args.num_nodes, fraction=args.fraction)
        

    print(dataset.__class__.__name__)

    # Set the sampling probability and number of runs/samples for the DropGIN
    n = dataset.num_nodes
    print(f'Number of nodes: {n}')
    print(f'LR: {args.lr}')
    gamma = n
    p_opt = 2 * 1 /(1+gamma)
    if args.prob >= 0:
        p = args.prob
    else:
        p = p_opt
    if args.num_runs > 0:
        num_runs = args.num_runs
    else:
        num_runs = gamma
    print(f'Number of runs: {num_runs}')
    print(f'Number of steps: {args.num_steps}')
    print(f'Sampling probability: {p}')

    degs = []
    for g in dataset.makedata():
        deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
        degs.append(deg.max())
    print(f'Mean Degree: {torch.stack(degs).float().mean()}')
    print(f'Max Degree: {torch.stack(degs).max()}')
    print(f'Min Degree: {torch.stack(degs).min()}')
    print(f'Number of graphs: {len(dataset.makedata())}')

    graph_classification = dataset.graph_class
    if graph_classification:
        print('Graph Clasification Task')
    else:
        print('Node Clasification Task')
    
    num_features = dataset.num_features
    Conv = GINConv
    if args.augmentation == 'ports':
        Conv = GINEConv
    elif args.augmentation == 'ids':
        num_features += 1
    elif args.augmentation == 'random':
        num_features += 1

    class GIN(nn.Module):
        def __init__(self):
            super(GIN, self).__init__()

            dim = args.hidden_units

            self.num_steps = args.num_steps

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(Conv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_steps-1):
                self.convs.append(Conv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, dataset.num_classes))
        
        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, Conv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, x, edge_index, batch):

            if args.augmentation == 'random':
                x = torch.cat([x, torch.randint(0, 100, (x.size(0), 1), device=x.device) / 100.0], dim=1)
            
            outs = [x]
            for i in range(self.num_steps):
                x = self.convs[i](x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x)
            
            out = None
            for i, x in enumerate(outs):
                if graph_classification:
                    x = global_add_pool(x, batch)
                x = self.fcs[i](x) # No dropout for these experiments
                if out is None:
                    out = x
                else:
                    out += x
            out = out / len(outs)
            return F.log_softmax(out, dim=-1), 0

    use_aux_loss = args.use_aux_loss

    class DropGIN(nn.Module):
        def __init__(self):
            super(DropGIN, self).__init__()

            dim = args.hidden_units

            self.num_steps = args.num_steps

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(Conv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_steps-1):
                self.convs.append(Conv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, dataset.num_classes))
            
            if use_aux_loss:
                self.aux_fcs = nn.ModuleList()
                self.aux_fcs.append(nn.Linear(num_features, dataset.num_classes))
                for i in range(self.num_steps):
                    self.aux_fcs.append(nn.Linear(dim, dataset.num_classes))
        
        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, Conv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, x, edge_index, batch):
            # Do runs in paralel, by repeating the graphs in the batch
            x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
            drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * p).bool()
            x[drop] = 0.0
            del drop
            outs = [x]
            x = x.view(-1, x.size(-1))
            run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)
            for i in range(self.num_steps):   
                x = self.convs[i](x, run_edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x.view(num_runs, -1, x.size(-1)))
            del run_edge_index

            out = None
            for i, x in enumerate(outs):
                x = x.mean(dim=0)
                if graph_classification:
                    x = global_add_pool(x, batch)
                x = self.fcs[i](x) # No dropout layer in these experiments
                if out is None:
                    out = x
                else:
                    out += x
            
            if use_aux_loss:
                aux_out = torch.zeros(num_runs, out.size(0), out.size(1), device=out.device)
                run_batch = batch.repeat(num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(batch.size(0)) * (batch.max() + 1)
                for i, x in enumerate(outs):
                    if graph_classification:
                        x = x.view(-1, x.size(-1))
                        x = global_add_pool(x, run_batch)
                    x = x.view(num_runs, -1, x.size(-1))
                    x = self.aux_fcs[i](x) # No dropout layer in these experiments
                    aux_out += x

                return F.log_softmax(out, dim=-1), F.log_softmax(aux_out, dim=-1)
            else:
                return F.log_softmax(out, dim=-1), 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if args.model_type == 'GIN':
        model = GIN()
    elif args.model_type == 'DropGIN':
        model = DropGIN()
    elif 'SMP' in args.model_type:
        model = SMP(num_input_features=dataset.num_features, num_classes=dataset.num_classes, num_layers=args.num_steps, hidden=args.hidden_units, layer_type=args.model_type,
                 hidden_final=args.hidden_units, dropout_prob=args.dropout, use_batch_norm=True, use_x=False, map_x_to_u=False,
                 num_towers=1, simplified=False) #  based on https://github.com/cvignac/SMP/blob/master/config_cycles.yaml and https://github.com/cvignac/SMP/blob/master/cycles_main.py
    elif args.model_type == 'PPGN':
        model = Powerful(dataset.num_classes, dataset.num_features, args.num_steps, args.hidden_units,
                          args.hidden_units, args.dropout, simplified=False, layers_per_conv=args.layers_per_conv, activation_function=args.activation_function, negative_slope=args.negative_slope)
    else:
        model = AgentNet(num_features=dataset.num_features, hidden_units=args.hidden_units, num_out_classes=dataset.num_classes, dropout=args.dropout, num_steps=args.num_steps,
                        num_agents=args.num_agents, reduce=args.reduce, node_readout=args.node_readout, use_step_readout_lin=args.use_step_readout_lin,
                        num_pos_attention_heads=args.num_pos_attention_heads, readout_mlp=args.readout_mlp, self_loops=args.self_loops, post_ln=args.post_ln,
                        attn_dropout=args.attn_dropout, no_time_cond=args.no_time_cond, mlp_width_mult=args.mlp_width_mult, activation_function=args.activation_function,
                        negative_slope=args.negative_slope, input_mlp=args.input_mlp, attn_width_mult=args.attn_width_mult, importance_init=args.importance_init,
                        random_agent=args.random_agent, test_argmax=args.test_argmax, global_agent_pool=args.global_agent_pool, agent_global_extra=args.agent_global_extra,
                        basic_global_agent=args.basic_global_agent, basic_agent=args.basic_agent, bias_attention=args.bias_attention, visited_decay=args.visited_decay,
                        sparse_conv=args.sparse_conv, mean_pool_only=args.mean_pool_only, edge_negative_slope=args.edge_negative_slope,
                        final_readout_only=args.final_readout_only)
    
    print(model.__class__.__name__)
    model = model.to(device)
    use_aux_loss = args.use_aux_loss

    def train(epoch, loader, optimizer):
        model.train()
        loss_all = 0
        n = 0
        correct = 0
        n_aux = 0 if use_aux_loss else 1
        correct_aux = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            if 'SMP' in args.model_type:
                    logs = model(data)
            else:
                logs, aux_logs = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(logs, data.y)
            n += len(data.y)
            if use_aux_loss:
                aux_loss = F.nll_loss(aux_logs.view(-1, aux_logs.size(-1)), data.y.unsqueeze(1).expand(-1,aux_logs.size(0)).clone().view(-1))
                loss = 0.75*loss + 0.25*aux_loss
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad, norm_type=2) # Like in most transformers
            optimizer.step()
            pred = logs.max(1)[1]
            correct += pred.eq(data.y).sum().item()
            if use_aux_loss:
                pred_aux = aux_logs.view(-1, aux_logs.size(-1)).max(1)[1]
                n_aux += len(pred_aux)
                correct_aux += pred_aux.eq(data.y.unsqueeze(1).expand(-1,aux_logs.size(0)).clone().view(-1)).sum().item()
        return loss_all / len(loader.dataset), correct / n, correct_aux / n_aux

    def test(loader):
        model.eval()
        n = 0
        with torch.no_grad():
            correct = 0
            for data in loader:
                data = data.to(device)
                if 'SMP' in args.model_type:
                    logs = model(data)
                else:
                    logs, aux_logs = model(data.x, data.edge_index, data.batch)
                pred = logs.max(1)[1]
                n += len(pred)
                correct += pred.eq(data.y).sum().item()
        return correct / n

    def train_and_test(num_nodes=-1, fraction=-1.0):
        train_accs = []
        best_val_test_accs = []
        test_accs = []
        for seed in range(args.num_seeds): # NOTE: Normally 10
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model.reset_parameters()
            lr = args.lr
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
            scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs, min_lr_mult=args.min_lr_mult)

            if num_nodes > 0 and fraction > 0: # For the ablation
                test_dataset = dataset.makedata(num_nodes, fraction, num_graphs=300)
                train_dataset = dataset.makedata(num_nodes, fraction)
                val_dataset = dataset.makedata(num_nodes, fraction, num_graphs=300)
            else:
                test_dataset = dataset.makedata(num_graphs=300)
                train_dataset = dataset.makedata()
                val_dataset = dataset.makedata(num_graphs=300)

            if args.batch_size < 1:
                batch_size = len(train_dataset)
            else:
                batch_size = args.batch_size
            test_loader = DataLoader(test_dataset, batch_size=300)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=300)

            best_val_acc = 0.0
            best_val_test_acc = 0.0

            print('---------------- Seed {} ----------------'.format(seed))
            for epoch in range(1, args.epochs+1):
                if args.verbose:
                    start = time.time()
                if args.gumbel_warmup < 0:
                    gumbel_warmup = args.warmup
                else:
                    gumbel_warmup = args.gumbel_warmup
                model.temp = cos_anneal(gumbel_warmup, gumbel_warmup + args.gumbel_decay_epochs, args.gumbel_temp, args.gumbel_min_temp, epoch)
                lr = scheduler.optimizer.param_groups[0]['lr']
                train_loss, train_acc, train_aux_acc = train(epoch, train_loader, optimizer)

                if args.track_val:
                    val_acc = test(val_loader)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_val_test_acc = test(test_loader)
                
                scheduler.step()
                if args.verbose:
                    print('Epoch: {:03d}, LR: {:7f}, Gumbel Temp: {:7f}, Train Loss: {:.7f}, Train Accuracy: {:.7f}, Train Aux Accuracy: {:.7f}, Time: {:7f}'.format(epoch, lr, model.temp, train_loss, train_acc, train_aux_acc, time.time() - start), flush=True)
            train_acc = test(train_loader)
            train_accs.append(train_acc)
            best_val_test_accs.append(best_val_test_acc)
            print('Train Acc: {:.7f}, Test Acc: {:7f}, Best Val: {:7f}, Test Acc (Best Val Epoch): {:7f}'.format(train_acc, test_acc, best_val_acc, best_val_test_acc), flush=True)            
        train_acc = torch.tensor(train_accs)
        test_acc = torch.tensor(test_accs)
        best_val_test_acc = torch.tensor(best_val_test_accs)
        print('---------------- Final Result ----------------')
        print('Train Mean: {:7f}, Train Std: {:7f}, Test Mean: {}, Test Std: {}, Test Mean (Best Val Epoch): {}, Test Std (Best Val Epoch): {}'.format(train_acc.mean(), train_acc.std(), test_acc.mean(dim=0), test_acc.std(dim=0),  best_val_test_acc.mean(dim=0), best_val_test_acc.std(dim=0)), flush=True)
        return test_acc.mean(dim=0), test_acc.std(dim=0)

    if args.density_ablation:
        print('Dropout probability ablation')
        num_nodes = [8, 16, 32, 64, 128, 256, 512, 1024]
        fractions_1 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        fractions_2 = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
        means = []
        stds = []
        for n, f in zip(num_nodes, fractions_1):
            print(f'Number of nodes: {n}, Fraction: {f}')
            mean, std = train_and_test()
            means.append(mean.item())
            stds.append(std.item())
        num_nodes = np.array(num_nodes)
        means_1 = np.array(means)
        stds_1 = np.array(stds)
        lower_1 = means_1 - stds_1
        lower_1 = [i if i > 0 else 0 for i in lower_1]
        upper_1 = means_1 + stds_1
        upper_1 = [i if i <= 1 else 1 for i in upper_1]
        print("Constant Fraction:")
        print("Means:", means_1)
        print("STDs:", stds_1)
        means = []
        stds = []
        for n, f in zip(num_nodes, fractions_2):
            print(f'Number of nodes: {n}, Fraction: {f}')
            mean, std = train_and_test(num_nodes=n, fraction=f)
            means.append(mean.item())
            stds.append(std.item())
        means_2 = np.array(means)
        stds_2 = np.array(stds)
        lower_2 = means_2 - stds_2
        lower_2 = [i if i > 0 else 0 for i in lower_2]
        upper_2 = means_2 + stds_2
        upper_2 = [i if i <= 1 else 1 for i in upper_2] 
        print("Constant Number:")
        print("Means:", means_2)
        print("STDs:", stds_2)
        plt.figure()
        plt.plot(num_nodes, means_1, color='blue', label='Constant fraction of sub-graphs')
        plt.fill_between(num_nodes, lower_1, upper_1, alpha=0.3, color='blue')
        plt.plot(num_nodes, means_2, color='red', label='Constant number of sub-graphs')
        plt.fill_between(num_nodes, lower_2, upper_2, alpha=0.3, color='red')
        plt.xlabel("Graph Size")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.ylim(bottom=0.4)
        plt.xlim(left=0.0)
        file_name = "ablation_{}.pdf".format(args.dataset)
        plt.savefig(file_name)
    else:
        train_and_test()    

if __name__ == '__main__':
    parser = add_model_args(None, hyper=True)
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--augmentation', type=str, default='none', help="Options are ['none', 'random']")
    parser.add_argument('--prob', type=float, default=0.1) # For DropGIN
    parser.add_argument('--num_runs', type=int, default=-1) # For DropGIN
    parser.add_argument('--model_type', type=str, default='agent')
    parser.add_argument('--num_nodes', type=int, default=16)
    parser.add_argument('--fraction', type=float, default=1.0) # Fraction of cells in Ladder graph that are turned into crossed cells
    parser.add_argument('--track_val', action='store_true', default=False)
    parser.add_argument('--layers_per_conv', type=int, default=1) # only for PPGN
    
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--density_ablation', action='store_true', default=False, help="Run probability ablation study")

    parser.add_argument('--dataset', type=str, default='fourcycles', help="Options are ['CSL', 'fourcycles', 'WL3', 'ladder']")

    parser.add_argument('--slurm', action='store_true', default=False)
    parser.add_argument('--grid_search', action='store_true', default=False)
    parser.add_argument('--gpu_jobs', action='store_true', default=False)

    # Grid search parms
    parser.opt_list('--lr', type=float, default=0.0001, tunable=True, options=[0.001, 0.0005, 0.0001])
    parser.opt_list('--batch_size', type=int, default=300, tunable=False, options=[50, 300])
    parser.opt_list('--hidden_units', type=int, default=128, tunable=True, options=[64, 128])
    parser.opt_list('--num_steps', type=int, default=16, tunable=True, options=[4, 8, 16, 64])
    parser.opt_list('--num_agents', type=int, default=16, tunable=True, options=[2, 16]) # For CSL it should be [2, 41]

    args = parser.parse_args()

    if args.slurm:
        print('Launching SLURM jobs')
        log_path = 'slurm_log/'
        cluster = SlurmCluster(
            hyperparam_optimizer=args,
            log_path=log_path,
            python_cmd='python'
        )
        cluster.job_time = '48:00:00'

        cluster.memory_mb_per_node = '24G'

        job_name = f'{args.dataset}_{args.model_type}'

        if args.gpu_jobs:
            cluster.per_experiment_nb_cpus = 2
            cluster.per_experiment_nb_gpus = 1
            cluster.optimize_parallel_cluster_gpu(main, nb_trials=None, job_name=job_name, job_display_name=args.dataset)
        else:
            cluster.per_experiment_nb_cpus = 8
            cluster.optimize_parallel_cluster_cpu(main, nb_trials=None, job_name=job_name, job_display_name=args.dataset)
    elif args.grid_search:
        for hparam_trial in args.trials(None):
            main(hparam_trial)
    else:
        main(args)

    print('Finished', flush=True)
