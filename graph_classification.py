# This implementation is based on https://github.com/KarolisMart/DropGNN/blob/main/gin-graph_classification.py which was basied on https://github.com/weihua916/powerful-gnns and https://github.com/chrsmrrs/k-gnn/tree/master/examples
import os.path as osp
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, global_add_pool
import torch_geometric.transforms as T
from sklearn.model_selection import StratifiedKFold
from test_tube.hpc import SlurmCluster

from util import cos_anneal, get_cosine_schedule_with_warmup, PTCDataset
from model import AgentNet, add_model_args
from smp_models.ppgn import Powerful

torch.set_printoptions(profile="full")

def main(args, cluster=None):
    print(args, flush=True)

    BATCH = args.batch_size

    if 'IMDB' in args.dataset: #IMDB-BINARY or #IMDB-MULTI
        class MyFilter(object):
            def __call__(self, data):
                return True
        if args.one_hot_degree:
            path = osp.join(
                osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}')
            max_degree = 88 if args.dataset == 'IMDB-MULTI' else 135
            dataset = TUDataset(
                path,
                name=args.dataset,
                pre_transform=T.OneHotDegree(max_degree),
                pre_filter=MyFilter())
        else:
            path = osp.join(
                osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}_constant')
            max_degree = 88 if args.dataset == 'IMDB-MULTI' else 135
            dataset = TUDataset(
                path,
                name=args.dataset,
                pre_transform=T.Constant(),
                pre_filter=MyFilter())
    elif 'MUTAG' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True
        if args.one_hot_degree:
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'MUTAG_one_hot_degree')
            dataset = TUDataset(path, name='MUTAG', pre_filter=MyFilter(), pre_transform=T.OneHotDegree(4))
        else:
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'MUTAG')
            dataset = TUDataset(path, name='MUTAG', pre_filter=MyFilter())
    elif 'PROTEINS' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PROTEINS')
        dataset = TUDataset(path, name='PROTEINS', pre_filter=MyFilter())
    elif 'PTC_GIN' in args.dataset: 
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PTC_GIN')
        dataset = PTCDataset(path, name='PTC')
    elif 'DD' == args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True 
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'DD')
        dataset = TUDataset(path, name='DD', pre_filter=MyFilter())
    elif 'REDDIT' in args.dataset: #REDDIT-BINARY or REDDIT-MULTI-5K
        class MyFilter(object):
            def __call__(self, data):
                return True
        if args.one_hot_degree:
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}')
            max_degree = 3062 if args.dataset == 'REDDIT-BINARY' else 2011
            dataset = TUDataset(
                path,
                name=args.dataset,
                pre_transform=T.OneHotDegree(max_degree),
                pre_filter=MyFilter())
        else:
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}_constant')
            dataset = TUDataset(
                path,
                name=args.dataset,
                pre_transform=T.Constant(),
                pre_filter=MyFilter())
    else:
        raise ValueError

    print(dataset)

    # Set the sampling probability and number of runs/samples for the DropGIN
    n = []
    degs = []
    for g in dataset:
        num_nodes = g.num_nodes
        deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
        n.append(g.num_nodes)
        degs.append(deg)
    print(f'Mean Degree: {torch.cat(degs).float().mean()}')
    print(f'Max Degree: {torch.cat(degs).max()}')
    print(f'Min Degree: {torch.cat(degs).min()}')
    mean_n = torch.tensor(n).float().mean().item()
    print(f'Mean number of nodes: {mean_n}')
    print(f'Max number of nodes: {torch.tensor(n).float().max().item()}')
    print(f'Min number of nodes: {torch.tensor(n).float().min().item()}')
    print(f'Number of graphs: {len(dataset)}')
    gamma = mean_n
    p = 2 * 1 /(1+gamma)
    num_runs = gamma
    print(f'Number of runs: {num_runs}')
    print(f'Sampling probability: {p}')


    def separate_data(dataset_len, seed=0):
        # Use same splitting/10-fold as GIN paper
        skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
        idx_list = []
        for idx in skf.split(np.zeros(dataset_len), np.zeros(dataset_len)):
            idx_list.append(idx)
        return idx_list

    class GIN(nn.Module):
        def __init__(self):
            super(GIN, self).__init__()

            num_features = dataset.num_features
            dim = args.hidden_units
            self.dropout = args.dropout

            self.num_layers = 4

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(GINConv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_layers-1):
                self.convs.append(GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, dataset.num_classes))
        
        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, GINConv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, x, edge_index, batch):
            outs = [x]
            # print(x.dtype, x.shape)
            for i in range(self.num_layers):
                x = self.convs[i](x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x)
            
            out = None
            for i, x in enumerate(outs):
                x = global_add_pool(x, batch)
                x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
                if out is None:
                    out = x
                else:
                    out += x
            return F.log_softmax(out, dim=-1), 0
            # return out, 0

    class Baseline(nn.Module):
        def __init__(self):
            super(Baseline, self).__init__()

            num_features = dataset.num_features
            dim = args.hidden_units

            if args.baseline_in_mlp:
                self.in_mlp = nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))

            self.out_mlp = nn.Sequential(nn.Linear((dim if args.baseline_in_mlp else num_features), dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))
        
        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, x, edge_index, batch):
            if args.baseline_in_mlp:
                x = self.in_mlp(x)
            x = global_add_pool(x, batch)
            out = self.out_mlp(x)
            return F.log_softmax(out, dim=-1), 0

    use_aux_loss = args.use_aux_loss

    class DropGIN(nn.Module):
        def __init__(self):
            super(DropGIN, self).__init__()

            num_features = dataset.num_features
            dim = args.hidden_units
            self.dropout = args.dropout

            self.num_layers = 4

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(GINConv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_layers-1):
                self.convs.append(GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, dataset.num_classes))
            
            if use_aux_loss:
                self.aux_fcs = nn.ModuleList()
                self.aux_fcs.append(nn.Linear(num_features, dataset.num_classes))
                for i in range(self.num_layers):
                    self.aux_fcs.append(nn.Linear(dim, dataset.num_classes))
        
        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, GINConv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, x, edge_index, batch):            
            # Do runs in paralel, by repeating the graphs in the batch
            x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
            drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device)*p).bool()
            x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
            del drop
            outs = [x]
            x = x.view(-1, x.size(-1))
            run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)
            for i in range(self.num_layers):
                x = self.convs[i](x, run_edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x.view(num_runs, -1, x.size(-1)))
            del  run_edge_index
            out = None
            for i, x in enumerate(outs):
                x = x.mean(dim=0)
                x = global_add_pool(x, batch)
                x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
                if out is None:
                    out = x
                else:
                    out += x

            if use_aux_loss:
                aux_out = torch.zeros(num_runs, out.size(0), out.size(1), device=out.device)
                run_batch = batch.repeat(num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(batch.size(0)) * (batch.max() + 1)
                for i, x in enumerate(outs):
                    x = x.view(-1, x.size(-1))
                    x = global_add_pool(x, run_batch)
                    x = x.view(num_runs, -1, x.size(-1))
                    x = F.dropout(self.aux_fcs[i](x), p=self.dropout, training=self.training)
                    aux_out += x

                return F.log_softmax(out, dim=-1), F.log_softmax(aux_out, dim=-1)
            else:
                return F.log_softmax(out, dim=-1), 0

    torch.manual_seed(0)
    np.random.seed(0)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if args.model_type == 'GIN':
        model = GIN()
    elif args.model_type == 'Baseline':
        model = Baseline()
    elif args.model_type == 'DropGIN':
        model = DropGIN()
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
            logs, aux_logs = model(data.x, data.edge_index, data.batch)
        
            loss = F.nll_loss(logs, data.y)
            n += len(data.y)
            if use_aux_loss:
                aux_loss = F.nll_loss(aux_logs.view(-1, aux_logs.size(-1)), data.y.unsqueeze(1).expand(-1,aux_logs.size(1)).clone().view(-1))
                
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
                correct_aux += pred_aux.eq(data.y.unsqueeze(1).expand(-1,aux_logs.size(1)).clone().view(-1)).sum().item()
        return loss_all / n, correct / n, correct_aux / n_aux

    def val(loader):
        model.eval()
        with torch.no_grad():
            loss_all = 0
            for data in loader:
                data = data.to(device)
                logs, aux_logs = model(data.x, data.edge_index, data.batch)
                loss_all += F.nll_loss(logs, data.y, reduction='sum').item()
        return loss_all / len(loader.dataset)

    def test(loader):
        model.eval()
        with torch.no_grad():
            correct = 0
            for data in loader:
                data = data.to(device)
                logs, aux_logs = model(data.x, data.edge_index, data.batch)
                pred = logs.max(1)[1]
                correct += pred.eq(data.y).sum().item()
        return correct / len(loader.dataset)

    acc = []
    splits = separate_data(len(dataset), seed=0)
    print(model.__class__.__name__)
    for i, (train_idx, test_idx) in enumerate(splits):
        model.reset_parameters()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs, min_lr_mult=args.min_lr_mult)

        test_dataset = dataset[test_idx.tolist()]
        train_dataset = dataset[train_idx.tolist()]

        test_loader = DataLoader(test_dataset, batch_size=BATCH)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(len(train_dataset)*args.iters_per_epoch/(len(train_dataset)/BATCH))), batch_size=BATCH, drop_last=False, collate_fn=Collater(follow_batch=[],exclude_keys=[]))	# GIN like epochs/batches - they do 50 radom batches per epoch

        print('---------------- Split {} ----------------'.format(i), flush=True)
        
        test_acc = 0
        acc_temp = []
        for epoch in range(1, args.epochs+1):
            if args.verbose or epoch == 350:
                start = time.time()
                torch.cuda.reset_peak_memory_stats(0)
            lr = scheduler.optimizer.param_groups[0]['lr']
            if args.gumbel_warmup < 0:
                gumbel_warmup = args.warmup
            else:
                gumbel_warmup = args.gumbel_warmup
            model.temp = cos_anneal(gumbel_warmup, gumbel_warmup + args.gumbel_decay_epochs, args.gumbel_temp, args.gumbel_min_temp, epoch)
            train_loss, train_acc, train_aux_acc = train(epoch, train_loader, optimizer)
            scheduler.step()
            test_acc = test(test_loader)
            if args.verbose or epoch == 350:
                print('Epoch: {:03d}, LR: {:.7f}, Gumbel Temp: {:.4f}, Train Loss: {:.7f}, Train Acc: {:.4f}, Train Aux Acc: {:.4f}, Test Acc: {:.4f}, Time: {:.4f}, Mem: {:.3f}, Cached: {:.3f}, Steps: {:02d}'.format(epoch, lr, model.temp, train_loss, train_acc, train_aux_acc, test_acc, time.time() - start, torch.cuda.max_memory_allocated()/1024.0**3, torch.cuda.max_memory_reserved()/1024.0**3, len(train_loader)), flush=True)
            acc_temp.append(test_acc)
        acc.append(torch.tensor(acc_temp))
    acc = torch.stack(acc, dim=0)
    acc_mean = acc.mean(dim=0)
    best_epoch = acc_mean.argmax().item()
    print('---------------- Final Epoch Result ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(acc[:,-1].mean(), acc[:,-1].std(), ))
    print(f'---------------- Best Epoch: {best_epoch} ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(acc[:,best_epoch].mean(), acc[:,best_epoch].std()), flush=True)

if __name__ == '__main__':
    parser = add_model_args(None, hyper=True)
    parser.opt_list('--batch_size', type=int, default=32, tunable=True, options=[32, 128])
    parser.opt_list('--hidden_units', type=int, default=64, tunable=True, options=[32, 64, 128])
    parser.opt_list('--num_steps', type=int, default=32, tunable=True, options=[8, 16])
    
    # REDDIT
    # parser.opt_list('--num_steps', type=int, default=32, tunable=True, options=[8, 4])
    # parser.opt_list('--batch_size', type=int, default=32, tunable=True, options=[32, 64])
    # parser.opt_list('--hidden_units', type=int, default=64, tunable=True, options=[32, 64])

    dd_ablation = False
    # DD Ablation - Uncomment
    # parser.opt_list('--num_agents', type=int, default=32, tunable=True, options=[512, 256, 128, 64, 32, 16, 8, 4, 2])
    # parser.opt_list('--num_steps', type=int, default=32, tunable=True, options=[32, 16, 8, 4, 2])
    # parser.opt_list('--batch_size', type=int, default=32, tunable=False, options=[32]) # 128 is best, but 32 is ok
    # parser.opt_list('--hidden_units', type=int, default=32, tunable=False, options=[32]) # 64 is best, but 32 is ok
    # dd_ablation = True

    parser.add_argument('--iters_per_epoch', type=int, default=50) # Number of steps/batches per epoch. Like in GIN

    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--slurm', action='store_true', default=False)
    parser.add_argument('--grid_search', action='store_true', default=False)
    parser.add_argument('--gpu_jobs', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='MUTAG', help="Options are ['MUTAG', 'PTC', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI']")
    parser.add_argument('--one_hot_degree', action='store_true', default=False)
    parser.add_argument('--baseline_in_mlp', action='store_true', default=False)


    parser.add_argument('--layers_per_conv', type=int, default=1) # for PPGN
 

    parser.add_argument('--model_type', type=str, default='agent')
    

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


        job_name = f'{args.dataset}_{args.model_type}{"_ablation_" if dd_ablation else ""}_{args.num_agents}'
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
