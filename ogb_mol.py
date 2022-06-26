# Based on https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol
import torch
from torch_geometric.loader import DataLoader
from test_tube.hpc import SlurmCluster
import random

from tqdm import tqdm
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from util import cos_anneal, get_cosine_schedule_with_warmup
from model import AgentNet, add_model_args

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type, use_aux_loss=False):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            if use_aux_loss:
                pred, aux_pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            else:
                pred, _ = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            is_labeled = is_labeled.view(-1)
            if "classification" in task_type: 
                try:
                    loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                except:
                    print('fail')
                    print(batch)
                    print(pred)
                    print(batch.y)
                    print(is_labeled)
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            if use_aux_loss:
                if "classification" in task_type: 
                    aux_loss = cls_criterion(aux_pred[is_labeled.unsqueeze(1).expand(-1,aux_pred.size(1),-1)].to(torch.float32).view(-1), batch.y.to(torch.float32)[is_labeled].unsqueeze(1).expand(-1,aux_pred.size(1)).clone().view(-1))
                else:
                    aux_loss = reg_criterion(aux_pred[is_labeled.unsqueeze(1).expand(-1,aux_pred.size(1),-1)].to(torch.float32).view(-1), batch.y.to(torch.float32)[is_labeled].unsqueeze(1).expand(-1,aux_pred.size(1)).clone().view(-1))
                loss = 0.75*loss + 0.25*aux_loss
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator, use_aux_loss=False):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, _ = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main(args, cluster=None):
    
    print(args)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # Seed things
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    model = AgentNet(num_features=dataset.num_features, hidden_units=args.hidden_units, num_out_classes=dataset.num_classes, dropout=args.dropout, num_steps=args.num_steps,
                    num_agents=args.num_agents, reduce=args.reduce, node_readout=args.node_readout, use_step_readout_lin=args.use_step_readout_lin,
                    num_pos_attention_heads=args.num_pos_attention_heads, readout_mlp=args.readout_mlp, self_loops=args.self_loops, post_ln=args.post_ln,
                    attn_dropout=args.attn_dropout, no_time_cond=args.no_time_cond, mlp_width_mult=args.mlp_width_mult, activation_function=args.activation_function,
                    negative_slope=args.negative_slope, input_mlp=args.input_mlp, attn_width_mult=args.attn_width_mult, importance_init=args.importance_init,
                    random_agent=args.random_agent, test_argmax=args.test_argmax, global_agent_pool=args.global_agent_pool, agent_global_extra=args.agent_global_extra,
                    basic_global_agent=args.basic_global_agent, basic_agent=args.basic_agent, bias_attention=args.bias_attention, visited_decay=args.visited_decay,
                    sparse_conv=args.sparse_conv, num_edge_features=args.num_edge_features, mean_pool_only=args.mean_pool_only, edge_negative_slope=args.edge_negative_slope,
                    regression=args.regression, final_readout_only=args.final_readout_only, ogb_mol=True).to(device)


    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs, min_lr_mult=args.min_lr_mult)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs, min_lr_mult=args.min_lr_mult)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        torch.cuda.reset_peak_memory_stats(0)
        print("=====Epoch {}".format(epoch))
        print('Training...')
        lr = scheduler.optimizer.param_groups[0]['lr']
        if args.gumbel_warmup < 0:
            gumbel_warmup = args.warmup
        else:
            gumbel_warmup = args.gumbel_warmup
        model.temp = cos_anneal(gumbel_warmup, gumbel_warmup + args.gumbel_decay_epochs, args.gumbel_temp, args.gumbel_min_temp, epoch)
        train(model, device, train_loader, optimizer, dataset.task_type, use_aux_loss=args.use_aux_loss)
        scheduler.step()

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator, use_aux_loss=args.use_aux_loss)
        valid_perf = eval(model, device, valid_loader, evaluator, use_aux_loss=args.use_aux_loss)
        test_perf = eval(model, device, test_loader, evaluator, use_aux_loss=args.use_aux_loss)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf, 'LR': lr, 'Mem': round(torch.cuda.max_memory_allocated()/1024.0**3, 3), 'Cached': round(torch.cuda.max_memory_reserved()/1024.0**3, 3)})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

if __name__ == "__main__":
    # AgentNet params
    parser = add_model_args(None, hyper=True)

    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')

    parser.add_argument('--slurm', action='store_true', default=False)
    parser.add_argument('--grid_search', action='store_true', default=False)
    parser.add_argument('--gpu_jobs', action='store_true', default=False)
    # Run all seeds
    parser.opt_list('--seed', type=int, default=0, tunable=True, options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

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

        job_name = f'{args.dataset}_{args.num_agents}_{args.num_steps}_{args.hidden_units}_{args.lr}_{args.edge_negative_slope}_{args.batch_size}_all_seeds'
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