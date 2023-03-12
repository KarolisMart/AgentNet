# This implementation is based on https://github.com/KarolisMart/DropGNN/blob/main/mpnn-qm9.py
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
import time

from util import cos_anneal, get_cosine_schedule_with_warmup
from model import AgentNet, add_model_args

parser = add_model_args()
parser.add_argument('--target', default=0)
parser.add_argument('--aux_loss', action='store_true', default=False)
parser.add_argument('--complete_graph', action='store_true', default=False)
args = parser.parse_args()
print(args)
target = int(args.target)
print('---- Target: {} ----'.format(target))

class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, int(args.target)]  # Specify target: 0 = mu
        return data

if args.complete_graph:
    class Complete(object):
        def __call__(self, data):
            device = data.edge_index.device

            row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
            col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

            row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
            col = col.repeat(data.num_nodes)
            edge_index = torch.stack([row, col], dim=0)

            edge_attr = None
            if data.edge_attr is not None:
                idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
                size = list(data.edge_attr.size())
                size[0] = data.num_nodes * data.num_nodes
                edge_attr = data.edge_attr.new_zeros(size)
                edge_attr[idx] = data.edge_attr

            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            data.edge_attr = edge_attr
            data.edge_index = edge_index

            return data
    
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'MPNN-QM9')
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = QM9(path, transform=transform)
else:
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', '1-QM9')
    dataset = QM9(path, transform=T.Compose([MyTransform(), T.Distance()]))



dataset = dataset.shuffle()

# Normalize targets to mean = 0 and std = 1.
tenpercent = int(len(dataset) * 0.1)
mean = dataset.data.y[tenpercent:].mean(dim=0)
std = dataset.data.y[tenpercent:].std(dim=0)
dataset.data.y = (dataset.data.y - mean) / std

test_dataset = dataset[:tenpercent]
val_dataset = dataset[tenpercent:2 * tenpercent]
train_dataset = dataset[2 * tenpercent:]
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=torch.get_num_threads())
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=torch.get_num_threads())
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=torch.get_num_threads())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_aux_loss = args.aux_loss
model = AgentNet(num_features=dataset.num_features, hidden_units=args.hidden_units, num_out_classes=1, dropout=args.dropout, num_steps=args.num_steps,
                num_agents=args.num_agents, reduce=args.reduce, node_readout=args.node_readout, use_step_readout_lin=args.use_step_readout_lin,
                num_pos_attention_heads=args.num_pos_attention_heads, readout_mlp=args.readout_mlp, self_loops=args.self_loops, post_ln=args.post_ln,
                attn_dropout=args.attn_dropout, no_time_cond=args.no_time_cond, mlp_width_mult=args.mlp_width_mult, activation_function=args.activation_function,
                negative_slope=args.negative_slope, input_mlp=args.input_mlp, attn_width_mult=args.attn_width_mult, importance_init=args.importance_init,
                random_agent=args.random_agent, test_argmax=args.test_argmax, global_agent_pool=args.global_agent_pool, agent_global_extra=args.agent_global_extra,
                basic_global_agent=args.basic_global_agent, basic_agent=args.basic_agent, bias_attention=args.bias_attention, visited_decay=args.visited_decay,
                sparse_conv=args.sparse_conv, mean_pool_only=args.mean_pool_only, edge_negative_slope=args.edge_negative_slope,
                final_readout_only=args.final_readout_only, num_edge_features=5, regression=True, qm9=True).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs, min_lr_mult=args.min_lr_mult)

mean, std = mean[target].to(device), std[target].to(device)

def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred, aux_pred = model(data.x, data.edge_index, data.batch, data.edge_attr)
        loss = F.mse_loss(pred.view(-1), data.y)
        if use_aux_loss:
            aux_loss = F.mse_loss(aux_pred.view(-1), data.y.unsqueeze(1).expand(-1,aux_pred.size(1)).clone().view(-1))
            loss = 0.75*loss + 0.25*aux_loss
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        pred, _ = model(data.x, data.edge_index, data.batch, data.edge_attr)
        error += ((pred.view(-1) * std) -
                  (data.y * std)).abs().sum().item() # MAE
    return error / len(loader.dataset)

print(model.__class__.__name__)
best_val_error = None
for epoch in range(1, args.epochs+1):
    torch.cuda.reset_peak_memory_stats(0)
    start = time.time()
    lr = scheduler.optimizer.param_groups[0]['lr']
    if args.gumbel_warmup < 0:
        gumbel_warmup = args.warmup
    else:
        gumbel_warmup = args.gumbel_warmup
    model.temp = cos_anneal(gumbel_warmup, gumbel_warmup + args.gumbel_decay_epochs, args.gumbel_temp, args.gumbel_min_temp, epoch)
    loss = train(epoch)
    val_error = test(val_loader)
    scheduler.step()#val_error

    if best_val_error is None:
        best_val_error = val_error
    if val_error <= best_val_error:
        test_error = test(test_loader)
        best_val_error = val_error
    
    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, Test MAE: {:.7f}, Time: {:.4f}, Mem: {:.3f}, Cached: {:.3f}'.format(
        epoch, lr, loss, val_error, test_error, time.time() - start, torch.cuda.max_memory_allocated()/1024.0**3, torch.cuda.max_memory_reserved()/1024.0**3), flush=True)
