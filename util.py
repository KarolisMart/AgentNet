import torch
import os
import math
import shutil
import numpy as np
import networkx as nx
from torch import Tensor, LongTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch_scatter import scatter_max, scatter_add, scatter_mean
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from typing import Optional


from torch_scatter import scatter_add
from torch_scatter.utils import broadcast

def print_mem():
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Allocated MAX:', round(torch.cuda.max_memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print('Cached MAX:   ', round(torch.cuda.max_memory_reserved(0)/1024**3,1), 'GB')
    torch.cuda.reset_peak_memory_stats(0)

def spmm(index: Tensor, value: Tensor, m: int, n: int,
         matrix: Tensor, reduce: str='sum') -> Tensor:
    """Matrix product of sparse matrix with dense matrix.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        n (int): The second dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.

    :rtype: :class:`Tensor`
    """

    assert n == matrix.size(-2)

    row, col = index[0], index[1]
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix.index_select(-2, col)
    out = out * value.unsqueeze(-1)
    if reduce == 'sum':
        out = scatter_add(out, row, dim=-2, dim_size=m)
    elif reduce == 'mean' or reduce == 'log' or reduce == 'sqrt':
        out = scatter_add(out, row, dim=-2, dim_size=m)

        # Modified scatter mean to the mean over values (one hot vector), counts only ones
        dim_size = out.size(-2)
        index_dim = -2
        if index_dim < 0:
            index_dim = index_dim + out.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1

        # Value is assumed to be in [0,1]
        count = scatter_add(value.detach(), index[0], index_dim, None, dim_size) #NOTE: could also not detach this
        count[count < 1] = 1
        if reduce == 'log':
            count = count / torch.log2(count + 1)
        elif reduce == 'sqrt':
            count = torch.sqrt(count)
        count = broadcast(count, out, -2)
        if out.is_floating_point():
            out.true_divide_(count)
        else:
            out.div_(count, rounding_mode='floor')
    
    else:
        out = scatter_max(out, row, dim=-2, dim_size=m)[0]

    return out

def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return scatter_add(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == 'log' or reduce == 'sqrt':
        out = scatter_add(src, index, dim, out, dim_size)

        count = scatter_add(torch.ones_like(index, dtype=src.dtype, device=src.device), index, dim, None, dim_size)
        count[count < 1] = 1
        if reduce == 'sqrt':
            count = torch.sqrt(count)
        else:
            count = count / torch.log2(count + 1)
        count = broadcast(count, out, -2)
        if out.is_floating_point():
            out.true_divide_(count)
        else:
            out.div_(count, rounding_mode='floor')
        return out
    else:
        return scatter_max(src, index, dim, out, dim_size)[0]

def gumbel_softmax(src: Tensor, index: LongTensor, num_nodes: int=0, hard: bool=True, tau: float=1.0, i: int=0):
    r"""Computes a sparsely evaluated gumbel softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the gubmel softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        hard (bool, optional): if True, returned samples will be onehot.
        tau (float, optional): Gumbel temperature.

    :rtype: :class:`Tensor`
    """
    num_nodes = maybe_num_nodes(index, num_nodes)
    # Could instead try introducing gumbel scale Beta and increase it during training. But this is probably contrary to what we want to do (get ~deterministic walks): https://openreview.net/pdf?id=S5Rv7Yv2THb
    gumbels = (-torch.empty_like(src, memory_format=torch.legacy_contiguous_format).exponential_().log())
    gumbels = (src + gumbels) / tau
    y_soft = scatter_softmax(gumbels , index, dim=-1, dim_size=num_nodes)
    if hard:
        max_index = scatter_max(y_soft, index, dim=-1, dim_size=num_nodes)[1]
        if max_index.max() >= y_soft.size(0):
            # Apparently scatter_max can sometimes fail for good inputs? In this case - try again
            torch.cuda.synchronize(device=y_soft.device)
            max_index = scatter_max(y_soft, index, dim=-1, dim_size=num_nodes)[1]
            # Check if now OK:
            if max_index.max() >= y_soft.size(0):
                print(i,y_soft.max(), y_soft.min(), src.max(), src.min(), max_index.max())
                print(y_soft[index == index[torch.argmax(y_soft)]], index[torch.argmax(y_soft)], index[index == index[torch.argmax(y_soft)]])
                print(y_soft, max_index, index, y_soft.shape, num_nodes, flush=True) #, flush=True
        y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(0, max_index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

def cos_anneal(e0: int, e1: int, t0: int, t1: int, e: int):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

def lin_anneal(e0: int, e1: int, t0: int, t1: int, e: int):
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1,
    min_lr_mult: float = 1e-7
):
    """
    From https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py#L104

    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr_mult (`float`, *optional*, defaults to -1e-7):
            Smallest LR multiplier to use. Set to 1 for no decay
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(min_lr_mult, float(current_step) / float(max(1, num_warmup_steps)))
        return cos_anneal(num_warmup_steps, num_training_steps, 1.0, min_lr_mult, current_step)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, min_lr_mult: float = 1e-7):
    """
    From https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py#L75

    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr_mult (`float`, *optional*, defaults to -1e-7):
            Smallest LR multiplier to use. Set to 1 for no decay
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return lin_anneal(num_warmup_steps, num_training_steps, 1.0, min_lr_mult, current_step)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# Code to import GIN PTC dataset version https://github.com/weihua916/powerful-gnns/blob/master/util.py
class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

def S2V_to_PyG(data):
    new_data = Data()
    setattr(new_data, 'edge_index', data.edge_mat)
    setattr(new_data, 'x', data.node_features)
    setattr(new_data, 'num_nodes', data.node_features.shape[0])
    setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).long())

    return new_data


def load_data(dataset, degree_as_tag, folder):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('%s/%s.txt' % (folder, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    return [S2V_to_PyG(datum) for datum in g_list]


class PTCDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            name,
            transform=None,
            pre_transform=None,
    ):
        self.name = name
        self.url = 'https://github.com/weihua916/powerful-gnns/raw/master/dataset.zip'

        super(PTCDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = 'raw'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed'
        return os.path.join(self.root, self.name, name)

    @property
    def num_tasks(self):
        return 1

    @property
    def eval_metric(self):
        return 'acc'

    @property
    def task_type(self):
        return 'classification'

    @property
    def raw_file_names(self):
        return ['PTC.mat', 'PTC.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        folder = os.path.join(self.root, self.name)
        path = download_url(self.url, folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)

        shutil.move(os.path.join(folder, f'dataset/{self.name}'), os.path.join(folder, self.name))
        shutil.rmtree(os.path.join(folder, 'dataset'))

        os.rename(os.path.join(folder, self.name), self.raw_dir)

    def process(self):
        data_list = load_data('PTC', degree_as_tag=False, folder=self.raw_dir)
        print(sum([data.num_nodes for data in data_list]))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
