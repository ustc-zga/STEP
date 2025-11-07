import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
       
        x=x.permute(0,1,3,2)
        x_in = self.align(x)[:, :, self.kt - 1:, :]  
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)  
        return torch.relu(self.conv(x) + x_in)  
    

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        '''Align the input and output.
        '''
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1), similar to fc

    def forward(self, x):  # x: (n,c,l,v)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  

    
class Pooler(nn.Module):
    '''Pooling the token representations of  time series.
    '''
    def __init__(self, n_query, d_model, agg='avg'):
        """
        :param n_query: number of query
        :param d_model: dimension of model 
        """
        super(Pooler, self).__init__()

        ## attention matirx
        self.att = nn.Conv2d(d_model, n_query,1) 
        self.align = Align(d_model, d_model)
        self.softmax = nn.Softmax(dim=2) # softmax on the seq_length dim, nclv

        self.d_model = d_model
        self.n_query = n_query 
        if agg == 'avg':
            self.agg = nn.AvgPool2d(kernel_size=(n_query, 1), stride=1)
        elif agg == 'max':
            self.agg = nn.MaxPool2d(kernel_size=(n_query, 1), stride=1)
        else:
            raise ValueError('Pooler supports [avg, max]')
        
    def forward(self, x):
        """
        :param x: key sequence of region embeding, nclv
        :return x: hidden embedding used for conv, ncqv
        :return x_agg: region embedding for spatial similarity, nvc
        :return A: temporal attention, lnv
        """
        x_in = self.align(x)[:, :, -self.n_query:, :] # ncqv
        # calculate the attention matrix A using key x  
        A = self.att(x) # x: nclv, A: nqlv 
        A = F.softmax(A, dim=2) # nqlv

      
        x = torch.einsum('nclv,nqlv->ncqv', x, A)
        x_agg = self.agg(x).squeeze(2) # ncqv->ncv
        x_agg = torch.einsum('ncv->nvc', x_agg) # ncv->nvc

        # calculate the temporal simlarity (prob)
        A = torch.einsum('nqlv->lnqv', A)
        A = self.softmax(self.agg(A).squeeze(2)) # A: lnqv->lnv
       
        return torch.relu(x + x_in), x_agg, A




def aug_edge(sim_mx, input_graph, percent=0.2):
    """Generate the data augumentation from edge perspective 
        for undirected graph without self-loop.
    :param sim_mx: tensor, symmetric similarity, [v,v]
    :param input_graph: tensor, adjacency matrix without self-loop, [v,v]
    :return aug_graph: tensor, augmented adjacency matrix on cuda, [v,v]
    """    
    ## edge dropping starts here
    drop_percent = percent / 2
    
    index_list = input_graph.nonzero() # list of edges [row_idx, col_idx]
    
    edge_num = int((index_list.shape[0]-input_graph.shape[0])/2)  
    

    edge_mask = (input_graph > 0).tril(diagonal=-1)
    
    add_drop_num = int(edge_num * drop_percent) 
    aug_graph = copy.deepcopy(input_graph) 

    drop_prob = torch.softmax(sim_mx[edge_mask], dim=0)
    drop_prob = (1. - drop_prob).cpu().detach().numpy() # normalized similarity to get sampling probability 
    drop_prob /= drop_prob.sum()
   
    drop_list = np.random.choice(edge_num, size=add_drop_num, p=drop_prob)
   
    drop_index = index_list[drop_list]
    
    zeros = torch.zeros_like(aug_graph[0, 0])
    aug_graph[drop_index[:, 0], drop_index[:, 1]] = zeros
    aug_graph[drop_index[:, 1], drop_index[:, 0]] = zeros

    ## edge adding starts here
    node_num = input_graph.shape[0]
    x, y = np.meshgrid(range(node_num), range(node_num), indexing='ij')
    mask = y < x
    x, y = x[mask], y[mask]

    add_prob = sim_mx[torch.ones(sim_mx.size(), dtype=bool).tril(diagonal=-1)] # .numpy()
    add_prob = torch.softmax(add_prob, dim=0).cpu().detach().numpy()
    add_list = np.random.choice(int((node_num * node_num - node_num) / 2), 
                                size=add_drop_num, p=add_prob)
   
    
    ones = torch.ones_like(aug_graph[0, 0])
    aug_graph[x[add_list], y[add_list]] = ones
    aug_graph[y[add_list], x[add_list]] = ones
    
    return aug_graph   


def aug_node(t_sim_mx, emission_data, percent=0.1):
   
    l, n, v = t_sim_mx.shape
    mask_num = int(n * l * v * percent)
    aug_emission = emission_data
    aug_emission=aug_emission.permute(0,3,2,1)
    mask_prob = (1. - t_sim_mx.permute(1, 0, 2).reshape(-1)).cpu().detach().numpy()
    mask_prob /= mask_prob.sum()

    x, y, z = np.meshgrid(range(n), range(l), range(v), indexing='ij')
    mask_list = np.random.choice(n * l * v, size=mask_num, p=mask_prob)
  

    zeros = torch.zeros_like(aug_emission[0, 0, 0])
    aug_emission[
        x.reshape(-1)[mask_list], 
        y.reshape(-1)[mask_list], 
        z.reshape(-1)[mask_list]] = zeros 
    aug_emission=aug_emission.permute(0,3,2,1)
    return aug_emission

def norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[1]
    W = W + np.identity(N)  # 为邻接矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix


def sim_global(emission_data, sim_type='cos'):
    """Calculate the global similarity of traffic emission data.
    :param emission_data: tensor, original emission [n,l,v,c] or location embedding [n,v,c]
    :param type: str, type of similarity, attention or cosine. ['att', 'cos']
    :return sim: tensor, symmetric similarity, [v,v]
    """
   
    if len(emission_data.shape) == 4:
        n,l,v,c = emission_data.shape
        att_scaling = n * l * c
        cos_scaling = torch.norm(emission_data, p=2, dim=(0, 1, 3)) ** -1 # cal 2-norm of each node, dim N
        sim = torch.einsum('btnc, btmc->nm', emission_data, emission_data)
    elif len(emission_data.shape) == 3:
        n,v,c = emission_data.shape
        att_scaling = n * c
        cos_scaling = torch.norm(emission_data, p=2, dim=(0, 2)) ** -1 # cal 2-norm of each node, dim N
        sim = torch.einsum('bnc, bmc->nm', emission_data, emission_data)
  

    if sim_type == 'cos':
        # cosine similarity
        scaling = torch.einsum('i, j->ij', cos_scaling, cos_scaling)
        sim = sim * scaling
    elif sim_type == 'att':
        # scaled dot product similarity
        scaling = float(att_scaling) ** -0.5 
        sim = torch.softmax(sim * scaling, dim=-1)
    else:
        raise ValueError('sim_global only support sim_type in [att, cos].')
    
    return sim



def gumbel_softmax(logits, tau, hard: bool = False, eps: float = 1e-10, dim: int = -1):
        
   

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


