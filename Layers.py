import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch.nn import  Conv2d
from torch import Tensor

os.environ['CUDA_VISIBLE_DEVICE']='0,1,2'





class cheby_conv(nn.Module):
   
    def __init__(self,c_in,c_out,K,Kt):
        super(cheby_conv,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 





class ST_BLOCK_6(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_6,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.gcn=cheby_conv(c_out,2*c_out,K,1)
        
        self.c_out=c_out
        self.conv_1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
    def forward(self,x,supports):
        x_input1=self.conv_1(x)
        x1=self.conv1(x)   
        x2=self.gcn(x1,supports)
        filter,gate=torch.split(x2,[self.c_out,self.c_out],1)
        x=(filter+x_input1)*torch.sigmoid(gate)
        return x,x1,x2
    

class GCN1(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, hop = 1):
        super(GCN1, self).__init__()
        self.in_features = in_features
        self.hop = hop
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.w_lot = nn.ModuleList()
    
        for i in range(hop):
            in_features = (self.in_features) if(i==0) else out_features 
            self.w_lot.append(nn.Linear(in_features, out_features, bias=True))
       
    def forward(self, h_c, adj):
       
        b,c,n,t=h_c.shape
      
        h_c=h_c.reshape(-1,n,c)
        # adj normalize
        adj_rowsum = torch.sum(adj,dim=-1,keepdim=True)
        adj = adj.div(torch.where(adj_rowsum>1e-8, adj_rowsum, 1e-8*torch.ones(1,1).cuda())) # row normalize
        
        
        for i in range(self.hop):
           
            h_c=torch.einsum('kk,mkc->mkc',adj,h_c)
            h_c = self.leakyrelu(self.w_lot[i](h_c)) #(B, N, F)
            
        h_c=h_c.reshape(b,c,n,-1)
        return h_c
    

class SpatialHeteroModel(nn.Module):
    '''Spatialtemporal heterogeneity modeling by using a soft-clustering paradigm.
    '''
    def __init__(self, c_in, nmb_prototype, tau=0.5):
        super(SpatialHeteroModel, self).__init__()
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.prototypes = nn.Linear(c_in, nmb_prototype, bias=False)
        self.latent=nmb_prototype
        self.tau = tau
        self.d_model = c_in
        

        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z1, z2):
      
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = self.l2norm(w)
            self.prototypes.weight.copy_(w)
        
       
        b,c,n,t=z1.size()
        z1=z1.narrow(3,1,1).squeeze()
        z2=z2.narrow(3,1,1).squeeze()
        zc1 = self.prototypes(self.l2norm(z1.reshape(-1, self.d_model))) # nd -> nk, assignment q, embedding z
        zc2 = self.prototypes(self.l2norm(z2.reshape(-1, self.d_model))) # nd -> nk
       
        with torch.no_grad():
            q1 = sinkhorn(zc1.detach())
            q2 = sinkhorn(zc2.detach())
     
        l1 = - torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = - torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
     
        clu=zc1.reshape(b,n,-1)
      
        
        
        return clu, l1+l2
    
@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes
    
    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    
    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

class SCConv(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, latend_num, gcn_hop):
        super(SCConv, self).__init__()
        self.in_features = in_features
        self.out_features=out_features
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.gcn_1=GCN1(in_features=out_features, out_features=out_features, \
                                       dropout=dropout, alpha=alpha, hop = gcn_hop)
     
        self.shm=SpatialHeteroModel(c_in=out_features, nmb_prototype=latend_num, tau=0.5)
      
        self.start_conv = nn.Conv2d(in_channels=in_features,
                                    out_channels=out_features,
                                    kernel_size=(1,1))
        self.start_conv_1 = nn.Conv2d(in_channels=in_features,
                                    out_channels=out_features,
                                    kernel_size=(1,1))
   

    def forward(self,x,x_aug,adj,adj_aug):
       
                               
        b,c,n,t=x.size()
        x=self.start_conv(x) 
        x_aug=self.start_conv(x_aug)
       
    
        z1=self.gcn_1(x,adj)
        z2=self.gcn_1(x,adj_aug)
     
        A,ssl_loss=self.shm(z1,z2)
     
        h_c=torch.einsum('bkn,bcnt->bckt',A.permute(0,2,1),z1)
       
       
        adj2 = torch.bmm(torch.matmul(A.permute(0,2,1),adj),A) # (B, K, K)
        adj2=adj2.narrow(0,1,1).squeeze()

       
        return h_c,adj2,ssl_loss,A,z1
