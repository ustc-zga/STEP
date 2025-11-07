import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BatchNorm2d, Conv2d
import numpy as np
from Layers import *
from utils import *

class Model(nn.Module):
    def __init__(self,device, num_nodes,dropout=0,  
                 in_dim=1,out_dim=12,residual_channels=64,dilation_channels=64,
                 skip_channels=256,end_channels=512,num_of_weeks=0,num_of_days=2, num_of_hours=1,num_for_predict=12,latend_num=30,
                 num_layer=2):
        super(Model, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        self.num_layer=num_layer
        
        self.device=device
        
        self.t_len= (num_of_weeks+num_of_days+num_of_hours)*num_for_predict
        self.channel=self.t_len*residual_channels
        
        
        self.tconv11 = TemporalConvLayer(kt=3, c_in=1,c_out=residual_channels//2 , act="GLU")
        self.pool=Pooler(self.t_len-2,residual_channels//2)
        
        self.block1=ST_BLOCK_6(dilation_channels,dilation_channels,num_nodes,self.t_len,K=3,Kt=3)
        
        self.block_c1=ST_BLOCK_6(dilation_channels,dilation_channels,latend_num,self.t_len,K=3,Kt=3)
     
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.SCconv=SCConv(in_features=in_dim, out_features=residual_channels, dropout=0.5,\
                                   alpha=0.2, latend_num=latend_num, gcn_hop = 1)
    

        self.skip_conv1=Conv2d(dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,self.t_len),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.bn=BatchNorm2d(in_dim,affine=False)
      

    def forward(self, input,graph):
        x=self.bn(input)
      
        skip=0
        x_1 = self.tconv11(x)    # nclv
        x_2, x_agg, t_sim_mx = self.pool(x_1)
        s_sim_mx = sim_global(x_agg, sim_type='cos')
      
        adj_mx1=np.load('HiCo-STEP_v1/data/adj_12.npy')
      
        adj1=torch.from_numpy(norm_Adj(adj_mx1)).type(torch.FloatTensor).to(self.device) 
        
        
        adj_mx1=torch.FloatTensor(adj_mx1).to(self.device)
        adj1_aug=aug_edge(s_sim_mx,adj1,percent=0.01)
        
        x_aug=aug_node(t_sim_mx,x, percent=0.001)
        
        x_cluster,adj2,s_loss,A,z1= self.SCconv(x,x_aug,adj1,adj1_aug)
       
       
        x=self.start_conv(x)
       
        skip=0
        
        #1
        # x_cluster,_,_=self.block_c1(x_cluster,adj2) 
        x,_,_=self.block1(x,adj1) 
        # x_cluster,_,_=self.block_c1(x_cluster,adj2)
        # x,_,_=self.block1(x,adj1) 
       
        x_s1_glb = torch.einsum('wvl, ndwl->ndvl', (graph, x_cluster)) 
        # x=torch.cat((x,(x_s1_glb)),1) 
        x= x_s1_glb+x        
        
        s1=self.skip_conv1(x)
        skip=s1+skip 
       
        #2    
        #output
        x = F.relu(skip)      
        x = F.relu(self.end_conv_1(x))            
        x = self.end_conv_2(x)        
      
        return x,s_loss,A
      