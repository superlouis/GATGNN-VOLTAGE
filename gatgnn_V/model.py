import torch, numpy as np
import torch.optim as optim
from   torch.optim import lr_scheduler 
from   torch.nn import Linear, Dropout, Parameter
import torch.nn.functional as F 
import torch.nn as nn

from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax
from torch_geometric.nn       import global_add_pool
from torch_geometric.nn.inits import glorot, zeros
torch.cuda.empty_cache()

class GAT_Crystal(MessagePassing):
    def __init__(self, in_features, out_features, edge_dim, heads, concat=False,
                 dropout=0, bias=True, **kwargs):
        '''
        Our Augmented Graph Attention Layer
        > Defined in paper as *AGAT*
        =======================================================================
        in_features    : input-features
        out_features   : output-features
        edge_dim       : edge-features
        heads          : attention-heads
        concat         : to concatenate the attention-heads or sum them
        dropout        : 0
        bias           : True
        '''    
        super(GAT_Crystal, self).__init__(aggr='add',flow='target_to_source', **kwargs)
        self.in_features       = in_features
        self.out_features      = out_features
        self.heads             = heads
        self.concat            = concat
        self.dropout           = dropout
        self.neg_slope         = 0.2
        self.prelu             = nn.PReLU()
        self.bn1               = nn.BatchNorm1d(heads)
        self.W                 = Parameter(torch.Tensor(in_features+edge_dim,heads*out_features))
        self.att               = Parameter(torch.Tensor(1,heads,2*out_features))

        if bias and concat       : self.bias = Parameter(torch.Tensor(heads * out_features))
        elif bias and not concat : self.bias = Parameter(torch.Tensor(out_features))
        else                     : self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x,edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i,edge_attr): 
        x_i   = torch.cat([x_i,edge_attr],dim=-1)
        x_j   = torch.cat([x_j,edge_attr],dim=-1)
        
        x_i   = F.softplus(torch.matmul(x_i,self.W))
        x_j   = F.softplus(torch.matmul(x_j,self.W))
        x_i   = x_i.view(-1, self.heads, self.out_features)
        x_j   = x_j.view(-1, self.heads, self.out_features)

        alpha = F.softplus((torch.cat([x_i, x_j], dim=-1)*self.att).sum(dim=-1))
        alpha = F.softplus(self.bn1(alpha))
        alpha = softmax(alpha,edge_index_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        x_j   = (x_j * alpha.view(-1, self.heads, 1)).transpose(0,1)
        return x_j

    def update(self, aggr_out,x):
        if self.concat is True:    aggr_out = aggr_out.view(-1, self.heads * self.out_features)
        else:                      aggr_out = aggr_out.mean(dim=0)
        if self.bias is not None:  aggr_out = aggr_out + self.bias
        return aggr_out

class COMPOSITION_Attention(torch.nn.Module):
    def __init__(self,neurons):
        '''
        Global-Attention Mechanism based on the crystal's elemental composition
        > Defined in paper as *GI-M1*
        =======================================================================
        neurons : number of neurons to use 
        '''
        super(COMPOSITION_Attention, self).__init__()
        self.node_layer1    = Linear(neurons+103,32)
        self.atten_layer    = Linear(32,1)

    def forward(self,x,batch,global_feat):
        counts      = torch.unique(batch,return_counts=True)[-1]
        graph_embed = global_feat
        graph_embed = torch.repeat_interleave(graph_embed, counts, dim=0)
        chunk       = torch.cat([x,graph_embed],dim=-1)
        x           = F.softplus(self.node_layer1(chunk))
        x           = self.atten_layer(x)
        weights     = softmax(x,batch)
        return weights

class GATGNN_R(torch.nn.Module):
    def __init__(self,heads,neurons=64,nl=3,xtra_layers=True
                     , concat_comp=False,neighborHood='small'):
        super(GATGNN_R, self).__init__()

        self.n_heads        = heads
        self.number_layers  = nl
        self.concat_comp    = concat_comp
        self.additional     = xtra_layers   

        n_h, n_hX2          = neurons, neurons*2
        self.neurons        = neurons
        self.neg_slope      = 0.2  

        self.embed_n        = Linear(92,n_h)
        self.embed_e        = Linear(41,n_h) if neighborHood == 'small' else Linear(9,n_h)
        self.embed_comp     = Linear(103,n_h)
 
        self.node_att       = nn.ModuleList([GAT_Crystal(n_h,n_h,n_h,self.n_heads) for i in range(nl)])
        self.batch_norm     = nn.ModuleList([nn.BatchNorm1d(n_h) for i in range(nl)])

        self.comp_atten     = COMPOSITION_Attention(n_h)

        if self.concat_comp : reg_h   = n_hX2
        else                : reg_h   = n_h

        if self.additional:
            self.linear1    = nn.Linear(reg_h,reg_h)
            self.linear2    = nn.Linear(reg_h,reg_h)

    def forward(self,x, edge_index, edge_attr,batch, global_feat,cluster):
        x           = self.embed_n(x)
        edge_attr   = F.leaky_relu(self.embed_e(edge_attr),self.neg_slope)

        for a_idx in range(len(self.node_att)):
            x     = self.node_att[a_idx](x,edge_index,edge_attr)
            x     = self.batch_norm[a_idx](x)
            x     = F.softplus(x)
        
        ag        = self.comp_atten(x,batch,global_feat)
        x         = (x)*ag
        
        # CRYSTAL FEATURE-AGGREGATION 
        y         = global_add_pool(x,batch).unsqueeze(1).squeeze()
        
        return y

class REACTION_PREDICTOR(torch.nn.Module):
    def __init__(self,inp_dim,module1,module2,neurons=64):
        super(REACTION_PREDICTOR, self).__init__()
        self.neurons        = neurons
        self.neg_slope      = 0.2  
        
        self.gatgnn0        = module1
        self.gatgnn1        = module2
        
        self.layer1         = Linear(inp_dim,neurons)
        self.layer2         = Linear(neurons,neurons)
        
        self.output         = Linear(neurons,1)     

    def forward(self, data0, data1):
        x0, x1                      = data0.x,data1.x
        edge_index0, edge_index1    = data0.edge_index, data1.edge_index
        edge_attr0, edge_attr1      = data0.edge_attr, data1.edge_attr
        batch0,batch1               = data0.batch,data1.batch
        global_feat0, global_feat1  = data0.global_feature, data1.global_feature
        cluster0,cluster1           = data0.cluster, data1.cluster
        
        Embedding0 = self.gatgnn0(x0,edge_index0,edge_attr0,batch0,global_feat0,cluster0)
        Embedding1 = self.gatgnn1(x1,edge_index1,edge_attr1,batch1,global_feat1,cluster1)
        
        x = torch.cat([Embedding0, Embedding1],dim=-1)
        
        x = F.leaky_relu(self.layer1(x),self.neg_slope)
        x = F.dropout(x,p=0.3,training=self.training)
        
        x = F.leaky_relu(self.layer2(x),self.neg_slope)
        x = F.dropout(x,p=0.3,training=self.training)
        
        y = self.output(x).squeeze()
        return y