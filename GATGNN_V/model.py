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

class GATGNN_R(torch.nn.Module):
    def __init__(self,heads,classification=None,neurons=64,nl=3,xtra_layers=True,global_attention='composition',
                 unpooling_technique='random',concat_comp=False,edge_format='CGCNN'):
        super(GATGNN_R, self).__init__()

        self.n_heads        = heads
        self.classification = True if classification != None else False 
        self.unpooling      = unpooling_technique
        self.g_a            = global_attention
        self.number_layers  = nl
        self.concat_comp    = concat_comp
        self.additional     = xtra_layers   

        n_h, n_hX2          = neurons, neurons*2
        self.neurons        = neurons
        self.neg_slope      = 0.2  

        self.embed_n        = Linear(92,n_h)
        self.embed_e        = Linear(41,n_h) if edge_format in ['CGCNN','NEW'] else Linear(9,n_h)
        self.embed_comp     = Linear(103,n_h)
 
        self.node_att       = nn.ModuleList([GAT_Crystal(n_h,n_h,n_h,self.n_heads) for i in range(nl)])
        self.batch_norm     = nn.ModuleList([nn.BatchNorm1d(n_h) for i in range(nl)])

        self.comp_atten     = COMPOSITION_Attention(n_h)

        if self.concat_comp : reg_h   = n_hX2
        else                : reg_h   = n_h

        if self.additional:
            self.linear1    = nn.Linear(reg_h,reg_h)
            self.linear2    = nn.Linear(reg_h,reg_h)

        if self.classification :    self.out  =  Linear(reg_h,2)
        else:                       self.out  =  Linear(reg_h,1)

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