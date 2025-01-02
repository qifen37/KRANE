import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from model.tools import ccorr_new, cconv, ccorr, rotate
import torch.nn.functional as F
class RMHA(MessagePassing):
    def __init__(self, in_channels, out_channels, rel_dim, drop, bias, op, beta):
        super(self.__class__, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = op
        self.bias = bias
        self.beta = beta
        
        # self.w_loop = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_in = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_out = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_rel = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_att = torch.nn.Linear(3*in_channels, out_channels).cuda()
        self.a = torch.nn.Linear(out_channels, 1, bias=False).cuda()
        # self.loop_rel = torch.nn.Parameter(torch.Tensor(1, in_channels)).cuda()
        # torch.nn.init.xavier_uniform_(self.loop_rel)
        
        self.drop_ratio = drop
        self.drop = torch.nn.Dropout(drop, inplace=False)
        self.bn = torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))
        
        self.res_w = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.activation = torch.nn.Tanh() #torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if bias:
            self.register_parameter("bias_value", torch.nn.Parameter(torch.zeros(out_channels)))
        
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, x, edge_index, edge_type, rel_embed, pre_alpha=None):
        # rel_emb = torch.cat([rel_emb, self.loop_rel], dim=0)
        
        num_ent = x.size(0)
        # num_edges = edge_index.size(1)//2
        # #print(x.shape)
        # # loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).cuda()
        # # loop_type   = torch.full((num_ent,), rel_emb.size(0)-1, dtype=torch.long).cuda()
        # self.in_index = edge_index[:, :num_edges]
        # self.in_type = edge_type[:num_edges]
        #print(edge_index)
        in_res = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_emb=rel_embed, pre_alpha=pre_alpha)
        # in_res = self.propagate(edge_index=self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed, pre_alpha=pre_alpha)
        # loop_res = self.propagate(edge_index=loop_index, x=x, edge_type=loop_type, rel_emb=rel_emb, pre_alpha=pre_alpha, mode="loop")
        loop_res = self.res_w(x)
        out = self.drop(in_res) + self.drop(loop_res)

        if self.bias:
            out = out + self.bias_value
        
        out = self.bn(out)
        out = self.activation(out)

        return out, self.w_rel(rel_embed)
    
    def message(self,x_i, x_j, edge_type, rel_emb, ptr, index, size_i, pre_alpha):
        
        rel_emb = torch.index_select(rel_emb, 0, edge_type)
        #print(x_i.shape, x_j.shape, rel_emb.shape)
        xj_rel = self.rel_transform(x_j, rel_emb)
        num_edge = xj_rel.size(0)//2

        in_message = xj_rel[:num_edge]
        out_message = xj_rel[num_edge:]
        #print(in_message.shape, out_message.shape)        
        trans_in = self.w_in(in_message)
        trans_out = self.w_out(out_message)
        #print(trans_in.shape, trans_out.shape)      
        out = torch.cat((trans_in, trans_out), dim=0)
        
        b = self.leaky_relu(self.w_att(torch.cat((x_i, rel_emb, x_j), dim=1))).cuda()
        b = self.a(b).float()
        #print(b.shape, index, size_i)
        alpha = softmax(b, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.drop_ratio, training=self.training, inplace=False)
        if pre_alpha!=None and self.beta != 0:
            self.alpha = alpha*(1-self.beta) + pre_alpha*(self.beta)
        else:
            self.alpha = alpha
        out = out * alpha.view(-1,1)


        return out

    def update(self, aggr_out):
        return aggr_out

    def rel_transform(self, ent_embed, rel_embed):
        
        if self.op == 'corr':
            #print(ent_embed.shape, rel_embed.shape)
            trans_embed  = ccorr(ent_embed, rel_embed)
        elif self.op == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.op == 'mult':
            trans_embed = ent_embed * rel_embed
        elif self.op == "corr_new":
            trans_embed = ccorr_new(ent_embed, rel_embed)
        # elif self.op == "conv":
        #     trans_embed = cconv(ent_embed, rel_emb)
        # elif self.op == "conv_new":
        #     trans_embed = cconv_new(ent_embed, rel_emb)
        # elif self.op == 'cross':
        #     trans_embed = ent_embed * rel_emb + ent_embed
        # elif self.op == "corr_plus":
        #     trans_embed = ccorr_new(ent_embed, rel_emb) + ent_embed
        elif self.op == "rotate":
            trans_embed = rotate(ent_embed, rel_embed)
        else:
            raise NotImplementedError
        
        return trans_embed
