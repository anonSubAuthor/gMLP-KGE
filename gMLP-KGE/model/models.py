import torch
from torch import nn
from torch.nn import functional as F

class InfoNCE(nn.Module):
    """
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.01):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(temperature), requires_grad=True)
    

    def forward(self, values,labels):
        return self.info_nce(values, labels)
    def normalize(self, x):
        return F.normalize(x, dim=-1)
    def info_nce(self,values, labels):
        logits = self.normalize(values)
        logits = logits/self.temperature
        logits = torch.softmax(logits, dim = -1)
        return F.binary_cross_entropy(logits, labels)

def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param


class LTEModel(nn.Module):
    def __init__(self, num_ents, num_rels, params=None):
        super(LTEModel, self).__init__()

        self.p = params
        self.bceloss = InfoNCE(self.p.temperature)
        # self.bceloss = torch.nn.BCELoss()
        self.init_embed = get_param((num_ents, self.p.embed_dim))
        self.device = "cuda"

        self.init_rel = get_param((num_rels * 2, self.p.init_dim))

        self.bias = nn.Parameter(torch.zeros(num_ents))

        self.h_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.embed_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.t_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.embed_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.r_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.x_ops = self.p.x_ops
        self.r_ops = self.p.r_ops
        self.diff_ht = False

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

    def exop(self, x, r, x_ops=None, r_ops=None, diff_ht=False):
        x_head = x_tail = x
        if len(x_ops) > 0:
            for x_op in x_ops.split("."):
                if diff_ht:
                    x_head = self.h_ops_dict[x_op](x_head)
                    x_tail = self.t_ops_dict[x_op](x_tail)
                else:
                    x_head = x_tail = self.h_ops_dict[x_op](x_head)

        if len(r_ops) > 0:
            for r_op in r_ops.split("."):
                r = self.r_ops_dict[r_op](r)

        return x_head, x_tail, r

class gMLPBlock(nn.Module):
    def __init__(self,params, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2) # (256, d_ffn * 2=1024)  [-1,256,1024]
        self.sgu = SpatialGatingUnit(params, d_ffn, seq_len)   #
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        
        self.drop = nn.Dropout(params.g_drop)#这里 0.4

    def forward(self, x):
        residual = x
        x = self.norm(x)     # [-1,256,256]
        x = self.drop(x)
        x = F.gelu(self.channel_proj1(x))  # GELU激活函数 [-1,256,256]
        x = self.sgu(x)   # [-1,256,256]
        x = self.channel_proj2(x)
        out = x + residual
        return out

class SpatialGatingUnit(nn.Module):  # [-1,256,256]
    def __init__(self, params, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.s_drop = params.s_drop
        self.norm = nn.BatchNorm1d(seq_len)   # [-1,256,256]->[-1,256,512]
        self.spatial_proj = nn.Conv1d(seq_len, seq_len*128, kernel_size=1) # [-1,256,512]->[-1,256,512]
        self.spatial_proj_1 = nn.Conv1d(seq_len*128, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)  # 偏差
        self.W = nn.Linear(d_ffn, d_ffn)
    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        u = self.W(F.dropout(F.gelu(u), self.s_drop, self.training))

        v = self.norm(v)
        v = self.spatial_proj(v)
        v = self.spatial_proj_1(v)
        out = u * v

        return out
        
class gMLP(nn.Module):
    def __init__(self, params, d_model=256, d_ffn=512, seq_len=256, num_layers=1):
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(params, d_model, d_ffn, seq_len) for _ in range(num_layers)]
        )

 
    def forward(self, x):
        return self.model(x)

class gMLP_KGE(LTEModel):
    def __init__(self, num_ents, num_rels, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, params)

        self.bn1 = nn.LayerNorm(self.p.embed_dim*2)
        self.bn2 = torch.nn.LayerNorm(self.p.embed_dim)

        
        self.drop1 = nn.Dropout(self.p.input_drop)
        self.drop2 = nn.Dropout(self.p.feature_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hidden_drop)
        self.gmlp = gMLP(params, self.p.k_h, self.p.k_h*self.p.k, 1, num_layers = 1)
        self.fc = nn.Linear(self.p.embed_dim*2, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, self.p.k_h))
        return stack_inp

    def forward(self,sub, rel):
        x = self.init_embed
        r = self.init_rel

        x_h, x_t, r = self.exop(x, r, self.x_ops, self.r_ops)

        sub_emb = torch.index_select(x_h, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)

        all_ent = x_t
        
        stk_inps = self.concat(sub_emb, rel_emb)
        stk_inp = self.drop1(stk_inps)
        
        x = self.gmlp(stk_inp).squeeze(1)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.drop2(x)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        return x
