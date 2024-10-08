﻿import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba
from mamba_blocks import MambaBlock
from layers import GraphConvolution, Shift_gcn #unit_gcn
from ctrgcn import unit_gcn
from graph.tools import normalize_digraph
from shiftgcn_model.shift_gcn import Shift_tcn

### torch version too old for timm
### https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

### torch version too old for timm
### https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)




def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=False,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation
                ),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        # print(len(self.branches))
        # initialize
        self.apply(weights_init)



    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), groups=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class MHSA(nn.Module):
    def __init__(self, dim_in, dim, A, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., insert_cls_layer=0, pe=False, num_point=25,
                 outer=True, layer=0,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_point = num_point
        self.layer = layer



        h1 = A.sum(0)
        h1[h1 != 0] = 1
        h = [None for _ in range(num_point)]
        h[0] = np.eye(num_point)
        h[1] = h1
        self.hops = 0*h[0]
        for i in range(2, num_point):
            h[i] = h[i-1] @ h1.transpose(0, 1)
            h[i][h[i] != 0] = 1

        for i in range(num_point-1, 0, -1):
            if np.any(h[i]-h[i-1]):
                h[i] = h[i] - h[i - 1]
                self.hops += i*h[i]
            else:
                continue

        self.hops = torch.tensor(self.hops).long()
        #
        self.rpe = nn.Parameter(torch.zeros((self.hops.max()+1, dim)))

        self.w1 = nn.Parameter(torch.zeros(num_heads, head_dim))



        A = A.sum(0)
        A[:, :] = 0

        self.outer = nn.Parameter(torch.stack([torch.eye(A.shape[-1]) for _ in range(num_heads)], dim=0), requires_grad=True)

        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.kv = nn.Conv2d(dim_in, dim * 2, 1, bias=qkv_bias)
        self.q = nn.Conv2d(dim_in, dim, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)


        self.proj = nn.Conv2d(dim, dim, 1, groups=6)

        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        self.insert_cls_layer = insert_cls_layer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, e):
        N, C, T, V = x.shape
        kv = self.kv(x).reshape(N, 2, self.num_heads, self.dim // self.num_heads, T, V).permute(1, 0, 4, 2, 5, 3)
        k, v = kv[0], kv[1]

        ## n t h v c
        q = self.q(x).reshape(N, self.num_heads, self.dim // self.num_heads, T, V).permute(0, 3, 1, 4, 2)

        e_k = e.reshape(N, self.num_heads, self.dim // self.num_heads, T, V).permute(0, 3, 1, 4, 2)
        #
        #
        pos_emb = self.rpe[self.hops]
        #
        k_r = pos_emb.view(V, V, self.num_heads, self.dim // self.num_heads)
        #
        b = torch.einsum("bthnc, nmhc->bthnm", q, k_r)
        #
        c = torch.einsum("bthnc, bthmc->bthnm", q, e_k)
        d = torch.einsum("hc, bthmc->bthm", self.w1, e_k).unsqueeze(-2)


        a = q @ k.transpose(-2, -1)

        attn = a + b + c + d


        attn = attn * self.scale

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)


        x = (self.alpha * attn + self.outer) @ v
        # x = attn @ v


        x = x.transpose(3, 4).reshape(N, T, -1, V).transpose(1, 2)
        x = self.proj(x)

        x = self.proj_drop(x)
        return x

# using conv2d implementation after dimension permutation
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 num_heads=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x = self.fc1(x.transpose(1,2)).transpose(1,2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # x = self.fc2(x.transpose(1,2)).transpose(1,2)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class unit_vit(nn.Module):
    def __init__(self, dim_in, dim, A, num_of_heads, add_skip_connection=True,  qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer=0,
                insert_cls_layer=0, pe=False, num_point=25, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim_in)
        self.dim_in = dim_in
        self.dim = dim
        self.add_skip_connection = add_skip_connection
        self.num_point = num_point
        self.attn = MHSA(dim_in, dim, A, num_heads=num_of_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                             proj_drop=drop, insert_cls_layer=insert_cls_layer, pe=pe, num_point=num_point, layer=layer, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.dim_in != self.dim:
            self.skip_proj = nn.Conv2d(dim_in, dim, (1, 1), padding=(0, 0), bias=False)
        self.pe_proj = nn.Conv2d(dim_in, dim, 1, bias=False)
        self.pe = pe

    def forward(self, x, joint_label, groups):
        ## more efficient implementation
        label = F.one_hot(torch.tensor(joint_label)).float().to(x.device)
        z = x @ (label / label.sum(dim=0, keepdim=True))

        # w/o proj
        # z = z.permute(3, 0, 1, 2)
        # w/ proj
        z = self.pe_proj(z).permute(3, 0, 1, 2)

        e = z[joint_label].permute(1, 2, 3, 0)

        if self.add_skip_connection:
            if self.dim_in != self.dim:
                x = self.skip_proj(x) + self.drop_path(self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e))
        else:
            x = self.drop_path(self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), e))

        # x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))

        return x

class TCN_ViT_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, num_of_heads=6, residual=True, kernel_size=5, dilations=[1,2], pe=False, num_point=25, layer=0):
        super(TCN_ViT_unit, self).__init__()
        self.mamba= MambaBlock(embed_dim=500)
        # self.mamba_2 = MambaBlock(embed_dim=512)
        self.shift_gcn = Shift_gcn(in_channels,out_channels,A)
        self.shift_gcn_2 = Shift_gcn(out_channels,108,A).to('cuda')
        self.shift_gcn_3 = Shift_gcn(108,54,A).to('cuda') 
        self.shift_gcn_4 = Shift_gcn(54,20,A).to('cuda')
        self.shift_gcn_5 = Shift_gcn(20,54,A).to('cuda')
        self.shift_gcn_6 = Shift_gcn(54,108,A).to('cuda')
        self.shift_gcn_7 = Shift_gcn(108,216,A).to('cuda')
        # self.unit_gcn = unit_gcn(in_channels,out_channels,A,adaptive =True)
        # self.unit_gcn_2 = unit_gcn(out_channels,108,A,adaptive =True).to('cuda')
        # self.unit_gcn_3 = unit_gcn(108,54,A,adaptive =True).to('cuda') 
        # self.unit_gcn_4 = unit_gcn(54,20,A,adaptive =True).to('cuda')
        # self.unit_gcn_5 = unit_gcn(20,54,A,adaptive =True).to('cuda')
        # self.unit_gcn_6 = unit_gcn(54,108,A,adaptive =True).to('cuda')
        # self.unit_gcn_7 = unit_gcn(108,216,A,adaptive =True).to('cuda')
        # self.vit1 = unit_vit(in_channels, out_channels, A, add_skip_connection=residual, num_of_heads=num_of_heads, pe=pe, num_point=num_point, layer=layer)
        # self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        # self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,dilations=dilations,
                                            # redisual=True has worse performance in the end
                                            # residual=False)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        # self.graph_conv = GraphConvolution()
        self.w = nn.Parameter(torch.empty(1, out_channels, 1, 1))
        torch.nn.init.xavier_uniform_(self.w)
        self.adj = A
        self.act = nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.pe_proj = nn.Conv2d(in_channels,out_channels, 1, bias=False)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, joint_label, groups):
        # y = self.act(self.tcn1(self.vit1(x, joint_label, groups)) + self.residual(x))
        # x_vit = self.vit1(x,joint_label,groups)
        label = F.one_hot(torch.tensor(joint_label)).float().to(x.device)
        # label = nn.Parameter(torch.randn(label.shape).to(x.device))
        # label = F.softmax(label, dim=0)
        z = x @ (label / label.sum(dim=0, keepdim=True))
        # # # w/o proj
        # # # z = z.permute(3, 0, 1, 2)
        # # # w/ proj
        z = self.pe_proj(z).permute(3, 0, 1, 2)
        # # e = z.permute(1,2,3,0)
        e = z[joint_label].permute(1, 2, 3, 0)
        # print(e.shape,z.shape)
        shift_gcn_out = self.shift_gcn(x)
        # unit_gcn_out = self.unit_gcn(x)
        # shift_gcn_out = unit_gcn(self.in_channels, self.out_channels, self.adj, 8, 25).to('cuda')(x)
        # shift_gcn_out = torch.cat((shift_gcn_out, e), dim=1)
        n,c,t,v = shift_gcn_out.size()
        # n,c,t,v = unit_gcn_out.size()
        # c = int(c_2/2) # c = c_2/2
        # shift_gcn_out = Shift_gcn(c_2,c,self.adj).to('cuda')(shift_gcn_out)
        # w = nn.Parameter(torch.empty(1, c, 1, 1))
        # torch.nn.init.xavier_uniform_(w)
        w = F.normalize(self.w, p=2, dim=1) 
        w = w.cuda()
        wx = shift_gcn_out * w
        one_minus_w = 1 - w
        we = e * one_minus_w
        f = wx + we
        # gate = torch.sigmoid(nn.Parameter(torch.randn(1, c, 1, 1)))  # Learnable gate
        # gate = gate.cuda()
        # shift_gcn_out = gate * shift_gcn_out + (1 - gate) * e

        # x_vit_1 = Shift_gcn(c,108,self.adj).to('cuda')(shift_gcn_out)
        # x_vit_2 = Shift_gcn(108,54,self.adj).to('cuda')(x_vit_1) 
        # x_vit = Shift_gcn(54,20,self.adj).to('cuda')(x_vit_2)
        # x_vit_1 = self.unit_gcn_2(f) #(shift_gcn_out)
        # x_vit_2 = self.unit_gcn_3(x_vit_1) 
        # x_vit = self.unit_gcn_4(x_vit_2)
        x_vit_1 = self.shift_gcn_2(f) #(shift_gcn_out)
        x_vit_2 = self.shift_gcn_3(x_vit_1) 
        x_vit = self.shift_gcn_4(x_vit_2)
        x_dash = x_vit.permute(0,2,3,1).contiguous()
        n,t,v,c =x_dash.size()
        x_dash = x_dash.view(n,t,v*c)
        # print(c,f.shape[1])
        # x_vit_1 = unit_gcn(c,108,self.adj,12,25).to('cuda')(shift_gcn_out)
        # x_vit_2 = unit_gcn(108,54,self.adj,9,25).to('cuda')(x_vit_1) 
        # x_vit = unit_gcn(54,20,self.adj,5,25).to('cuda')(x_vit_2)
        # x_dash = x_vit.permute(0,2,3,1).contiguous()
        # n,t,v,c =x_dash.size()
        # x_dash = x_dash.view(n,t,v*c)
        # linear_vit = nn.Linear(v*c, 512, dtype=torch.float16)
        # linear_vit = linear_vit.to('cuda')
        # x_dash = linear_vit(x_dash)
        x_dash = self.mamba(x_dash)
        # linear_vit_2=nn.Linear(512,v*c, dtype=torch.float16)
        # linear_vit_2=linear_vit_2.to('cuda')
        # x_dash=linear_vit_2(x_dash)
        # x_dash = x_dash.view(n,t,v,c).permute(0,3,1,2).contiguous() + x_vit
        # x_dash = unit_gcn(c,54,self.adj,9,25).to('cuda')(x_dash) + x_vit_2
        # x_dash = unit_gcn(54,108,self.adj,12,25).to('cuda')(x_dash) + x_vit_1
        # x_dash = unit_gcn(108,216,self.adj,8,25).to('cuda')(x_dash) + shift_gcn_out
        x_dash = x_dash.view(n,t,v,c).permute(0,3,1,2).contiguous() #+ x_vit
        # x_dash = Shift_gcn(c,54,self.adj).to('cuda')(x_dash) + x_vit_2
        # x_dash = Shift_gcn(54,108,self.adj).to('cuda')(x_dash) + x_vit_1
        # x_dash = Shift_gcn(108,216,self.adj).to('cuda')(x_dash) + shift_gcn_out
        # x_dash = self.unit_gcn_5(x_dash) + x_vit_2
        # x_dash = self.unit_gcn_6(x_dash) + x_vit_1
        # x_dash = self.unit_gcn_7(x_dash) + f #shift_gcn_out
        x_dash = self.shift_gcn_5(x_dash) + x_vit_2
        x_dash = self.shift_gcn_6(x_dash) + x_vit_1
        x_dash = self.shift_gcn_7(x_dash) + f #+ shift_gcn_out
        y = self.act(self.tcn1(x_dash) + self.residual(x))

        # vit_out = self.vit1(x, joint_label, groups)
        # vit_out_prime = vit_out.permute(0,2,3,1).contiguous()
        # N_v,T_v,V_v,C_v = vit_out_prime.size()
        # linear_vit = nn.Linear(C_v, 3, dtype=torch.float16)
        # linear_vit = linear_vit.to('cuda')
        # vit_flat = vit_out_prime.view(N_v*T_v*V_v,C_v)
        # vit_flat = linear_vit(vit_flat.half())
        # x_prime = vit_flat.view(N_v,T_v,V_v,3)
        # print(x.shape)
        # shift_gcn_out = self.shift_gcn(x)
        # x_prime = x.permute(0,2,3,1).contiguous()
        # N,T,V,C = x_prime.size()
        # N,T,V,C= vit_out_prime.size()
        # print(x_prime.size())
        # print(N,T,V,C)
        # x_reshaped = x_prime.view(N * T, V, C)

        # D=16
        # graph_convolution_layer = GraphConvolution(input_dim=C,output_dim=D,num_vetex=25)
        # adj_tensor = torch.tensor(self.adj)
        # adjacency_matrix = np.sum(self.adj, axis=0)
        # Apply a threshold to get a binary adjacency matrix
        # adjacency_matrix[adjacency_matrix > 0] = 1
        # normalize_adj = torch.tensor(normalize_digraph(adjacency_matrix))
        # adj_mat_rep = normalize_adj.unsqueeze(0).repeat(N * T, 1, 1)

        # gph_out = graph_convolution_layer(normalize_adj,x_prime)
        # gph_out = output_reshaped.view(N, T, V, D)
        # flattened_out = gph_out.contiguous().view(N, T, V * D)
        # N1, T1, VD = flattened_out.size()
        # flattened_out = vit_out_prime.contiguous().view(N,T, V*C)
        # N1, T1, VD = flattened_out.size()
        # V_prime = 512
        # # D_prime = 512
        # linear_layer = nn.Linear(VD, V_prime, dtype=torch.float16)
        # linear_layer = nn.Linear(VD, V_prime, dtype=torch.float16)
        # linear_layer = linear_layer.to('cuda')
        # flattened_output_reshaped = flattened_out.view(N1*T1, VD)
        # # print(flattened_output_reshaped.device)
        # linear_output = linear_layer(flattened_output_reshaped.half())
        # linear_output = linear_output.view(N1, T1, V_prime)
        # mamba_out = self.mamba(linear_output.float())
        # mamba_out = linear_output
        # mamba_out = self.mamba(flattened_out.float())
        # mamba_out = self.mamba_2(mamba_out_1)
        # lin_layer_2 = nn.Linear(V_prime,VD,dtype=torch.float16)
        # lin_layer_2 = lin_layer_2.to('cuda')
        # lin_layer_3 = nn.Linear(VD,V*216,dtype=torch.float16)
        # lin_layer_3 = lin_layer_3.to('cuda')
        # mamba_out_reshaped = mamba_out.view(N1*T1,V_prime).half()
        # # mamba_out_reshaped = linear_output.view(N1*T1,V_prime).half()
        # lin_out_2 = lin_layer_2(mamba_out_reshaped)
        # lin_out_3 = lin_layer_3(lin_out_2)
        # lin_out_3 = lin_out_3.view(N1,T1,V*216)
        # lin_out_3 = lin_out_3.view(N,T,V,216)
        # # lin_out_3 = mamba_out.view(N,T,V,216) 

        # tcn_inp = lin_out_3.permute(0,3,1,2).float()
        # y = self.act(self.tcn1(tcn_inp).half() + self.residual(x))

        return y


class Model(nn.Module):
    def __init__(self, num_class=400, num_point=18, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, num_of_heads=9, joint_label=[], **kwargs):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  #3.18,18 # 3,25,25

        self.num_of_heads = num_of_heads
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.joint_label = joint_label


        self.l1 = TCN_ViT_unit(3, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=1)
        # * num_heads, effect of concatenation following the official implementation
        # self.l2 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=2)
        self.l3 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=3)
        # self.l4 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=4)
        self.l5 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, stride=2, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=5)
        # self.l6 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=6)
        self.l7 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=7)
        # self.l8 = TCN_ViT_unit(24 * num_of_heads, 24 * num_of_heads, A, num_of_heads=num_of_heads)
        self.l8 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, stride=2, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=8)
        # self.l9 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=9)
        self.l10 = TCN_ViT_unit(24*num_of_heads, 24*num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=10)
        # standard ce loss
        self.fc = nn.Linear(24*num_of_heads, num_class)
        
        # ## larger model
        # self.l1 = TCN_ViT_unit(3, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads, pe=True,
        #                        num_point=num_point, layer=1)
        # # * num_heads, effect of concatenation following the official implementation
        # self.l2 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                        pe=True, num_point=num_point, layer=2)
        # self.l3 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                        pe=True, num_point=num_point, layer=3)
        # self.l4 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                        pe=True, num_point=num_point, layer=4)
        # self.l5 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, stride=2,
        #                        num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=5)
        # self.l6 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                        pe=True, num_point=num_point, layer=6)
        # self.l7 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                        pe=True, num_point=num_point, layer=7)
        # # self.l8 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, num_of_heads=num_of_heads)
        # self.l8 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, stride=2,
        #                        num_of_heads=num_of_heads, pe=True, num_point=num_point, layer=8)
        # self.l9 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                        pe=True, num_point=num_point, layer=9)
        # self.l10 = TCN_ViT_unit(36 * num_of_heads, 36 * num_of_heads, A, residual=True, num_of_heads=num_of_heads,
        #                         pe=True, num_point=num_point, layer=10)
        # # standard ce loss
        # self.fc = nn.Linear(36 * num_of_heads, num_class)
        # self.z_prior = torch.empty(num_class, 24*num_of_heads)
        # nn.init.orthogonal_(self.z_prior, gain=3)
        # self.fc_mu = nn.Linear(24*num_of_heads, 24*num_of_heads)
        # self.fc_logvar = nn.Linear(24*num_of_heads, 24*num_of_heads)
        # self.cls_linear = nn.Linear(77,60,dtype=torch.float32).to('cuda') #bs 60
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        # self.transformer = Transformer(
        #     width=transformer_width,
        #     layers=transformer_layers,
        #     heads=transformer_heads,
        #     attn_mask=self.build_attention_mask(),
        #     dropout=dpr
        # )
        # self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        # self.ln_final = LayerNorm(transformer_width)
        
        # self.dropout = nn.Dropout(emb_dropout)
        # self.emb_dropout = emb_dropout
        
        # self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.positional_embedding, std=0.01)
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            # nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            # nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            # nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if self.text_projection is not None:
            # nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # def latent_sample(self, mu, logvar):
    #     if self.training:
    #         std = logvar.mul(0.5).exp()
    #         # std = logvar.exp()
    #         std = torch.clamp(std, max=100)
    #         # std = std / (torch.norm(std, 2, dim=1, keepdim=True) + 1e-4)
    #         eps = torch.empty_like(std).normal_()
    #         return eps.mul(std) + mu
    #     else:
    #         return mu

    def forward(self, x, y):
        groups = []
        for num in range(max(self.joint_label)+1):
            groups.append([ind for ind, element in enumerate(self.joint_label) if element==num])

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)

        ## n, c, t, v
        x = x.view(N, M, V, C, T).contiguous().view(N * M, V, C, T).permute(0, 2, 3, 1)

        x = self.l1(x, self.joint_label, groups)
        # x = self.l2(x, self.joint_label, groups)
        x = self.l3(x, self.joint_label, groups)
        # x = self.l4(x, self.joint_label, groups)
        x = self.l5(x, self.joint_label, groups)
        # x = self.l6(x, self.joint_label, groups)
        x = self.l7(x, self.joint_label, groups)
        x = self.l8(x, self.joint_label, groups)
        # x = self.l9(x, self.joint_label, groups)
        x = self.l10(x, self.joint_label, groups)

        # Added code for mamba 
        # x = x.repeat(1, int(self.fc.weight.shape[1]/x.shape[1]), 1, 1)

        # N*M, C, T, V
        _ , C, T, V = x.size()
        # spatial temporal average pooling
        x = x.contiguous().view(N, M, C, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        # z_mu = self.fc_mu(x)
        # z_logvar = self.fc_logvar(x)
        # z = self.latent_sample(z_mu, z_logvar)

        x = self.fc(x)
        # print(x.shape)
        # cls_emb = self.cls_linear(texts)
        # d_type = x.dtype
        # t = self.token_embedding(texts).type(d_type)  # [batch_size, n_ctx, d_model]

        # t = t + self.positional_embedding.type(d_type)
        # if self.emb_dropout > 0:
        #     t = self.dropout(x)
        # t = t.permute(1, 0, 2)  # NLD -> LND
        # t = self.transformer(t)
        # t = t.permute(1, 0, 2)  # LND -> NLD
        # t = self.ln_final(t).type(d_type)  # eg, [400 77 512]

        # text_token = t @ self.text_projection   # eg, [400 77 512]

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # t = t[torch.arange(t.shape[0]), texts.argmax(dim=-1)] @ self.text_projection   # 400 512  


        return x, y #z
