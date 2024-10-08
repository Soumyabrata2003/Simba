import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from torch.autograd import Variable
import numpy as np
import math
from scipy.stats import norm
import scipy
from collections import OrderedDict

# class GraphConvolution(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """

#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         # self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         self.weight = Parameter(torch.cuda.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.cuda.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input, adj):
#         # print(input.device,self.weight.device)
#         # self.weight = self.weight.to(input.device)
#         NT, V, C = input.size()
#         input_reshaped = input.view(NT*V, C)
    #     support = torch.mm(input_reshaped, self.weight)
    #     # support = support.view(NT,V, -1)
    #     # print(adj.device,support.device)
    #     adj = adj.to(support.device)
    #     output = torch.spmm(adj, support)
    #     print(output.shape)
    #     if self.bias is not None:
    #         # self.bias = self.bias.to(input.device)
    #         return output + self.bias
    #     else:
    #         return output

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #            + str(self.in_features) + ' -> ' \
    #            + str(self.out_features) + ')'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(nn.Module):
	def __init__(self, input_dim, output_dim, num_vetex, act=F.relu, dropout=0.5, bias=True):
		super(GraphConvolution, self).__init__()

		self.alpha = 1.

		self.act = act
		self.dropout = nn.Dropout(dropout)
		self.weight = nn.Parameter(torch.randn(input_dim, output_dim)).to(device)
		if bias:
			self.bias = nn.Parameter(torch.randn(output_dim)).to(device)
		else:
			self.bias = None

		for w in [self.weight]:
			nn.init.xavier_normal_(w)

	def normalize(self, m):
		rowsum = torch.sum(m, 0)
		r_inv = torch.pow(rowsum, -0.5)
		r_mat_inv = torch.diag(r_inv).double()
        
		m_norm = torch.mm(r_mat_inv, m)
		m_norm = torch.mm(m_norm, r_mat_inv)

		return m_norm

	def forward(self, adj, x):

		x = self.dropout(x)

		# K-ordered Chebyshev polynomial
		# print(adj.shape)
		# print(adj.dtype,x.dtype)
		adj_norm = self.normalize(adj)
		# print(adj_norm.shape)
		sqr_norm = self.normalize(torch.mm(adj,adj))
		m_norm = self.alpha*adj_norm.half() + (1.-self.alpha)*sqr_norm.half()
		x_tmp = torch.einsum('abcd,de->abce', x.half(), self.weight.half())
		# print(m_norm.device,x_tmp.device)
		# print(m_norm.dtype,x_tmp.dtype)
		m_norm = m_norm.to(device)
		x_out = torch.einsum('ij,abid->abjd', m_norm, x_tmp)
		if self.bias is not None:
			x_out += self.bias.half()

		x_out = self.act(x_out)
		
		return x_out
     
def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
	
class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        # if self.in_channels == 3:
        #     self.group = 25
        # else:
        #     self.group = 25

        # self.Linear_weight = nn.Parameter(torch.zeros((in_channels*25)//self.group ,(out_channels*25)//self.group,requires_grad=True,device='cuda'),requires_grad=True)
        # nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0 / ((out_channels*25)//self.group)))

        # self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        # nn.init.constant_(self.Linear_bias, 0)

        # self.Feature_Mask = nn.Parameter(torch.ones(1,25,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        # nn.init.constant_(self.Feature_Mask, 0)


        # self.bn = nn.BatchNorm1d(25*out_channels)
        # self.relu = nn.ReLU()
        if self.in_channels == 3:
            self.group = 20
        else:
            self.group = 20

        self.Linear_weight = nn.Parameter(torch.zeros((in_channels*20)//self.group ,(out_channels*20)//self.group,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0 / ((out_channels*20)//self.group)))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant_(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,20,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant_(self.Feature_Mask, 0)


        self.bn = nn.BatchNorm1d(20*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


        # index_array = np.empty(25*in_channels).astype(np.int64)
        # for i in range(25):
        #     for j in range(in_channels):
        #         index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*25)
        # self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        # index_array = np.empty(25*out_channels).astype(np.int64)
        # for i in range(25):
        #     for j in range(out_channels):
        #         index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*25)
        # self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        index_array = np.empty(20*in_channels).astype(np.int64)
        for i in range(20):
            for j in range(in_channels):
                index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*20)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = np.empty(20*out_channels).astype(np.int64)
        for i in range(20):
            for j in range(out_channels):
                index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*20)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0,2,3,1).contiguous() #[n,t,v,c]

        x = x.view(n*t,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n*t,v,c)

        x = x * (torch.tanh(self.Feature_Mask)+1)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c

        x = x + self.Linear_bias

        x = x.view(n*t,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,t,v,self.out_channels).permute(0,3,1,2) # n,c,t,v

        x = x + self.down(x0)
        x = self.relu(x)
        return x

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        self.num_subset = num_subset
        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [
                                      3, 1, num_point, num_point]), dtype=torch.float32, requires_grad=True).repeat(1, groups, 1, 1), requires_grad=True)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels * num_subset)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.Linear_weight = nn.Parameter(torch.zeros(
            in_channels, out_channels * num_subset, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * num_subset)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels * num_subset, 1, 1, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.constant_(self.Linear_bias, 1e-6)

        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(num_point))
        self.eyes = nn.Parameter(torch.tensor(torch.stack(
            eye_array), requires_grad=False, device='cuda'), requires_grad=False)  # [c,25,25]

    def norm(self, A):
        b, c, h, w = A.size()
        A = A.view(c, self.num_point, self.num_point)
        D_list = torch.sum(A, 1).view(c, 1, self.num_point)
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = self.eyes * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A

    def forward(self, x0):
        learn_A = self.DecoupleA.repeat(
            1, self.out_channels // self.groups, 1, 1)
        norm_learn_A = torch.cat([self.norm(learn_A[0:1, ...]), self.norm(
            learn_A[1:2, ...]), self.norm(learn_A[2:3, ...])], 0)

        x = torch.einsum(
            'nctw,cd->ndtw', (x0, self.Linear_weight)).contiguous()
        x = x + self.Linear_bias
        x = self.bn0(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum('nkctv,kcvw->nctw', (x, norm_learn_A))

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        return x

