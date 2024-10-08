from mamba_ssm import Mamba
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import wandb
from einops import rearrange
from einops.layers.torch import Rearrange
import torchmetrics
import torchvision
import torchvision.transforms as transforms
# from generic_transformer.transformer_blocks import Tower
from mamba_ssm.ops.triton.layernorm import RMSNorm
# from performer_pytorch.performer_pytorch import FixedPositionalEmbedding

class MambaBlock(nn.Module):
    def __init__(self, embed_dim, dropout_level=0):
        super().__init__()

        self.mamba =  Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        self.norm = RMSNorm(embed_dim) #nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_level)

    def forward(self, x):
        x = self.norm(self.mamba(x) + x)
        return self.dropout(x)


class MambaTower(nn.Module):
    def __init__(self, embed_dim, n_layers, seq_len=None, global_pool=False, dropout=0 , use_pos_embeddings=True,rotary_position_emb=False):
        super().__init__()
        self.blocks = nn.Sequential(*[MambaBlock(embed_dim, dropout_level=dropout) for _ in range(n_layers)])
        self.global_pool = global_pool #for classification or other supervised learning.
        self.use_pos_embeddings = use_pos_embeddings
        self.rotary_position_emb = rotary_position_emb

        if rotary_position_emb:
            self.token_emb = nn.Embedding(seq_len, embed_dim)
            self.pos_emb = FixedPositionalEmbedding(embed_dim, seq_len)
            # self.layer_pos_emb = FixedPositionalEmbedding(dim_head, seq_len)

        if use_pos_embeddings:
            #simple fixed learned positional encodings for now:
            self.pos_embed = nn.Embedding(seq_len, embed_dim)
            self.register_buffer('precomputed_pos_enc', torch.arange(0, seq_len).long())


    def forward(self, x):
        #for input (bs, n, d) it returns either (bs, n, d) or (bs, d) is global_pool
        # out = self.blocks(x) if not self.global_pool else torch.mean(self.blocks(x),1)
        # return out
        if self.use_pos_embeddings:
            pos_enc = self.precomputed_pos_enc[:x.size(1)].expand(x.size(0), -1)
            out = self.blocks(x+self.pos_embed(pos_enc))
        elif self.rotary_position_emb:
            # token and positional embeddings
            # x = x.long()
            x = self.token_emb(x.int())
            x += self.pos_emb(x)
            out = self.blocks(x)
        else:
            out = self.blocks(x)

        if self.global_pool:
            return torch.mean(out, dim=1)
        else:
            return out
        
class ImgClassifier(nn.Module):
    def __init__(self, patch_size=16, img_size=224, n_channels=3, embed_dim=768, joint=False, n_layers=12, dropout=0):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size         
        self.n_channels = 3
        self.embed_dim = embed_dim       #Replace transformer width with embed_dim in mamba for positional encodings
        self.dropout = dropout
        self.n_layers = n_layers
        seq_len = int((self.img_size/self.patch_size)*((self.img_size/self.patch_size)))
        patch_dim = self.n_channels*self.patch_size*self.patch_size

        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                                   p1=self.patch_size, p2=self.patch_size)
        
        # scale = embed_dim ** -0.5
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        # self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        # self.positional_embedding = nn.Parameter(scale * torch.randn((img_size // patch_size) ** 2 + 1, embed_dim))
        # self.joint = joint
        # if joint:
        #     print('=====using space-time attention====')
        #     self.T = T
        #     self.time_embedding = nn.Parameter(scale * torch.randn(T, embed_dim))  # pos emb
        
        # self.visual = VisualTransformer(
        #         input_resolution=img_size,
        #         patch_size=patch_size,
        #         width=embed_dim,
        #         layers=n_layers,
        #         heads=embed_dim//64,
        #         output_dim=embed_dim,
        #     )

        self.func = nn.Sequential(self.rearrange,
                                  nn.LayerNorm(patch_dim),
                                  nn.Linear(patch_dim, embed_dim),
                                  nn.LayerNorm(embed_dim),
                                  Tower(embed_dim=embed_dim, n_layers=n_layers, seq_len=seq_len, global_pool=True, dropout=dropout,
                                        use_pos_embeddings=True)
                                #   MambaTower(embed_dim=embed_dim, n_layers=n_layers, seq_len=seq_len, global_pool=True, dropout=dropout, use_pos_embeddings=False,
                                            #  rotary_position_emb=False)
                                  )#   nn.Linear(embed_dim, 100))
        # self.func = nn.Sequential(self.visual, nn.Linear(embed_dim,100))

    def forward(self, x):
        # x = self.conv1(x)  # shape = [*, width, grid, grid]
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # x = x + self.positional_embedding.to(x.dtype)

        # if self.joint:
        #     from einops import rearrange
        #     B = x.shape[0] // self.T
        #     cls_tokens = x[:B, 0, :].unsqueeze(1)  # only one cls_token
        #     x = x[:,1:]
        #     x = rearrange(x, '(b t) n c -> (b n) t c',b=B,t=self.T)
        #     x = x + self.time_embedding.to(x.dtype)   # temporal pos emb
        #     x = rearrange(x, '(b n) t c -> b (n t) c',b=B,t=self.T)
        #     x = torch.cat((cls_tokens, x), dim=1)

        # x= self.func(x)

        return self.func(x)
