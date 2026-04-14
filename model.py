import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width, gamma=10000):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model
        
        pe = torch.zeros(d_model, height, width)
        d_p = d_model // 4
        div_term = torch.exp(torch.arange(0, d_p, 2).float() * -(np.log(gamma) / d_p))
        
        pos_h = torch.arange(0, height).unsqueeze(1).float()
        pos_w = torch.arange(0, width).unsqueeze(1).float()
        
        pe[0:d_p:2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[1:d_p:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_p:d_p*2:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_p+1:d_p*2:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe.view(1, self.d_model, -1).transpose(1, 2)


class MSWSA(nn.Module):
    def __init__(self, d_model, num_heads, window_sizes):
        super(MSWSA, self).__init__()
        self.num_heads = num_heads
        self.window_sizes = window_sizes 
        self.d_head = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, num_heads, N, d_head]

        q = q.reshape(B, self.num_heads, H, W, self.d_head)
        k = k.reshape(B, self.num_heads, H, W, self.d_head)
        v = v.reshape(B, self.num_heads, H, W, self.d_head)

        out_heads = []
        for i, w in enumerate(self.window_sizes):

            qi = q[:, i:i+1] 
            ki = k[:, i:i+1]
            vi = v[:, i:i+1]
            
            attn = (qi @ ki.transpose(-2, -1)) * (self.d_head ** -0.5)
            attn = F.softmax(attn, dim=-1)
            out_i = (attn @ vi)
            out_heads.append(out_i)

        out = torch.cat(out_heads, dim=1).reshape(B, N, C)
        return self.proj(out)

class STIMI(nn.Module):
    def __init__(self, img_size=20, patch_size=1, embed_dim=512, 
                 enc_depth=12, dec_depth=8, num_heads=8, 
                 mask_ratio=0.75, window_sizes=[3, 5, 7, 9, 11, 13, 15, 17]):
        super().__init__()

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Linear(patch_size**2, embed_dim)
        self.pos_embed = PositionalEncoding2D(embed_dim, img_size, img_size)
        
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(enc_depth)
        ])

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        
        self.decoder = nn.ModuleList()
        for _ in range(dec_depth):
            layer = nn.ModuleDict({
                'mswsa': MSWSA(embed_dim, num_heads, window_sizes),
                'norm1': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                ),
                'norm2': nn.LayerNorm(embed_dim)
            })
            self.decoder.append(layer)

        self.reconstruct_head = nn.Linear(embed_dim, patch_size**2)

    def random_masking(self, x, mask_ratio):
        B, N, L = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, L))

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        
        x_visible, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        for layer in self.encoder:
            x_visible = layer(x_visible)

        mask_tokens = self.mask_token.repeat(x_visible.shape[0], ids_restore.shape[1] - x_visible.shape[1], 1)
        x_full = torch.cat([x_visible, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        x_full = self.decoder_embed(x_full)
        x_full = self.pos_embed(x_full)

        H = W = int(self.num_patches**0.5)
        for layer in self.decoder:
            x_full = x_full + layer['mswsa'](layer['norm1'](x_full), H, W)
            x_full = x_full + layer['mlp'](layer['norm2'](x_full))

        pred = self.reconstruct_head(x_full)
        return pred, mask

def stimi_loss(pred, target, mask, lambd=0.05):
    mse_loss = (pred - target) ** 2
    mse_loss = (mse_loss.mean(dim=-1) * mask).sum() / mask.sum()

    p = F.softmax(target, dim=-1)
    q = F.log_softmax(pred, dim=-1)
    kl_loss = F.kl_div(q, p, reduction='batchmean')

    return mse_loss + lambd * kl_loss

