import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        attn_pdrop,
        resid_pdrop
    ):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        assert(self.head_dim * self.num_heads == self.hidden_dim), "The hidden_dim should be divisible by num_heads"

        self.query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_pdrop)

        self.fc_out = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.resid_drop = nn.Dropout(resid_pdrop)

    def forward(self, x):
        N, T, C = x.shape

        queries = self.query(x).view(N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = self.key(x).view(N, T, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        values = self.value(x).view(N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = queries @ keys / np.sqrt(self.hidden_dim)
        attn = torch.softmax(energy, -1)
        # N, H, query_len, key_len
        attn = self.attn_drop(attn)

        out = attn @ values
        out = self.fc_out(out.permute(0, 2, 1, 3).contiguous().view(N, T, self.hidden_dim))
        out = self.resid_drop(out)

        return out


class Block(nn.Module):
    def __init__(
        self,
        hidden_dim, 
        num_heads, 
        attn_pdrop,
        resid_pdrop,
    ):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(
            hidden_dim = hidden_dim,
            num_heads = num_heads,
            attn_pdrop = attn_pdrop,
            resid_pdrop = resid_pdrop
        )

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(resid_pdrop)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class ViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        hidden_dim,
        num_heads,
        num_layers,
        embd_pdrop,
        attn_pdrop,
        resid_pdrop
    ):
        super(ViT, self).__init__()
        self.path_embedding = PatchEmbed(
            img_size = img_size,
            patch_size = patch_size,
            embed_dim = hidden_dim,
        ) 
        num_patches = self.path_embedding.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.embd_drop = nn.Dropout(embd_pdrop)

        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden_dim = hidden_dim,
                    num_heads = num_heads,
                    attn_pdrop = attn_pdrop,
                    resid_pdrop = resid_pdrop,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        patch_embedding = self.path_embedding(x)
        cls_token = self.cls_token.expand(patch_embedding.shape[0], -1, -1)
        patch_embedding = torch.cat((cls_token, patch_embedding), dim=1)
        embedding =  self.embd_drop(patch_embedding + self.pos_embedding)

        last_hidden = embedding
        for block in self.blocks:
            last_hidden = block(last_hidden)

        return last_hidden, last_hidden[:, 0]


class ViTClsHead(nn.Module):
    def __init__(
        self, 
        img_size, 
        patch_size, 
        hidden_dim,
        num_heads,
        num_layers,
        embd_pdrop,
        attn_pdrop,
        resid_pdrop,
        classes=10,
    ):
        super(ViTClsHead, self).__init__()
        self.vit = ViT(
            img_size = img_size,
            patch_size = patch_size,
            hidden_dim = hidden_dim,
            num_heads = num_heads,
            num_layers = num_layers,
            embd_pdrop = embd_pdrop,
            attn_pdrop = attn_pdrop,
            resid_pdrop = resid_pdrop,
        )

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, classes)

    def forward(self, x, targets=None):
        last_hidden, cls_hidden = self.vit(x)
        logits = self.head(self.ln_f(cls_hidden))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

