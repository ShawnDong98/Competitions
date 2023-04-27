import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config.cfg import CFG


class CasualSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head, 
        block_size,
        attn_pdrop,
        resid_pdrop
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = self.embed_dim // self.n_head
        assert (self.head_dim * self.n_head == self.embed_dim), "embed_dim should be divisable by n_head"

        self.query = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.key = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

    def forward(self, x):
        N, T, C = x.shape

        queries = self.query(x).view(N, T, self.n_head, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(N, T, self.n_head, self.head_dim).transpose(1, 2)
        values = self.value(x).view(N, T, self.n_head, self.head_dim).transpose(1, 2)

        energy = queries @ keys.transpose(-1, -2)
        energy = energy / (self.embed_dim ** (1 / 2))
        energy = energy.masked_fill(self.mask[:, :, :T, :T]==0, -float("inf"))
        attn = torch.softmax(energy, -1)
        attn = self.attn_drop(attn)

        out = attn @ values
        out = self.fc_out(out.transpose(1, 2).contiguous().view(N, T, self.embed_dim))
        out = self.resid_drop(out)

        return out


class Block(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_head,
        block_size,
        attn_pdrop,
        resid_pdrop,

    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head

        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.attn = CasualSelfAttention(
            embed_dim = embed_dim,
            n_head = n_head,
            block_size = block_size,
            attn_pdrop = attn_pdrop,
            resid_pdrop = resid_pdrop,
        )

        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.GELU(),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
            nn.Dropout(resid_pdrop)
        )


    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x



class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        n_head,
        block_size,
        num_layers,
        attn_pdrop,
        resid_pdrop,
        embd_pdrop,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.block_size = block_size
        self.num_layers = num_layers
        
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        self.embd_drop = nn.Dropout(embd_pdrop)

        self.blocks = nn.Sequential(
            *[
                Block( 
                    embed_dim = self.embed_dim, 
                    n_head = self.n_head, 
                    block_size = self.block_size, 
                    attn_pdrop = attn_pdrop, 
                    resid_pdrop = resid_pdrop,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, x, targets=None):
        N, T = x.shape

        word_embedding = self.word_embedding(x)
        pos_embedding = self.pos_embedding[:, :T, :]
        embedding = self.embd_drop(word_embedding + pos_embedding)

        last_hidden = self.blocks(embedding)

        logits = self.head(self.ln_f(last_hidden))

        loss = None 


        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.contiguous().view(-1))
        return loss, logits


    def get_trg_idx_hidden(self, text):
        N, T = text.shape

        word_embedding = self.word_embedding(text)
        pos_embedding = self.pos_embedding[:, :T, :]
        embedding = self.embd_drop(word_embedding + pos_embedding)

        last_hidden = self.blocks(embedding)

        trg_token_idx = torch.nonzero(torch.where(text==3, torch.tensor(3).to(text.device), torch.tensor(0).to(text.device)))[:, 1]

        trg_hideen_state = torch.stack([last_hidden[i, idx, :] for i, idx in enumerate(trg_token_idx)])
        
        return trg_hideen_state



    def train_caption(self, image_embedding, text):
        inputs, targets = text[:, :-1], text
        N, T = inputs.shape

        word_embedding = self.word_embedding(inputs)
        pos_embedding = self.pos_embedding[:, :T, :]
        embedding = self.embd_drop(word_embedding + pos_embedding)

        input_embeddings = torch.cat((image_embedding.unsqueeze(1),embedding), dim=1)

        last_hidden = self.blocks(input_embeddings)

        logits = self.head(self.ln_f(last_hidden))

        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return loss, logits

    def caption_step(self, text, image_embedding):
        N, T = text.shape
        word_embedding = self.word_embedding(text)
        pos_embedding = self.pos_embedding[:, :T, :]
        text_embedding = self.embd_drop(word_embedding + pos_embedding)
        embedding = torch.cat((image_embedding.unsqueeze(1),text_embedding), dim=1)

        last_hidden = self.blocks(embedding)
        logits = self.head(self.ln_f(last_hidden))

        return logits


    def caption_image(self, image_embedding, max_length):
        text = torch.tensor([2]).unsqueeze(0).to(image_embedding.device)

        for step in range(max_length-1):
            N, T = text.shape
            word_embedding = self.word_embedding(text)
            pos_embedding = self.pos_embedding[:, :T, :]
            text_embedding = self.embd_drop(word_embedding + pos_embedding)
            embedding = torch.cat((image_embedding.unsqueeze(1),text_embedding), dim=1)

            last_hidden = self.blocks(embedding)
            logits = self.head(self.ln_f(last_hidden))

            token = logits.argmax(-1)[:, -1]
            text = torch.cat((text, token.unsqueeze(0)), dim=1)
            # 如果新生成的token为 [SEP]
            if token == 3:
                return text.cpu().numpy().squeeze()
        return text.cpu().numpy().squeeze()
