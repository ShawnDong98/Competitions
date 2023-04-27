import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.gpt import GPT

class ImageEncoder(nn.Module):
    def __init__(
        self,
        image_model,
        pretrained,
        trainable,
    ):
        super().__init__()
        self.model = timm.create_model(
            image_model,
            pretrained = pretrained,
            num_classes = 0,
        )

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, image):
        features = self.model(image)
        return features

class ProjectionHead(nn.Module):
    def __init__(
        self,
        features_dim,
        proj_dim,
        resid_pdrop
    ):
        super().__init__()
        self.features_dim = features_dim
        self.proj_dim = proj_dim

        self.projection = nn.Linear(self.features_dim, self.proj_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.proj_dim, self.proj_dim)
        self.ln = nn.LayerNorm(self.proj_dim)
        self.resid_drop = nn.Dropout(resid_pdrop)

    def forward(self, features):
        projected = self.projection(features)
        out = self.gelu(projected)
        out = projected + self.resid_drop(self.fc(out))
        out = self.ln(out)

        return out

def cross_entropy(preds, targets, reduce="None"):
    log_softmax = nn.LogSoftmax(-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduce == "mean":
        return loss.mean()
    if reduce == "None":
        return loss

class CLIP(nn.Module):
    def __init__(
        self,
        image_model,
        pretrained,
        trainable,
        vocab_size,
        embed_dim,
        n_head,
        block_size,
        num_layers,
        attn_pdrop,
        resid_pdrop,
        embd_pdrop,
        image_features_dim,
        text_features_dim,
        proj_dim
    ):
        super().__init__()
        self.ImageEncoder = ImageEncoder( 
            image_model = image_model, 
            pretrained = pretrained, 
            trainable = trainable,
        )

        self.TextEncoder = GPT( 
            vocab_size = vocab_size,
            embed_dim = embed_dim,
            n_head = n_head,
            block_size = block_size,
            num_layers = num_layers,
            attn_pdrop = attn_pdrop,
            resid_pdrop = resid_pdrop,
            embd_pdrop = embd_pdrop
        )

        self.ImageProjectionHead = ProjectionHead(
            features_dim = image_features_dim,
            proj_dim = proj_dim,
            resid_pdrop = resid_pdrop
        )

        self.TextProjectionHead = ProjectionHead(
            features_dim = text_features_dim,
            proj_dim = proj_dim,
            resid_pdrop = resid_pdrop
        )

    def forward(self, image, text, mode="clip", train_image_model=False):
        if mode == "clip":
            image_features = self.ImageEncoder(image)
            text_features = self.TextEncoder.get_trg_idx_hidden(text)

            image_embeddings = self.ImageProjectionHead(image_features)
            text_embeddings = self.TextProjectionHead(text_features)

            logits = text_embeddings @ image_embeddings.T
            text_similarity = text_embeddings @ text_embeddings.T
            image_similarity = image_embeddings @ image_embeddings.T
            targets = torch.softmax((text_similarity + image_similarity) / 2, dim = -1)

            text_loss = cross_entropy(logits, targets)
            image_loss = cross_entropy(logits.T, targets.T)

            loss = (text_loss + image_loss) / 2

            return loss.mean()

        if mode == "caption":
            loss, logits = self.train_caption(image, text, train_image_model)
            return loss, logits
            
    def train_caption(self, image, text, train_image_model=False):
        image_feature = self.ImageEncoder(image)
        image_embedding = self.ImageProjectionHead(image_feature)

        if train_image_model:
            loss, logits = self.TextEncoder.train_caption(image_embedding, text)
        else:
            loss, logits = self.TextEncoder.train_caption(image_embedding.detach(), text)

        return loss, logits

    @torch.no_grad()
    def caption_step(self, image, text):
        image_feature = self.ImageEncoder(image)
        image_embedding = self.ImageProjectionHead(image_feature)

        logits = self.TextEncoder.caption_step(text, image_embedding)

        return logits[:, -1, :]

    @torch.no_grad()
    def caption_image(self, image, tokenizer, max_length):
        image_feature = self.ImageEncoder(image)
        image_embedding = self.ImageProjectionHead(image_feature)

        caption_list = self.TextEncoder.caption_image(image_embedding, max_length)

        return tokenizer.decode(caption_list)