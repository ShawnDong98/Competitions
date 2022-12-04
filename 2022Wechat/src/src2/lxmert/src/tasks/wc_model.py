# coding=utf-8
# Copyleft 2019 project LXRT.
import torch
import torch.nn as nn
import torch.nn.functional as F

from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU



class WCModel(nn.Module):
    def __init__(self,
                config,
    ):
        super().__init__()
        self.config = config
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args = self.config,
        )

        self.embedding_attn = EmbeddingAttention(
            hidden_size=self.lxrt_encoder.dim,
        )
        
        self.logit_fc = nn.Sequential(
            nn.Linear(self.lxrt_encoder.dim, self.lxrt_encoder.dim * 2),
            GeLU(),
            BertLayerNorm(self.lxrt_encoder.dim * 2, eps=1e-12),
            nn.Linear(self.lxrt_encoder.dim * 2, self.config.num_classes)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, 
                input_ids,
                input_masks,
                segment_ids,
                feat, 
                visual_attention_mask
    ):
        pos = torch.arange(32, dtype=torch.long).to(input_ids.device)
        x = self.lxrt_encoder(
            input_ids, 
            input_masks,
            segment_ids,
            (feat, pos),
            visual_attention_mask
        )


        x = self.embedding_attn(x)

        logit = self.logit_fc(x)

        return logit


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad



class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = self.fusion_dropout(inputs)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, 
                out_size,
                linear_layer_size = [1024,512],
                hidden_dropout_prob = 0.2,
                num_label = 200,
    ):
        super().__init__()
        self.norm= nn.BatchNorm1d(out_size)
        self.dense = nn.Linear(out_size, linear_layer_size[0])
        self.norm_1= nn.BatchNorm1d(linear_layer_size[0])
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dense_1 = nn.Linear(linear_layer_size[0], linear_layer_size[1])  
        self.norm_2= nn.BatchNorm1d(linear_layer_size[1])
        self.out_proj = nn.Linear(linear_layer_size[1], num_label)

    def forward(self, features, **kwargs):
        x = self.norm(features)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(self.norm_1(x))
        x = self.dropout(x)
        x = self.dense_1(x)
        x = torch.relu(self.norm_2(x))
        x = self.dropout(x)        
        x = self.out_proj(x)
        return x


class FGM:
    def __init__(self, model, epsilon, emb_name):
        self.model = model
        self.backup = {}
        self.emb_name = emb_name
        self.epsilon = epsilon

    def attack(self):
        for name, param in self.model.lxrt_encoder.named_parameters():
            if param.requires_grad and self.emb_name in name and 'video_embeddings' not in name:
                self.backup[name] = param.data.clone()
                # print(f"{name}: {param.grad}")
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.lxrt_encoder.named_parameters():
            if param.requires_grad and self.emb_name in name and 'video_embeddings' not in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

  

class PGD:
    def __init__(self, 
                 model,
                 epsilon, 
                 emb_name, 
                 alpha        
    ):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.alpha = alpha

    def attack(self, is_first_attack=False):
        for name, param in self.model.lxrt_encoder.named_parameters():
            if param.requires_grad and self.emb_name in name and 'video_embeddings' not in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.lxrt_encoder.named_parameters():
            if param.requires_grad and self.emb_name in name and 'video_embeddings' not in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.lxrt_encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.lxrt_encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class EmbeddingAttention(nn.Module):
    def __init__(self, hidden_size):
        super(EmbeddingAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def get_attn(self, inputs, mask = None):
        inputs = torch.unsqueeze(inputs, 1)
        attn_scores = self.attn(inputs).squeeze(2)
        if mask is not None:
            attn_scores = mask * attn_scores
        attn_weights = attn_scores.unsqueeze(2)
        attn_out = torch.sum(inputs * attn_weights, dim = 1)

        return attn_out

    def forward(self, inputs, mask=None):
        attn_out = self.get_attn(inputs, mask)
        return attn_out