import torch
from torch import nn
import torch.nn.functional as F

from model.modeling import BertOnlyMLMHead, VisualOnlyMLMHead, UniBert, UniBertForMaskedLM
from transformers.models.bert.modeling_bert import BertConfig



class WeChatModel(nn.Module):
    def __init__(
        self,
        model_path,
        num_classes,
        frame_embedding_size,
        vlad_cluster_size,
        vlad_hidden_size,
        fc_size,
        dropout,
        se_ratio,
        mean_pooling = True,
        max_pooling = True,
        median_pooling = True,
        tasks = ['mlm', 'mfm', 'cls']
    ):
        super().__init__()
        print(tasks)
        self.mean_pooling = mean_pooling
        self.max_pooling = max_pooling
        self.median_pooling = median_pooling
        self.tasks = tasks

        uni_bert_cfg = BertConfig.from_pretrained(f'{model_path}/config.json')

        self.bert = UniBertForMaskedLM.from_pretrained(
            model_path, 
            config=uni_bert_cfg
        )

        if 'cls' in self.tasks:
            bert_output_size = 768

            self.nextvlad = NeXtVLAD(
                frame_embedding_size, 
                vlad_cluster_size,
                output_size=vlad_hidden_size, 
                dropout=dropout
            )
            self.enhance = SENet(channels=vlad_hidden_size, ratio=se_ratio)

            output_features_cnt = 0
            if self.mean_pooling:
                output_features_cnt += 1
            if self.max_pooling:
                output_features_cnt += 1
            if self.median_pooling:
                output_features_cnt += 1
                
            self.fusion = ConcatDenseSE(vlad_hidden_size + bert_output_size * output_features_cnt, fc_size, se_ratio, dropout)
            self.classifier = nn.Linear(fc_size, num_classes)
        
        if 'mfm' in self.tasks:
            self.bert_mvm_lm_header = VisualOnlyMLMHead(
                uni_bert_cfg,
                frame_embedding_size=768
            ) 
        
        if 'mlm' in self.tasks:
            self.vocab_size = uni_bert_cfg.vocab_size

    def forward(self, 
                text_input_ids, 
                text_mask, 
                video_feature, 
                video_mask, 
                tasks = ['cls']
        ):
        return_mlm = False
        if 'mlm' in tasks: return_mlm = True
        
        features, mask, lm_prediction_scores = self.bert(video_feature, video_mask, text_input_ids, text_mask, return_mlm=return_mlm)

        outputs = {}
        if 'mlm' in tasks:
            pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            outputs["mlm"] = pred

        if 'mfm' in tasks:
            vm_output = self.bert_mvm_lm_header(features[:, 1:video_feature.size()[1] + 1, :])
            outputs["mfm"] = vm_output

        if 'cls' in tasks:

            vision_embedding = self.nextvlad(video_feature, video_mask)
            vision_embedding = self.enhance(vision_embedding)


            features_list = [vision_embedding]
            if self.mean_pooling: 
                features_mean = (features*mask.unsqueeze(-1)).sum(1)/mask.sum(1).unsqueeze(-1)
                features_mean = features_mean.float()
                features_list.append(features_mean)
            if self.max_pooling:
                features_max = features+(1-mask).unsqueeze(-1)*(-1e10)
                features_max = features_max.max(1)[0].float()
                features_list.append(features_max)
            if self.median_pooling:
                features_median, _ = torch.median(features, 1)
                features_list.append(features_median)
            
            final_features = self.fusion(features_list)
            logits = self.classifier(final_features)
            outputs['cls'] = logits
            

        return outputs

    # calc mfm loss 
    def calculate_mfm_loss(self, video_feature_output, video_feature_input, 
                           video_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
            video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

        afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])

        video_tr = video_feature_input.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss





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
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
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
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name and 'video_embeddings' not in name:
                self.backup[name] = param.data.clone()
                # print(f"{name}: {param.grad}")
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.bert.named_parameters():
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
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name and 'video_embeddings' not in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.bert.named_parameters():
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
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]