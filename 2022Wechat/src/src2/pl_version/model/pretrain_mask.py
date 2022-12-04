import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataset.tokenization import BertTokenizer


class MaskLM(object):
    def __init__(self, 
                tokenizer_path='bert-base-chinese',
                mlm_probability=0.15
    ):
        self.mlm_probability = mlm_probability
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_path,
            do_lower_case=True
        )

    def torch_mask_tokens(self, 
                         inputs: Any, 
                         special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone() # batch, seq_len


        probability_matrix = torch.full(labels.shape, self.mlm_probability) # batch, seq_len: value = 0.15

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val ,already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0) # 如果是 special_tokens 被换掉的概率为 0
        
        # Draws binary random numbers (0 or 1) from a Bernoulli distribution. 
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100 #  We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(["[MASK]"])[0]

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels




class MaskVideo(object):
    def __init__(self, mfm_probability=0.15):
        self.mfm_probability = mfm_probability

    def torch_mask_frames(self, video_feature, video_mask):
        probability_matrix = torch.full(video_mask.shape, 0.9 * self.mfm_probability) # batch, frame_len: value=0.15
        # mask 的地方不会再被 mask
        probability_matrix = probability_matrix * video_mask
        # 根据概率矩阵进行伯努利采样
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # ?
        video_labels_index = torch.arange(video_feature.size(0) * video_feature.size(1)).view(-1, video_feature.size(1))
        video_labels_index = -100 * ~masked_indices + video_labels_index * masked_indices

        # 将 mask 矩阵广播为 video_feature 形状
        masked_indices_unsqueeze = masked_indices.unsqueeze(-1).expand_as(video_feature)
        # 将要mask的地方填0
        inputs = video_feature.data.masked_fill(masked_indices_unsqueeze, 0.0)
        # 
        labels = video_feature[masked_indices_unsqueeze].contiguous().view(-1, video_feature.size(2))

        return inputs, video_labels_index