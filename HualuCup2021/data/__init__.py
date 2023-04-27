from data.icap_dataset import ImageCaptionDataset
from data.hlb_dataset import HLBCaptionInferenceDataset
from data.tokenizer import CNTokenizer

import pandas as pd
from config.cfg import CFG


def make_train_val_split_dataframe(filepath):
    import numpy as np
    np.random.seed(42)
    df = pd.read_csv(filepath)
    max_id = max(df['ids']) + 1 if not CFG.Debug else 200
    choices = np.arange(0, max_id)
    valid = np.random.choice(choices, int(0.1 * len(choices)), replace=False)
    train = [id_ for id_ in choices if id_ not in valid]

    valid_df = df[df['ids'].isin(valid)].reset_index(drop=True)
    train_df = df[df['ids'].isin(train)].reset_index(drop=True)

    return train_df, valid_df