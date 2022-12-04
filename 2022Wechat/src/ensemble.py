import os
import pandas as pd
import numpy as np
from src2.category_id_map import lv2id_to_category_id


# logits1 = np.load("./data/result_1.npy")
logits2 = np.load("./data/result_2.npy")
# logits3 = np.load("./data/result_3.npy")

# res = logits1 + 0.8 * logits2 + 0.2 * logits3
res = 0.8 * logits2
predictions = res.argmax(1)
pred = [lv2id_to_category_id(p) for p in predictions]

test_df = pd.read_csv("./data/annotations/test_b.csv")
test_df = test_df.drop(columns=["Unnamed: 0", "title", "asr", "ocr"])
test_df['pred'] = pred
test_df = test_df.set_index('id')
print(test_df)
test_df.to_csv(os.path.join("./data/", 'result.csv'), header=None)