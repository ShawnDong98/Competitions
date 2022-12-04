import os
import sys
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
print(sys.path)
from src2.pl_version.category_id_map import category_id_to_lv2id
from tqdm import tqdm



if not os.path.exists("./data/zip_feats/labeled"):
    os.mkdir("./data/zip_feats/labeled")
    os.system("unzip ./data/zip_feats/labeled.zip -d ./data/zip_feats/labeled")

if not os.path.exists("./data/zip_feats/test_b"):
    os.mkdir("./data/zip_feats/test_b")
    os.system("unzip ./data/zip_feats/test_b.zip -d ./data/zip_feats/test_b")

if not os.path.exists("./data/zip_feats/unlabeled"):
    os.mkdir("./data/zip_feats/unlabeled")
    os.system("unzip ./data/zip_feats/unlabeled.zip -d ./data/zip_feats/unlabeled")


# 制作 labeled csv
root_dir = "./data/annotations"

count_null = 0
with open(os.path.join(root_dir, "labeled.json"), "r") as f:
    labeled_data = json.load(f)

cls_1 = []
cls_2 = []
for data in tqdm(labeled_data):
    if len(data['title']) == 0: count_null += 1
    cls_1.append(int(data['category_id'][:2]))
    cls_2.append(int(data['category_id'][2:]))

df = pd.DataFrame(labeled_data)

df["cls_1"] = cls_1

df['fold'] = -1


N_FOLDS = 4
strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=999, shuffle=True)
for i, (_, train_index) in enumerate(strat_kfold.split(df.index, df["cls_1"])):
    df.iloc[train_index, -1] = i
    
df['fold'] = df['fold'].astype('int')

df.to_csv(os.path.join(root_dir, "label.csv"))


# 制作 test_a csv
root_dir = "./data/annotations"

count_null = 0
with open(os.path.join(root_dir, "test_a.json"), "r") as f:
    test_a_data = json.load(f)

# pprint(unlabeled_data)
for data in tqdm(test_a_data):
    if len(data['title']) == 0: count_null += 1
    
print(count_null) # 14619

df = pd.DataFrame(test_a_data)
df.to_csv(os.path.join(root_dir, "test_a.csv"))

# 制作 test_b csv
root_dir = "./data/annotations"

count_null = 0
with open(os.path.join(root_dir, "test_b.json"), "r") as f:
    test_a_data = json.load(f)

# pprint(unlabeled_data)
for data in tqdm(test_a_data):
    if len(data['title']) == 0: count_null += 1
    
print(count_null) # 14619

df = pd.DataFrame(test_a_data)
df.to_csv(os.path.join(root_dir, "test_b.csv"))