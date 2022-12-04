# pretrain vlbert
python src/src2/pl_version/main_pretrain.py --label_csv ./data/annotations/label.csv \
                                       --train_dir ./data/zip_feats/unlabeled \
                                       --val_dir ./data/zip_feats/labeled \
                                       --test_dir ./data/zip_feats/test_b \
                                       --train_json ./data/annotations/unlabeled.json \
                                       --val_json ./data/annotations/labeled.json \
                                       --test_json ./data/annotations/test_b.json \
                                       --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                       --checkpoints_dir ./src/src2/pl_version/checkpoints_pretrained \
                                       

#****************************************************************

# fintune pl
python src/src2/pl_version/main_finetune.py --fold 0 \
                                       --label_csv ./data/annotations/label.csv \
                                       --train_dir ./data/zip_feats/labeled \
                                       --test_dir ./data/zip_feats/test_b \
                                       --train_json ./data/annotations/labeled.json \
                                       --test_json ./data/annotations/test_b.json \
                                       --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                       --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
                                       --checkpoints_dir ./src/src2/pl_version/checkpoints_finetune_fold0 \
                                       


python src/src2/pl_version/main_finetune.py --fold 1 \
                                       --label_csv ./data/annotations/label.csv \
                                       --train_dir ./data/zip_feats/labeled \
                                       --test_dir ./data/zip_feats/test_b \
                                       --train_json ./data/annotations/labeled.json \
                                       --test_json ./data/annotations/test_b.json \
                                       --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                       --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
                                       --checkpoints_dir ./src/src2/pl_version/checkpoints_finetune_fold1 \
                                       

python src/src2/pl_version/main_finetune.py --fold 2 \
                                       --label_csv ./data/annotations/label.csv \
                                       --train_dir ./data/zip_feats/labeled \
                                       --test_dir ./data/zip_feats/test_b \
                                       --train_json ./data/annotations/labeled.json \
                                       --test_json ./data/annotations/test_b.json \
                                       --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                       --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
                                       --checkpoints_dir ./src/src2/pl_version/checkpoints_finetune_fold2 \
                                       


python src/src2/pl_version/main_finetune.py --fold 3 \
                                       --label_csv ./data/annotations/label.csv \
                                       --train_dir ./data/zip_feats/labeled \
                                       --test_dir ./data/zip_feats/test_b \
                                       --train_json ./data/annotations/labeled.json \
                                       --test_json ./data/annotations/test_b.json \
                                       --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                       --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
                                       --checkpoints_dir ./src/src2/pl_version/checkpoints_finetune_fold3 \
                                       


# ***************************************************************
# finetune tez
python src/src2/tez_version/main.py --fold 0 \
                                       --label_csv ./data/annotations/label.csv \
                                       --train_dir ./data/zip_feats/labeled \
                                       --test_dir ./data/zip_feats/test_b \
                                       --train_json ./data/annotations/labeled.json \
                                       --test_json ./data/annotations/test_b.json \
                                       --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                       --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
                                       --checkpoints_dir ./src/src2/tez_version/checkpoints_finetune_fold0 \
                                       

python src/src2/tez_version/main.py --fold 1 \
                                       --label_csv ./data/annotations/label.csv \
                                       --train_dir ./data/zip_feats/labeled \
                                       --test_dir ./data/zip_feats/test_b \
                                       --train_json ./data/annotations/labeled.json \
                                       --test_json ./data/annotations/test_b.json \
                                       --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                       --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
                                       --checkpoints_dir ./src/src2/tez_version/checkpoints_finetune_fold1 \
                                       

python src/src2/tez_version/main.py --fold 2 \
                                       --label_csv ./data/annotations/label.csv \
                                       --train_dir ./data/zip_feats/labeled \
                                       --test_dir ./data/zip_feats/test_b \
                                       --train_json ./data/annotations/labeled.json \
                                       --test_json ./data/annotations/test_b.json \
                                       --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                       --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
                                       --checkpoints_dir ./src/src2/tez_version/checkpoints_finetune_fold2 \
                                       

python src/src2/tez_version/main.py --fold 3 \
                                       --label_csv ./data/annotations/label.csv \
                                       --train_dir ./data/zip_feats/labeled \
                                       --test_dir ./data/zip_feats/test_b \
                                       --train_json ./data/annotations/labeled.json \
                                       --test_json ./data/annotations/test_b.json \
                                       --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                       --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
                                       --checkpoints_dir ./src/src2/tez_version/checkpoints_finetune_fold3 \
                                       



# ***************************************************************

# pretrain lxmert
python src/src2/lxmert/src/pretrain/lxmert_pretrain.py --train unlabeled,labeled \
                                                  --batch_size 64 \
                                                  --valid test_b \
                                                  --valid_batch_size 128 \
                                                  --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                                  --output ./src/src2/lxmert/checkpoints_pretrained \
                                                  --root_path ./data/annotations/ \
                                                  --tiny False \
                                                  --fast False \
                                                  


# finetune lxmert

python src/src2/lxmert/src/tasks/wc.py --fold 0 \
                                       --label_csv ./data/annotations/label.csv \
                                       --train_dir ./data/zip_feats/labeled \
                                       --test_dir ./data/zip_feats/test_b \
                                       --train_json ./data/annotations/labeled.json \
                                       --test_json ./data/annotations/test_b.json \
                                       --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                       --pretrained_path ./src/src2/lxmert/checkpoints_pretrained/BEST_EVAL_LOSS \
                                       --checkpoints_dir ./src/src2/lxmert/checkpoints_finetune_fold0 \
                                       --mode train \
                                       

python src/src2/lxmert/src/tasks/wc.py --fold 1 \
                                       --label_csv ./data/annotations/label.csv \
                                       --train_dir ./data/zip_feats/labeled \
                                       --test_dir ./data/zip_feats/test_b \
                                       --train_json ./data/annotations/labeled.json \
                                       --test_json ./data/annotations/test_b.json \
                                       --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                       --pretrained_path ./src/src2/lxmert/checkpoints_pretrained/BEST_EVAL_LOSS \
                                       --checkpoints_dir ./src/src2/lxmert/checkpoints_finetune_fold1 \
                                       --mode train \
                                       

python src/src2/lxmert/src/tasks/wc.py --fold 2 \
                                       --label_csv ./data/annotations/label.csv \
                                       --train_dir ./data/zip_feats/labeled \
                                       --test_dir ./data/zip_feats/test_b \
                                       --train_json ./data/annotations/labeled.json \
                                       --test_json ./data/annotations/test_b.json \
                                       --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                       --pretrained_path ./src/src2/lxmert/checkpoints_pretrained/BEST_EVAL_LOSS \
                                       --checkpoints_dir ./src/src2/lxmert/checkpoints_finetune_fold2 \
                                       --mode train \
                                       

python src/src2/lxmert/src/tasks/wc.py --fold 3 \
                                       --label_csv ./data/annotations/label.csv \
                                       --train_dir ./data/zip_feats/labeled \
                                       --test_dir ./data/zip_feats/test_b \
                                       --train_json ./data/annotations/labeled.json \
                                       --test_json ./data/annotations/test_b.json \
                                       --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
                                       --pretrained_path ./src/src2/lxmert/checkpoints_pretrained/BEST_EVAL_LOSS \
                                       --checkpoints_dir ./src/src2/lxmert/checkpoints_finetune_fold3 \
                                       --mode train \
                                       
