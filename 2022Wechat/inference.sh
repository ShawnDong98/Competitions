# # evaluate pl
# python src/src2/pl_version/main_finetune.py --fold 0 \
#                                        --mode evaluate \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/pl_version/checkpoints_finetune_fold0

# python src/src2/pl_version/main_finetune.py --fold 1 \
#                                        --mode evaluate \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/pl_version/checkpoints_finetune_fold1

# python src/src2/pl_version/main_finetune.py --fold 2 \
#                                        --mode evaluate \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/pl_version/checkpoints_finetune_fold2


# python src/src2/pl_version/main_finetune.py --fold 3 \
#                                        --mode evaluate \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/pl_version/checkpoints_finetune_fold3

# # inference pl

# python src/src2/pl_version/main_finetune.py --fold 0 \
#                                        --mode inference \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/pl_version/checkpoints_finetune_fold0

# python src/src2/pl_version/main_finetune.py --fold 1 \
#                                        --mode inference \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/pl_version/checkpoints_finetune_fold1


# python src/src2/pl_version/main_finetune.py --fold 2 \
#                                        --mode inference \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/pl_version/checkpoints_finetune_fold2

# python src/src2/pl_version/main_finetune.py --fold 3 \
#                                        --mode inference \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/pl_version/checkpoints_finetune_fold3


# # ***************************************************************

# # evaluate tez

# python src/src2/pl_version/main_finetune.py --fold 0 \
#                                        --mode evaluate \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/tez_version/checkpoints_finetune_fold0

# python src/src2/pl_version/main_finetune.py --fold 1 \
#                                        --mode evaluate \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/tez_version/checkpoints_finetune_fold1


# python src/src2/pl_version/main_finetune.py --fold 2 \
#                                        --mode evaluate \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/tez_version/checkpoints_finetune_fold2


# python src/src2/pl_version/main_finetune.py --fold 3 \
#                                        --mode evaluate \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/tez_version/checkpoints_finetune_fold3

# # inference tez
# python src/src2/pl_version/main_finetune.py --fold 0 \
#                                        --mode inference \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/tez_version/checkpoints_finetune_fold0

# python src/src2/pl_version/main_finetune.py --fold 1 \
#                                        --mode inference \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/tez_version/checkpoints_finetune_fold1


# python src/src2/pl_version/main_finetune.py --fold 2 \
#                                        --mode inference \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/tez_version/checkpoints_finetune_fold2


# python src/src2/pl_version/main_finetune.py --fold 3 \
#                                        --mode inference \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/pl_version/checkpoints_pretrained/best_pth.ckpt \
#                                        --checkpoints_dir ./src/src2/tez_version/checkpoints_finetune_fold3
# # ****************************************************************

# # evaluate lxmert

# python src/src2/lxmert/src/tasks/wc.py --fold 0 \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/lxmert/checkpoints_pretrained/BEST_EVAL_LOSS \
#                                        --checkpoints_dir ./src/src2/lxmert/checkpoints_finetune_fold0 \
#                                        --mode evaluate

# python src/src2/lxmert/src/tasks/wc.py --fold 1 \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/lxmert/checkpoints_pretrained/BEST_EVAL_LOSS \
#                                        --checkpoints_dir ./src/src2/lxmert/checkpoints_finetune_fold1 \
#                                        --mode evaluate

# python src/src2/lxmert/src/tasks/wc.py --fold 2 \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/lxmert/checkpoints_pretrained/BEST_EVAL_LOSS \
#                                        --checkpoints_dir ./src/src2/lxmert/checkpoints_finetune_fold2 \
#                                        --mode evaluate


# python src/src2/lxmert/src/tasks/wc.py --fold 3 \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/lxmert/checkpoints_pretrained/BEST_EVAL_LOSS \
#                                        --checkpoints_dir ./src/src2/lxmert/checkpoints_finetune_fold3 \
#                                        --mode evaluate

# # inference lxmert 

# python src/src2/lxmert/src/tasks/wc.py --fold 0 \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/lxmert/checkpoints_pretrained/BEST_EVAL_LOSS \
#                                        --checkpoints_dir ./src/src2/lxmert/checkpoints_finetune_fold0 \
#                                        --mode inference

# python src/src2/lxmert/src/tasks/wc.py --fold 1 \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/lxmert/checkpoints_pretrained/BEST_EVAL_LOSS \
#                                        --checkpoints_dir ./src/src2/lxmert/checkpoints_finetune_fold1 \
#                                        --mode inference

# python src/src2/lxmert/src/tasks/wc.py --fold 2 \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/lxmert/checkpoints_pretrained/BEST_EVAL_LOSS \
#                                        --checkpoints_dir ./src/src2/lxmert/checkpoints_finetune_fold2 \
#                                        --mode inference


# python src/src2/lxmert/src/tasks/wc.py --fold 3 \
#                                        --label_csv ./data/annotations/label.csv \
#                                        --train_dir ./data/zip_feats/labeled \
#                                        --test_dir ./data/zip_feats/test_b \
#                                        --train_json ./data/annotations/labeled.json \
#                                        --test_json ./data/annotations/test_b.json \
#                                        --model_path ./src/src2/input/pretrain-model/chinese-macbert-base \
#                                        --pretrained_path ./src/src2/lxmert/checkpoints_pretrained/BEST_EVAL_LOSS \
#                                        --checkpoints_dir ./src/src2/lxmert/checkpoints_finetune_fold3 \
#                                        --mode inference

# # ensemble
# python src/src2/ensemble.py

python src/ensemble.py