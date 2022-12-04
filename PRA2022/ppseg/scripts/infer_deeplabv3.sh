python predict.py \
       --config configs/pra2022/deeplabv3p_resnet50.yml \
       --model_path ../output/pp_liteseg/output/best_model/model.pdparams \
       --image_path ../datasets/extra/image/ \
       --save_dir output/result