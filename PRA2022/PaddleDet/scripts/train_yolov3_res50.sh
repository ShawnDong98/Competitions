export ROCR_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/yolov3/yolov3_res50.yml \
                       --eval \
                       --enable_ce \
                       --amp \
                       --classwise \
                       --resume output/yolov3_res50/12.pdparams