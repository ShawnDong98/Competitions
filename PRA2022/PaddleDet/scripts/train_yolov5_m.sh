export ROCR_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/yolov5/yolov5_m.yml \
                       --eval \
                       --enable_ce \
                       --amp