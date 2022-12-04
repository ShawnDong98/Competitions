export ROCR_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyoloe/ppyoloe_plus_crn_s.yml \
                       --eval \
                       --enable_ce \
                       --amp \
                       --classwise 