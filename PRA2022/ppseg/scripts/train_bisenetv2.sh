python3 -m paddle.distributed.launch train.py --config configs/pra2022/bisenetv2.yml \
                                              --do_eval \
                                              --save_interval 400 \
                                              --precision fp16 \
                                              --amp_level O1
