python3 -m paddle.distributed.launch train.py --config configs/pra2022/hardnet.yml \
                                              --save_interval 400 \
                                              --do_eval \
                                              --precision fp16 \
                                              --amp_level O1
