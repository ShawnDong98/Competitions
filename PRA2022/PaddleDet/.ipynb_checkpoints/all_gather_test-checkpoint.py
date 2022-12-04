import os
import random
import numpy as np
import paddle
from paddle.distributed import init_parallel_env
paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)

def init_parallel_env():
    env = os.environ
    dist = 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env
    if dist:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        local_seed = (99 + trainer_id)
        random.seed(local_seed)
        np.random.seed(local_seed)

    paddle.distributed.init_parallel_env()

def main():
    init_parallel_env()
    tensor_list = []
    if paddle.distributed.ParallelEnv().local_rank == 0:
        np_data1 = np.array([[4, 5, 6], [4, 5, 6]])
        np_data2 = np.array([[4, 5, 6], [4, 5, 6]])
        data1 = paddle.to_tensor(np_data1)
        data2 = paddle.to_tensor(np_data2)
        paddle.distributed.all_gather(tensor_list, data1, use_calc_stream=False)
    else:
        np_data1 = np.array([[1, 2, 3, 1], [1, 2, 3, 1]])
        np_data2 = np.array([[1, 2, 3, 1], [1, 2, 3, 1]])
        data1 = paddle.to_tensor(np_data1)
        data2 = paddle.to_tensor(np_data2)
        paddle.distributed.all_gather(tensor_list, data2, use_calc_stream=False)

    print(len(tensor_list))
    print(tensor_list)
    
if __name__ == "__main__":
    paddle.distributed.spawn(main, nprocs=4)