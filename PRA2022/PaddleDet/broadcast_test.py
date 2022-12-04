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
    if paddle.distributed.ParallelEnv().local_rank == 0:
        np_data = np.array([[4, 5, 6], [4, 5, 6]])
    else:
        np_data = np.array([[1, 2, 3], [1, 2, 3]])
    data = paddle.to_tensor(np_data)
    paddle.distributed.broadcast(data, 1)
    out = data.numpy()