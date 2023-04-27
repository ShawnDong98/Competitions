from train_clip import main as main_clip
from train_cap import main as main_cap
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIPCap')
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--clip_save_checkpoint_path', type=str, default="./checkpoints/AiO_clip_best_model_0923.pth.tar")
    parser.add_argument('--cap_save_checkpoint_path', type=str, default="./checkpoints/AiO_cap_best_model_0923.pth.tar")
    parser.add_argument('--debug', action="store_true", help="Debug.")
    args = parser.parse_args()
    
    main_clip(args)
    main_cap(args)