from pprint import pprint
from box import Box

config = {
    'debug': False,
    'name': 'Petfinder',
    'seed': 3407,
    'n_splits': 5,
    'epochs': 5,
    'root': "../datasets/kaggle/petfinder_clean",
    'image_size_tta': 440,
    'image_size': 384,
    'work_dir': './checkpoint',
    'log_level': 'INFO',
    'log_average_filter':['pred', 'label'],
    'log_config': {
        'interval': 10,
        'hooks' : [
            dict(name='PetfinderLoggerHook'),
            # dict(name='TextLoggerHook'),
        ]
    },
    'model': {
        'name': 'swin_large_patch4_window12_384',
        'feature_dim': 128,
        'output_dim': 1,
    },
    'train_loader': {
        'batch_size': 16,
        'num_workers': 10,
        'shuffle': True,
        'drop_last': True,
        'pin_memory': False,
    },
    'val_loader': {
        'batch_size': 16,
        'num_workers': 10,
        'shuffle': False,
        'drop_last': False,
        'pin_memory': False,
    },
    'optimizer': {
        'name': 'AdamW',
    },
    'optimizer_config': {
        'grad_clip': None
    },
    'lr_config': {
        'policy': 'OneCycle',
        'max_lr': 2e-5,
        'by_epoch': False,
    },
    'workflow': [('train', 1), ('val', 1)],
    'checkpoint_config': {
        'interval': 1,
    },
    'earlystopping_config': {
        'monitor': 'mse',
        'patience': 2,
        'mode': 'min',
    }
}


config = Box(config)
pprint(config)
