from pprint import pprint
from box import Box

config = {
    'debug': False,
    'name': 'Petfinder',
    'seed': 3407,
    'n_splits': 10,
    'epochs': 5,
    'root': "../datasets/kaggle/petfinder",
    'image_size_tta': 256,
    'image_size': 224,
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
        'name': 'swin_large_patch4_window7_224',
        'output_dim': 1,
    },
    'train_loader': {
        'batch_size': 80,
        'num_workers': 10,
        'shuffle': True,
        'drop_last': True,
        'pin_memory': False,
    },
    'val_loader': {
        'batch_size': 80,
        'num_workers': 10,
        'shuffle': False,
        'drop_last': False,
        'pin_memory': False,
    },
    'optimizer': {
        'name': 'AdamW',
        'lr': 3e-5,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.01,
    },
    'optimizer_config': {
        'grad_clip': None
    },
    'momentum_config': {
        'policy': 'OneCycle',
        'by_epoch': False
    },
    'lr_config': {
        'policy': 'OneCycle',
        'max_lr': 3e-5,
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
