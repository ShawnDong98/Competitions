from pprint import pprint
from box import Box

config = {
    'debug': False,
    'seed': 3407,
    'n_splits': 5,
    'epochs': 10,
    'root': "../datasets/fackface_det/",
    'image_size': 224,
    'work_dir': './checkpoint',
    'log_level': 'INFO',
    'log_config': {
        'interval': 50,
        'hooks' : [
            dict(name='WandBLoggerHook'),
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
        'lr': 1e-5,
    },
    'optimizer_config': {
        'grad_clip': None
    },
    'scheduler': {
        'name': 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts',
        'params': {
            'T_0': 10,
            'eta_min': 1e-4,
        }
    },
    'lr_config': {
        'policy': 'step',
        'step': 2,
    },
    'loss': 'torch.nn.BCEWithLogitsLoss',
    'workflow': [('train', 1), ('val', 1)],
    'checkpoint_config': {
        'interval': 1,
    },
    'earlystopping_config': {
        'monitor': 'f1_score',
        'patience': 2,
        'mode': 'max',
    }
}

config = Box(config)
pprint(config)