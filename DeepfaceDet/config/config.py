from pprint import pprint
from box import Box

config = {
    'debug': False,
    'name': 'FackFaceDet',
    'seed': 999,
    'n_splits': 10,
    'epochs': 5,
    'root': "../datasets/fackface_det/",
    'image_size': 384,
    'image_size_tta': 440,
    'work_dir': './checkpoint',
    'log_level': 'INFO',
    'log_average_filter': [],
    'log_config': {
        'interval': 50,
        'hooks' : [
            dict(name='FackFaceDetLoggerHook'),
            # dict(name='TextLoggerHook'),
        ]
    },
    'model': {
        'name': 'swin_large_patch4_window12_384',
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
    'momentum_config': {
        'policy': 'OneCycle',
        'by_epoch': False,
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
        'monitor': 'f1_score',
        'patience': 2,
        'mode': 'max',
    }
}

config = Box(config)
pprint(config)
