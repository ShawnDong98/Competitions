import wandb
from ...utils import master_only
from .base import LoggerHook


class WandBLoggerHook(LoggerHook):
    def __init__(
        self,
        init_kwargs = None,
        interval = 10,
        ignore_last = True,
        reset_flag = True
    ):
        super().__init__(interval, ignore_last, reset_flag)
        self.import_wandb()
        self.init_kwargs = init_kwargs

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "WandB is not installed. Please install WandB to use this hook."
            )
        self.wandb = wandb

    @master_only
    def before_run(self, trainer):
        for hook in trainer.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break
        wandb.init(config=trainer.config, project=trainer.config.name, entity="shawndong98")
        wandb.watch(trainer.model, log_freq=self.interval)

    @master_only
    def log(self, trainer):
        if trainer.mode == 'train':
            lr_str = ', '.join(
                ['{:.7f}'.format(lr) for lr in trainer.current_lr()])
            log_str = 'Epoch [{}][{}/{}]\tlr: {}, '.format(
                trainer.epoch + 1, trainer.inner_iter + 1,
                len(trainer.data_loader), lr_str)
            wandb.log(
                {
                    'lr': float(lr_str),
                    'train_epoch': trainer.epoch + 1,
                }
            )

            if 'time' in trainer.log_buffer.output:
                log_str += (
                    'time: {log[time]:.3f}, data_time: {log[data_time]:.3f}, '.
                    format(log=trainer.log_buffer.output))
            log_items = []
            wandb_log_buffer = {}
            for name, val in trainer.log_buffer.output.items():
                if name in ['time', 'data_time']:
                    continue
                log_items.append('train_{}: {:.4f}'.format(name, val))
                wandb_log_buffer['train_{}'.format(name)] = val
            log_str += ', '.join(log_items)
            trainer.logger.info(log_str)
            wandb.log(wandb_log_buffer)
        else:
            log_str = 'Epoch({}) [{}][{}]\t'.format(trainer.mode, trainer.epoch, trainer.inner_iter + 1)
            wandb.log(
                {
                    'val_epoch': trainer.epoch + 1,
                }
            )

            if 'time' in trainer.log_buffer.output:
                log_str += (
                    'time: {log[time]:.3f}, data_time: {log[data_time]:.3f}, '.
                    format(log=trainer.log_buffer.output))
            log_items = []
            wandb_log_buffer = {}
            for name, val in trainer.log_buffer.output.items():
                if name in ['time', 'data_time']:
                    continue
                log_items.append('val_{}: {:.4f}'.format(name, val))
                wandb_log_buffer['val_{}'.format(name)] = val
            log_str += ', '.join(log_items)
            trainer.logger.info(log_str)
            wandb.log(wandb_log_buffer)
