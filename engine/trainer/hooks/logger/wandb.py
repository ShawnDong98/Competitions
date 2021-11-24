import wandb
from .base import LoggerHook


class WandBLoggerHook(LoggerHook):
    def before_run(self, trainer):
        for hook in trainer.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break
        wandb.init(config=trainer.config, project="FackFaceDet", entity="shawndong98")
        wandb.watch(trainer.model, log_freq=self.interval)

    def log(self, trainer):
        if trainer.mode == 'train':
            lr_str = ', '.join(
                ['{:.5f}'.format(lr) for lr in trainer.current_lr()])
            log_str = 'Epoch [{}][{}/{}]\tlr: {}, '.format(
                trainer.epoch + 1, trainer.inner_iter + 1,
                len(trainer.data_loader), lr_str)
            wandb.log(
                {
                    'lr': float(lr_str),
                    'train_epoch': trainer.epoch + 1,
                }
            )
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
        for name, val in trainer.log_buffer.output.items():
            if name in ['time', 'data_time']:
                continue
            log_items.append('{}: {:.4f}'.format(name, val))
        log_str += ', '.join(log_items)
        trainer.logger.info(log_str)
        wandb.log(trainer.log_buffer.output)