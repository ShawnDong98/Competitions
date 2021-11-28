import torch
import wandb
from .base import LoggerHook

class PetfinderLoggerHook(LoggerHook):

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
                if name in ['time', 'data_time', 'pred', 'label']:
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
                if name in ['time', 'data_time', 'pred', 'label']:
                    continue
                log_items.append('val_{}: {:.4f}'.format(name, val))
                wandb_log_buffer['val_{}'.format(name)] = val
            log_str += ', '.join(log_items)
            trainer.logger.info(log_str)
            wandb.log(wandb_log_buffer)

    def before_run(self, trainer):
        for hook in trainer.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break
        wandb.init(config=trainer.config, project=trainer.config.name, entity="shawndong98")
        wandb.watch(trainer.model, log_freq=self.interval)

    def before_train_epoch(self, trainer):
        trainer.log_buffer.clear()  # clear logs of last epoch

    def before_val_epoch(self, trainer):
        trainer.log_buffer.clear()  # clear logs of last epoch
        self.log(trainer)

    def after_train_iter(self, trainer):
        if self.every_n_inner_iters(trainer, self.interval):
            trainer.log_buffer.average(self.interval)
        elif self.end_of_epoch(trainer) and not self.ignore_last:
            # not precise but more stable
            trainer.log_buffer.average(self.interval)

        if trainer.log_buffer.ready:
            self.log(trainer)
            if self.reset_flag:
                trainer.log_buffer.clear_output()

    def after_train_epoch(self, trainer):
        if trainer.log_buffer.ready:
            self.log(trainer)
            if self.reset_flag:
                trainer.log_buffer.clear_output()

    def after_val_epoch(self, trainer):

        trainer.log_buffer.average()
        preds = torch.cat(trainer.log_buffer.val_history['pred'])
        labels = torch.cat(trainer.log_buffer.val_history['label'])
        trainer.log_buffer.output['mse'] = torch.sqrt(((preds - labels) ** 2).mean())
        self.log(trainer)
