from .base import LoggerHook


class TextLoggerHook(LoggerHook):

    def log(self, trainer):
        if trainer.mode == 'train':
            lr_str = ', '.join(
                ['{:.5f}'.format(lr) for lr in trainer.current_lr()])
            log_str = 'Epoch [{}][{}/{}]\tlr: {}, '.format(
                trainer.epoch + 1, trainer.inner_iter + 1,
                len(trainer.data_loader), lr_str)
        else:
            log_str = 'Epoch({}) [{}][{}]\t'.format(trainer.mode, trainer.epoch,
                                                    trainer.inner_iter + 1)
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