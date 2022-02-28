#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2020    Zeng Xingui(zengxingui@baidu.com)
# Copyright     2020    Zhang Yinhui(zhangyinhui@baidu.com)
#
########################################################################

"""
config callbacks
"""
from __future__ import print_function
import os 
import sys
import numbers

import paddle
from paddle.fluid.dygraph.parallel import ParallelEnv
from visualdl import LogWriter

def update_callbacks(cbk_list,
                     batch_size=None,
                     epochs=None,
                     steps=None,
                     verbose=2,
                     metrics=None,
                     ):
    """
    update CallbackList params

    Args:
        cbk_list: an isinstance of CallbackList
        batch_size: batch size
        epochs: training epochs
        steps: steps per epoch
        verbose: The verbosity mode
        metrics: a list of metrics name

    Returns:
        cbk_list: a list of Callback
    """
    params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps,
        'verbose': verbose,
        'metrics': metrics,
    }
    cbk_list.set_params(params)
    return cbk_list


class VisualCallback(paddle.callbacks.Callback):
    """using VisualDL callback function. using tips: 
       visualdl --logdir ./log --model ./log/inference_model/__model__  --port 8080 --host 0.0.0.0
    Args:
        save_dir(str): the address of saving model
        log_dir(str): the address of log

    Examples:
        .. code-block:: python

            import paddle
            from paddle.static import InputSpec
            from sidt.callbacks.callbacks import VisualCallback

            inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
            labels = [InputSpec([None, 1], 'int64', 'label')]

            train_dataset = paddle.vision.datasets.MNIST(mode='train')

            model = paddle.Model(paddle.vision.LeNet(classifier_activation=None),
                inputs, labels)

            optim = paddle.optimizer.Adam(0.001)
            model.prepare(optimizer=optim,
                        loss=paddle.nn.CrossEntropyLoss(),
                        metrics=paddle.metric.Accuracy())

            callback = VisualCallback(save_dir='./log/', log_dir='./log/')
            model.fit(train_dataset, batch_size=64, callbacks=callback)
    """
    def __init__(self, log_freq=1, save_dir=None, log_dir=None):
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.log_freq = log_freq

    def _is_save(self):
        return self.model and self.save_dir and ParallelEnv().local_rank == 0

    def _is_log(self):
        return self.log_dir and ParallelEnv().local_rank == 0 

    def _updates(self, logs, mode):
        """update the log_writer.

        Args:
            logs (dict): The logs is a dict or None.
            mode(str): train, eval, test
        """
        if not self._is_log():
            return
        if not hasattr(self, "log_writer"):
            # init an instance of LogWriter for logging training
            self.log_writer = LogWriter(logdir=self.log_dir)

        metrics = getattr(self, '%s_metrics' % (mode))
        current_step = getattr(self, '%s_step' % (mode))

        if mode == 'train':
            total_step = current_step
        else:
            total_step = self.epoch


        # log {'loss': [1.4611504], 'acc': 0.9575, 'step': 312, 'batch_size': 16}
        for k in metrics:
            if k in logs:
                temp_tag = mode + '/' + k

                if isinstance(logs[k], (list, tuple)):
                    temp_value = logs[k][0]
                elif isinstance(logs[k], numbers.Number):
                    temp_value = logs[k]
                else:
                    continue

                self.log_writer.add_scalar(tag=temp_tag, step=total_step, value=temp_value)

    def on_train_begin(self, logs=None):
        """Called at the start of training.

        Args:
            logs (dict): The logs is a dict or None.
        """
        self.epochs = self.params['epochs']
        assert self.epochs
        # self.train_metrics is a list contain str ['loss', 'acc']
        self.train_metrics = self.params['metrics']
        assert self.train_metrics
        self.train_step = 0
        self.eval_step = 0

    def on_train_end(self, logs=None):
        """Called at the end of training.

        Args:
            logs (dict): The logs is a dict or None.
        """
        self.epochs = self.params['epochs']
        # Only the inference model can be display in visualdl, If `model.save` is 
        # set to False, only inference model will be saved. It 
        # should be noted that before using `save`, you should run the model, and 
        # the shape of input you saved is as same as the input of its running.
        # `@paddle.jit.to_static` must be added on `forward` function of your layer 
        # in dynamic mode now and these will be optimized later.
        if self._is_save():
            path = '{}/inference_model'.format(self.save_dir)
            print('save checkpoint at {}'.format(os.path.abspath(path)))
            self.model.save(path, False)

    def on_eval_begin(self, logs=None):
        """Called at the start of evaluation.

        Args:
            logs (dict): The logs is a dict or None. The keys of logs
                passed by paddle.Model contains 'steps' and 'metrics',
                The `steps` is number of total steps of validation dataset.
                The `metrics` is a list of str including 'loss' and the names
                of paddle.metric.Metric.
        """
        self.eval_steps = logs.get('steps', None)
        self.eval_metrics = logs.get('metrics', [])
        self.eval_step = 0
        self.evaled_samples = 0

    def on_test_begin(self, logs=None):
        """Called at the beginning of predict.

        Args:
            logs (dict): The logs is a dict or None.
        """
        self.test_steps = logs.get('steps', None)
        self.test_metrics = logs.get('metrics', [])
        self.test_step = 0

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch.

        Args:
            epoch (int): The index of epoch.
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is None.
        """
        self.steps = self.params['steps']
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch.

        Args:
            epoch (int): The index of epoch.
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict, contains 'loss', metrics and 'batch_size'
                of last batch.
        """
        logs = logs or {}
        if self._is_log() and (self.steps is not None):
            self._updates(logs, 'train')

        # using the histogram of visualdl to display the parameters of every layer in steps
        for param in self.model.network.parameters():
            self.log_writer.add_histogram(tag='parameters/%s' % (param.name),
                                          values=param.flatten().numpy(),
                                          step=self.epoch,
                                          buckets=100)

        for param in self.model.network.parameters():
            grad = param.grad
            if grad is not None:
                self.log_writer.add_histogram(tag='Gradient/%s' % (param.name),
                                              values=grad.flatten(),
                                              step=self.epoch,
                                              buckets=200)

        # get_lr() only works in Dygraph mode, Get current step learning rate.
        # The return value is all the same When _LRScheduler is not used,
        # otherwise return the current step learning rate.
        self.log_writer.add_scalar(tag="learning rate",
                                   step=self.epoch,
                                   value=self.model._optimizer.get_lr())

    def on_train_batch_end(self, step, logs=None):
        """Called at the end of each batch in training.

        Args:
            step (int): The index of step (or iteration).
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict, contains 'loss', metrics and 'batch_size'
                of current batch.
        """
        logs = logs or {}
        self.train_step += 1

        if self._is_log() and self.train_step % self.log_freq == 0:
            if self.steps is None or self.train_step < (self.epoch + 1) * self.steps:
                self._updates(logs, 'train')

    def on_eval_end(self, logs=None):
        """Called at the end of evaluation.

        Args:
            logs (dict): The logs is a dict or None. The `logs` passed by
                paddle.Model is a dict, contains 'loss', metrics and 'batch_size'
                of current batch.
        """
        logs = logs or {}

        if self._is_log():
            self._updates(logs, 'eval')

    def on_test_end(self, step, logs=None):
        """Called at the end of prediction.

        Args:
            logs (dict): The logs is a dict or None.
        """
        logs = logs or {}

        if self._is_log():
            self._updates(logs, 'test')


class LRUpdateCallback(paddle.callbacks.Callback):
    """
    LR update callback function
    """

    def __init__(self, scheduler):
        self._scheduler = scheduler

    def on_epoch_end(self, epoch, logs=None):
        """
        Update learning rate at the end of each epoch
        """
        if isinstance(self._scheduler, paddle.optimizer.lr.ReduceOnPlateau):
            metrics = getattr(self, 'train_metrics')
            self._scheduler.step(metrics['loss'][0])
        else:
            self._scheduler.step()
        new_lr = self._scheduler.get_lr()
        self.model._optimizer.set_lr(new_lr)


