import gc
import os
import os.path as osp
from typing import Sequence, Union, List, Generator, Optional, Tuple
import math

import numpy as np

from tqdm import tqdm, trange
import torch
import torch.cuda.amp as amp
from torch import nn
from torch import Tensor

import matplotlib.pyplot as plt
import pandas as pd

from ._misc import text_styles, TrivialContext

class DefaultConfig:
    def __init__(self):
        self.checkpoint_dir = ''
        self.run_name = f'test'
        # training
        self.train_batch_size = 32
        self.val_batch_size = 32
        self.start_epoch = 0
        self.end_epoch = 1
        self.optimizers = torch.optim.AdamW
        self.optimizer_params = {'lr': 3e-4, 'weight_decay': 1e-3}
        self.schedulers = []
        self.scheduler_params = []
        # means we step the scheduler at each training iteration
        #  the other option is 'epoch'
        self.scheduler_interval = 'step'
        # whether or not the scheduler requires the step or epoch as an input
        #  argument
        self.scheduler_interval_eval = '[]'
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.clip_grad_norm = -1
        # automatic mixed precision https://pytorch.org/docs/stable/notes/amp_examples.html
        self.use_amp = False
        # whether we are aiming to maximize or minimize score
        self.score_objective = 'max'


def merge_config(custom_config):
    config = DefaultConfig()
    for prop, value in vars(custom_config).items():
        if prop.startswith('_'):
            continue
        setattr(config, prop, value)
    # for backwards compatibility
    try:
        config.schedulers = config.scheduler
        config.optimizers = config.optimizer
    except AttributeError:
        pass
    return config


class Fitter:
    """
    May need to override:
     - prepare_inputs_and_targets
     - compute_loss
     - compute_scores
     - collate_targets_and_outputs (for validation)
     - forward_train (if you have more than one model)
     - forward_validate (if you have more than one model)
     - on_start_epoch
     - on_validate_end
    """
    def __init__(self, models, data_loaders, device, config=None, n_val_iters=0,
                    id_key='', load=''):
        """
        a list of models may be provided, but then a list of optimizers
         and a list of schedulers of the same lengths are also expected
        `n_val_iters` lets us specify how often to do validation in intervals
         of train iterations
        `id_key` is used during validation with inspect=True. The idea is that
         one of the keys in the dataset __getitem__ return value corresponds to
         a datapoint id. This key is then returned alongside a list of vaidation
         results. It lets us inspect a particular data point. See `validate`
         method for more info
        """
        self.models = models
        if not isinstance(self.models, Sequence):
            self.models = [self.models]
        [model.to(device) for model in self.models]
        self.data_loaders = data_loaders
        self.device = device
        # keep original config in case we want to reset it after lr sweep
        self.original_config = config
        if config is not None:
            self.config = merge_config(config)
        else:
            self.config = DefaultConfig()
        self.reset_fitter_state()
        self._n_train_iters = len(self.data_loaders.train_loader)
        if n_val_iters > 0 and n_val_iters <= len(self.data_loaders.train_loader):
            self.n_val_iters = n_val_iters
        else:
            if n_val_iters > len(self.data_loaders.train_loader):
                print("Warning: Clipping n_val_iters to max train iters")
            self.n_val_iters = len(self.data_loaders.train_loader)
        if load:
            self.load(load)
        # for validation debug
        self.id_key = id_key
        # automatic mixed precision
        if self.config.use_amp:
            self.scaler = amp.GradScaler()
            self.train_forward_context = amp.autocast
        else:
            self.train_forward_context = TrivialContext
        # check score objective
        assert self.config.score_objective in ['min', 'max'], \
                        "config.score_objective must be either 'min' or 'max'"

    def reset_fitter_state(self):
        """
        NOTE: I haven't done a thorough check to make sure I'm not missing anythin
        """
        self.reset_history()
        self.reset_optimizers()
        self.reset_schedulers()
        self.best_running_val_loss = np.float('inf')
        if self.config.score_objective == 'max':
            self.best_running_val_score = -np.float('inf')
        elif self.config.score_objective == 'min':
            self.best_running_val_score = np.float('inf')
        self.epoch = self.config.start_epoch

    def reset_optimizers(self):
        optimizers = self.config.optimizers
        if not isinstance(optimizers, Sequence):
            optimizers = [optimizers]
        assert len(optimizers) == len(self.models), \
                    "Must provide as many optimizers as models"
        optimizer_params = self.config.optimizer_params
        if not isinstance(optimizer_params, Sequence):
            optimizer_params = [optimizer_params]
        self.optimizers = []
        for opt, ops, m in zip(optimizers, optimizer_params, self.models):
            self.optimizers.append(opt(m.parameters(), **ops))
        
    def set_lrs(self, lrs: Union[Sequence[float], float]):
        """ manually set the lrs of the optimizers
        """
        if not isinstance(lrs, Sequence):
            lrs = [lrs]
        assert len(lrs) == len(self.optimizers), \
            "Must provide as many lrs as there are optimizers"
        for lr, optim in zip(lrs, self.optimizers):
            for g in optim.param_groups:
                g['lr'] = lr

    def reset_schedulers(self):
        schedulers = self.config.schedulers
        if not isinstance(schedulers, Sequence):
            schedulers = [schedulers]
        if len(schedulers) > 0:
            assert len(schedulers) == len(self.models), \
                    "Must provide as many schedulers as models"
        else:
            self.schedulers = []
            return
        scheduler_params = self.config.scheduler_params
        if not isinstance(scheduler_params, Sequence):
            scheduler_params = [scheduler_params]
        assert len(scheduler_params) == len(schedulers), \
                "Must provide as many sets of scheduler_params as schedulers"
        self.schedulers = []
        for sched, sps, opt in zip(schedulers, scheduler_params, self.optimizers):
            self.schedulers.append(sched(opt, **sps))

    def reset_history(self):
        self.history = {}
        optimizers = self.config.optimizers
        if not isinstance(optimizers, Sequence):
            optimizers = [optimizers]
        for i in range(len(optimizers)):
            self.history[f'lr_{i}'] = []
            self.history[f'grad_norm_{i}'] = []

    def forward_train(self, inputs, targets):
        """
        a single forward pass for training. usually this is straight forward
        but one might want to be more specific where there are multiple models
        """
        return self.models[0](*inputs)

    def train_iteration(self, inputs, targets):
        """
        what needs to happen during one train loop iteration
        """
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        with self.train_forward_context():
            outputs = self.forward_train(inputs, targets)
            losses = self.compute_losses(targets, outputs, mode='train')
            if not isinstance(losses, Sequence):
                # backwards compatibility
                losses = [losses]
            loss = sum(losses)

        if self.config.use_amp:
            self.scaler.scale(loss).backward()
            # this next line makes sure getting/clipping grad norm works
            for optimizer in self.optimizers:
                self.scaler.unscale_(optimizer)
        else:
            loss.backward()
        
        grad_norms = []
        for model in self.models:
            if self.config.clip_grad_norm > 0:
                grad_norms.append(nn.utils.clip_grad_norm_(model.parameters(),
                                    self.config.clip_grad_norm))
            else:
                # NOTE assumes l2 norm
                grad_norms.append(torch.norm(torch.stack([torch.norm(
                        p.grad.detach(), 2) for p in model.parameters() \
                            if p.grad is not None]), 2))

        if self.config.use_amp:
            # scaler.step knows that I previously used scaler.unscale_
            for optimizer in self.optimizers:
                self.scaler.step(optimizer)
            self.scaler.update()
        else:
            for optimizer in self.optimizers:
                optimizer.step()
            
        return losses, grad_norms

    def fit(self, overfit=False, skip_to_step=0, save=True,
                    bail=float('inf'), verbose=0):
        """
        `skip_to_step` cycles through the train loader without training
            until that step is reached. Useful for diagnosing a bug appearing
            at a specific location in the train cycle
        """

        overfit_data = None # in case we want to try overfitting to one batch

        epoch_bar = trange(self.epoch, self.config.end_epoch, disable=(verbose != 1))

        for epoch in epoch_bar:
            # hook for tasks to do when starting epoch
            self.on_start_epoch()
            # train
            [model.train() for model in self.models]
            total_train_losses = []
            train_preds = 0
            train_bar = tqdm(self.data_loaders.train_loader, disable=(verbose != 2))
            train_bar.set_description(f'Epoch {epoch:03d}')
            self.train_step = 0
            for data in train_bar:
                if skip_to_step > 0 and self.train_step < skip_to_step:
                    self.train_step += 1
                    continue

                if overfit:
                    if overfit_data is None:
                        overfit_data = data
                    else:
                        data = overfit_data

                # get inputs and targets
                inputs, targets = self.prepare_inputs_and_targets(data, mode='train')

                # train step
                losses, grad_norms = self.train_iteration(inputs, targets)

                # scheduler
                # first keep track of lr to report on it
                lrs = []
                for optimizer in self.optimizers:
                    lrs.append(optimizer.param_groups[0]['lr'])
                for scheduler in self.schedulers:
                    if self.config.scheduler_interval == 'step':
                        args = eval(self.config.scheduler_interval_eval)
                        scheduler.step(*args)

                # logging
                batch_size = inputs[0].shape[0]
                with torch.no_grad():
                    # train_losses is just the item() version of losses for reporting
                    train_losses = [l.item() for l in losses]
                for i, tl in enumerate(train_losses):
                    if len(total_train_losses) == 0:
                        # make train losses the right length
                        total_train_losses = [0.] * len(train_losses)
                    while len(total_train_losses) < i+1:
                        total_train_losses.append(0.)
                    total_train_losses[i] += tl * batch_size # unaverage
                train_preds += batch_size
                kwargs = {f'lr_{i}': f'{lr:.2E}' for i, lr in enumerate(lrs)}
                kwargs.update({f'grad_norm_{i}': f'{grad_norm:.3f}' \
                                for i, grad_norm in enumerate(grad_norms)})
                kwargs.update({f'train_loss_{i}': f'{l:.3f}' \
                                for i, l in enumerate(train_losses)})
                train_bar.set_postfix(**kwargs)
                for i, l in enumerate(train_losses):
                    if not f'train_loss_{i}' in self.history:
                        self.history[f'train_loss_{i}'] = []
                    self.history[f'train_loss_{i}'].append(l)
                for i, (lr, grad_norm) in enumerate(zip(lrs, grad_norms)):
                    self.history[f'lr_{i}'].append(lr)
                    self.history[f'grad_norm_{i}'].append(grad_norm.item())

                # bail out if loss gets too high
                for i, l in enumerate(train_losses):
                    if l > bail:
                        print(f"loss_{i} blew up. Bailed training.")
                        return

                    if math.isnan(l):
                        msg = f"WARNING: NaN loss_{i}"
                        if self.config.use_amp:
                            msg += " Heads up: this may be a side effect of using AMP."
                            msg += " If so, the gradient scaler should skip this step."
                        print(msg)
                        # if bail is set to a specific number, bail
                        if bail < float('inf'):
                            return

                # validate every val_itrs
                if (self.train_step + 1) % self.n_val_iters == 0:
                    if self.data_loaders.val_loader is not None:
                        val_losses, val_scores = \
                                        self.validate(verbose=(verbose == 2))
                    else:
                        val_losses, val_scores = [], []

                    for i, l in enumerate(val_losses):
                        if not f'val_loss_{i}' in self.history:
                            self.history[f'val_loss_{i}'] = []
                        self.history[f'val_loss_{i}'].append(l.item())
                    for i, s in enumerate(val_scores):
                        if not f'val_score_{i}' in self.history:
                            self.history[f'val_score_{i}'] = []
                        self.history[f'val_score_{i}'].append(s)

                    # best loss is based on sum of losses
                    # TODO decide if I want to use running avg
                    #  and if so, make it possible to decide on length
                    running_val_loss = 0
                    for i in range(len(val_losses)):
                        running_val_loss += np.mean(self.history[f'val_loss_{i}'][-1:])                        
                    if save and running_val_loss <= self.best_running_val_loss:
                        self.best_running_val_loss = running_val_loss
                        self.save(f'{self.config.run_name}_best_loss.pt')

                    if 'val_score_0' in self.history:
                        running_val_score = np.mean(self.history['val_score_0'][-1:])
                        sign = 1 if self.config.score_objective == 'min' else -1
                        if save and sign*running_val_score <= sign*self.best_running_val_score:
                            self.best_running_val_score = running_val_score
                            self.save(f'{self.config.run_name}_best_score.pt')
                        
                    [model.train() for model in self.models]

                self.train_step += 1

            # step scheduler on epoch
            if self.config.scheduler_interval == 'epoch':
                for scheduler in self.schedulers:
                    args = eval(self.config.scheduler_interval_eval)
                    scheduler.step(*args)

            if verbose == 1:
                kwargs = {f'val_loss_{i}': f"{l.item():.3f}" for i, l \
                                                in enumerate(val_losses)}
                kwargs.update({f'val_score_{i}': f"{s:.3f}" for i, s \
                                                in enumerate(val_scores)})
                kwargs.update({f'train_loss_{i}': f"{l/train_preds:.3f}" \
                                    for i, l in enumerate(total_train_losses)})
                kwargs.update({f'lr_{i}': f"{lr:.2E}" for i, lr in enumerate(lrs)})
                epoch_bar.set_postfix(**kwargs)
            
            if verbose == 2:
                for i, (tl, vl)  in enumerate(zip(total_train_losses, val_losses)):
                    msg = f"\nAverage train / val loss {i}"
                    msg += f"{text_styles.BOLD}{(tl/train_preds):.5f}{text_styles.ENDC}"
                    msg += f" / {text_styles.BOLD}{(vl):.5f}{text_styles.ENDC}. "
                    print(msg + '\n', flush=True)

            self.epoch = epoch + 1

            if save and self.config.run_name:
                self.save(f'{self.config.run_name}_last.pt')

        if save and self.config.run_name:
            self.save(f'{self.config.run_name}_epoch{self.epoch:02d}.pt')

    def on_start_epoch(self):
        """ tasks to do when starting in epoch, overwrite me
        """
        pass

    def prepare_inputs_and_targets(self, data, mode='val'):
        """
        this method probably needs to be overriden
        return a list of batches of inputs, and a single batch of targets
        """
        assert mode in ['train', 'val'], "`mode` must be either 'train' or 'val'"
        return [data['inp'].to(self.device)], data['target'].to(self.device)

    def forward_validate(self, inputs):
        """
        a single forward pass for validation. usually this is straight forward
        but one might want to be more specific where there are multiple models
        """
        return self.models[0](*inputs)

    def validate(self, inspect=False, loader=None, use_train_loader=False,
                    verbose=True) -> Tuple[float, List[float]]:
        """
        inspects lets us retrieve more information than just the validation score and loss
        if `loader` is not provide, `self.val_loader` is used by default
        if `use_train_loader` is set, `self.train_loader` is used
        """
        [model.eval() for model in self.models]
        ls_outputs = []
        ls_targets = []
        if inspect and self.id_key != '':
            ids = [] # for debugging purposes
        if loader is None:
            if not use_train_loader:
                loader = self.data_loaders.val_loader
            else:
                loader = self.data_loaders.train_loader
        elif use_train_loader:
            print("Warning: You have provided a loader but also set `use_train_loader` to true")
        val_bar = tqdm(loader, disable=(not verbose))
        for data in val_bar:
            inputs, targets = self.prepare_inputs_and_targets(data, mode='val')
            with torch.no_grad():
                outputs = self.forward_validate(inputs)
            ls_targets.append(targets)
            ls_outputs.append(outputs)
            if inspect and self.id_key != '':
                ids += list(data[self.id_key])
        targets, outputs = self.collate_targets_and_outputs(ls_targets, ls_outputs)

        with torch.no_grad():
            losses = self.compute_losses(targets, outputs, mode='val')
            if not isinstance(losses, Sequence):
                # backwards compatibility
                losses = [losses]

        scores = self.compute_scores(targets, outputs, mode='val')
        if not isinstance(scores, Sequence):
            # backwards compatibility
            scores = [scores]

        if verbose:
            print(''.join([f', val_loss_{i}: {s:.3f}' \
                                for i, s in enumerate(losses)]) + \
                    ''.join([f', val_score_{i}: {s:.3f}' \
                                    for i, s in enumerate(scores)]),
                  flush=True)
        if inspect:
            ret = {f'loss_{i}': l for i, l in enumerate(losses)}
            ret.update({'targets': targets, 'outputs': outputs})
            ret.update({f'score_{i}': s for i, s in enumerate(scores)})
            if self.id_key != '':
                ret['id'] = ids
            return ret

        self.on_validate_end(targets, outputs)

        return losses, scores

    def on_validate_end(self, targets, outputs):
        """
        post-validation hook, might be useful for showing something like an 
        example
        """
        pass

    def collate_targets_and_outputs(self, ls_targets, ls_outputs):
        """
        during validation targets and outputs are concatenated into a list
        depending on the nature of these, we might want to overwrite the way
        that the are collated prior to computing loss and score
        by default, we have the naive implementation where ls_targets and ls_outputs
        are lists of tensors
        note that we send targets and outputs to cpu
        """
        targets = torch.cat([t.cpu() for t in ls_targets], axis=0)
        outputs = torch.cat([o.cpu() for o in ls_outputs], axis=0)
        return targets, outputs

    def compute_loss(self, targets, outputs, mode='train'):
        # backwards compatibility
        return [self.config.criterion(outputs, targets)]

    def compute_losses(self, targets, outputs, mode='train'):
        # backwards compatibility
        return self.compute_loss(targets, outputs, mode=mode)

    def compute_score(self, targets, outputs, mode='val') -> Union[Sequence[float], float]:
        """
        backwards compatibility
        """
        return []

    def compute_scores(self, targets, outputs, mode='val') -> Union[Sequence[float], float]:
        """
        return a list of scores
        Note that the order of scores affects two things:
        1. For the purpose of saving on best score, the first of the scores is
            used.
        2. For the purposes of plotting scores, teh first of the scores is used
        """
        # backwards compatibility if not overriden
        return self.compute_score(targets, outputs, mode='val')

    def test(self, loader=None, verbose=True) -> Generator[
                            Tuple[Tensor, Optional[List[str]]], None, None]:
        """
        similar to `validate` method, but not expecting targets. simply
         a generator of outputs and ids if self.id_key is specified
        Note that this uses `forward_validate` method
        """
        [model.eval() for model in self.models]
        if self.id_key == '':
            print("Warning: You have not set an id_key. Results won't have ids.")
        if loader is None:
            print("Warning: No loader provided. Default to val_loader")
            loader = self.data_loaders.val_loader
        # test_bar = tqdm(loader, disable=(not verbose))
        for data in loader:
            inputs = self.prepare_inputs_and_targets(data, mode='test')
            with torch.no_grad():
                outputs = self.forward_validate(inputs).cpu()
            if self.id_key != '':
                yield outputs, list(data[self.id_key])
            else:
                yield outputs

    def plot_history(self, plot_from=0, sma_period=5):
        fig, ax = plt.subplots(2, 2, figsize=(20,10))
        ax = ax.flatten()

        # train loss
        max_losses = 3 # maximum number of loss traces to plot
        x_axis = np.arange(1, len(self.history[f'train_loss_0'])+1)/self._n_train_iters
        for i in range(max_losses+1):
            if f'train_loss_{i}' not in self.history:
                break
            if i == max_losses:
                print(f"Warning: Max number of loss traces ({max_losses}) " \
                            + "exceeded. Not all losses are plotted")
                break
            train_loss = pd.Series(self.history[f'train_loss_{i}'][plot_from:])
            ax[0].plot(x_axis[plot_from:], train_loss, alpha=0.5, label=f'train_loss_{i}')
            ax[0].plot(x_axis[plot_from:][sma_period-1:],
                train_loss.rolling(window=sma_period).mean().iloc[sma_period-1:].values,
                label=f'train_loss_{i}_smoothed')
        
        # val loss
        vals_per_epoch = self._n_train_iters//self.n_val_iters
        x_axis = np.arange(1, len(self.history['val_loss_0']) + 1)/vals_per_epoch
        for i in range(max_losses+1) or i == max_losses:
            if f'val_loss_{i}' not in self.history:
                break
            ax[0].plot(x_axis[(vals_per_epoch * plot_from)//self._n_train_iters:],
                self.history[f'val_loss_{i}'][(vals_per_epoch * plot_from)//self._n_train_iters:],
                label=f'val_loss_{i}')

        ax[0].legend()
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('loss')
        ax[0].grid()

        title = f"Best train_loss_0: {min(self.history['train_loss_0']):0.3f}"
        if len(self.history['val_loss_0']):
            title += f". Best val_loss_0: {min(self.history['val_loss_0']):0.3f}"
        ax[0].set_title(title)

        # val metrics
        if len(self.history['val_score_0']):
            ax[1].plot(x_axis[(vals_per_epoch * plot_from)//self._n_train_iters:],
                    self.history['val_score_0'][(vals_per_epoch * plot_from)//self._n_train_iters:])
            ax[1].set_xlabel('epoch')
            ax[1].set_ylabel('score')
            ax[1].grid()
            if self.config.score_objective == 'max':
                title = f"Best val_score_0: {max(self.history['val_score_0']):0.3f}"
            elif self.config.score_objective == 'min':
                title = f"Best val_score_0: {min(self.history['val_score_0']):0.3f}"
            ax[1].set_title(title)

        # lrs
        x_axis = np.arange(1, len(self.history['train_loss_0'])+1)/self._n_train_iters
        legend = []
        for i in range(len(self.optimizers)):
            ax[2].plot(x_axis[plot_from:], self.history[f'lr_{i}'][plot_from:])
            legend.append(f'lr_{i}')
        ax[2].set_xlabel('epoch')
        ax[2].set_ylabel('lr')
        if len(legend):
            ax[2].legend(legend)
        ax[2].grid()

        # grad norms
        x_axis = np.arange(1, len(self.history['train_loss_0'])+1)/self._n_train_iters
        legend = []
        for i in range(len(self.optimizers)):
            ax[3].plot(x_axis[plot_from:], self.history[f'grad_norm_{i}'][plot_from:])
            legend.append(f'grad_norm_{i}')
        ax[3].set_xlabel('epoch')
        ax[3].set_ylabel('grad_norm')
        if len(legend):
            ax[3].legend(legend)
        ax[3].grid()
        return fig, ax

    def lr_sweep(self, start_lrs: Sequence[float], gamma: float, bail=np.float('inf')):
        """
        run an lr sweep starting from `start_lrs` (provide as many lrs as there
         are optimizers) and with exponential growth rate `gamma` (> 1)
        `bail` is the loss at which training stops
        """
        if not isinstance(start_lrs, Sequence):
            start_lrs = [start_lrs]
        assert len(start_lrs) == (n := len(self.optimizers)), \
            f"Must provide as many starting lrs as there are optimizers: {n}"
        if len(start_lrs) > 1:
            for i, (lr, _) in enumerate(zip(start_lrs, self.config.optimizer_params)):
                self.config.optimizer_params[i]['lr'] = lr
        else:
            self.config.optimizer_params['lr'] = start_lrs[0]
        self.config.schedulers = [torch.optim.lr_scheduler.ExponentialLR]*len(start_lrs)
        self.config.scheduler_params = [{'gamma': gamma}]*len(start_lrs)
        self.reset_fitter_state()
        self.fit(verbose=2, save=False, bail=bail)
        self.plot_lr_sweep()
        # now clean up
        if self.original_config is not None:
            self.config = merge_config(self.original_config)
        else:
            self.config = DefaultConfig()
        self.reset_fitter_state()
        print("LR sweep done and fitter state has been reset")

    def plot_lr_sweep(self, sma_period=5):
        num_lrs = len(self.optimizers)
        fig, ax = plt.subplots(1, num_lrs, figsize=(10*num_lrs,5))
        if num_lrs > 1:
            ax = ax.flatten()
        else:
            ax = [ax]
        
        max_losses = 3 # maximum number of loss traces to plot
        for i in range(num_lrs):
            for j in range(max_losses+1):
                if f'train_loss_{j}' not in self.history:
                    break
                if i == max_losses:
                    print(f"Warning: Max number of loss traces ({max_losses}) " \
                                + "exceeded. Not all losses are plotted")
                    break
                loss = pd.Series(self.history[f'train_loss_{j}'])
                ax[i].plot(self.history[f'lr_{i}'], loss, label=f'loss_{j}')
                # ax[i].plot(self.history[f'lr_{i}'][sma_period-1:],
                #     loss.rolling(window=sma_period).mean().iloc[sma_period-1:].values)
            ax[i].set_xlabel(f'lr_{i}')
            ax[i].set_ylabel('loss')
            ax[i].set_yscale('log')
            ax[i].set_xscale('log')
            ax[i].legend()
            ax[i].grid()
            ax[i].xaxis.grid(True, which='minor')
        plt.tight_layout()
        plt.show()
        return fig, ax

    def save(self, filename):
        """
        saves under checkpoint_dir/run_name
        """
        save_dct = {
            'epoch': self.epoch,
            'best_running_val_score': self.best_running_val_score,
            'best_running_val_loss': self.best_running_val_loss,
            'history': self.history,
        }
        for i, model in enumerate(self.models):
            save_dct[f'model_{i}_state_dict'] = model.state_dict()
        for i, optimizer in enumerate(self.optimizers):
            save_dct[f'optimizer_{i}_state_dict'] = optimizer.state_dict()
        for i, scheduler in enumerate(self.schedulers):
            save_dct[f'scheduler_{i}_state_dict'] = scheduler.state_dict()
        save_dir = osp.join(self.config.checkpoint_dir, self.config.run_name)
        if not osp.isdir(save_dir):
            os.mkdir(save_dir)
        torch.save(save_dct, osp.join(save_dir, filename))

    def load(self, file_path: str):
        """
        `file_path` is path to checkpoint file under the config's checkpoint
         dir. Unlike the `save` method, `run_name` is not assumed
        """
        checkpoint = torch.load(osp.join(self.config.checkpoint_dir, file_path),
                                    map_location=self.device)
        for i, model in enumerate(self.models):
            try:
                model.load_state_dict(checkpoint[f'model_{i}_state_dict'],
                                        strict=False)
            except RuntimeError:
                msg = f"RuntimeError when loading model_{i}_state_dict"
                msg += ". Suspecting that top layers did not match"
                msg += ". Dropping them and trying again."
                print(msg)
                keys = [k for k in checkpoint[f'model_{i}_state_dict'] if 'top.' in k]
                for key in keys:
                    del checkpoint[f'model_{i}_state_dict'][key]
                model.load_state_dict(checkpoint[f'model_{i}_state_dict'],
                                        strict=False)
        for i, optimizer in enumerate(self.optimizers):
            try:
                optimizer.load_state_dict(checkpoint[f'optimizer_{i}_state_dict'])
            except:
                print(f"Warning: Could not load optimizer_{i}_state_dict")
        for i, scheduler in enumerate(self.schedulers):
            try:
                scheduler.load_state_dict(checkpoint[f'scheduler_{i}_state_dict'])
            except:
                print(f"Warning: Could not load scheduler_{i}_state_dict")
        try:
            self.epoch = checkpoint['epoch']
        except:
            print("Warning: Could not load epoch number")
        try:
            self.history = checkpoint['history']
            # backwards compatibility
            if 'val_score' in self.history:
                self.history['val_score_0'] = self.history['val_score']
            if self.config.score_objective == 'max':
                self.best_running_val_score = max(self.history['val_score_0'])
            else:
                self.best_running_val_score = min(self.history['val_score_0'])
            # backwards compatibility
            if 'val_loss' in self.history:
                self.history['val_loss_0'] = self.history['val_loss']
            self.best_running_val_loss = min(self.history['val_loss_0'])
            # backwards compatibility
            if 'train_loss' in self.history:
                self.history['train_loss_0'] = self.history['train_loss']
        except:
            print("Warning: Could not load history")

    def clear_optimizers(self):
        try:
            self._optimizers_to(torch.device('cpu'))
            for optimizer in self.optimizers:
                del optimizer
            gc.collect()
            torch.cuda.empty_cache()
        except TypeError:
            pass
                    
    def _optimizers_to(self, device):
        for optimizer in self.optimizers:
            for param in optimizer.state.values():
                # Not sure there are any global tensors in the state dict
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(device)
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(device)
                elif isinstance(param, dict):
                    for subparam in param.values():
                        if isinstance(subparam, torch.Tensor):
                            subparam.data = subparam.data.to(device)
                            if subparam._grad is not None:
                                subparam._grad.data = subparam._grad.data.to(device)