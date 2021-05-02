import gc
import os

import numpy as np

from tqdm import tqdm, trange
import torch
import torch.cuda.amp as amp
from torch import nn

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
        self.optimizer = torch.optim.AdamW
        self.optimizer_params = {'lr': 3e-4, 'weight_decay': 1e-3}
        self.scheduler = None
        # means we step the scheduler at each training iteration
        #  the other option is 'epoch'
        self.scheduler_interval = 'step'
        # whether or not the scheduler requires the step or epoch as an input
        #  argument
        self.scheduler_interval_arg = False
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.clip_grad_norm = -1
        # automatic mixed precision https://pytorch.org/docs/stable/notes/amp_examples.html
        self.use_amp = False


def merge_config(custom_config):
    config = DefaultConfig()
    for prop, value in vars(custom_config).items():
        if prop.startswith('_'):
            continue
        setattr(config, prop, value)
    return config


class Fitter:
    """
    Must implement:
     - compute_score
    Recommended to override:
     - prepare_inputs_and_targets
    TODO - config is ugly because it requires inside knowledge
    """
    def __init__(self, model, data_loaders, device, config=None, n_val_iters=0,
                    id_key='', load=''):
        """
        `n_val_iters` lets us specify how often to do validation in intervals
         of train iterations
        `id_key` is used during validation with inspect=True. The idea is that
         one of the keys in the dataset __getitem__ return value corresponds to
         a datapoint id. This key is then returned alongside a list of vaidation
         results. It lets us inspect a particular data point. See `validate`
         method for more info
        """
        self.model = model
        self.model.to(device)
        self.data_loaders = data_loaders
        self.device = device
        if config is not None:
            self.config = merge_config(config)
        else:
            config = DefaultConfig()
        self.reset_history()
        self.reset_optimizer()
        self.reset_scheduler()
        self.best_running_val_loss = np.float('inf')
        self.best_running_val_score = -np.float('inf')
        self.epoch = self.config.start_epoch
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
        
    def reset_optimizer(self):
        self.optimizer = self.config.optimizer(self.model.parameters(),
                                               **self.config.optimizer_params)
        
    def reset_scheduler(self):
        if self.config.scheduler is not None:
            self.scheduler = self.config.scheduler(self.optimizer,
                                                   **self.config.scheduler_params)
        else:
            self.scheduler = None

    def reset_history(self):
        self.history = {'train_loss': [], 'val_loss': [], 'val_score': [],
                        'lr': [], 'grad_norm': []}

    def train_iteration(self, inputs, targets):
        """
        what needs to happen during one train loop iteration
        """
        self.optimizer.zero_grad()

        with self.train_forward_context():
            outputs = self.model(*inputs)
            loss = self.compute_loss(targets, outputs)

        if self.config.use_amp:
            self.scaler.scale(loss).backward()
            # this next line makes sure getting/clipping grad norm works
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        
        if self.config.clip_grad_norm > 0: 
            grad_norm = nn.utils.clip_grad_norm_(
                                        self.model.parameters(),
                                        self.config.clip_grad_norm)
        else:
            # NOTE assumes l2 norm
            grad_norm = torch.norm(torch.stack([torch.norm(
                p.grad.detach(), 2) for p in self.model.parameters()]), 2)

        if self.config.use_amp:
            # scaler.step knows that I previously used scaler.unscale_
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        return loss, grad_norm

    def fit(self, cont=True, overfit=False, save=True, bail=False, verbose=0):

        if self.epoch == 0 or not cont:
            self.reset_optimizer()
            self.reset_scheduler()
            self.reset_history()
            self.best_running_val_loss = np.float('inf')
            self.best_running_val_score = -np.float('inf')

        criterion = self.config.criterion

        overfit_data = None # in case we want to try overfitting to one batch

        epoch_bar = trange(self.epoch, self.config.end_epoch, disable=(verbose != 1))

        for epoch in epoch_bar:
            # hook for tasks to do when starting epoch
            self.on_start_epoch()
            # train
            self.model.train()
            total_train_loss = 0.0
            train_preds = 0
            train_bar = tqdm(self.data_loaders.train_loader, disable=(verbose != 2))
            train_bar.set_description(f'Epoch {epoch:03d}')
            self.train_step = 0
            for data in train_bar:
                if overfit:
                    if overfit_data is None:
                        overfit_data = data
                    else:
                        data = overfit_data

                # get inputs and targets
                inputs, targets = self.prepare_inputs_and_targets(data, mode='train')

                # train step
                loss, grad_norm = self.train_iteration(inputs, targets)

                # scheduler
                # first keep track of lr to report on it
                lr = self.optimizer.param_groups[0]['lr']
                if self.scheduler is not None:
                    if self.config.scheduler_interval == 'step':
                        args = [self.train_step] if self.config.scheduler_interval_arg else []
                        self.scheduler.step(*args)

                # logging
                batch_size = inputs[0].shape[0]
                train_loss = loss.item()
                total_train_loss += train_loss * batch_size # unaverage
                train_preds += batch_size
                train_bar.set_postfix(train_loss=f'{train_loss:.5f}',
                                lr=f'{lr:.2E}', grad_norm=f'{grad_norm:.3f}')
                self.history['train_loss'].append(train_loss)
                self.history['lr'].append(lr)
                self.history['grad_norm'].append(grad_norm)

                # bail out
                if bail and train_loss > 100000:
                    print("Loss blew up. Bailed training.")
                    return

                # validate every val_itrs
                if (self.train_step + 1) % self.n_val_iters == 0:
                    if self.data_loaders.val_loader is not None:
                        avg_val_loss, avg_val_score = self.validate(
                                                        verbose=(verbose == 2))
                    else:
                        avg_val_loss, avg_val_score = np.nan, np.nan

                    # save best
                    self.history['val_loss'].append(avg_val_loss.item())
                    self.history['val_score'].append(avg_val_score)

                    # TODO decide if I want to use running avg
                    #  and if so, make it possible to decide on length
                    running_val_loss = np.mean(self.history['val_loss'][-1:])
                    running_val_score = np.mean(self.history['val_score'][-1:])

                    if save and running_val_loss <= self.best_running_val_loss:
                        self.best_running_val_loss = running_val_loss
                        # print("Saving new best loss")
                        self.save(f'{self.config.run_name}_best_loss.pt')

                    # TODO - make it so that I can decide whether I'm going for
                    #  max or min
                    if save and running_val_score >= self.best_running_val_score:
                        self.best_running_val_score = running_val_score
                        # print("Saving new best score")
                        self.save(f'{self.config.run_name}_best_score.pt')
                        
                    self.model.train()

                self.train_step += 1

            # step scheduler on epoch
            if self.scheduler is not None and self.config.scheduler_interval == 'epoch':
                args = [self.epoch] if self.config.scheduler_interval_arg else []
                self.scheduler.step(*args)

            if verbose == 1:
                epoch_bar.set_postfix(train_loss=f"{(total_train_loss/train_preds):0.3f}",
                                val_loss=f"{avg_val_loss:.3f}", lr=f"{lr:.2E}",
                                val_score=f"{avg_val_score:.3f}",
                                val_acc=f"{avg_val_score:.3f}")
            
            if verbose == 2:
                msg = "\nAverage train / val loss was "
                msg += f"{text_styles.BOLD}{(total_train_loss/train_preds):0.5f}{text_styles.ENDC}"
                msg += f" / {text_styles.BOLD}{(avg_val_loss):0.5f}{text_styles.ENDC}. "
                print(msg + '\n', flush=True)

            self.epoch = epoch + 1

            if save and self.config.run_name:
                self.save(f'{self.config.run_name}_last.pt')

        if save and self.config.run_name:
            self.save(f'{self.config.run_name}_epoch{epoch:02d}.pt')

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

    def validate(self, inspect=False, loader=None, use_train_loader=False, verbose=True):
        """
        inspects lets us retrieve more information than just the validation score and loss
        if `loader` is not provide, `self.val_loader` is used by default
        if `use_train_loader` is set, `self.train_loader` is used
        """
        criterion = self.config.criterion
        self.model.eval()
        ls_outputs = []
        ls_targets = []
        if inspect and len(self.id_key):
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
                outputs = self.model(*inputs)
            ls_targets.append(targets)
            ls_outputs.append(outputs)
            if inspect and len(self.id_key):
                ids += list(data[self.id_key])
        targets, outputs = self.collate_targets_and_outputs(ls_targets, ls_outputs)

        with torch.no_grad():
            avg_val_loss = self.compute_loss(targets, outputs)

        avg_val_score = self.compute_score(targets, outputs)

        if verbose:
            print(f'avg_val_loss: {avg_val_loss:.3f}, ' +
                  f'avg_val_score: {avg_val_score:.3f}',
                  flush=True)
        if inspect:
            ret = {'loss': avg_val_loss, 'score': avg_val_score,
                    'targets': targets, 'outputs': outputs}
            if len(self.id_key):
                ret[self.id_key] = ids
            return ret

        self.on_validate_end(targets, outputs)

        return avg_val_loss, avg_val_score

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
        targets = torch.cat(ls_targets, axis=0).cpu()
        outputs = torch.cat(ls_outputs, axis=0).cpu()
        return targets, outputs


    def compute_loss(self, targets, outputs):
        """ could override this
        """
        return self.config.criterion(outputs, targets)

    def compute_score(self, targets, outputs):
        """ must be overidden
        """
        raise NotImplementedError()

    def plot_history(self, plot_from=0, sma_period=5):
        fig, ax = plt.subplots(1, 3, figsize=(20,5))

        # train loss
        x_axis = np.arange(1, len(self.history['train_loss'])+1)/self._n_train_iters
        train_loss = pd.Series(self.history['train_loss'][plot_from:])
        ax[0].plot(x_axis[plot_from:], train_loss, alpha=0.5)
        ax[0].plot(x_axis[plot_from:][sma_period-1:],
                   train_loss.rolling(window=sma_period).mean().iloc[sma_period-1:].values)
        
        # val loss
        vals_per_epoch = self._n_train_iters//self.n_val_iters
        x_axis = np.arange(1, len(self.history['val_loss']) + 1)/vals_per_epoch
        ax[0].plot(x_axis[(vals_per_epoch * plot_from)//self._n_train_iters:],
                   self.history['val_loss'][(vals_per_epoch * plot_from)//self._n_train_iters:])
        ax[0].legend(['train loss', 'train loss smoothed', 'val loss'])
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('loss')
        ax[0].grid()

        title = f"Best train loss: {min(self.history['train_loss']):0.3f}"
        if len(self.history['val_loss']):
            title += f". Best val loss: {min(self.history['val_loss']):0.3f}"
        ax[0].set_title(title)

        # val metrics
        if len(self.history['val_score']):
            ax[1].plot(x_axis[(vals_per_epoch * plot_from)//self._n_train_iters:],
                    self.history['val_score'][(vals_per_epoch * plot_from)//self._n_train_iters:])
            ax[1].set_xlabel('epoch')
            ax[1].set_ylabel('score')
            ax[1].grid()
            title = f"Best val score: {max(self.history['val_score']):0.3f}"
            ax[1].set_title(title)

        # lr
        x_axis = np.arange(1, len(self.history['train_loss'])+1)/self._n_train_iters
        ax[2].plot(x_axis[plot_from:], self.history['lr'][plot_from:])
        ax[2].set_xlabel('epoch')
        ax[2].set_ylabel('lr')
        ax[2].grid()

    def plot_lr_sweep(self, sma_period=5):
        fig, ax = plt.subplots(figsize=(15,5))
        loss = pd.Series(self.history['train_loss'])
        ax.plot(self.history['lr'], loss)
        ax.plot(self.history['lr'][sma_period-1:],
                loss.rolling(window=sma_period).mean().iloc[sma_period-1:].values)
        legend = ['exact', 'smoothed']
        ax.set_xlabel('lr')
        ax.set_ylabel('loss')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid()
        ax.xaxis.grid(True, which='minor')

    def save(self, filename):
        save_dct = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'best_running_val_score': self.best_running_val_score,
            'best_running_val_loss': self.best_running_val_loss,
            'history': self.history,
            'config': self.config,
        }
        if self.scheduler is not None:
            save_dct['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(save_dct, os.path.join(self.config.checkpoint_dir, filename))

    def load(self, filename):
        checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, filename),
                                    map_location= self.device)
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except RuntimeError:
            msg = "RuntimeError when loading model state dict"
            msg += ". Suspecting that top layers did not match"
            msg += ". Dropping them and trying again."
            print(msg)
            keys = [k for k in checkpoint['model_state_dict'] if 'top.' in k]
            for key in keys:
                del checkpoint['model_state_dict'][key]
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("Warning: Could not load optimizer_state_dict")
        try:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print("Warning: Could not load scheduler_state_dict")
        try:
            self.epoch = checkpoint['epoch']
        except:
            print("Warning: Could not load epoch number")
        try:
            self.history = checkpoint['history']
            self.best_running_val_loss = min(self.history['val_loss'])
            self.best_running_val_score = max(self.history['val_score'])
        except:
            print("Warning: Could not load history")

    def clear_optimizer(self):
        try:
            self._optimizer_to(torch.device('cpu'))
            del self.optimizer
            gc.collect()
            torch.cuda.empty_cache()
        except TypeError:
            pass
                    
    def _optimizer_to(self, device):
        for param in self.optimizer.state.values():
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