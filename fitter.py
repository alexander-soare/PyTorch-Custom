import gc
import os

import numpy as np

from tqdm.auto import tqdm, trange
import torch
import matplotlib.pyplot as plt
import pandas as pd

from ._misc import text_styles

class Fitter:
    # TODO - config is ugly because it requires inside knowledge
    def __init__(self, model, data_loaders, device, config, n_val_iters=0, load=''):
        self.model = model
        self.model.to(device)
        self.data_loaders = data_loaders
        self.device = device
        self.config = config
        self.reset_history()
        self.reset_optimizer()
        self.reset_scheduler()
        self.epoch = self.config.start_epoch
        self._n_train_iters = len(self.data_loaders.train_loader)
        if n_val_iters > 0:
            self.n_val_iters = n_val_iters
        else:
            self.n_val_iters = len(self.data_loaders.train_loader)
        if load:
            self.load(load)
        # for validation debug
        self.id_key = ''
        
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
        self.history = {'train_loss': [], 'val_loss': [], 'val_score': [], 'lr': []}

    def fit(self, cont=True, overfit=False, save=True, bail=False, verbose=0):

        if self.epoch == 0 or not cont:
            self.reset_optimizer()
            self.reset_scheduler()
            self.reset_history()

        criterion = self.config.criterion

        overfit_data = None # in case we want to try overfitting to one batch

        best_running_val_loss = np.float('inf')
        best_running_val_score = -np.float('inf')

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

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(*inputs)

                loss = self.compute_loss(targets, outputs)

                loss.backward()
                self.optimizer.step()

                # scheduler
                lr = self.optimizer.param_groups[0]['lr']
                if self.scheduler is not None:
                    self.step_scheduler()

                # logging
                train_loss = loss.item()
                total_train_loss += train_loss * targets.shape[0] # unaverage
                train_preds += targets.shape[0]
                train_bar.set_postfix(train_loss=f'{train_loss:.5f}',
                                lr=f'{lr:.2E}')
                self.history['train_loss'].append(train_loss)
                self.history['lr'].append(lr)

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
                    self.history['val_loss'].append(avg_val_loss)
                    self.history['val_score'].append(avg_val_score)

                    # TODO decide if I want to use running avg
                    #  and if so, make it possible to decide on length
                    running_val_loss = np.mean(self.history['val_loss'][-1:])
                    running_val_score = np.mean(self.history['val_score'][-1:])

                    if save and running_val_loss <= best_running_val_loss:
                        best_running_val_loss = running_val_loss
                        # print("Saving new best loss")
                        self.save(f'{self.config.run_name}_best_loss.pt')

                    # TODO - make it so that I can decide whether I'm going for
                    #  max or min
                    if save and running_val_score >= best_running_val_score:
                        best_running_val_score = running_val_score
                        # print("Saving new best score")
                        self.save(f'{self.config.run_name}_best_score.pt')

                self.train_step += 1

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

    def step_scheduler(self):
        self.scheduler.step()

    def validate(self, inspect=False, use_train_loader=False, verbose=True):
        criterion = self.config.criterion
        self.model.eval()
        y_pred = []
        y_true = []
        if inspect and len(self.id_key):
            ids = [] # for debugging purposes
        if not use_train_loader:
            loader = self.data_loaders.val_loader
        else:
            loader = self.data_loaders.train_loader
        val_bar = tqdm(loader, disable=(not verbose))
        for data in val_bar:
            inputs, targets = self.prepare_inputs_and_targets(data, mode='val')
            with torch.no_grad():
                outputs = self.model(*inputs)
            y_true.append(targets.cpu())
            y_pred.append(outputs.cpu())
            if inspect and len(self.id_key):
                ids += list(data[self.id_key])
            
        y_true = torch.cat(y_true, axis=0)
        y_pred = torch.cat(y_pred, axis=0)

        avg_val_loss = self.compute_loss(y_true, y_pred)

        avg_val_score = self.compute_score(y_true, y_pred)

        if verbose:
            print(f'avg_val_loss: {avg_val_loss:.3f}, ' +
                  f'avg_val_score: {avg_val_score:.3f}',
                  flush=True)
        if inspect:
            ret = {'loss': avg_val_loss, 'score': avg_val_score,
                    'y_true': y_true, 'y_pred': y_pred}
            if len(self.id_key):
                ret[self.id_key] = ids
            return ret
        return avg_val_loss, avg_val_score

    def compute_loss(self, y_true, y_pred):
        """ could override this
        """
        return self.config.criterion(y_pred, y_true)

    def compute_score(self, y_true, y_pred):
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
            'history': self.history,
            'config': self.config,
        }
        if self.scheduler is not None:
            save_dct['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(save_dct, os.path.join(self.config.weights_dir, filename))

    def load(self, filename):
        checkpoint = torch.load(os.path.join(self.config.weights_dir, filename),
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