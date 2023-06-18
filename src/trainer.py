import itertools
import timm
import time
import os
import pandas as pd

from tqdm.auto import tqdm
from pathlib import Path
from shutil import rmtree

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum, nn
from torch.utils.data import DataLoader, Dataset, random_split

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from accelerate import (Accelerator, DistributedDataParallelKwargs,
                        DistributedType)
from beartype.door import is_bearable
from beartype.typing import Dict, List, Literal, Optional, Union
from beartype.vale import Is

from functions import compute_binary_scores, compute_multi_scores
from helper import get_set_seed, exists, beartype_jit
from dataset import CNN2dDataset, train_augment, valid_augment_lstm, get_folds_lstm

from optimizer import get_linear_scheduler, get_optimizer
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pickle

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
        return log
    
def cycle(dl):
    while True:
        for data in dl:
            yield data
            
def noop(*args, **kwargs):
    pass
        
class ModelTrainer(nn.Module):
    def __init__(
    self,
    *,
    Model,
    num_train_epoches,
    batch_size,
    time_to,
    train_df,
    valid_df,
    training_config,
    model_config,
    lr=3e-4,
    lr_warmup=0,
    grad_accum_every=2,
    wd=1e-4,
    max_grad_norm = 0.5,
    save_results_every=1,
    save_model_every=1,
    results_folder='results',
    accelerate_kwargs: dict = {},
    config_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.fold = training_config['fold']
        kwargs_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        date = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
        self.results_folder = Path(results_folder)
        self.results_folder = str(self.results_folder) + '/' + 'fold_'+ str(self.fold) + '/' + date
        os.makedirs(self.results_folder, exist_ok=True)
        accelerate_kwargs = {
            **accelerate_kwargs,
            'log_with' : 'wandb',
            'project_dir' : self.results_folder
        }
        self.accelerator = Accelerator(**accelerate_kwargs, kwargs_handlers=[kwargs_handler])
        hps = {"learning_rate": lr}
        self.accelerator.init_trackers("tlvmc", config=hps) 
        self.log_with = accelerate_kwargs['log_with'] if 'log_with' in accelerate_kwargs else None
        
        self.model = Model(
            **model_config
        )
        
        self.register_buffer('steps', torch.Tensor([0]))
        self.register_buffer('epoches', torch.Tensor([0]))
        self.num_train_epoches = num_train_epoches
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        
        # Optimizer
        
        self.optim = get_optimizer(self.model.parameters(), lr = lr, wd =  wd)
        if lr_warmup > 0:
            self.scheduler = get_linear_scheduler(
                self.optim,
                total_iters = lr_warmup
            )
        else : 
            self.scheduler = None
        self.max_grad_norm = max_grad_norm
        
        # Dataset
        
        with open('non_norm.pickle', 'rb') as p:
            npy = pickle.load(p)
        
        # train_df.to_csv(self.results_folder + '/' + 'train_df.csv', index=False)
        # valid_df.to_csv(self.results_folder + '/' + 'valid_df.csv', index=False)
        self.train_ds = LSTMModelDataset(
            df = train_df,
            npy = npy,
            augment = train_augment(),
            time_to = time_to, model_type='current',
        ) 
        self.valid_ds = LSTMModelDataset(
            df = valid_df,
            npy = npy,
            augment = None,
            time_to = time_to, model_type='current',
        )
        

        self.train_dl = DataLoader(self.train_ds, 
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=12 
                                   )
        
        self.valid_dl = DataLoader(self.valid_ds, 
                                   batch_size=batch_size, 
                                   shuffle=False,
                                    num_workers=12
                                   )
        
        (
            self.model,
            self.optim,
            self.train_dl,
            self.valid_dl,
        ) = self.accelerator.prepare(
            self.model,
            self.optim,
            self.train_dl,
            self.valid_dl
        )
        
        if exists(self.scheduler):
            self.scheduler = self.accelerator.prepare(self.scheduler)

        # dataloader iterators

        self.train_dl_iter = cycle(self.train_dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every        
        
    def save(self, model_path, optim_path, scheduler_path=None):
        model_state_dict = self.accelerator.get_state_dict(self.model)
        torch.save(model_state_dict, model_path)

        optim_state_dict = self.optim.state_dict()
        torch.save(optim_state_dict, optim_path)

        if exists(self.scheduler):
            assert exists(scheduler_path)
            scheduler_state_dict = self.scheduler.state_dict()
            torch.save(scheduler_state_dict, scheduler_path)
            
    def load(self, model_path, optim_path, scheduler_path=None, steps=0):
        model_path = Path(model_path)
        optim_path = Path(optim_path)
        assert model_path.exists() and optim_path.exists()

        model_state_dict = torch.load(model_path, map_location=self.device)
        optim_state_dict = torch.load(optim_path, map_location=self.device)
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(model_state_dict)
        self.optim.load_state_dict(optim_state_dict)

        if exists(self.scheduler):
            assert exists(scheduler_path), 'the config specifies lr warmup is used, but no scheduler checkpoint is given. try setting lr_warmup to 0.'
            scheduler_path = Path(scheduler_path)
            assert scheduler_path.exists()
            scheduler_state_dict = torch.load(scheduler_path, map_location=self.device)
            self.scheduler.load_state_dict(scheduler_state_dict)

        if steps > 0:
            assert int(self.steps.item()) == 0, 'steps should be 0 when loading a checkpoint for the first time'
            self.steps += steps 
                    
    def print(self, msg):
        self.accelerator.print(msg)
        
        
    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process
    
    def train_one_step(self, ep):
        steps = int(self.steps.item())
        self.model.train()

        # logs

        logs = {}

        # update 

        for _ in range(self.grad_accum_every):
            batch = next(self.train_dl_iter)
            non_empty_batch = False
            while non_empty_batch is False:
                if len(batch) == 0:
                    continue
                
                batch['signals'] = batch['signals'].to(self.device)
                batch['label'] = batch['label'].to(self.device)
                non_empty_batch = True
                batch = self.model(batch)
                
                #loss = self.accelerator.reduce(batch['bce_loss'], 'mean')
                loss2 = self.accelerator.reduce(batch['focal_loss'], 'mean').mean()
                #loss2 = self.accelerator.reduce(batch['arc_loss'], 'mean').mean()
                self.accelerator.backward(loss = loss2 / self.grad_accum_every)
                accum_log(logs, {'loss': loss2.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()
        
        if exists(self.scheduler):
            self.scheduler.step()
        
        self.accelerator.log({
            "train_loss" : logs['loss'],
            "epoches" : ep,
        },step = steps)
        
        self.steps += 1
        
        return logs, logs['loss']
    
    def valid_one_step(self, ep):
        steps = int(self.steps.item())
        valid_loss = None
        valid_accuracy = None
        non_empty_batch = False
        
        while non_empty_batch is False:
            batch = next(self.valid_dl_iter)
            if len(batch) == 0:
                continue
            non_empty_batch = True

            with torch.no_grad():
                self.model.eval()
                batch['signals'] = batch['signals'].to(self.device)
                batch['label'] = batch['label'].to(self.device)
                batch = self.model(batch)
            #loss1 = self.accelerator.reduce(batch['bce_loss'], 'mean')
            valid_loss = self.accelerator.reduce(batch['focal_loss'], 'mean').mean()
            #valid_loss = self.accelerator.reduce(batch['arc_loss'], 'mean').mean()
            probs = self.accelerator.gather_for_metrics(batch['probability'].contiguous())
            labels = self.accelerator.gather_for_metrics(batch['label'].contiguous())
                        
        self.accelerator.log({
                    "valid_loss" : valid_loss,
                    "epoches" : ep,
                    },step = steps)
                
        self.steps+= 1
        ret = {
            'probability' : probs,
            'labels' : labels,
            'valid_loss' : valid_loss
        }            

        return ret
    
    
    def train_start(self, log_fn = noop):
        print('train start')
        debug = False
        gpus = 1
        for ep in tqdm(range(self.num_train_epoches)):
            losses = []
            #all_steps = len(self.valid_ds) // self.batch_size // gpus
            for i in tqdm(range(len(self.train_ds) // self.batch_size // gpus + 1)):
                logs, loss = self.train_one_step(ep)
                log_fn(logs)
                losses.append(loss)
            print(f'epoch {ep} train_loss : ', np.mean(losses))
            
            if not (ep % self.save_results_every) and not (debug):
                probs = []
                labels = []                
                losses=[]
                for i in tqdm(range(len(self.valid_ds) // self.batch_size // gpus + 1)):
                    ret = self.valid_one_step(ep)
                    probs.append(ret['probability'].detach().cpu().numpy())
                    labels.append(ret['labels'].detach().cpu().numpy())
                    losses.append(ret['valid_loss'].detach().cpu().numpy())                    
                probs = np.concatenate(probs)
                labels = np.concatenate(labels)
                np.save(self.results_folder + '/' + str(ep) + '_prob_fold_' + str(self.fold), probs)
                np.save(self.results_folder + '/' + str(ep) + '_label_fold_' + str(self.fold), labels)
                b_scores = compute_binary_scores(labels, probs)
                losses = np.mean(losses)
                print(f'valid_loss : {losses}')
                #acc_score,  f1_score, recall_score = scores['acc'], scores['f1'], scores['recall']
                m_acc_score, m_f1_score, m_roc_score, m_recall_score, m_map_score = b_scores['acc'], b_scores['roc'], b_scores['f1'], b_scores['recall'], b_scores['score']

                self.accelerator.log({
                "valid_acc_score" : m_acc_score,
                "valid_roc_score" : m_roc_score,
                "valid_f1_score"  : m_f1_score,
                "valid_recall_score" : m_recall_score,
                "valid_map_score" : m_map_score
                },step = ep)
                
                #print(f'ep {ep} acc : {acc_score}, f1 : {f1_score}, recall : {recall_score}')
                print(f'ep {ep} CPC_score acc : {m_acc_score}, roc : {m_roc_score}, f1 : {m_f1_score}, recall : {m_recall_score}')
                print(f'metric_score : {m_map_score}')
            if self.is_main and not (ep % self.save_model_every) and not (debug):
                self.print(f'{ep}: saving model to {str(self.results_folder)}')

                model_path = str(self.results_folder + f'/model_ep{ep}.pt')
                optim_path = str(self.results_folder + f'/optimizer_ep{ep}.pt')
                scheduler_path = str(self.results_folder + f'/scheduler_ep{ep}.pt')

                self.save(model_path, optim_path, scheduler_path)
            self.epoches += 1
        self.print('training complete')
        
