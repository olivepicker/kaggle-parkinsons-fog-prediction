import sys
import torch
import argparse
import os
import pandas as pd
from trainer import ModelTrainer
from models.model_cls import Net as clsModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from helper import get_set_seed, exists, beartype_jit
from dataset import CNN2dModelDataset, train_augment, valid_augment_lstm, get_folds_lstm



if __name__ == "__main__":
    seed = get_set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True, type=int)
    df = pd.read_parquet('train_5fold.parquet')
    args = parser.parse_args()

    train_df = df[df['fold']!=args.fold].reset_index(drop=True)
    valid_df = df[df['fold']==args.fold].reset_index(drop=True)
    train_df = pd.concat([train_df, valid_df]).reset_index(drop=True)
    print(f'fold {args.fold} train len : {len(train_df)}, valid len : {len(valid_df)}')
    trainer = ModelTrainer(
        Model = clsModel,
        train_df = train_df,
        valid_df = valid_df,
        time_to = 3600,
        num_train_epoches = 4,
        batch_size = 1024,
        training_config={
            'fold' : args.fold,
        },
        model_config = {
            'model_cfg' : {
                'model_name' : 'convnext_tiny',
                'pretrained' : True,
                'in_chans' : 3,
                'num_classes' : 3,
                'drop_rate' : 0.7,
                'drop_path_rate' : 0.4}
            },
        lr=1e-4,
        lr_warmup=0,
        grad_accum_every=2,
        wd=1e-4,
        max_grad_norm = 0.5,
        save_results_every=1,
        save_model_every=1,
        results_folder='results/cls/',
        accelerate_kwargs ={
            'project_dir' : './logs/'
        } 
    ).to(device)
    print('start')
    trainer.train_start()