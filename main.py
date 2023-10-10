import argparse
import os
import time

import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

from common.utils.utils import name_with_datetime
from model.rulformer import RULformer

from dataloader.CMAPSS_Transformer_Dataloader import CMAPSSTrainDataset, CMAPSSTestDataset
from torch.nn.utils.rnn import pad_sequence
from mask import rul_evaluation
import torch.nn.functional as F
from loss import js_div
import seaborn as sns
from common.utils.utils import exists


def collate_fn_old(batch_data):
    """

    Args:
        batch_data:

    Returns:
        texta: full cycle
        textb: window-piece
        label: rul
    """
    texta, textb, label = list(zip(*batch_data))

    texta = list(texta)
    # texta.sort(key=lambda x: len(x), reverse=True)

    texta = [torch.flip(t, dims=[0]) for t in texta]  # texta: list, t size is (L, D)
    texta = pad_sequence(texta, batch_first=True, padding_value=PAD_IDX)  
    texta = torch.flip(texta, dims=[1])

    textb = torch.stack(textb)
    label = torch.stack(label)

    return texta, textb, label


def collate_fn(batch_data):
    """

    Args:
        batch_data:

    Returns:
        texta: full cycle
        textb: window-piece
        label: rul
    """
    texta, texta2, textb, label = list(zip(*batch_data))

    texta = list(texta)
    # texta.sort(key=lambda x: len(x), reverse=True)  
    texta = [torch.flip(t, dims=[0]) for t in texta]  # texta: list, t size is (L, D)
    texta = pad_sequence(texta, batch_first=True, padding_value=PAD_IDX)  # tensor
    texta = torch.flip(texta, dims=[1])

    texta2 = list(texta2)
    # texta.sort(key=lambda x: len(x), reverse=True)  
    texta2 = [torch.flip(t, dims=[0]) for t in texta2]  # texta: list, t size is (L, D)
    texta2 = pad_sequence(texta2, batch_first=True, padding_value=PAD_IDX)  # tensor
    texta2 = torch.flip(texta2, dims=[1])

    textb = torch.stack(textb)
    label = torch.stack(label)

    return texta, texta2, textb, label


def train(model, train_loader, test_ds, optimizer, criterion,
          n_epochs: int = 100, step_size: Optional[int] = 100, step_gamma: Optional[float] = 0.1,
          tqdm_interval: Optional[float] = 1., verbose: bool = False, device=None, dtype=None) -> None:
    """ Train the model.

    Args:
        model: the model need to be trained (required).
        train_loader: train_dataset in Dataloader type (required)
        test_ds: test_dataset in Dataset type (required)
        optimizer: optimizer (required)
        criterion: the loss function or the score (required).
        n_epochs: training epochs (optional)
        step_size: step period to cut the lr(optional)
        step_gamma: cut learning rate (optional)
        tqdm_interval: (optional)
        verbose: print after each epoch (optional).
        device: device name (optional)
        dtype: data dtype (optional)
    """
    factory_kwargs = {'device': device, 'dtype': dtype}

    scheduler = StepLR(optimizer, step_size=step_size, gamma=step_gamma)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.
        for step, (batch_src, batch_tgt, batch_y) in enumerate(
                tqdm(train_loader, mininterval=tqdm_interval)
        ):
            batch_src, batch_tgt, batch_y = batch_src.to(device), batch_tgt.to(device), batch_y.to(device)

            src_mask, tgt_mask, src_padding_mask, _ = RULformer.create_mask(batch_src,
                                                                            batch_tgt,
                                                                            batch_first=True,
                                                                            **factory_kwargs)

            # when there is batch_y, model is in training.
            # empirical: (layers, B, T, L), T: window size, L: cycle length
            # sigma: (layers, mini-batch, T, 1)
            output, prior, empirical, _ = model(batch_src, batch_tgt, batch_y,
                                                src_key_padding_mask=src_padding_mask,
                                                memory_key_padding_mask=src_padding_mask,
                                                need_ass_dis=True)

            loss_d = 0
            for i in range(empirical.size(0)):
                loss_d += js_div(empirical[i], prior[i])  # do not need log anymore
            loss_d /= empirical.size(0)
            loss_p = criterion(output, batch_y.unsqueeze(dim=-1))

            loss = loss_p + 0.5 * loss_d
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().detach().numpy()

        epoch_loss = epoch_loss / len(train_loader)
        scheduler.step()

        if test_ds is not None:
            rmse_loss, score = rul_evaluation(model, train_data, test_ds, criterion, max_life=max_life, **factory_kwargs)
            # Print loss information after every epoch
            print('\r epoch: {0:d} / {1:d}, train_loss: {2:.6f}, eval_rmse: {3:.6f}, eval_score: {4:.6f}'.format(
                epoch, n_epochs, epoch_loss, rmse_loss, score))
        else:
            print('\r epoch: {0:d} / {1:d}, err_rul: {2:.6f}'.format(
                epoch, n_epochs, epoch_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transformer RUL')
    # dataset args
    parser.add_argument('--train_dataset', type=str, default='FD001',
                        help='The train_dataset name, FD001_prior - FD004')
    parser.add_argument('--test_dataset', type=str, default='FD001',
                        help='The test_dataset name, FD001_prior - FD004')
    parser.add_argument('--run_name', default=str(time.time()),
                        help='The folder name used to save model, save_path and evaluation metrics. '
                             'This can be set to any word')
    parser.add_argument('--normalization', type=str, default='minmax',
                        help='The normalization method')
    parser.add_argument('--window_size', type=int, default=30,
                        help='Length of sliding time window. decoder input')
    parser.add_argument('--max_life', type=int, default=125,
                        help='mac cycle time in CMAPSS')
    # Training args
    parser.add_argument('--gpu', type=str, default='cuda:0',
                        help='The gpu name used for training and inference, e.g. cuda:0, defaults:cuda:0')
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size, turn down ``batch_size`` when memory is limited. defaults: 64 ')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='The learning rate, defaults: 0.0001')
    parser.add_argument('--epochs', type=int, default=50,
                        help='The number of epochs, default: 100')
    # config args
    parser.add_argument('--max_threads', type=int, default=None,
                        help='The maximum allowed number of threads used by this process')
    parser.add_argument('--save_model', action="store_true",
                        help='Whether save trained model')
    parser.add_argument('--save_every', type=int, default=None,
                        help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--eval', action="store_true",
                        help='Whether to perform evaluation after training')
    parser.add_argument('--showfig', action="store_true",
                        help='Whether to show figure')
    args = parser.parse_args()

    # load dataset
    path = './data/CMAPSS'
    PAD_IDX = 0  # cope with variant length cycle, padding to the same length, used in colloan_fn in dataloader.
    max_life = 125

    print("Loading training sub-dataset {0}...".format(args.train_dataset))
    train_data = CMAPSSTrainDataset(readpath=path, subdataset=args.train_dataset,
                                    norm='minmax',
                                    window_size=args.window_size,
                                    max_life=max_life,
                                    # smoothing=True,
                                    # select_plat=20
                                    )  # only latest 125 sample in each cycle if `drop_plat == True`, call 125 the plat
    print("Loading test sub-dataset {0}...".format(args.test_dataset))

    statistic_kwargs = train_data.statistic_num

    test_data = CMAPSSTestDataset(readpath=path, subdataset=args.test_dataset,
                                  window_size=args.window_size, max_life=max_life,
                                  norm='minmax', **statistic_kwargs)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_old,
                              drop_last=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_old)

    # training configuration parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    factory_kwargs = {'device': device, 'dtype': dtype}
    print('Running device is {0}'.format(device))

    start_time = time.time()

    # get model
    RULformer = RULformer(d_model=14, d_emb=128, nhead=8,
                          num_encoder_layers=1, num_decoder_layers=1,
                          dim_flatten=128 * args.window_size,
                          norm='batch',
                          max_life=max_life,
                          window_size=args.window_size,
                          batch_first=True, **factory_kwargs).to(device)

    # train
    Adam_optimizer = Adam(RULformer.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    train(RULformer, train_loader, test_data, Adam_optimizer, criterion, step_size=100, n_epochs=args.epochs,
          **factory_kwargs)

    # save trained model
    if args.save_model:
        run_dir = os.path.join('training', '{0}_{1}'.format(args.train_dataset, name_with_datetime()))  # Save path
        if not Path(run_dir).is_dir():
            os.makedirs(run_dir, exist_ok=True)
        torch.save(RULformer, os.path.join(run_dir, '{0}.pth'.format(start_time)))

    eval_loss = rul_evaluation(RULformer, train_data, test_data, criterion, **factory_kwargs)
    print('RMSE loss is', eval_loss)
