# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:20:37 2020

Created on 2018/12
Author: Kaituo XU

Edited by: yoonsanghyu  2020/04
Edited by: Yichi Wang  2024/08

"""

import os
import time

import torch
import numpy as np
import torch.distributed as dist
from pit_criterion import cal_loss
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr

def neg_si_sdr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch_size = target.shape[0]
    si_snr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_snr_val.view(batch_size, -1), dim=1)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Solver(object):
    
    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer

        
        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        # logging
        self.print_freq = args.print_freq
        self.local_rank = args.local_rank
        self.nprocs = args.nprocs
        # loss
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            cont = torch.load(self.continue_from, map_location="cpu")
            self.start_epoch = cont['epoch']
            self.model.load_state_dict(cont['model_state_dict'])
            self.optimizer.load_state_dict(cont['optimizer_state'])
            torch.set_rng_state(cont['trandom_state'])
            np.random.set_state(cont['nrandom_state'])
            del cont
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.best_val_loss = float("inf")
        # self.best_val_loss = -21.754
        self.halving = False
        self.val_no_impv = 0

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            optim_state = self.optimizer.state_dict()
            print('epoch start Learning rate: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
            print("Training...")
            
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0:5d} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            if self.local_rank == 0:
                if self.checkpoint:
                    file_path = os.path.join(
                        self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'trandom_state': torch.get_rng_state(),
                        'nrandom_state': np.random.get_state()}, file_path)
                    print('Saving checkpoint model to %s' % file_path)

            
            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Adjust learning rate (halving)
            if self.half_lr:
                if val_loss >= self.best_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 3:
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No imporvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_file_path = os.path.join(
                    self.save_folder, 'temp_best.pth.tar')
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, best_file_path)
                print("Find better validated model, saving to %s" % best_file_path)

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        losses = AverageMeter('Loss')
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        end = time.time()
        for i, (data) in enumerate(data_loader):
            padded_mixture, mixture_lengths, padded_source, tgt_theta_angle, tgt_fi_angle = data 
            if self.use_cuda:
                padded_mixture = padded_mixture.cuda(self.local_rank, non_blocking=True)
                mixture_lengths = mixture_lengths.cuda(self.local_rank, non_blocking=True)
                padded_source = padded_source.cuda(self.local_rank, non_blocking=True)
                tgt_theta_angle = tgt_theta_angle.cuda(self.local_rank, non_blocking=True)
                tgt_fi_angle = tgt_fi_angle.cuda(self.local_rank, non_blocking=True)
            
            # print(tgt_fi_angle)
            estimate_source = self.model(padded_mixture, tgt_theta_angle, tgt_fi_angle)
            padded_source1 = torch.unsqueeze(padded_source[:, 0,:], dim=1)
            padded_source2 = torch.unsqueeze(padded_source[:, 1,:], dim=1)
            estimate_source1 = torch.unsqueeze(estimate_source[:, 0,:], dim=1)
            estimate_source2 = torch.unsqueeze(estimate_source[:, 1,:], dim=1)
            neg_sisdr_loss1, best_perm = pit(preds=estimate_source1, target=padded_source1, metric_func=neg_si_sdr, eval_func='min')
            neg_sisdr_loss2, best_perm = pit(preds=estimate_source2, target=padded_source2, metric_func=neg_si_sdr, eval_func='min')
            loss = neg_sisdr_loss1.mean() + neg_sisdr_loss2.mean()
            # estimate_source = self.model(padded_mixture, none_mic.long(), tgt_theta_angle, tgt_fi_angle)
            # padded_source = torch.unsqueeze(padded_source[:, 0,:], dim=1)

            # loss, max_snr, estimate_source, reorder_estimate_source = \
            #     cal_loss(padded_source, estimate_source, mixture_lengths)

            torch.distributed.barrier()
            reduced_loss = self.reduce_mean(loss, self.nprocs)
            losses.update(reduced_loss.item(), padded_mixture.size(0))

            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            if i % self.print_freq == 0:
                print('Epoch {0:3d} | Iter {1:5d} | Average Loss {2:3.3f} | '
                      'Current Loss {3:3.6f} | {4:5.1f} ms/batch'.format(
                          epoch + 1, i + 1, losses.avg,
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)
            
        del padded_mixture, mixture_lengths, padded_source, \
            loss
            
        if self.use_cuda: torch.cuda.empty_cache()

        return losses.avg

    def reduce_mean(self, tensor, nprocs):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt
