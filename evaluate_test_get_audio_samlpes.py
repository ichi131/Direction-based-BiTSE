#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:09:09 2020

@author: yoonsanghyu
Edited by: Yichi Wang  2024/08
"""

import argparse
# from mir_eval.separation import bss_eval_sources
from pit_criterion import cal_loss
from collections import OrderedDict
import numpy as np
import torch
import os
import soundfile as sf
from NBSS import NBSS

from data import AudioDataset, EvalAudioDataLoader
from FaSNet import FaSNet_TAC
from pathlib import Path

def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3: # [B, C, T]
            results.append(input[:,:length].view(C, -1).cpu().numpy())
        elif dim == 2:  # [B, T]  
            results.append(input[:length].view(-1).cpu().numpy())
    return results


parser = argparse.ArgumentParser('Evaluate separation performance using FaSNet + TAC')
parser.add_argument('--model_path', type=str, default='/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/exp/prop2/epoch64.pth.tar', help='Path to model file created by training') # prop2
parser.add_argument('--cal_sdr', type=int, default=0, help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--use_cuda', type=int, default=1, help='Whether use GPU to separate speech')

# General config
# Task related
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')


# Network architecture
parser.add_argument('--enc_dim', default=64, type=int, help='Number of filters in autoencoder')
parser.add_argument('--win_len', default=4, type=int, help='Number of convolutional blocks in each repeat')
parser.add_argument('--context_len', default=16, type=int, help='context window size')
parser.add_argument('--feature_dim', default=64, type=int, help='feature dimesion')
parser.add_argument('--hidden_dim', default=128, type=int, help='Hidden dimension')
parser.add_argument('--layer', default=4, type=int, help='Number of layer in dprnn step')
parser.add_argument('--segment_size', default=50, type=int, help="segment_size")
parser.add_argument('--nspk', default=1, type=int, help='Maximum number of speakers')
parser.add_argument('--mic', default=6, type=int, help='number of microphone')


def evaluate(args):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0
    tag = args.model_path.split("/")[-2]
    print(tag)
    # Load model

    model = NBSS(n_channel=2,
                n_speaker=2,
                arch="NBC2",
                arch_kwargs={
                    "n_layers": 8, # 12 for large
                    "dim_hidden": 96, # 192 for large
                    "dim_ffn": 192, # 384 for large
                    "block_kwargs": {
                        'n_heads': 2,
                        'dropout': 0,
                        'conv_kernel_size': 3,
                        'n_conv_groups': 8,
                        'norms': ("LN", "GBN", "GBN"),
                        'group_batch_norm_kwargs': {
                            'group_size': 257, # 129 for 8k Hz
                            'share_along_sequence_dim': False,
                        },
                    }
                },)
    
    if args.use_cuda:
        #model = torch.nn.DataParallel(model)
        model.cuda()
    
    # model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    model_info = torch.load(args.model_path)
    new_state_dict = OrderedDict()
    for k, v in model_info["model_state_dict"].items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name] = v
    model_info["model_state_dict"] = new_state_dict

    try:
        model.load_state_dict(model_info['model_state_dict'])
    except KeyError:
        state_dict = OrderedDict()
        for k, v in model_info['model_state_dict'].items():
            # name = k.replace("module.", "")    # remove 'module.'
            state_dict[k] = v
        model.load_state_dict(state_dict)
    
    # print(model)
    model.eval()

    # Load data    
    dataset = AudioDataset('tt', batch_size = 1, sample_rate = args.sample_rate, nmic = args.mic)
    data_loader = EvalAudioDataLoader(dataset, batch_size=1, num_workers=8)
    
    sisnr_array=[]
    sdr_array=[]
    with torch.no_grad():
        folder_path_enhance = Path("/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/"+ tag + "/output_audio_enhance")
        folder_path_enhance.mkdir(parents=True, exist_ok=True)

        folder_path_label = Path("/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/"+ tag + "/output_audio_label")
        folder_path_label.mkdir(parents=True, exist_ok=True)

        folder_path_noisy = Path("/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/"+ tag + "/output_audio_noisy")
        folder_path_noisy.mkdir(parents=True, exist_ok=True)
        for i, (data) in enumerate(data_loader):
            # Get batch data
            padded_mixture, mixture_lengths, padded_source, theta, fi = data
            tmp = padded_mixture
            if args.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
                theta = theta.cuda()
                fi = fi.cuda()
            
            x = torch.rand(2, 6, 32000)
            none_mic = torch.zeros(1).type(x.type())        
            # Forward
            estimate_source = model(padded_mixture.float(), theta, fi)  # [M, C, T]    
            # padded_source = torch.unsqueeze(padded_source[:, 0,:], dim=1)    
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
                
            # M,_,T = padded_mixture.shape    
            # mixture_ref = torch.chunk(padded_mixture, args.mic, dim =1)[0] #[M, ch, T] -> [M, 1, T]
            # mixture_ref = mixture_ref.view(M,T) #[M, 1, T] -> [M, T]
            
            mixture = remove_pad(padded_mixture, mixture_lengths)
            source = remove_pad(padded_source, mixture_lengths)
            estimate_source = remove_pad(reorder_estimate_source, mixture_lengths)
            
            # for each utterance
            for mix, src_ref, src_est in zip(mixture, source, estimate_source):
                sample_rate = 16000  # You can adjust this according to your needs

                # Write the NumPy array to a WAV file
                sf.write('/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/'+ tag + '/output_audio_enhance/'+str(total_cnt)+'l.wav', src_est[0], sample_rate)
                sf.write('/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/'+ tag + '/output_audio_enhance/'+str(total_cnt)+'r.wav', src_est[1], sample_rate)
                sf.write('/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/'+ tag + '/output_audio_label/'+str(total_cnt)+'l.wav', src_ref[0], sample_rate)
                sf.write('/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/'+ tag + '/output_audio_label/'+str(total_cnt)+'r.wav', src_ref[1], sample_rate)
                sf.write('/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/'+ tag + '/output_audio_noisy/'+str(total_cnt)+'l.wav', mix[0], sample_rate)
                sf.write('/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/'+ tag + '/output_audio_noisy/'+str(total_cnt)+'r.wav', mix[1], sample_rate)
                
                avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                total_SISNRi += avg_SISNRi
                sisnr_array.append(avg_SISNRi)
                total_cnt += 1
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
        
    # np.save('sisnr.npy',np.array(sisnr_array))
    # np.save('sdr.npy',np.array(sdr_array))
    print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix[0])
    sisnr2b = cal_SISNR(src_ref[1], mix[1])
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    evaluate(args)
