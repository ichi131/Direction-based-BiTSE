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
import pickle
from data import AudioDataset, EvalAudioDataLoader
from pesq import pesq
import numpy as np
import math

def gcc_phat(sig1, sig2, fs, max_tau=None, interp=16):
    # 计算信号的FFT
    n = sig1.size + sig2.size
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    # 计算互相关
    R = SIG1 * np.conj(SIG2)
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    # 找到最大值所在位置
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau

def calculate_itd_error(s_l, s_r, ts_l, ts_r, fs=16000):
    itd_clean = gcc_phat(s_l, s_r, fs)
    itd_separated = gcc_phat(ts_l, ts_r, fs)
    return np.abs(itd_clean - itd_separated)

def calculate_ild_error(s_l, s_r, ts_l, ts_r):
    ild_clean = 10 * np.log10(np.sum(s_l**2) / np.sum(s_r**2))
    ild_separated = 10 * np.log10(np.sum(ts_l**2) / np.sum(ts_r**2))
    return np.abs(ild_clean - ild_separated)

def stft(x, fs, frame_size, hop_size):
    # 窗函数
    window = np.hamming(frame_size)
    # 总帧数
    num_frames = 1 + int((len(x) - frame_size) / hop_size)
    # 频域结果
    frames = np.zeros((num_frames, frame_size), dtype=complex)
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frames[i, :] = np.fft.fft(window * x[start:end])
    
    return frames

def istft(X, fs, frame_size, hop_size):
    # 输出信号
    output_signal = np.zeros((X.shape[0] * hop_size + frame_size))
    window = np.hamming(frame_size)
    
    for i in range(X.shape[0]):
        start = i * hop_size
        end = start + frame_size
        output_signal[start:end] += np.real(np.fft.ifft(X[i, :])) * window
    
    return output_signal

def estoi(clean, denoised, fs=16000, frame_size=512, hop_size=256):
    Zxx_clean = stft(clean, fs, frame_size, hop_size)
    Zxx_denoised = stft(denoised, fs, frame_size, hop_size)

    P_clean = np.abs(Zxx_clean) ** 2
    P_denoised = np.abs(Zxx_denoised) ** 2

    num = np.sum(np.sqrt(P_clean * P_denoised))
    denom = np.sqrt(np.sum(P_clean) * np.sum(P_denoised))
    estoi_val = num / denom if denom != 0 else 0

    return estoi_val



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
parser.add_argument('--model_path', type=str, default='/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/exp/cdf/epoch53.pth.tar', help='Path to model file created by training')  # cdf
# parser.add_argument('--model_path', type=str, default='/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/exp/cdf-ipd/epoch48.pth.tar', help='Path to model file created by training') # cdf ipd
# parser.add_argument('--model_path', type=str, default='/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/exp/cdf-ipd-stft/epoch53.pth.tar', help='Path to model file created by training') # cdf ipd stft
# parser.add_argument('--model_path', type=str, default='/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/exp/prop2/epoch64.pth.tar', help='Path to model file created by training') # prop2
# parser.add_argument('--model_path', type=str, default='/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/exp/prop1-2/epoch49.pth.tar', help='Path to model file created by training') # prop1
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
    # total_SDRi = 0

    # total_SISNRi25 = 0
    # total_pesq25 = 0
    # total_cnt25 = 0

    # total_SISNRi50 = 0
    # total_pesq50 = 0
    # total_cnt50 = 0

    # total_SISNRi75 = 0
    # total_pesq75 = 0
    # total_cnt75 = 0

    # total_SISNRi100 = 0
    # total_pesq100 = 0
    # total_cnt100 = 0

    total_SISNRiall = 0
    total_pesqnall = 0
    total_pesqwall = 0
    total_stoiall = 0
    total_cntall = 0 

    total_SISNRia15 = 0
    total_pesqna15 = 0
    total_pesqwa15 = 0
    total_stoia15 = 0
    total_cnta15 = 0

    total_SISNRia45 = 0
    total_pesqna45 = 0
    total_pesqwa45 = 0
    total_stoia45 = 0
    total_cnta45 = 0

    total_SISNRia90 = 0
    total_pesqna90 = 0
    total_pesqwa90 = 0
    total_stoia90 = 0
    total_cnta90 = 0

    total_SISNRia180 = 0
    total_pesqna180 = 0
    total_pesqwa180 = 0
    total_stoia180 = 0
    total_cnta180 = 0

    none_counter = 0

    # Load model
    with open("/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/data_script/tt.scp", 'r') as f:
        lines = f.readlines()

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
        for i, (data) in enumerate(data_loader):
            # Get batch data
            padded_mixture, mixture_lengths, padded_source, tgt_theta_angle, tgt_fi_angle = data

            spk_angle = abs(int(os.path.basename(lines[i]).split("_")[3]) - int(os.path.basename(lines[i]).split("_")[6].replace(".wav", "")))
            tmp = padded_mixture
            if args.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
                tgt_theta_angle = tgt_theta_angle.cuda()
                tgt_fi_angle = tgt_fi_angle.cuda()
            
            x = torch.rand(2, 6, 32000)
            none_mic = torch.zeros(1).type(x.type())        
            # Forward
            estimate_source = model(padded_mixture.float(), tgt_theta_angle, tgt_fi_angle)  # [M, C, T]    
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
                # Compute SDRi
                # if args.cal_sdr:
                #     avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                #     total_SDRi += avg_SDRi
                #     sdr_array.append(avg_SDRi)
                #     print("\tSDRi={0:.2f}".format(avg_SDRi))
                # # Compute SI-SNRi
                # if configs[i]['overlap_ratio'] < 0.25:
                #     avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                #     score = (pesq(src_ref[0], src_est[0]) + pesq(src_ref[1], src_est[1]))/2
                #     # print("\tPESQ=",score)
                #     # print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                #     total_SISNRi25 += avg_SISNRi
                #     total_pesq25 += score
                #     sisnr_array.append(avg_SISNRi)
                #     total_cnt25 += 1

                # elif 0.25 <= configs[i]['overlap_ratio'] < 0.5:
                #     avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                #     score = (pesq(src_ref[0], src_est[0]) + pesq(src_ref[1], src_est[1]))/2
                #     # print("\tPESQ=",score)
                #     # print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                #     total_SISNRi50 += avg_SISNRi
                #     total_pesq50 += score
                #     sisnr_array.append(avg_SISNRi)
                #     total_cnt50 += 1

                # elif 0.5 <= configs[i]['overlap_ratio'] < 0.75:
                #     avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                #     score = (pesq(src_ref[0], src_est[0]) + pesq(src_ref[1], src_est[1]))/2
                #     # print("\tPESQ=",score)
                #     # print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                #     total_SISNRi75 += avg_SISNRi
                #     total_pesq75 += score
                #     sisnr_array.append(avg_SISNRi)
                #     total_cnt75 += 1

                # else:
                #     avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                #     score = (pesq(src_ref[0], src_est[0]) + pesq(src_ref[1], src_est[1]))/2
                #     # print("\tPESQ=",score)
                #     # print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                #     total_SISNRi100 += avg_SISNRi
                #     total_pesq100 += score
                #     sisnr_array.append(avg_SISNRi)
                #     total_cnt100 += 1

                if spk_angle < 15:
                    avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                    scorew = (pesq(16000, src_ref[0], src_est[0],'wb') + pesq(16000, src_ref[1], src_est[1],'wb'))/2
                    scoren = (pesq(16000, src_ref[0], src_est[0],'nb') + pesq(16000, src_ref[1], src_est[1],'nb'))/2
                    scores = (estoi(src_ref[0], src_est[0]) + estoi(src_ref[1], src_est[1]))/2
                    # itd_error = calculate_itd_error(src_ref[0], src_ref[1], src_est[0], src_est[1])
                    # ild_error = calculate_ild_error(src_ref[0], src_ref[1], src_est[0], src_est[1])
                    # print("\tPESQ=",score)
                    # print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                    total_SISNRia15 += avg_SISNRi
                    total_pesqwa15 += scorew
                    total_pesqna15 += scoren
                    total_stoia15 += scores
                    # total_itd15 += itd_error
                    # total_ild15 += ild_error
                    sisnr_array.append(avg_SISNRi)
                    total_cnta15 += 1

                elif 15 <= spk_angle < 45:
                    avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                    scorew = (pesq(16000, src_ref[0], src_est[0],'wb') + pesq(16000, src_ref[1], src_est[1],'wb'))/2
                    scoren = (pesq(16000, src_ref[0], src_est[0],'nb') + pesq(16000, src_ref[1], src_est[1],'nb'))/2
                    scores = (estoi(src_ref[0], src_est[0]) + estoi(src_ref[1], src_est[1]))/2
                    # itd_error = calculate_itd_error(src_ref[0], src_ref[1], src_est[0], src_est[1])
                    # ild_error = calculate_ild_error(src_ref[0], src_ref[1], src_est[0], src_est[1])
                    # print("\tPESQ=",score)
                    # print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                    total_SISNRia45 += avg_SISNRi
                    total_pesqwa45 += scorew
                    total_pesqna45 += scoren
                    total_stoia45 += scores
                    # total_itd45 += itd_error
                    # total_ild45 += ild_error
                    sisnr_array.append(avg_SISNRi)
                    total_cnta45 += 1
                
                elif 45 <= spk_angle < 90:
                    avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                    scorew = (pesq(16000, src_ref[0], src_est[0],'wb') + pesq(16000, src_ref[1], src_est[1],'wb'))/2
                    scoren = (pesq(16000, src_ref[0], src_est[0],'nb') + pesq(16000, src_ref[1], src_est[1],'nb'))/2
                    scores = (estoi(src_ref[0], src_est[0]) + estoi(src_ref[1], src_est[1]))/2
                    # itd_error = calculate_itd_error(src_ref[0], src_ref[1], src_est[0], src_est[1])
                    # ild_error = calculate_ild_error(src_ref[0], src_ref[1], src_est[0], src_est[1])
                    # print("\tPESQ=",score)
                    # print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                    total_SISNRia90 += avg_SISNRi
                    total_pesqwa90 += scorew
                    total_pesqna90 += scoren
                    total_stoia90 += scores
                    # total_itd90 += itd_error
                    # total_ild90 += ild_error
                    sisnr_array.append(avg_SISNRi)
                    total_cnta90 += 1

                else:
                    avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)

                    scorew = (pesq(16000, src_ref[0], src_est[0],'wb') + pesq(16000, src_ref[1], src_est[1],'wb'))/2
                    scoren = (pesq(16000, src_ref[0], src_est[0],'nb') + pesq(16000, src_ref[1], src_est[1],'nb'))/2
                    scores = (estoi(src_ref[0], src_est[0]) + estoi(src_ref[1], src_est[1]))/2
                    # itd_error = calculate_itd_error(src_ref[0], src_ref[1], src_est[0], src_est[1])
                    # ild_error = calculate_ild_error(src_ref[0], src_ref[1], src_est[0], src_est[1])
                    # print("\tPESQ=",score)
                    # print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                    total_SISNRia180 += avg_SISNRi
                    total_pesqwa180 += scorew
                    total_pesqna180 += scoren
                    total_stoia180 += scores
                    # total_itd180 += itd_error
                    # total_ild180 += ild_error
                    sisnr_array.append(avg_SISNRi)
                    total_cnta180 += 1

                total_SISNRiall += avg_SISNRi
                total_pesqwall += scorew
                total_pesqnall += scoren
                total_stoiall += scores
                # total_itd += itd_error
                # total_ild += ild_error
                total_cntall += 1
                print(total_cntall, avg_SISNRi)
                # print(total_SISNRiall/total_cntall)
                # print(total_pesqwall/total_cntall)
                # print(total_pesqnall/total_cntall)
                # print(total_stoiall/total_cntall)
                # print(total_itd/total_cntall)
                # print(total_ild/total_cntall)
   
    # if args.cal_sdr:
    #     print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
        
    np.save('sisnr.npy',np.array(sisnr_array))
    np.save('sdr.npy',np.array(sdr_array))

    # print("Utt", total_cnt25)
    # print("Average SISNR improvement 25: {0:.2f}".format(total_SISNRi25 / total_cnt25))
    # print("Average PESQ improvement 25: {0:.2f}".format(total_pesq25 / total_cnt25))

    # print("Utt", total_cnt50)
    # print("Average SISNR improvement 50: {0:.2f}".format(total_SISNRi50 / total_cnt50))
    # print("Average PESQ improvement 50: {0:.2f}".format(total_pesq50 / total_cnt50))

    # print("Utt", total_cnt75)
    # print("Average SISNR improvement 75: {0:.2f}".format(total_SISNRi75 / total_cnt75))
    # print("Average PESQ improvement 75: {0:.2f}".format(total_pesq75 / total_cnt75))

    # print("Utt", total_cnt100)     
    # print("Average SISNR improvement 100: {0:.2f}".format(total_SISNRi100 / total_cnt100))
    # print("Average PESQ improvement 100: {0:.2f}".format(total_pesq100 / total_cnt100))

    print("Utt", total_cnta15)
    print("Average SISNR improvement 15: {0:.2f}".format(total_SISNRia15 / total_cnta15))
    print("Average WB-PESQ improvement 15: {0:.2f}".format(total_pesqwa15 / total_cnta15))
    print("Average NB-PESQ improvement 15: {0:.2f}".format(total_pesqna15 / total_cnta15))
    print("Average STOI improvement 15: {0:.2f}".format(total_stoia15 / total_cnta15))
    # print("Average ILD improvement 15: {0:.2f}".format(total_ild15 / total_cnta15))
    # print("Average ITD improvement 15: {0:.2f}".format(total_itd15 / total_cnta15))

    print("Utt", total_cnta45)
    print("Average SISNR improvement 45: {0:.2f}".format(total_SISNRia45 / total_cnta45))
    print("Average WB-PESQ improvement 45: {0:.2f}".format(total_pesqwa45 / total_cnta45))
    print("Average NB-PESQ improvement 45: {0:.2f}".format(total_pesqna45 / total_cnta45))
    print("Average STOI improvement 45: {0:.2f}".format(total_stoia45 / total_cnta45))
    # print("Average ILD improvement 45: {0:.2f}".format(total_ild45 / total_cnta45))
    # print("Average ITD improvement 45: {0:.2f}".format(total_itd45 / total_cnta45))

    print("Utt", total_cnta90)
    print("Average SISNR improvement 90: {0:.2f}".format(total_SISNRia90 / total_cnta90))
    print("Average WB-PESQ improvement 90: {0:.2f}".format(total_pesqwa90 / total_cnta90))
    print("Average NB-PESQ improvement 90: {0:.2f}".format(total_pesqna90 / total_cnta90))
    print("Average STOI improvement 90: {0:.2f}".format(total_stoia90 / total_cnta90))
    # print("Average ILD improvement 90: {0:.2f}".format(total_ild90 / total_cnta90))
    # print("Average ITD improvement 90: {0:.2f}".format(total_itd90 / total_cnta90))

    print("Utt", total_cnta180)     
    print("Average SISNR improvement 180: {0:.2f}".format(total_SISNRia180 / total_cnta180))
    print("Average WB-PESQ improvement 180: {0:.2f}".format(total_pesqwa180 / total_cnta180))
    print("Average NB-PESQ improvement 180: {0:.2f}".format(total_pesqna180 / total_cnta180))
    print("Average STOI improvement 180: {0:.2f}".format(total_stoia180 / total_cnta180))
    # print("Average ILD improvement 180: {0:.2f}".format(total_ild180 / total_cnta180))
    # print("Average ITD improvement 180: {0:.2f}".format(total_itd180 / total_cnta180))

    print("Utt", total_cntall)     
    print("Average SISNR improvement all: {0:.2f}".format(total_SISNRiall / total_cntall))
    print("Average WB-PESQ improvement all: {0:.2f}".format(total_pesqwall / total_cntall))
    print("Average NB-PESQ improvement all: {0:.2f}".format(total_pesqnall / total_cntall))
    print("Average STOI improvement all: {0:.2f}".format(total_stoiall / total_cntall))
    # print("Average ILD improvement all: {0:.2f}".format(total_ildall / total_cntall))
    # print("Average ITD improvement all: {0:.2f}".format(total_itdall / total_cntall))

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