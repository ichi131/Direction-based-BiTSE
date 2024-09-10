# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:33:56 2020

@author: yoonsanghyu
Edited by: Yichi Wang  2024/08
"""

import random
import os
import numpy as np
import torch
import torch.utils.data as data
import soundfile as sf

def extract_samples(*arrays):
    """
    从多个长度相同的numpy数组中随机抽取连续的64000个点。如果数组长度小于64000，使用0填充。
    
    Parameters:
        *arrays (numpy.ndarray): 可变数量的numpy数组。
    
    Returns:
        list of numpy.ndarray: 抽取并可能填充后的数组列表。
    """
    # 确定数组长度
    N = arrays[0].size
    
    # 确定起始索引
    if N < 64000:
        start_index = 0  # 如果数组长度小于64000，则从0开始抽取
        extract_length = N  # 实际抽取长度小于64000时为N
    else:
        start_index = np.random.randint(0, N - 64000 + 1)
        extract_length = 64000
    
    # 从每个数组中抽取64000个点，如果长度不够则用0填充
    extracted_arrays = []
    for arr in arrays:
        if N < 64000:
            # 如果数组长度小于64000，用0填充
            padded_array = np.zeros(64000, dtype=arr.dtype)
            padded_array[:extract_length] = arr
            extracted_arrays.append(padded_array)
        else:
            # 抽取从start_index开始的64000个点
            extracted_arrays.append(arr[start_index:start_index + 64000])
    
    return extracted_arrays

# read 'tr' or 'val' or 'test' mixture path
def read_scp(opt_data):

    mix_scp = '/train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare/data_script/{0}.scp'.format(opt_data)
    lines = open(mix_scp, 'r').readlines()

    scp_dict= []
    for l in lines:
        scp_parts = l.strip()
        scp_dict.append(scp_parts)
        
    return scp_dict

# put data path in batch
class AudioDataset(data.Dataset):
    def __init__(self, opt_data, batch_size = 3, sample_rate=16000, nmic=2):
        super(AudioDataset, self).__init__()
        '''
        opt_data : 'tr', 'val', 'test'
        batch_size : default 3
        sample_rate : 16000
        nmic : # of channel ex) fixed :6mic
        nsample : all sample/nmic
        
        '''
      
        #read data path
        mix_scp = read_scp(opt_data)

        if opt_data == 'tr':
            nsample = 20000
        if opt_data == 'tt':
            nsample = 2999
        if opt_data == 'cv':
            nsample = 5000

        minibatch = []
        mix = []
        end = 0
        while end < nsample:
            num_segments = 0  
            mix = []
            while num_segments < batch_size and end < nsample:
                # mix.append(os.path.join(mix_path,'sample{0}'.format(end)))
                mix.append(mix_scp[end])
                num_segments += 1
                end += 1
            minibatch.append([mix])
            
        self.minibatch = minibatch
                            
    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)
    
    
# read wav file in batch for tr, val      
class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
        
def _collate_fn(batch):
    
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mix_torch: B x ch x T, torch.Tensor
        ilens_torch : B, torch.Tentor
        src_torch: B x C x T, torch.Tensor
        
    ex)
    torch.Size([3, 6, 64000])
    tensor([64000, 64000, 64000], dtype=torch.int32)
    torch.Size([3, 2, 64000])
    """
    
    sr = 16000
    nmic = 2

    #assert len(batch) == 1
    
    total_mix = []
    total_src_tgt = []
    total_theta_tgt_angle = []
    total_fi_tgt_angle = []
    for i in batch[0][0]:
        tgt_theta_angle_list=[]
        tgt_fi_angle_list=[]
        # angle_tgt_path = os.path.join(i, 'tgt_angle.txt')

        # with open(angle_tgt_path, "r") as f:
        #     tmp = f.read()
        #     spk_tgt_idx = int(tmp.split("\n")[0].split(":")[1])+1
        #     tgt_theta_angle = float(tmp.split("\n")[1].split("\t")[0])
        #     tgt_fi_angle = float(tmp.split("\n")[1].split("\t")[1])
        #     tgt_theta_angle_list.append(tgt_theta_angle)
        #     tgt_fi_angle_list.append(tgt_fi_angle)
        spk_tgt_num = random.randint(0,1)
        spk_tgt, _ = sf.read(i.replace("mix", "s"+str(spk_tgt_num+1)))
        mix, _ = sf.read(i)
        src_tgt_l, src_tgt_r, mix_l, mix_r =extract_samples(spk_tgt[:, 0], spk_tgt[:, 1], mix[:, 0], mix[:, 1])
        
        src_tgt_list = [src_tgt_l, src_tgt_r]
        mix_list = [mix_l, mix_r]

        if spk_tgt_num == 0:
            tgt_theta_angle = np.mod(float(os.path.basename(i).split("_")[2]), 360)
            tgt_fi_angle = np.mod(float(os.path.basename(i).split("_")[3]), 360)
            tgt_theta_angle_list.append(tgt_theta_angle)
            tgt_fi_angle_list.append(tgt_fi_angle)
        else:
            tgt_theta_angle = np.mod(float(os.path.basename(i).split("_")[5]), 360)
            tgt_fi_angle = np.mod(float(os.path.basename(i).split("_")[6].replace(".wav","")), 360)
            tgt_theta_angle_list.append(tgt_theta_angle)
            tgt_fi_angle_list.append(tgt_fi_angle)
        

        # for n in range(1, 7):
        #     if n == 1 or n == 4:
        #         spk_tgt_idx_path = os.path.join(i,'spk' + str(spk_tgt_idx) + '_mic{0}.wav'.format(n))
        #         spk_tgt, _ = sf.read(spk_tgt_idx_path)
        #         src_tgt_list.append(spk_tgt)
        #         mix_path = os.path.join(i,'mixture_mic{0}.wav'.format(n)) 
        #         mix, _ = sf.read(mix_path)
        #         mix_list.append(mix)
                
        src_tgt_np = np.asarray(src_tgt_list,dtype=np.float32)
        mix_np = np.asarray(mix_list,dtype=np.float32)
        tgt_theta_angle_np = np.asarray(tgt_theta_angle_list, dtype=np.float32)
        tgt_fi_angle_np = np.asarray(tgt_fi_angle_list, dtype=np.float32)
    
        total_mix.append(mix_np)
        total_src_tgt.append(src_tgt_np)
        total_theta_tgt_angle.append(tgt_theta_angle_np)
        total_fi_tgt_angle.append(tgt_fi_angle_np)
        
    total_mix_np = np.asarray(total_mix,dtype=np.float32)
    total_src_tgt_np = np.asarray(total_src_tgt,dtype=np.float32)
    total_theta_tgt_angle_np = np.asarray(total_theta_tgt_angle, dtype=np.float32)
    total_fi_tgt_angle_np =np.asarray(total_fi_tgt_angle, dtype=np.float32)

    mix_torch = torch.from_numpy(total_mix_np)
    src_tgt_torch = torch.from_numpy(total_src_tgt_np)
    angle_theta_tgt_torch = torch.from_numpy(total_theta_tgt_angle_np)
    angle_fi_tgt_torch = torch.from_numpy(total_fi_tgt_angle_np)

    ilens = np.array([mix.shape[1] for mix in mix_torch])
    ilens_torch = torch.from_numpy(ilens)

    return mix_torch, ilens_torch, src_tgt_torch, angle_theta_tgt_torch, angle_fi_tgt_torch


# read wav file in batch for test   
# class EvalAudioDataLoader(data.DataLoader):
    
#     """
#     NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
#     """

#     def __init__(self, *args, **kwargs):
#         super(EvalAudioDataLoader, self).__init__(*args, **kwargs)
#         self.collate_fn = _collate_fn_eval
        
# def _collate_fn_eval(batch):

#     # batch should be located in list    
#     """
#     Args:
#         batch: list, len(batch) = 1. See AudioDataset.__getitem__()
#     Returns:
#         mix_torch: B x ch x T, torch.Tensor
#         ilens_torch : B, torch.Tentor
#         src_torch: B x C x T, torch.Tensor
        
#     ex)
#     torch.Size([3, 6, 64000])
#     tensor([64000, 64000, 64000], dtype=torch.int32)
#     torch.Size([3, 2, 64000])
#     """
    
#     sr = 16000
#     nmic = 2

#     #assert len(batch) == 1
    
#     total_mix = []
#     total_src_tgt = []
#     total_theta_tgt_angle = []
#     total_fi_tgt_angle = []
#     for i in batch[0][0]:
#         src_tgt_list = []
#         mix_list = []
#         tgt_theta_angle_list=[]
#         tgt_fi_angle_list=[]

#         # angle_tgt_path = os.path.join(i, 'tgt_angle.txt')

#         # with open(angle_tgt_path, "r") as f:
#         #     tmp = f.read()
#         #     spk_tgt_idx = int(tmp.split("\n")[0].split(":")[1])+1
#         #     tgt_theta_angle = float(tmp.split("\n")[1].split("\t")[0])
#         #     tgt_fi_angle = float(tmp.split("\n")[1].split("\t")[1])
#         #     tgt_theta_angle_list.append(tgt_theta_angle)
#         #     tgt_fi_angle_list.append(tgt_fi_angle)
#         spk_tgt_num = random.randint(0,1)
#         spk_tgt, _ = sf.read(i.replace("mix", "s"+str(spk_tgt_num+1)))
#         mix, _ = sf.read(i)
#         # # src_tgt_l, src_tgt_r, mix_l, mix_r =extract_samples(spk_tgt[:, 0], spk_tgt[:, 1], mix[:, 0], mix[:, 1])
        
#         for len in range(spk_tgt.shape[1]):
#             src_tgt_list.append(spk_tgt[:, len])
#             mix_list.append(mix[:, len])
#         # src_tgt_l, src_tgt_r, mix_l, mix_r =extract_samples(spk_tgt[:, 0], spk_tgt[:, 1], mix[:, 0], mix[:, 1])
        
#         # src_tgt_list = [src_tgt_l, src_tgt_r]
#         # mix_list = [mix_l, mix_r]

#         if spk_tgt_num == 0:
#             tgt_theta_angle = np.mod(float(os.path.basename(i).split("_")[2]), 360)
#             tgt_fi_angle = np.mod(float(os.path.basename(i).split("_")[3]), 360)
#             tgt_theta_angle_list.append(tgt_theta_angle)
#             tgt_fi_angle_list.append(tgt_fi_angle)
#         else:
#             tgt_theta_angle = np.mod(float(os.path.basename(i).split("_")[5]), 360)
#             tgt_fi_angle = np.mod(float(os.path.basename(i).split("_")[6].replace(".wav","")), 360)
#             tgt_theta_angle_list.append(tgt_theta_angle)
#             tgt_fi_angle_list.append(tgt_fi_angle)
        

#         # for n in range(1, 7):
#         #     if n == 1 or n == 4:
#         #         spk_tgt_idx_path = os.path.join(i,'spk' + str(spk_tgt_idx) + '_mic{0}.wav'.format(n))
#         #         spk_tgt, _ = sf.read(spk_tgt_idx_path)
#         #         src_tgt_list.append(spk_tgt)
#         #         mix_path = os.path.join(i,'mixture_mic{0}.wav'.format(n)) 
#         #         mix, _ = sf.read(mix_path)
#         #         mix_list.append(mix)
                
#         src_tgt_np = np.asarray(src_tgt_list,dtype=np.float32)
#         mix_np = np.asarray(mix_list,dtype=np.float32)
#         tgt_theta_angle_np = np.asarray(tgt_theta_angle_list, dtype=np.float32)
#         tgt_fi_angle_np = np.asarray(tgt_fi_angle_list, dtype=np.float32)
    
#         total_mix.append(mix_np)
#         total_src_tgt.append(src_tgt_np)
#         total_theta_tgt_angle.append(tgt_theta_angle_np)
#         total_fi_tgt_angle.append(tgt_fi_angle_np)
        
#     total_mix_np = np.asarray(total_mix,dtype=np.float32)
#     total_src_tgt_np = np.asarray(total_src_tgt,dtype=np.float32)
#     total_theta_tgt_angle_np = np.asarray(total_theta_tgt_angle, dtype=np.float32)
#     total_fi_tgt_angle_np =np.asarray(total_fi_tgt_angle, dtype=np.float32)

#     mix_torch = torch.from_numpy(total_mix_np)
#     src_tgt_torch = torch.from_numpy(total_src_tgt_np)
#     angle_theta_tgt_torch = torch.from_numpy(total_theta_tgt_angle_np)
#     angle_fi_tgt_torch = torch.from_numpy(total_fi_tgt_angle_np)

#     ilens = np.array([mix.shape[1] for mix in mix_torch])
#     ilens_torch = torch.from_numpy(ilens)

#     return mix_torch, ilens_torch, src_tgt_torch, angle_theta_tgt_torch, angle_fi_tgt_torch

class EvalAudioDataLoader(data.DataLoader):
    
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalAudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval
        
def _collate_fn_eval(batch):

    # batch should be located in list    
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mix_torch: B x ch x T, torch.Tensor
        ilens_torch : B, torch.Tentor
        src_torch: B x C x T, torch.Tensor
        
    ex)
    torch.Size([3, 6, 64000])
    tensor([64000, 64000, 64000], dtype=torch.int32)
    torch.Size([3, 2, 64000])
    """
    
    total_mix = []
    total_src_tgt = []
    total_theta_tgt_angle = []
    total_fi_tgt_angle = []
    for i in batch[0][0]:
        tgt_theta_angle_list=[]
        tgt_fi_angle_list=[]
        # angle_tgt_path = os.path.join(i, 'tgt_angle.txt')

        # with open(angle_tgt_path, "r") as f:
        #     tmp = f.read()
        #     spk_tgt_idx = int(tmp.split("\n")[0].split(":")[1])+1
        #     tgt_theta_angle = float(tmp.split("\n")[1].split("\t")[0])
        #     tgt_fi_angle = float(tmp.split("\n")[1].split("\t")[1])
        #     tgt_theta_angle_list.append(tgt_theta_angle)
        #     tgt_fi_angle_list.append(tgt_fi_angle)
        # spk_tgt_num = random.randint(0,1)
        spk_tgt_num = 0
        spk_tgt, _ = sf.read(i.replace("mix", "s"+str(spk_tgt_num+1)))
        mix, _ = sf.read(i)
        # src_tgt_l, src_tgt_r, mix_l, mix_r = extract_samples(spk_tgt[:, 0], spk_tgt[:, 1], mix[:, 0], mix[:, 1])
        src_tgt_l, src_tgt_r, mix_l, mix_r = spk_tgt[:, 0], spk_tgt[:, 1], mix[:, 0], mix[:, 1]
        
        src_tgt_list = [src_tgt_l, src_tgt_r]
        mix_list = [mix_l, mix_r]

        if spk_tgt_num == 0:
            tgt_theta_angle = np.mod(float(os.path.basename(i).split("_")[2]), 360)
            tgt_fi_angle = np.mod(float(os.path.basename(i).split("_")[3]), 360)
            tgt_theta_angle_list.append(tgt_theta_angle)
            tgt_fi_angle_list.append(tgt_fi_angle)
        else:
            tgt_theta_angle = np.mod(float(os.path.basename(i).split("_")[5]), 360)
            tgt_fi_angle = np.mod(float(os.path.basename(i).split("_")[6].replace(".wav","")), 360)
            tgt_theta_angle_list.append(tgt_theta_angle)
            tgt_fi_angle_list.append(tgt_fi_angle)
        

        # for n in range(1, 7):
        #     if n == 1 or n == 4:
        #         spk_tgt_idx_path = os.path.join(i,'spk' + str(spk_tgt_idx) + '_mic{0}.wav'.format(n))
        #         spk_tgt, _ = sf.read(spk_tgt_idx_path)
        #         src_tgt_list.append(spk_tgt)
        #         mix_path = os.path.join(i,'mixture_mic{0}.wav'.format(n)) 
        #         mix, _ = sf.read(mix_path)
        #         mix_list.append(mix)
                
        src_tgt_np = np.asarray(src_tgt_list,dtype=np.float32)
        mix_np = np.asarray(mix_list,dtype=np.float32)
        tgt_theta_angle_np = np.asarray(tgt_theta_angle_list, dtype=np.float32)
        tgt_fi_angle_np = np.asarray(tgt_fi_angle_list, dtype=np.float32)
    
        total_mix.append(mix_np)
        total_src_tgt.append(src_tgt_np)
        total_theta_tgt_angle.append(tgt_theta_angle_np)
        total_fi_tgt_angle.append(tgt_fi_angle_np)
        
    total_mix_np = np.asarray(total_mix,dtype=np.float32)
    total_src_tgt_np = np.asarray(total_src_tgt,dtype=np.float32)
    total_theta_tgt_angle_np = np.asarray(total_theta_tgt_angle, dtype=np.float32)
    total_fi_tgt_angle_np =np.asarray(total_fi_tgt_angle, dtype=np.float32)

    mix_torch = torch.from_numpy(total_mix_np)
    src_tgt_torch = torch.from_numpy(total_src_tgt_np)
    angle_theta_tgt_torch = torch.from_numpy(total_theta_tgt_angle_np)
    angle_fi_tgt_torch = torch.from_numpy(total_fi_tgt_angle_np)

    ilens = np.array([mix.shape[1] for mix in mix_torch])
    ilens_torch = torch.from_numpy(ilens)

    return mix_torch, ilens_torch, src_tgt_torch, angle_theta_tgt_torch, angle_fi_tgt_torch