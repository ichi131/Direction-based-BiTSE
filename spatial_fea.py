import torch
import numpy as np
import math
C = 340
def stft_fun(input):
    sr = 16000
    # Define window and hop size in samples
    win_size = int(512)
    hop_size = int(256)

    # Compute STFT using a 32ms sqrt hann window with 16ms hop size and 512-point FFT size
    window = np.sqrt(np.hanning(win_size))

    magnitude_list = []
    phase_list = []
    lps_list = []
    stft_data_list = []
    pad_context = torch.zeros(input.size(0), input.size(1), 16).type(input.type())
    input = torch.cat([pad_context, input, pad_context], 2)
    for batch in range(input.size(0)):
        for i, channel in enumerate(range(input.size(1))):
            stft_data = torch.stft(input[batch][channel].unsqueeze(0), n_fft=512, hop_length=hop_size, win_length=win_size, window=torch.from_numpy(window).cuda(non_blocking=True), return_complex=True)
            # 计算幅度谱
            magnitude = torch.abs(stft_data).cuda(non_blocking=True)

            # 计算相位谱
            phase = torch.angle(stft_data).cuda(non_blocking=True)
            magnitude_list.append(magnitude[0])
            phase_list.append(phase[0])
            stft_data_list.append(stft_data[0])

            if i==0:
                # Convert to logarithmic power spectrum (LPS)
                lps = torch.log10(magnitude**2 + 1e-8).cuda(non_blocking=True)
                lps_list.append(lps[0])      
    # Get frequency and time arrays
    freqs = torch.fft.rfftfreq(512, 1/sr).cuda(non_blocking=True)
    times = torch.arange(lps.shape[2]) * hop_size / sr
    times_gpu = times.cuda(non_blocking=True)
    magnitude_trc = torch.stack(magnitude_list).view(input.size(0), input.size(1), len(magnitude[0]), len(magnitude[0][0])).cuda(non_blocking=True)
    phase_trc = torch.stack(phase_list).view(input.size(0), input.size(1), len(magnitude[0]), len(magnitude[0][0])).cuda(non_blocking=True)
    lps_trc = torch.stack(lps_list).view(input.size(0), len(magnitude[0]), len(magnitude[0][0])).cuda(non_blocking=True)
    stft_data_list = torch.stack(stft_data_list).view(input.size(0), input.size(1), len(magnitude[0]), len(magnitude[0][0])).cuda(non_blocking=True)
    # Return results as GPU tensors
    return freqs, times_gpu, magnitude_trc, phase_trc, lps_trc, stft_data_list

def get_spatial_fea(input, angle_tgt_theta, angle_tgt_fi):
    batch_size = input.size(0)
    nmic = input.size(1)
    
    f, t, mag_spec, phase_spec, lps, stft_data = stft_fun(input)
    # print(t)
    # print(stft_data.size())

    mic_list = [[1, 2], [1, 2]]
    mic_distance = [0.1449, 0.1449]
    mic_angle_fi = [angle_tgt_fi, angle_tgt_fi]
    
    ipd_list = []
    direction_diff = []
    log_mag = []
    complex_diff = []

    nchannel = len(mic_list)
    assert nmic == nchannel

    F, TF = stft_data.size(2), stft_data.size(3)
    stft_data = stft_data.view(batch_size, -1, F, TF).permute(0,2,3,1).contiguous()
    
    Xr = stft_data[..., 0].clone()  # copy
    XrMM = torch.abs(Xr).mean(dim=2)  # Xr_magnitude_mean: mean of the magnitude of the ref channel of X
    stft_data[:, :, :, :] /= (XrMM.reshape(batch_size, F, 1, 1) + 1e-8)
    stft_data = stft_data.view(batch_size, F, TF, -1).permute(0,3,1,2).contiguous()

    for i, pair in enumerate(mic_list):
        ipd_list.append(phase_spec[:,pair[0]-1,:,:] - phase_spec[:,pair[1]-1,:,:])
        direction_diff.append(mic_distance[i]*torch.cos(mic_angle_fi[i]/180*torch.pi)*torch.cos(angle_tgt_theta/180*torch.pi))
        log_mag.append(torch.log10(mag_spec[:,pair[0]-1,:,:] / (mag_spec[:,pair[1]-1,:,:] + 1e-8) + 1e-8))
        complex_diff.append(stft_data[:,pair[0]-1,:,:] - stft_data[:,pair[1]-1,:,:])
    
    ipd_list = torch.stack(ipd_list).permute(1,0,2,3).contiguous().cuda(non_blocking=True)
    direction_diff = torch.stack(direction_diff).squeeze().unsqueeze(dim=-1).permute(1,0).contiguous().cuda(non_blocking=True)
    # direction_diff = torch.stack(direction_diff).squeeze().permute(1,0).contiguous().cuda(non_blocking=True)
    log_mag = torch.stack(log_mag).permute(1,0,2,3).contiguous().cuda(non_blocking=True)
    complex_diff = torch.stack(complex_diff).permute(1,0,2,3).contiguous().cuda(non_blocking=True)

    cos_ipd = torch.cos(ipd_list)
    cos_ipd_cat = cos_ipd.view(batch_size, nchannel*f.size(0), -1)
    sin_ipd = torch.sin(ipd_list)
    sin_ipd_cat = sin_ipd.view(batch_size, nchannel*f.size(0), -1)

    TPD_list = []
    for single_batch in direction_diff:
        for channel in single_batch:
            TPD_list.append(2 * math.pi * f * channel / C)

    TPD_list = torch.stack(TPD_list).view(batch_size, nchannel, -1).cuda(non_blocking=True)
    TPD_tch = torch.unsqueeze(TPD_list, dim=-1).repeat(1, 1, 1, t.shape[0])

    deta = TPD_tch - ipd_list
    df_new = torch.sin(deta)
    # deta_pi = torch.remainder(deta, 2 * torch.pi) - 2 * torch.pi
    # ipd_norm = torch.remainder(ipd_list, 2 * torch.pi) - 2 * torch.pi

    # 三角形
    # df_new = -2 / (torch.pi) * torch.abs(deta_pi) + 1
    # ipd_new = -2 / (torch.pi) * torch.abs(ipd_norm) + 1

    # df_new = -2 / (torch.pi * torch.pi) * (torch.abs(deta_pi) * torch.abs(deta_pi)) + 1
    # ipd_new = -2 / (torch.pi * torch.pi) * (torch.abs(ipd_norm) * torch.abs(ipd_norm)) + 1
    
    df_all = torch.mul(torch.cos(TPD_tch), cos_ipd) + torch.mul(torch.sin(TPD_tch), sin_ipd)
    # df = torch.sum(df_all, dim=1)

    df_new = df_new.view(batch_size, -1, cos_ipd.size(2), cos_ipd.size(3)).permute(0,2,3,1).contiguous().view(batch_size*cos_ipd.size(2), cos_ipd.size(3), -1)
    log_mag = log_mag.view(batch_size, -1, cos_ipd.size(2), cos_ipd.size(3)).permute(0,2,3,1).contiguous().view(batch_size*cos_ipd.size(2), cos_ipd.size(3), -1)
    DF_all_spk_norm = df_all.view(batch_size, -1, cos_ipd.size(2), cos_ipd.size(3)).permute(0,2,3,1).contiguous().view(batch_size*cos_ipd.size(2), cos_ipd.size(3), -1)
    complex_diff = complex_diff.view(batch_size, -1, cos_ipd.size(2), cos_ipd.size(3)).permute(0,2,3,1).contiguous().view(batch_size*cos_ipd.size(2), cos_ipd.size(3), -1)
    complex_diff = torch.view_as_real(complex_diff).view(batch_size*cos_ipd.size(2), cos_ipd.size(3), -1)
    #print(DF_all_spk_norm)
    ipd_feature = torch.cat([cos_ipd, sin_ipd], 2).view(batch_size, -1, cos_ipd.size(2), cos_ipd.size(3)).permute(0,2,3,1).contiguous().view(batch_size*cos_ipd.size(2), cos_ipd.size(3), -1)
    # return DF_all_spk_norm, ipd_feature, complex_diff# log-division.log
    return DF_all_spk_norm, df_new, ipd_feature, complex_diff

# input = torch.randn(3,2,64000).cuda(non_blocking=True)
# thete = torch.randn(3,1).cuda(non_blocking=True)
# fi = torch.randn(3,1).cuda(non_blocking=True)
# DF_all_spk_norm, ipd_feature, log_mag= get_spatial_fea(input, thete, fi)
# print(DF_all_spk_norm.size())
# print(ipd_feature.size())
