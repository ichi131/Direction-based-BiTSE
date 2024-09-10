from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
# from torchmetrics.functional.audio import permutation_invariant_training as pit
# from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr

from blstm2_fc1 import BLSTM2_FC1
from NBC import NBC
from NBC2 import NBC2
from spatial_fea import get_spatial_fea

def neg_si_sdr(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch_size = target.shape[0]
    si_snr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_snr_val.view(batch_size, -1), dim=1)


class NBSS(nn.Module):
    """Multi-channel Narrow-band Deep Speech Separation with Full-band Permutation Invariant Training.

    A module version of NBSS which takes time domain signal as input, and outputs time domain signal.

    Arch could be NB-BLSTM or NBC
    """

    def __init__(
            self,
            n_channel: int = 2,
            n_speaker: int = 2,
            n_fft: int = 512,
            n_overlap: int = 256,
            ref_channel: int = 0,
            doa_input_shape: int = 36,
            arch: str = "NB_BLSTM",  # could also be NBC, NBC2
            arch_kwargs: Dict[str, Any] = dict(),
    ):
        super().__init__()

        if arch == "NB_BLSTM":
            self.arch: nn.Module = BLSTM2_FC1(input_size=n_channel * 2, output_size=n_speaker * 2, **arch_kwargs)
        elif arch == "NBC":
            self.arch = NBC(input_size=n_channel * 2, output_size=n_speaker * 2, **arch_kwargs)
        elif arch == 'NBC2':
            self.arch = NBC2(input_size=n_channel * 3, output_size=n_speaker * 2, **arch_kwargs)
        else:
            raise Exception(f"Unkown arch={arch}")

        self.register_buffer('window', torch.hann_window(n_fft), False)  # self.window, will be moved to self.device at training time
        self.n_fft = n_fft
        self.n_overlap = n_overlap
        self.ref_channel = ref_channel
        self.n_channel = n_channel
        self.n_speaker = n_speaker
        self.doa_input_shape = doa_input_shape
        # Define layers
        # self.theta_linear = nn.Linear(self.doa_input_shape, 96)  # DOA linear layer
        # self.fi_linear = nn.Linear(self.doa_input_shape, 96)  # DOA linear layer
        # self.layer_norm = torch.nn.LayerNorm(96)
        # self.direction_linear = nn.Linear(96, 96)
        # self.theta_conv = nn.Conv2d(in_channels=36, out_channels=96, kernel_size=(1, 1))
        # self.fi_conv = nn.Conv2d(in_channels=36, out_channels=96, kernel_size=(1, 1))
        # self.layer_norm = torch.nn.LayerNorm(96)
        # self.direction_linear = nn.Linear(96, 96)

    def forward(self, x: Tensor, angle_tgt_theta: Tensor, angle_tgt_fi: Tensor) -> Tensor:
        """forward

        Args:
            x: time domain signal, shape [Batch, Channel, Time]

        Returns:
            y: the predicted time domain signal, shape [Batch, Speaker, Time]
        """
        ##### 得到空间特征 ##### DF: (batch, freq, time frame, channel)  IPD: (batch, freq, time frame, 2*channel)
        DF_all_spk_norm, df_new, ipd_feature, complex_diff = get_spatial_fea(x, angle_tgt_theta, angle_tgt_fi)
        # DF_all_spk_norm, ipd_feature, log_mag = get_spatial_fea(x, angle_tgt_theta, angle_tgt_fi)
        # STFT
        B, C, T = x.shape
        # theta_input = self.angle_to_onehot(angle_tgt_theta)
        # fi_input = self.angle_to_onehot(angle_tgt_fi)

        x = x.reshape((B * C, T))
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.n_overlap, window=self.window, win_length=self.n_fft, return_complex=True)
        X = X.reshape((B, C, X.shape[-2], X.shape[-1]))  # (batch, channel, freq, time frame)
        X = X.permute(0, 2, 3, 1)  # (batch, freq, time frame, channel)

        # normalization by using ref_channel
        F, TF = X.shape[1], X.shape[2]
        Xr = X[..., self.ref_channel].clone()  # copy
        XrMM = torch.abs(Xr).mean(dim=2)  # Xr_magnitude_mean: mean of the magnitude of the ref channel of X
        X[:, :, :, :] /= (XrMM.reshape(B, F, 1, 1) + 1e-8)

        # to real
        X = torch.view_as_real(X)  # [B, F, T, C, 2]
        X = X.reshape(B * F, TF, C * 2)

        # angle_tgt_theta_expand = theta_input.unsqueeze(1).expand(B, TF, 36).unsqueeze(1)  # 新的形状将会是 [B, 1, TF, 36]

        # # 创建一个乘法因子向量，根据 F 的大小
        # factors_theta = torch.arange(1, F+1, dtype=torch.float32, device=theta_input.device).view(1, F, 1, 1)

        # # 将 x_expanded 与 factors 相乘以生成新的维度
        # angle_tgt_theta_new = angle_tgt_theta_expand * factors_theta  # 结果形状为 [B, F, TF, 36]

        # angle_tgt_fi_expand = fi_input.unsqueeze(1).expand(B, TF, 36).unsqueeze(1)  # 新的形状将会是 [B, 1, TF, 36]

        # # 创建一个乘法因子向量，根据 F 的大小
        # factors_fi = torch.arange(1, F+1, dtype=torch.float32, device=fi_input.device).view(1, F, 1, 1)

        # # 将 x_expanded 与 factors 相乘以生成新的维度
        # angle_tgt_fi_new = angle_tgt_fi_expand * factors_fi  # 结果形状为 [B, F, TF, 36]
        
        # # Process DOA input through conv layer
        # theta_output = self.theta_conv(angle_tgt_theta_new.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        # fi_output = self.fi_conv(angle_tgt_fi_new.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        # added_vector = theta_output + fi_output
        # # Compute the norm of the resulting vector
        # direction = self.layer_norm(added_vector)
        # direction = self.direction_linear(direction)

        # print("X", X.size())
        # print("IPD", ipd_feature.size())
        # X = torch.cat([X, DF_all_spk_norm, df_new, ipd_feature, complex_diff, SCM_list], dim=-1).float()
        # X = torch.cat([X, DF_all_spk_norm, df_new, ipd_feature, complex_diff], dim=-1).float()
        # X = torch.cat([X, DF_all_spk_norm, df_new, ipd_feature], dim=-1).float()
        # X = torch.cat([X, DF_all_spk_norm, ipd_feature, complex_diff], dim=-1).float()
        if X.size(1) != DF_all_spk_norm.size(1):
            min_dim = min(DF_all_spk_norm.size(1), X.size(1))
            X = X[:, :min_dim, :]
            DF_all_spk_norm = DF_all_spk_norm[:, :min_dim, :]
            df_new = df_new[:, :min_dim, :]
            ipd_feature = ipd_feature[:, :min_dim, :]
            complex_diff = complex_diff[:, :min_dim, :]
            # print(X.size(), DF_all_spk_norm.size())

        # X = torch.cat([X, DF_all_spk_norm, df_new, ipd_feature, complex_diff], dim=-1).float()
        # X = torch.cat([X, DF_all_spk_norm, df_new, ipd_feature], dim=-1).float()
        # X = torch.cat([X, DF_all_spk_norm, ipd_feature, complex_diff], dim=-1).float()
        # X = torch.cat([X, DF_all_spk_norm, ipd_feature], dim=-1).float()
        X = torch.cat([X, DF_all_spk_norm], dim=-1).float()
        # network processing
        output = self.arch(X)

        # to complex
        output = output.reshape(B, F, TF, self.n_speaker, 2)
        output = torch.view_as_complex(output)  # [B, F, TF, S]

        # inverse normalization
        Ys_hat = torch.empty(size=(B, self.n_speaker, F, TF), dtype=torch.complex64, device=output.device)
        XrMM = torch.unsqueeze(XrMM, dim=2).expand(-1, -1, TF)
        for spk in range(self.n_speaker):
            Ys_hat[:, spk, :, :] = output[:, :, :, spk] * XrMM[:, :, :]

        # iSTFT with frequency binding
        ys_hat = torch.istft(Ys_hat.reshape(B * self.n_speaker, F, TF), n_fft=self.n_fft, hop_length=self.n_overlap, window=self.window, win_length=self.n_fft, length=T)
        ys_hat = ys_hat.reshape(B, self.n_speaker, T)
        return ys_hat

    def angle_to_onehot(self, angles):
        """
        Convert angles in degrees to one-hot encoded vectors.
        Parameters:
        - angles: A tensor of shape [batch size, 1] containing angles in degrees.
        Returns:
        - A tensor of shape [batch size, 360] with one-hot encoded angles.
        """
        # 创建一个全零one-hot编码张量
        one_hot = torch.zeros(angles.size(0), 36).cuda(non_blocking=True)
        
        # 确保角度在0到360度之间，然后转换为整数索引
        indices = torch.remainder(angles, 36).floor().long().cuda(non_blocking=True)
        
        # 使用scatter方法在正确的位置置1
        one_hot.scatter_(1, indices, 1)
        return one_hot

if __name__ == '__main__':
    x = torch.randn(size=(4, 2, 64000)).cuda(non_blocking=True)
    ys = torch.randn(size=(4, 1, 64000)).cuda(non_blocking=True)
    # theta = torch.remainder((160 * torch.randn(size=(4, 1)) - 80), 360).cuda(non_blocking=True)
    theta = torch.Tensor([[65.4447], [152.5546], [30.7319], [294.0567]]).cuda(non_blocking=True)
    # fi = torch.remainder((360 * torch.randn(size=(4, 1)) - 90), 360).cuda(non_blocking=True)
    # fi = torch.Tensor([[265.7369], [164.4917], [359.0181], [75.8706]]).cuda(non_blocking=True)
    fi = torch.Tensor([[359.0181], [359.0181], [359.0181], [359.0181]]).cuda(non_blocking=True)
    print(theta, fi)
    # fi = torch.Tensor([[180], [326], [39], [197]]).cuda(non_blocking=True)
    # NBSS_with_NB_BLSTM = NBSS(n_channel=8, n_speaker=2, arch="NB_BLSTM")
    # ys_hat = NBSS_with_NB_BLSTM(x)
    # neg_sisdr_loss, best_perm = pit(preds=ys_hat, target=ys, metric_func=neg_si_sdr, eval_func='min')
    # print(ys_hat.shape, neg_sisdr_loss.mean())

    # NBSS_with_NBC = NBSS(n_channel=8, n_speaker=2, arch="NBC")
    # ys_hat = NBSS_with_NBC(x)
    # neg_sisdr_loss, best_perm = pit(preds=ys_hat, target=ys, metric_func=neg_si_sdr, eval_func='min')
    # print(ys_hat.shape, neg_sisdr_loss.mean())

    model = NBSS(n_channel=2,
                n_speaker=1,
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
                },).cuda()

    ys_hat = model(x, theta, fi)
    neg_sisdr_loss, best_perm = pit(preds=ys_hat, target=ys, metric_func=neg_si_sdr, eval_func='min')
    print(ys_hat.shape, neg_sisdr_loss.mean())
