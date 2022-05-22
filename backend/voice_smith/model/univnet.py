"""  
BSD 3-Clause License

Copyright (c) 2019, Seungwon Park 박승원
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from voice_smith.config.vocoder_model_config import vocoder_model_config as hp
from voice_smith.config.preprocess_config import preprocess_config

MAX_WAV_VALUE = 32768.0


class KernelPredictor(torch.nn.Module):
    """Kernel predictor for the location-variable convolutions"""

    def __init__(
        self,
        cond_channels,
        conv_in_channels,
        conv_out_channels,
        conv_layers,
        conv_kernel_size=3,
        kpnet_hidden_channels=64,
        kpnet_conv_size=3,
        kpnet_dropout=0.0,
        kpnet_nonlinear_activation="LeakyReLU",
        kpnet_nonlinear_activation_params={"negative_slope": 0.1},
    ):
        """
        Args:
            cond_channels (int): number of channel for the conditioning sequence,
            conv_in_channels (int): number of channel for the input sequence,
            conv_out_channels (int): number of channel for the output sequence,
            conv_layers (int): number of layers
        """
        super().__init__()

        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        kpnet_kernel_channels = (
            conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers
        )  # l_w
        kpnet_bias_channels = conv_out_channels * conv_layers  # l_b

        self.input_conv = nn.Sequential(
            nn.utils.weight_norm(
                nn.Conv1d(cond_channels, kpnet_hidden_channels, 5, padding=2, bias=True)
            ),
            getattr(nn, kpnet_nonlinear_activation)(
                **kpnet_nonlinear_activation_params
            ),
        )

        self.residual_convs = nn.ModuleList()
        padding = (kpnet_conv_size - 1) // 2
        for _ in range(3):
            self.residual_convs.append(
                nn.Sequential(
                    nn.Dropout(kpnet_dropout),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            kpnet_hidden_channels,
                            kpnet_hidden_channels,
                            kpnet_conv_size,
                            padding=padding,
                            bias=True,
                        )
                    ),
                    getattr(nn, kpnet_nonlinear_activation)(
                        **kpnet_nonlinear_activation_params
                    ),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            kpnet_hidden_channels,
                            kpnet_hidden_channels,
                            kpnet_conv_size,
                            padding=padding,
                            bias=True,
                        )
                    ),
                    getattr(nn, kpnet_nonlinear_activation)(
                        **kpnet_nonlinear_activation_params
                    ),
                )
            )
        self.kernel_conv = nn.utils.weight_norm(
            nn.Conv1d(
                kpnet_hidden_channels,
                kpnet_kernel_channels,
                kpnet_conv_size,
                padding=padding,
                bias=True,
            )
        )
        self.bias_conv = nn.utils.weight_norm(
            nn.Conv1d(
                kpnet_hidden_channels,
                kpnet_bias_channels,
                kpnet_conv_size,
                padding=padding,
                bias=True,
            )
        )

    def forward(self, c):
        """
        Args:
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)
        """
        batch, _, cond_length = c.shape
        c = self.input_conv(c)
        for residual_conv in self.residual_convs:
            residual_conv.to(c.device)
            c = c + residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)
        kernels = k.contiguous().view(
            batch,
            self.conv_layers,
            self.conv_in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            cond_length,
        )
        bias = b.contiguous().view(
            batch,
            self.conv_layers,
            self.conv_out_channels,
            cond_length,
        )

        return kernels, bias

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.input_conv[0])
        nn.utils.remove_weight_norm(self.kernel_conv)
        nn.utils.remove_weight_norm(self.bias_conv)
        for block in self.residual_convs:
            nn.utils.remove_weight_norm(block[1])
            nn.utils.remove_weight_norm(block[3])


class LVCBlock(torch.nn.Module):
    """the location-variable convolutions"""

    def __init__(
        self,
        in_channels,
        cond_channels,
        stride,
        dilations=[1, 3, 9, 27],
        lReLU_slope=0.2,
        conv_kernel_size=3,
        cond_hop_length=256,
        kpnet_hidden_channels=64,
        kpnet_conv_size=3,
        kpnet_dropout=0.0,
    ):
        super().__init__()

        self.cond_hop_length = cond_hop_length
        self.conv_layers = len(dilations)
        self.conv_kernel_size = conv_kernel_size

        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            conv_layers=len(dilations),
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout,
            kpnet_nonlinear_activation_params={"negative_slope": lReLU_slope},
        )

        self.convt_pre = nn.Sequential(
            nn.LeakyReLU(lReLU_slope),
            nn.utils.weight_norm(
                nn.ConvTranspose1d(
                    in_channels,
                    in_channels,
                    2 * stride,
                    stride=stride,
                    padding=stride // 2 + stride % 2,
                    output_padding=stride % 2,
                )
            ),
        )

        self.conv_blocks = nn.ModuleList()
        for dilation in dilations:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(lReLU_slope),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            in_channels,
                            in_channels,
                            conv_kernel_size,
                            padding=dilation * (conv_kernel_size - 1) // 2,
                            dilation=dilation,
                        )
                    ),
                    nn.LeakyReLU(lReLU_slope),
                )
            )

    def forward(self, x, c):
        """forward propagation of the location-variable convolutions.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length)
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)

        Returns:
            Tensor: the output sequence (batch, in_channels, in_length)
        """
        _, in_channels, _ = x.shape  # (B, c_g, L')

        x = self.convt_pre(x)  # (B, c_g, stride * L')
        kernels, bias = self.kernel_predictor(c)

        for i, conv in enumerate(self.conv_blocks):
            output = conv(x)  # (B, c_g, stride * L')

            k = kernels[:, i, :, :, :, :]  # (B, 2 * c_g, c_g, kernel_size, cond_length)
            b = bias[:, i, :, :]  # (B, 2 * c_g, cond_length)

            output = self.location_variable_convolution(
                output, k, b, hop_size=self.cond_hop_length
            )  # (B, 2 * c_g, stride * L'): LVC
            x = x + torch.sigmoid(output[:, :in_channels, :]) * torch.tanh(
                output[:, in_channels:, :]
            )  # (B, c_g, stride * L'): GAU

        return x

    def location_variable_convolution(self, x, kernel, bias, dilation=1, hop_size=256):
        """perform location-variable convolution operation on the input sequence (x) using the local convolution kernl.
        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length).
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length)
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length)
            dilation (int): the dilation of convolution.
            hop_size (int): the hop_size of the conditioning sequence.
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        """
        batch, _, in_length = x.shape
        batch, _, out_channels, kernel_size, kernel_length = kernel.shape
        assert in_length == (
            kernel_length * hop_size
        ), "length of (x, kernel) is not matched"

        padding = dilation * int((kernel_size - 1) / 2)
        x = F.pad(
            x, (padding, padding), "constant", 0
        )  # (batch, in_channels, in_length + 2*padding)
        x = x.unfold(
            2, hop_size + 2 * padding, hop_size
        )  # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = F.pad(x, (0, dilation), "constant", 0)
        x = x.unfold(
            3, dilation, dilation
        )  # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(
            3, 4
        )  # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)
        x = x.unfold(
            4, kernel_size, 1
        )  # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum("bildsk,biokl->bolsd", x, kernel)
        o = o.to(memory_format=torch.channels_last_3d)
        bias = bias.unsqueeze(-1).unsqueeze(-1).to(memory_format=torch.channels_last_3d)
        o = o + bias
        o = o.contiguous().view(batch, out_channels, -1)

        return o

    def remove_weight_norm(self):
        self.kernel_predictor.remove_weight_norm()
        nn.utils.remove_weight_norm(self.convt_pre[1])
        for block in self.conv_blocks:
            nn.utils.remove_weight_norm(block[1])


class Generator(nn.Module):
    """UnivNet Generator"""

    def __init__(self, cond_channels):
        super(Generator, self).__init__()
        self.mel_channel = cond_channels
        self.noise_dim = hp["gen"]["noise_dim"]
        self.hop_length = preprocess_config["stft"]["hop_length"]
        channel_size = hp["gen"]["channel_size"]
        kpnet_conv_size = hp["gen"]["kpnet_conv_size"]

        self.res_stack = nn.ModuleList()
        hop_length = 1
        for stride in hp["gen"]["strides"]:
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    cond_channels,
                    stride=stride,
                    dilations=hp["gen"]["dilations"],
                    lReLU_slope=hp["gen"]["lReLU_slope"],
                    cond_hop_length=hop_length,
                    kpnet_conv_size=kpnet_conv_size,
                )
            )

        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(
                hp["gen"]["noise_dim"],
                channel_size,
                7,
                padding=3,
                padding_mode="reflect",
            )
        )

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(hp["gen"]["lReLU_slope"]),
            nn.utils.weight_norm(
                nn.Conv1d(channel_size, 1, 7, padding=3, padding_mode="reflect")
            ),
            nn.Tanh(),
        )

    def forward_train(self, c):
        """
        Args:
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length)

        """
        z = torch.randn(c.shape[0], self.noise_dim, c.shape[2]).to(c.device)
        z = self.conv_pre(z)  # (B, c_g, L)

        for res_block in self.res_stack:
            res_block.to(z.device)
            z = res_block(z, c)  # (B, c_g, L * s_0 * ... * s_i)

        z = self.conv_post(z)  # (B, 1, L * 256)

        return z

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        print("Removing weight norm...")

        nn.utils.remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    def forward(self, c):
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(c.device)
        mel = torch.cat((c, zero), dim=2)

        audio = self.forward_train(mel)
        audio = audio.squeeze()  # collapse all dimension except time axis
        audio = audio[: -(self.hop_length * 10)]

        return audio


class DiscriminatorP(nn.Module):
    def __init__(self, period):
        super(DiscriminatorP, self).__init__()

        self.LRELU_SLOPE = hp["mpd"]["lReLU_slope"]
        self.period = period

        kernel_size = hp["mpd"]["kernel_size"]
        stride = hp["mpd"]["stride"]
        norm_f = (
            weight_norm if hp["mpd"]["use_spectral_norm"] == False else spectral_norm
        )

        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv2d(
                        1,
                        64,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        64,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        128,
                        256,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        256,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        512, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0)
                    )
                ),
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return fmap, x


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period) for period in hp["mpd"]["periods"]]
        )

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret


class DiscriminatorR(torch.nn.Module):
    def __init__(self, resolution):
        super(DiscriminatorR, self).__init__()

        self.resolution = resolution
        self.LRELU_SLOPE = hp["mpd"]["lReLU_slope"]

        norm_f = (
            weight_norm if hp["mrd"]["use_spectral_norm"] == False else spectral_norm
        )

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.stft(
            x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False
        )  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiResolutionDiscriminator, self).__init__()
        self.resolutions = hp["mrd"]["resolutions"]
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(resolution) for resolution in self.resolutions]
        )

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score)]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator()
        self.MPD = MultiPeriodDiscriminator()

    def forward(self, x):
        return self.MRD(x), self.MPD(x)


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self,
        device,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length).to(device)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y, mask):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
            
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)

        """"b = x.shape[0]
        x_mag = x_mag.masked_select(~mask.bool())
        y_mag = y_mag.masked_select(~mask.bool())
        x_mag = x_mag.reshape((b, 1, -1))
        y_mag = y_mag.reshape((b, 1, -1))"""

        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self, device, resolutions, window="hann_window"):
        """Initialize Multi resolution STFT loss module.
        Args:
            resolutions (list): List of (FFT size, hop size, window length).
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in resolutions:
            self.stft_losses += [STFTLoss(device, fs, ss, wl, window)]

    def forward(self, x, y, mask):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0

        for f in self.stft_losses:
            sc_l, mag_l = f(x, y, mask)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss
