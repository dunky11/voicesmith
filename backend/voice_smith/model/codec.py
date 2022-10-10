import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from typing import Tuple
from voice_smith.model.univnet import Generator as CodecDecoder
from voice_smith.utils.tools import init_weights, get_padding

LRELU_SLOPE = 0.3


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class CodecEncoder(torch.nn.Module):
    def __init__(
        self,
        channels_out,
        downsample_rates=[2, 2, 8, 8],
        downsample_kernel_sizes=[3, 3, 15, 15],
        downsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_downsamples = len(downsample_rates)

        downsample_channels = [
            downsample_initial_channel // (2 ** i)
            for i in range(len(downsample_rates) + 1)
        ][::-1]

        self.conv_pre = weight_norm(Conv1d(1, downsample_channels[0], 7, 1, padding=3))

        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(downsample_rates, downsample_kernel_sizes)):
            self.downs.append(
                weight_norm(
                    nn.Conv1d(
                        downsample_channels[i],
                        downsample_channels[i + 1],
                        kernel_size=k,
                        stride=u,
                        padding=(k) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = downsample_channels[i]
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = weight_norm(
            Conv1d(downsample_channels[-1], channels_out, 7, 1, padding=3)
        )
        self.downs.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_downsamples):
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.downs[i](x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.downs:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class CodecVQAutoEncoder(nn.Module):
    def __init__(self, model_config, preprocess_config):
        super().__init__()
        preprocess_config.stft.n_mel_channels = 128
        self.encoder = CodecEncoder(channels_out=preprocess_config.stft.n_mel_channels)
        self.vector_quantizer = MultistageVectorQuantizer()
        self.decoder = CodecDecoder(
            model_config=model_config, preprocess_config=preprocess_config
        )

    def forward(self, x):
        x = self.encoder(x)
        x, loss, perplexity = self.vector_quantizer(x)
        x = self.decoder(x)
        return x, loss, perplexity


class VectorQuantizer(nn.Module):
    def __init__(self, n_e: int = 8192, e_dim: int = 32, beta: float = 0.25):
        """ Quantizes latents by using a codebook. Default hyperparameters are taken
        from 

        Args:
            n_e (int, optional): Number of vectors in codebook. Defaults to 8192.
            e_dim (int, optional): Dimensions of each vector in codebook. Defaults to 32.
            beta (float, optional): Commitment cost. Defaults to 0.25.
        """
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z (torch.Tensor): Latents that will be quantized.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of the quantized
                latents, the quantization loss and the perplexity.
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, device=z.device
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 2, 1).contiguous()

        return z_q, loss, perplexity


class MultistageVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_codebooks: int = 16,
        n_e: int = 1024,
        e_dim: int = 128,
        beta: float = 0.25,
    ):
        """ Multi stage vector quantizer as described in https://www.isca-speech.org/archive/pdfs/interspeech_2021/vali21_interspeech.pdf
        and used in DelightfulTTS 2 (https://arxiv.org/pdf/2207.04646.pdf)

        Args:
            n_codebooks (int, optional): Number of codebooks. Defaults to 16,
                as used in DelightfulTTS 2.
            n_e (int, optional): Number of vectors in codebook. Defaults to 1024,
                as used in DelightfulTTS 2.
            e_dim (int, optional): Dimensions of each vector in codebook. Defaults to 128,
                as used in DelightfulTTS 2.
            beta (float, optional): Commitment cost. Defaults to 0.25.
        """
        super().__init__()
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(n_e=n_e, e_dim=e_dim, beta=beta)
                for _ in range(n_codebooks)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Latents that will be quantized.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of the quantized
                latents, the quantization loss and the perplexity.
        """
        average_perplexity, average_loss, y_out = 0.0, 0.0, 0.0
        y_k = x
        for quantizer in self.quantizers:
            y_k_hat, loss, perplexity = quantizer(y_k)
            average_perplexity += perplexity / len(self.quantizers)
            average_loss += loss / len(self.quantizers)
            y_out += y_k_hat
            y_k = y_k - y_k_hat
        return y_out, average_loss, average_perplexity


if __name__ == "__main__":
    from voice_smith.config.configs import VocoderModelConfig, PreprocessingConfig

    auto_enc = CodecVQAutoEncoder(
        model_config=VocoderModelConfig(),
        preprocess_config=PreprocessingConfig(language="multilingual"),
    ).cuda()
    """quantizer = VectorQuantizer(n_e=1000, e_dim=100, beta=1.0).cuda()
    inp = torch.randn((10, 100, 192)).cuda()
    y_pred = quantizer(inp)"""

    inp = torch.randn((2, 1, 25600)).cuda()
    y_pred, loss, perplexity = auto_enc(inp)
    print(y_pred.shape)
    print(loss)
    print(perplexity)

