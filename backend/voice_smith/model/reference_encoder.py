import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from voice_smith.utils import tools
from voice_smith.model.layers import CoordConv1d
from voice_smith.model.attention import (
    ConformerMultiHeadedSelfAttention,
    StyleEmbedAttention,
)
from voice_smith.config.configs import PreprocessingConfig, AcousticModelConfig


class ReferenceEncoder(nn.Module):
    def __init__(
        self, preprocess_config: PreprocessingConfig, model_config: AcousticModelConfig
    ):
        super().__init__()

        E = model_config.encoder.n_hidden
        n_mel_channels = preprocess_config.stft.n_mel_channels
        ref_enc_filters = model_config.reference_encoder.ref_enc_filters
        ref_enc_size = model_config.reference_encoder.ref_enc_size
        ref_enc_strides = model_config.reference_encoder.ref_enc_strides
        ref_enc_gru_size = model_config.reference_encoder.ref_enc_gru_size

        self.n_mel_channels = n_mel_channels
        K = len(ref_enc_filters)
        filters = [self.n_mel_channels] + ref_enc_filters
        strides = [1] + ref_enc_strides
        # Use CoordConv at the first layer to better preserve positional information: https://arxiv.org/pdf/1811.02122.pdf
        convs = [
            CoordConv1d(
                in_channels=filters[0],
                out_channels=filters[0 + 1],
                kernel_size=ref_enc_size,
                stride=strides[0],
                padding=ref_enc_size // 2,
                with_r=True,
            )
        ]
        convs2 = [
            nn.Conv1d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=ref_enc_size,
                stride=strides[i],
                padding=ref_enc_size // 2,
            )
            for i in range(1, K)
        ]
        convs.extend(convs2)
        self.convs = nn.ModuleList(convs)

        self.norms = nn.ModuleList(
            [
                nn.InstanceNorm1d(num_features=ref_enc_filters[i], affine=True)
                for i in range(K)
            ]
        )

        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1],
            hidden_size=ref_enc_gru_size,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, mel_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        inputs --- [N,  n_mels, timesteps]
        outputs --- [N, E//2]
        """

        mel_masks = tools.get_mask_from_lengths(mel_lens).unsqueeze(1)
        x = x.masked_fill(mel_masks, 0)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = F.leaky_relu(x, 0.3)  # [N, 128, Ty//2^K, n_mels//2^K]
            x = norm(x)

        for _ in range(2):
            mel_lens = tools.stride_lens(mel_lens)

        mel_masks = tools.get_mask_from_lengths(mel_lens)

        x = x.masked_fill(mel_masks.unsqueeze(1), 0)
        x = x.permute((0, 2, 1))
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, mel_lens.cpu().int(), batch_first=True, enforce_sorted=False
        )

        self.gru.flatten_parameters()
        x, memory = self.gru(x)  # memory --- [N, Ty, E//2], out --- [1, N, E//2]
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x, memory, mel_masks

    def calculate_channels(
        self, L: int, kernel_size: int, stride: int, pad: int, n_convs: int
    ) -> int:
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class UtteranceLevelProsodyEncoder(nn.Module):
    def __init__(self, preprocess_config: Dict[str, Any], model_config: Dict[str, Any]):
        super().__init__()

        self.E = model_config["encoder"]["n_hidden"]
        self.d_q = self.d_k = model_config["encoder"]["n_hidden"]
        ref_enc_gru_size = model_config["reference_encoder"]["ref_enc_gru_size"]
        ref_attention_dropout = model_config["reference_encoder"][
            "ref_attention_dropout"
        ]
        bottleneck_size = model_config["reference_encoder"]["bottleneck_size_u"]

        self.encoder = ReferenceEncoder(preprocess_config, model_config)
        self.encoder_prj = nn.Linear(ref_enc_gru_size, self.E // 2)
        self.stl = STL(preprocess_config, model_config)
        self.encoder_bottleneck = nn.Linear(self.E, bottleneck_size)
        self.dropout = nn.Dropout(ref_attention_dropout)

    def forward(self, mels: torch.Tensor, mel_lens: torch.Tensor) -> torch.Tensor:
        """
        mels --- [N, Ty/r, n_mels*r], r=1
        out --- [N, seq_len, E]
        """
        _, embedded_prosody, _ = self.encoder(mels, mel_lens)

        # Bottleneck
        embedded_prosody = self.encoder_prj(embedded_prosody)

        # Style Token
        out = self.encoder_bottleneck(self.stl(embedded_prosody))
        out = self.dropout(out)

        out = out.view((-1, 1, out.shape[3]))
        return out


class STL(nn.Module):
    """Style Token Layer"""

    def __init__(self, preprocess_config: Dict[str, Any], model_config: Dict[str, Any]):
        super(STL, self).__init__()

        num_heads = 1
        E = model_config["encoder"]["n_hidden"]
        self.token_num = model_config["reference_encoder"]["token_num"]
        self.embed = nn.Parameter(torch.FloatTensor(self.token_num, E // num_heads))
        d_q = E // 2
        d_k = E // num_heads
        self.attention = StyleEmbedAttention(
            query_dim=d_q, key_dim=d_k, num_units=E, num_heads=num_heads
        )

        torch.nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        query = x.unsqueeze(1)  # [N, 1, E//2]

        keys_soft = (
            torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        )  # [N, token_num, E // num_heads]

        # Weighted sum
        emotion_embed_soft = self.attention(query, keys_soft)

        return emotion_embed_soft


class PhonemeLevelProsodyEncoder(nn.Module):
    def __init__(
        self, preprocess_config: PreprocessingConfig, model_config: AcousticModelConfig
    ):
        super().__init__()

        self.E = model_config.encoder.n_hidden
        self.d_q = self.d_k = model_config.encoder.n_hidden
        bottleneck_size = model_config.reference_encoder.bottleneck_size_p
        ref_enc_gru_size = model_config.reference_encoder.ref_enc_gru_size

        self.encoder = ReferenceEncoder(preprocess_config, model_config)
        self.encoder_prj = nn.Linear(ref_enc_gru_size, model_config.encoder.n_hidden)
        self.attention = ConformerMultiHeadedSelfAttention(
            d_model=model_config.encoder.n_hidden,
            num_heads=model_config.encoder.n_heads,
            dropout_p=model_config.encoder.p_dropout,
        )
        self.encoder_bottleneck = nn.Linear(
            model_config.encoder.n_hidden, bottleneck_size
        )

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        mels: torch.Tensor,
        mel_lens: torch.Tensor,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        x --- [N, seq_len, encoder_embedding_dim]
        mels --- [N, Ty/r, n_mels*r], r=1
        out --- [N, seq_len, bottleneck_size]
        attn --- [N, seq_len, ref_len], Ty/r = ref_len
        """
        embedded_prosody, _, mel_masks = self.encoder(mels, mel_lens)

        # Bottleneck
        embedded_prosody = self.encoder_prj(embedded_prosody)

        attn_mask = mel_masks.view((mel_masks.shape[0], 1, 1, -1))
        x, _ = self.attention(
            query=x,
            key=embedded_prosody,
            value=embedded_prosody,
            mask=attn_mask,
            encoding=encoding,
        )
        x = self.encoder_bottleneck(x)
        x = x.masked_fill(src_mask.unsqueeze(-1), 0.0)
        return x
