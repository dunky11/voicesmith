
import json
from lib2to3.pgen2 import token
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from numba import jit, prange
from voice_smith.config.configs import PreprocessingConfig, AcousticModelConfigType
from voice_smith.config.symbols import symbols, symbol2id
from voice_smith.model.layers import (
    ConvTransposed,
    Conv1dGLU,
    GLUActivation,
    DepthWiseConv1d,
    PointwiseConv1d,
)
from voice_smith.model.attention import ConformerMultiHeadedSelfAttention
from voice_smith.model.position_encoding import positional_encoding
from voice_smith.model.reference_encoder import (
    PhonemeLevelProsodyEncoder,
    UtteranceLevelProsodyEncoder,
)
from voice_smith.utils import tools
from voice_smith.config.langs import SUPPORTED_LANGUAGES

LRELU_SLOPE = 0.3


class AcousticModel(nn.Module):
    def __init__(
        self,
        data_path: str,
        preprocess_config: PreprocessingConfig,
        model_config: AcousticModelConfigType,
        fine_tuning: bool,
        n_speakers: int,
    ):
        super().__init__()
        self.emb_dim = model_config.encoder.n_hidden
        self.encoder = Conformer(
            dim=model_config.encoder.n_hidden,
            n_layers=model_config.encoder.n_layers,
            n_heads=model_config.encoder.n_heads,
            embedding_dim=model_config.speaker_embed_dim + model_config.lang_embed_dim,
            p_dropout=model_config.encoder.p_dropout,
            kernel_size_conv_mod=model_config.encoder.kernel_size_conv_mod,
            with_ff=model_config.encoder.with_ff,
        )
        self.pitch_adaptor = PitchAdaptor(model_config, data_path=data_path)
        self.length_regulator = LengthAdaptor(model_config)

        self.utterance_prosody_encoder = UtteranceLevelProsodyEncoder(
            preprocess_config, model_config,
        )
        self.utterance_prosody_predictor = PhonemeProsodyPredictor(
            model_config=model_config, phoneme_level=False
        )
        self.phoneme_prosody_encoder = PhonemeLevelProsodyEncoder(
            preprocess_config, model_config,
        )
        self.phoneme_prosody_predictor = PhonemeProsodyPredictor(
            model_config=model_config, phoneme_level=True
        )
        self.u_bottle_out = nn.Linear(
            model_config.reference_encoder.bottleneck_size_u,
            model_config.encoder.n_hidden,
        )
        self.u_norm = nn.LayerNorm(
            model_config.reference_encoder.bottleneck_size_u, elementwise_affine=False,
        )
        self.p_bottle_out = nn.Linear(
            model_config.reference_encoder.bottleneck_size_p,
            model_config.encoder.n_hidden,
        )
        self.p_norm = nn.LayerNorm(
            model_config.reference_encoder.bottleneck_size_p, elementwise_affine=False,
        )

        self.aligner = Aligner(
            d_enc_in=model_config.encoder.n_hidden,
            d_dec_in=preprocess_config.stft.n_mel_channels,
            d_hidden=model_config.encoder.n_hidden,
        )

        self.decoder = Conformer(
            dim=model_config.decoder.n_hidden,
            n_layers=model_config.decoder.n_layers,
            n_heads=model_config.decoder.n_heads,
            embedding_dim=model_config.speaker_embed_dim + model_config.lang_embed_dim,
            p_dropout=model_config.decoder.p_dropout,
            kernel_size_conv_mod=model_config.decoder.kernel_size_conv_mod,
            with_ff=model_config.decoder.with_ff,
        )

        self.src_word_emb = Parameter(
            tools.initialize_embeddings((len(symbols), model_config.encoder.n_hidden))
        )

        self.to_mel = nn.Linear(
            model_config.decoder.n_hidden, preprocess_config.stft.n_mel_channels,
        )

        self.speaker_embed = Parameter(
            tools.initialize_embeddings((n_speakers, model_config.speaker_embed_dim))
        )
        self.lang_embed = Parameter(
            tools.initialize_embeddings(
                (len(SUPPORTED_LANGUAGES), model_config.lang_embed_dim)
            )
        )

    def get_embeddings(
        self,
        token_idx: torch.Tensor,
        speaker_idx: torch.Tensor,
        src_mask: torch.Tensor,
        lang_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_embeddings = F.embedding(token_idx, self.src_word_emb)
        speaker_embeds = F.embedding(speaker_idx, self.speaker_embed)
        lang_embeds = F.embedding(lang_idx, self.lang_embed)
        embeddings = torch.cat([speaker_embeds, lang_embeds], dim=2)
        embeddings = embeddings.masked_fill(src_mask.unsqueeze(-1), 0.0)
        token_embeddings = token_embeddings.masked_fill(src_mask.unsqueeze(-1), 0.0)
        return token_embeddings, embeddings

    def prepare_for_export(self) -> None:
        del self.phoneme_prosody_encoder
        del self.utterance_prosody_encoder

    def freeze(self) -> None:
        for par in self.parameters():
            par.requires_grad = False
        self.speaker_embed.requires_grad = True
        self.pitch_adaptor.pitch_embedding.requires_grad = True

    def unfreeze(self, freeze_text_embed: bool, freeze_lang_embed: bool) -> None:
        for par in self.parameters():
            par.requires_grad = True
        if freeze_text_embed:
            for par in self.src_word_emb.parameters():
                self.src_word_emb.requires_grad = False
        if freeze_lang_embed:
            self.lang_embed.requires_grad = False

    def average_utterance_prosody(
        self, u_prosody_pred: torch.Tensor, src_mask: torch.Tensor
    ) -> torch.Tensor:
        lengths = ((~src_mask) * 1.0).sum(1)
        u_prosody_pred = u_prosody_pred.sum(1, keepdim=True) / lengths.view(-1, 1, 1)
        return u_prosody_pred

    def forward_train(
        self,
        x: torch.Tensor,
        speakers: torch.Tensor,
        src_lens: torch.Tensor,
        mels: torch.Tensor,
        mel_lens: torch.Tensor,
        pitches: torch.Tensor,
        langs: torch.Tensor,
        attn_priors: torch.Tensor,
        use_ground_truth: bool = True,
    ) -> Dict[str, torch.Tensor]:
        src_mask = tools.get_mask_from_lengths(src_lens)
        mel_mask = tools.get_mask_from_lengths(mel_lens)

        x, embeddings = self.get_embeddings(
            token_idx=x, speaker_idx=speakers, src_mask=src_mask, lang_idx=langs
        )

        encoding = positional_encoding(
            self.emb_dim, max(x.shape[1], max(mel_lens)), device=x.device,
        )

        x = self.encoder(x, src_mask, embeddings=embeddings, encoding=encoding)

        u_prosody_ref = self.u_norm(
            self.utterance_prosody_encoder(mels=mels, mel_lens=mel_lens)
        )
        u_prosody_pred = self.u_norm(
            self.average_utterance_prosody(
                u_prosody_pred=self.utterance_prosody_predictor(x=x, mask=src_mask),
                src_mask=src_mask,
            )
        )

        p_prosody_ref = self.p_norm(
            self.phoneme_prosody_encoder(
                x=x, src_mask=src_mask, mels=mels, mel_lens=mel_lens, encoding=encoding
            )
        )
        p_prosody_pred = self.p_norm(
            self.phoneme_prosody_predictor(x=x, mask=src_mask,)
        )
        if use_ground_truth:
            x = x + self.u_bottle_out(u_prosody_ref)
            x = x + self.p_bottle_out(p_prosody_ref)
        else:
            x = x + self.u_bottle_out(u_prosody_pred)
            x = x + self.p_bottle_out(p_prosody_pred)
        x_res = x
        attn_logprob, attn_soft, attn_hard, attn_hard_dur = self.aligner(
            enc_in=x_res.permute((0, 2, 1)),
            dec_in=mels,
            enc_len=src_lens,
            dec_len=mel_lens,
            enc_mask=src_mask,
            attn_prior=attn_priors,
        )
        pitches = pitch_phoneme_averaging(
            durations=attn_hard_dur, pitches=pitches, max_phoneme_len=x.shape[1]
        )
        x, pitch_prediction, _, _ = self.pitch_adaptor.add_pitch_train(
            x=x,
            pitch_target=pitches,
            src_mask=src_mask,
            use_ground_truth=use_ground_truth,
        )
        """assert torch.equal(attn_hard_dur.sum(1).long(), mel_lens), (
            attn_hard_dur.sum(1),
            mel_lens,
        )"""
        x, log_duration_prediction, embeddings = self.length_regulator.upsample_train(
            x=x,
            x_res=x_res,
            duration_target=attn_hard_dur,
            src_mask=src_mask,
            embeddings=embeddings,
        )
        x = self.decoder(x, mel_mask, embeddings=embeddings, encoding=encoding)
        x = self.to_mel(x)

        x = x.permute((0, 2, 1))

        return {
            "y_pred": x,
            "pitch_prediction": pitch_prediction,
            "pitch_target": pitches,
            "log_duration_prediction": log_duration_prediction,
            "u_prosody_pred": u_prosody_pred,
            "u_prosody_ref": u_prosody_ref,
            "p_prosody_pred": p_prosody_pred,
            "p_prosody_ref": p_prosody_ref,
            "attn_logprob": attn_logprob,
            "attn_soft": attn_soft,
            "attn_hard": attn_hard,
            "attn_hard_dur": attn_hard_dur,
        }

    def forward(
        self,
        x: torch.Tensor,
        speakers: torch.Tensor,
        langs: torch.Tensor,
        p_control: float,
        d_control: float,
    ) -> torch.Tensor:
        src_mask = tools.get_mask_from_lengths(
            torch.tensor([x.shape[1]], dtype=torch.int64, device=x.device)
        )

        x, embeddings = self.get_embeddings(
            token_idx=x, speaker_idx=speakers, src_mask=src_mask, lang_idx=langs
        )
        encoding = positional_encoding(self.emb_dim, x.shape[1], device=x.device)

        x = self.encoder(x, src_mask, embeddings=embeddings, encoding=encoding)

        u_prosody_pred = self.u_norm(
            self.average_utterance_prosody(
                u_prosody_pred=self.utterance_prosody_predictor(x=x, mask=src_mask),
                src_mask=src_mask,
            )
        )
        p_prosody_pred = self.p_norm(
            self.phoneme_prosody_predictor(x=x, mask=src_mask,)
        )
        x = x + self.u_bottle_out(u_prosody_pred).expand_as(x)
        x = x + self.p_bottle_out(p_prosody_pred).expand_as(x)
        x_res = x
        x = self.pitch_adaptor.add_pitch(x=x, src_mask=src_mask, control=p_control)
        x, duration_rounded, embeddings = self.length_regulator.upsample(
            x=x,
            x_res=x_res,
            src_mask=src_mask,
            control=d_control,
            embeddings=embeddings,
        )
        mel_mask = tools.get_mask_from_lengths(
            torch.tensor([x.shape[1]], dtype=torch.int64, device=x.device)
        )
        if x.shape[1] > encoding.shape[1]:
            encoding = positional_encoding(self.emb_dim, x.shape[1], device=x.device)

        x = self.decoder(x, mel_mask, embeddings=embeddings, encoding=encoding)
        x = self.to_mel(x)
        x = x.permute((0, 2, 1))
        return x


class Conformer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        embedding_dim: int,
        p_dropout: float,
        kernel_size_conv_mod: int,
        with_ff: bool,
    ):
        super().__init__()
        d_k = d_v = dim // n_heads
        self.layer_stack = nn.ModuleList(
            [
                ConformerBlock(
                    dim,
                    n_heads,
                    d_k,
                    d_v,
                    kernel_size_conv_mod=kernel_size_conv_mod,
                    dropout=p_dropout,
                    embedding_dim=embedding_dim,
                    with_ff=with_ff,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        embeddings: torch.Tensor,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = mask.view((mask.shape[0], 1, 1, mask.shape[1]))
        for enc_layer in self.layer_stack:
            x = enc_layer(
                x,
                mask=mask,
                slf_attn_mask=attn_mask,
                embeddings=embeddings,
                encoding=encoding,
            )
        return x


class ConformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_k: int,
        d_v: int,
        kernel_size_conv_mod: int,
        embedding_dim: int,
        dropout: float,
        with_ff: bool,
    ):
        super().__init__()
        self.with_ff = with_ff
        self.conditioning = Conv1dGLU(
            d_model=d_model,
            kernel_size=kernel_size_conv_mod,
            padding=kernel_size_conv_mod // 2,
            embedding_dim=embedding_dim,
        )
        if self.with_ff:
            self.ff = FeedForward(d_model=d_model, dropout=dropout, kernel_size=3)
        self.conformer_conv_1 = ConformerConvModule(
            d_model, kernel_size=kernel_size_conv_mod, dropout=dropout
        )
        self.ln = nn.LayerNorm(d_model)
        self.slf_attn = ConformerMultiHeadedSelfAttention(
            d_model=d_model, num_heads=n_head, dropout_p=dropout
        )
        self.conformer_conv_2 = ConformerConvModule(
            d_model, kernel_size=kernel_size_conv_mod, dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        slf_attn_mask: torch.Tensor,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conditioning(x, embeddings=embeddings)
        if self.with_ff:
            x = self.ff(x) + x
        x = self.conformer_conv_1(x) + x
        res = x
        x = self.ln(x)
        x, _ = self.slf_attn(
            query=x, key=x, value=x, mask=slf_attn_mask, encoding=encoding
        )
        x = x + res
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = self.conformer_conv_2(x) + x
        return x


class FeedForward(nn.Module):
    def __init__(
        self, d_model: int, kernel_size: int, dropout: float, expansion_factor: int = 4
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
        self.conv_1 = nn.Conv1d(
            d_model,
            d_model * expansion_factor,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.LeakyReLU(LRELU_SLOPE)
        self.conv_2 = nn.Conv1d(d_model * expansion_factor, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = x.permute((0, 2, 1))
        x = self.conv_1(x)
        x = x.permute((0, 2, 1))
        x = self.act(x)
        x = self.dropout(x)
        x = x.permute((0, 2, 1))
        x = self.conv_2(x)
        x = x.permute((0, 2, 1))
        x = self.dropout(x)
        x = 0.5 * x
        return x


class PhonemeProsodyPredictor(nn.Module):
    """Non-parallel Prosody Predictor inspired by Du et al., 2021"""

    def __init__(self, model_config: Dict[str, Any], phoneme_level: bool):
        super().__init__()
        self.d_model = model_config.encoder.n_hidden
        kernel_size = model_config.reference_encoder.predictor_kernel_size
        dropout = model_config.encoder.p_dropout
        bottleneck_size = (
            model_config.reference_encoder.bottleneck_size_p
            if phoneme_level
            else model_config.reference_encoder.bottleneck_size_u
        )
        self.layers = nn.ModuleList(
            [
                ConvTransposed(
                    self.d_model,
                    self.d_model,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(LRELU_SLOPE),
                nn.LayerNorm(self.d_model),
                nn.Dropout(dropout),
                ConvTransposed(
                    self.d_model,
                    self.d_model,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(LRELU_SLOPE),
                nn.LayerNorm(self.d_model),
                nn.Dropout(dropout),
            ]
        )
        self.predictor_bottleneck = nn.Linear(self.d_model, bottleneck_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x -- [B, src_len, d_model]
        mask -- [B, src_len]
        outputs -- [B, src_len, 2 * d_model]
        """
        mask = mask.unsqueeze(2)
        for layer in self.layers:
            x = layer(x)
        x = x.masked_fill(mask, 0.0)
        x = self.predictor_bottleneck(x)
        return x


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        inner_dim = d_model * expansion_factor
        self.ln_1 = nn.LayerNorm(d_model)
        self.conv_1 = PointwiseConv1d(d_model, inner_dim * 2)
        self.conv_act = GLUActivation()
        self.depthwise = DepthWiseConv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=tools.calc_same_padding(kernel_size)[0],
        )
        self.ln_2 = nn.GroupNorm(1, inner_dim)
        self.activation = nn.LeakyReLU(LRELU_SLOPE)
        self.conv_2 = PointwiseConv1d(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_1(x)
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = self.conv_act(x)
        x = self.depthwise(x)
        x = self.ln_2(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x


class DepthwiseConvModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, expansion: int = 4):
        super().__init__()
        padding = tools.calc_same_padding(kernel_size)
        self.depthwise = nn.Conv1d(
            dim,
            dim * expansion,
            kernel_size=kernel_size,
            padding=padding[0],
            groups=dim,
        )
        self.act = nn.LeakyReLU(LRELU_SLOPE)
        self.out = nn.Conv1d(dim * expansion, dim, 1, 1, 0)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = x.permute((0, 2, 1))
        x = self.depthwise(x)
        x = self.act(x)
        x = self.out(x)
        x = x.permute((0, 2, 1))
        return x


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = F.embedding(idx, self.embeddings)
        return x


class PitchAdaptor(nn.Module):
    def __init__(self, model_config: AcousticModelConfigType, data_path: str):
        super().__init__()
        self.pitch_predictor = VariancePredictor(
            channels_in=model_config.encoder.n_hidden,
            channels=model_config.variance_adaptor.n_hidden,
            channels_out=1,
            kernel_size=model_config.variance_adaptor.kernel_size,
            p_dropout=model_config.variance_adaptor.p_dropout,
        )

        n_bins = model_config.variance_adaptor.n_bins

        # Always use stats from preprocessing data, even in fine-tuning
        with open(Path(data_path) / "stats.json") as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]

        print(f"Min pitch: {pitch_min} - Max pitch: {pitch_max}")

        self.register_buffer(
            "pitch_bins", torch.linspace(pitch_min, pitch_max, n_bins - 1)
        )
        self.pitch_embedding = Embedding(n_bins, model_config.encoder.n_hidden)

    def get_pitch_embedding_train(
        self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = self.pitch_predictor(x, mask)
        embedding_true = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        embedding_pred = self.pitch_embedding(
            torch.bucketize(prediction, self.pitch_bins)
        )
        return prediction, embedding_true, embedding_pred

    def get_pitch_embedding(
        self, x: torch.Tensor, mask: torch.Tensor, control: float
    ) -> torch.Tensor:
        prediction = self.pitch_predictor(x, mask)
        prediction = prediction * control
        embedding = self.pitch_embedding(torch.bucketize(prediction, self.pitch_bins))
        return embedding

    def add_pitch_train(
        self,
        x: torch.Tensor,
        pitch_target: torch.Tensor,
        src_mask: torch.Tensor,
        use_ground_truth: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            pitch_prediction,
            pitch_embedding_true,
            pitch_embedding_pred,
        ) = self.get_pitch_embedding_train(x, pitch_target, src_mask)
        if use_ground_truth:
            x = x + pitch_embedding_true
        else:
            x = x + pitch_embedding_pred
        return x, pitch_prediction, pitch_embedding_true, pitch_embedding_pred

    def add_pitch(
        self, x: torch.Tensor, src_mask: torch.Tensor, control: float
    ) -> torch.Tensor:
        pitch_embedding_pred = self.get_pitch_embedding(x, src_mask, control=control)
        x = x + pitch_embedding_pred
        return x


@jit(nopython=True)
def mas_width1(attn_map: np.ndarray) -> np.ndarray:
    """mas with hardcoded width=1"""
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):  # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j

            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1

            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_attn_map)

    for b in prange(b_attn_map.shape[0]):
        out = mas_width1(b_attn_map[b, 0, : out_lens[b], : in_lens[b]])
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out
    return attn_out


def pitch_phoneme_averaging(durations, pitches, max_phoneme_len):
    """
    :param durations: (batch_size, n_phones) durations for each phone
    :param pitches: (batch_size, n_mel_timesteps) per frame pitch values    
    """
    pitches_averaged = torch.zeros(
        (pitches.shape[0], max_phoneme_len), device=pitches.device
    )
    for batch_idx in prange(durations.shape[0]):
        start_idx = 0
        for i, duration in enumerate(durations[batch_idx]):
            duration = duration.int().item()
            if duration != 0:
                mean = torch.mean(pitches[batch_idx, start_idx : start_idx + duration])
                pitches_averaged[batch_idx][i] = mean
                start_idx += duration

    return pitches_averaged


class Aligner(nn.Module):
    def __init__(
        self,
        d_enc_in,
        d_dec_in,
        d_hidden,
        kernel_size_enc=3,
        kernel_size_dec=7,
        attn_channels=80,
        temperature=0.0005,
    ):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.key_proj = nn.Sequential(
            nn.Conv1d(
                d_enc_in,
                d_hidden,
                kernel_size=kernel_size_enc,
                padding=kernel_size_enc // 2,
            ),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(
                d_hidden,
                attn_channels,
                kernel_size=1,
                padding=0,
            ),
        )

        self.query_proj = nn.Sequential(
            nn.Conv1d(
                d_dec_in,
                d_hidden,
                kernel_size=kernel_size_dec,
                padding=kernel_size_dec // 2,
            ),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(
                d_hidden,
                d_hidden,
                kernel_size=kernel_size_dec,
                padding=kernel_size_dec // 2,
            ),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(
                d_hidden,
                attn_channels,
                kernel_size=1,
                padding=0,
            ),
        )

    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
        These will no longer recieve a gradient.
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(
                attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1
            )
        return torch.from_numpy(attn_out).to(attn.device)

    def forward(self, enc_in, dec_in, enc_len, dec_len, enc_mask, attn_prior):
        """
        :param enc_in: (B, C_1, T_1) Text encoder outputs.
        :param dec_in: (B, C_2, T_2) Data to align encoder outputs to.
        :speaker_emb: (B, C_3) Batch of speaker embeddings.
        """
        queries = dec_in
        keys = enc_in
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries)

        # Simplistic Gaussian Isotopic Attention
        attn = (
            queries_enc[:, :, :, None] - keys_enc[:, :, None]
        ) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            # print(f"AlignmentEncoder \t| mel: {queries.shape} phone: {keys.shape} mask: {mask.shape} attn: {attn.shape} attn_prior: {attn_prior.shape}")
            attn = self.log_softmax(attn) + torch.log(
                attn_prior.permute((0, 2, 1))[:, None] + 1e-8
            )
            # print(f"AlignmentEncoder \t| After prior sum attn: {attn.shape}")""""""

        attn_logprob = attn.clone()

        if enc_mask is not None:
            attn.masked_fill(enc_mask.unsqueeze(1).unsqueeze(1), -float("inf"))

        attn_soft = self.softmax(attn)  # softmax along T2
        attn_hard = self.binarize_attention_parallel(attn_soft, enc_len, dec_len)
        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        return attn_logprob, attn_soft, attn_hard, attn_hard_dur


class LengthAdaptor(nn.Module):
    """Length Regulator"""

    def __init__(self, model_config: AcousticModelConfigType):
        super().__init__()
        self.duration_predictor = VariancePredictor(
            channels_in=model_config.encoder.n_hidden,
            channels=model_config.variance_adaptor.n_hidden,
            channels_out=1,
            kernel_size=model_config.variance_adaptor.kernel_size,
            p_dropout=model_config.variance_adaptor.p_dropout,
        )

    def length_regulate(
        self, x: torch.Tensor, duration: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = torch.jit.annotate(List[torch.Tensor], [])
        mel_len = torch.jit.annotate(List[int], [])
        max_len = 0
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            if expanded.shape[0] > max_len:
                max_len = expanded.shape[0]
            output.append(expanded)
            mel_len.append(expanded.shape[0])
        output = tools.pad(output, max_len)
        return output, torch.tensor(mel_len, dtype=torch.int64)

    def expand(self, batch: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        out = torch.jit.annotate(List[torch.Tensor], [])
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)
        return out

    def upsample_train(
        self,
        x: torch.Tensor,
        x_res: torch.Tensor,
        duration_target: torch.Tensor,
        embeddings: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_res = x_res.detach()
        log_duration_prediction = self.duration_predictor(x_res, src_mask,)
        x, _ = self.length_regulate(x, duration_target)
        embeddings, _ = self.length_regulate(embeddings, duration_target)
        return x, log_duration_prediction, embeddings

    def upsample(
        self,
        x: torch.Tensor,
        x_res: torch.Tensor,
        src_mask: torch.Tensor,
        embeddings: torch.Tensor,
        control: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_duration_prediction = self.duration_predictor(x_res, src_mask,)
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * control), min=0,
        )
        x, _ = self.length_regulate(x, duration_rounded)
        embeddings, _ = self.length_regulate(embeddings, duration_rounded)
        return x, duration_rounded, embeddings


class VariancePredictor(nn.Module):
    """Duration and Pitch predictor"""

    def __init__(
        self,
        channels_in: int,
        channels: int,
        channels_out: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ConvTransposed(
                    channels_in,
                    channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(LRELU_SLOPE),
                nn.LayerNorm(channels),
                nn.Dropout(p_dropout),
                ConvTransposed(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(LRELU_SLOPE),
                nn.LayerNorm(channels),
                nn.Dropout(p_dropout),
            ]
        )

        self.linear_layer = nn.Linear(channels, channels_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.linear_layer(x)
        x = x.squeeze(-1)
        x = x.masked_fill(mask, 0.0)
        return x