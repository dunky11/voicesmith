import json
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from pathlib import Path
from typing import Dict, Any, Tuple
from voice_smith.config.configs import (
    PreprocessingConfig,
    AcousticModelConfig,
    VocoderModelConfig,
)
from voice_smith.model.layers import EmbeddingPadded
from voice_smith.config.symbols import symbols, symbol2id, pad
from voice_smith.model.layers import (
    ConvTransposed,
    Conv1dGLU,
    EmbeddingProjBlock,
    GLUActivation,
    DepthWiseConv1d,
    PointwiseConv1d,
)
from voice_smith.model.attention import ConformerMultiHeadedSelfAttention
from voice_smith.model.position_encoding import positional_encoding
from voice_smith.model.reference_encoder import PhonemeLevelProsodyEncoder
from voice_smith.utils import tools
from voice_smith.model.univnet import Generator as UnivNet

LRELU_SLOPE = 0.3

padding_idx = symbol2id[pad]


class AcousticModel(nn.Module):
    def __init__(
        self,
        data_path: str,
        preprocess_config: PreprocessingConfig,
        model_config: AcousticModelConfig,
        n_speakers: int,
    ):
        super().__init__()
        n_src_vocab = len(symbols) + 1
        self.emb_dim = model_config.encoder.n_hidden
        self.encoder_1 = Conformer(
            dim=model_config.encoder.n_hidden,
            n_layers=model_config.encoder.n_layers // 2,
            n_heads=model_config.encoder.n_heads,
            embedding_dim=model_config.speaker_embed_dim,
            p_dropout=model_config.encoder.p_dropout,
            kernel_size_conv_mod=model_config.encoder.kernel_size_conv_mod,
        )
        self.encoder_2 = Conformer(
            dim=model_config.encoder.n_hidden,
            n_layers=model_config.encoder.n_layers // 2,
            n_heads=model_config.encoder.n_heads,
            embedding_dim=model_config.speaker_embed_dim,
            p_dropout=model_config.encoder.p_dropout,
            kernel_size_conv_mod=model_config.encoder.kernel_size_conv_mod,
        )
        self.pitch_adaptor = PitchAdaptor(model_config, data_path=data_path)
        self.length_regulator = LengthAdaptor(model_config)
        self.bert_attention = BertAttention(
            d_model=model_config.encoder.n_hidden,
            num_heads=model_config.encoder.n_heads,
            p_dropout=model_config.encoder.p_dropout,
        )

        self.phoneme_prosody_encoder = PhonemeLevelProsodyEncoder(
            preprocess_config, model_config,
        )
        self.phoneme_prosody_predictor = PhonemeProsodyPredictor(
            model_config=model_config, phoneme_level=True
        )
        self.p_bottle_out = nn.Linear(
            model_config.reference_encoder.bottleneck_size_p,
            model_config.encoder.n_hidden,
        )
        self.p_norm = nn.LayerNorm(
            model_config.reference_encoder.bottleneck_size_p, elementwise_affine=False,
        )

        self.decoder = Conformer(
            dim=model_config.decoder.n_hidden,
            n_layers=model_config.decoder.n_layers,
            n_heads=model_config.decoder.n_heads,
            embedding_dim=model_config.speaker_embed_dim,
            p_dropout=model_config.decoder.p_dropout,
            kernel_size_conv_mod=model_config.decoder.kernel_size_conv_mod,
        )

        self.src_word_emb = EmbeddingPadded(
            n_src_vocab, model_config.encoder.n_hidden, padding_idx=padding_idx
        )
        self.to_mel = nn.Linear(
            model_config.decoder.n_hidden, preprocess_config.stft.n_mel_channels,
        )
        # TODO can be removed
        self.proj_speaker = EmbeddingProjBlock(model_config.speaker_embed_dim)
        self.speaker_embed = Parameter(
            tools.initialize_embeddings((n_speakers, model_config.speaker_embed_dim))
        )
        self.segment_size = preprocess_config.segment_size
        self.vocoder = UnivNet(
            model_config=VocoderModelConfig(), preprocess_config=preprocess_config
        )

    def get_embeddings(
        self, token_idx: torch.Tensor, speaker_ids: torch.Tensor, src_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_embeddings = self.src_word_emb(token_idx)
        speaker_embeds = torch.index_select(self.speaker_embed, 0, speaker_ids).squeeze(
            1
        )
        speaker_embeds = self.proj_speaker(speaker_embeds)
        token_embeddings = token_embeddings.masked_fill(src_mask.unsqueeze(-1), 0.0)
        return token_embeddings, speaker_embeds

    def prepare_for_export(self) -> None:
        del self.phoneme_prosody_encoder

    def freeze(self) -> None:
        for par in self.parameters():
            par.requires_grad = False
        self.speaker_embed.requires_grad = True
        self.pitch_adaptor.pitch_embedding.requires_grad = True

    def unfreeze(self) -> None:
        for par in self.parameters():
            par.requires_grad = True

    def forward_train(
        self,
        x: torch.Tensor,
        speakers: torch.Tensor,
        src_lens: torch.Tensor,
        mels: torch.Tensor,
        mel_lens: torch.Tensor,
        style_embeds_pred: torch.Tensor,
        attention_mask: torch.Tensor,
        pitches: torch.Tensor,
        durations: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        src_mask = tools.get_mask_from_lengths(src_lens)
        mel_mask = tools.get_mask_from_lengths(mel_lens)

        x, embeddings = self.get_embeddings(x, speakers, src_mask)

        encoding = positional_encoding(
            self.emb_dim,
            max(x.shape[1], style_embeds_pred.shape[1], max(mel_lens)),
            device=x.device,
        )

        x = self.encoder_1(x, src_mask, embeddings=embeddings, encoding=encoding)
        x, bert_attention = self.bert_attention(
            x=x,
            style_pred=style_embeds_pred,
            mask=src_mask,
            attention_mask=~attention_mask.bool(),
            encoding=encoding,
        )
        x = self.encoder_2(x, src_mask, embeddings=embeddings, encoding=encoding)

        p_prosody_ref = self.p_norm(
            self.phoneme_prosody_encoder(
                x=x, src_mask=src_mask, mels=mels, mel_lens=mel_lens, encoding=encoding
            )
        )
        p_prosody_pred = self.p_norm(
            self.phoneme_prosody_predictor(x=x, mask=src_mask,)
        )
        x = x + self.p_bottle_out(p_prosody_ref)

        x_res = x
        x, pitch_prediction, _, _ = self.pitch_adaptor.add_pitch_train(
            x=x, pitch_target=pitches, src_mask=src_mask, use_ground_truth=True,
        )
        (x, log_duration_prediction) = self.length_regulator.upsample_train(
            x=x, x_res=x_res, duration_target=durations, src_mask=src_mask
        )
        x = self.decoder(x, mel_mask, embeddings=embeddings, encoding=encoding)

        x = self.to_mel(x)

        x = x.permute((0, 2, 1))

        x, ids_slice = tools.rand_slice_segments(x, mel_lens, self.segment_size)

        x = self.vocoder.forward_train(x)

        return {
            "y_pred": x,
            "ids_slice": ids_slice,
            "pitch_prediction": pitch_prediction,
            "log_duration_prediction": log_duration_prediction,
            "p_prosody_pred": p_prosody_pred,
            "p_prosody_ref": p_prosody_ref,
            "bert_attention": bert_attention,
        }

    def forward(
        self,
        x: torch.Tensor,
        speakers: torch.Tensor,
        style_embeds_pred: torch.Tensor,
        p_control: float,
        d_control: float,
    ) -> torch.Tensor:
        src_mask = tools.get_mask_from_lengths(
            torch.tensor([x.shape[1]], dtype=torch.int64, device=x.device)
        )

        x, embeddings = self.get_embeddings(x, speakers, src_mask)

        encoding = positional_encoding(
            self.emb_dim, max(x.shape[1], style_embeds_pred.shape[1]), device=x.device
        )

        x = self.encoder_1(x, src_mask, embeddings=embeddings, encoding=encoding)

        attention_mask = tools.get_mask_from_lengths(
            torch.tensor(
                [style_embeds_pred.shape[1]],
                dtype=torch.int64,
                device=style_embeds_pred.device,
            )
        )
        attention_mask = attention_mask.view((attention_mask.shape[0], 1, 1, -1))

        x, _ = self.bert_attention(
            x=x,
            style_pred=style_embeds_pred,
            mask=src_mask,
            attention_mask=attention_mask,
            encoding=encoding,
        )

        x = self.encoder_2(x, src_mask, embeddings=embeddings, encoding=encoding)

        p_prosody_pred = self.p_norm(
            self.phoneme_prosody_predictor(x=x, mask=src_mask,)
        )
        x = x + self.p_bottle_out(p_prosody_pred).expand_as(x)
        x_res = x
        x = self.pitch_adaptor.add_pitch(x=x, src_mask=src_mask, control=p_control,)
        x, duration_rounded = self.length_regulator.upsample(
            x=x, x_res=x_res, src_mask=src_mask, control=d_control
        )
        mel_mask = tools.get_mask_from_lengths(
            torch.tensor([x.shape[1]], dtype=torch.int64, device=x.device)
        )
        if x.shape[1] > encoding.shape[1]:
            encoding = positional_encoding(self.emb_dim, x.shape[1], device=x.device)

        x = self.decoder(x, mel_mask, embeddings=embeddings, encoding=encoding)
        x = self.to_mel(x)
        x = x.permute((0, 2, 1))
        x = self.vocoder(x)
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
    ):
        super().__init__()
        self.conditioning = Conv1dGLU(
            d_model=d_model,
            kernel_size=kernel_size_conv_mod,
            padding=kernel_size_conv_mod // 2,
            embedding_dim=embedding_dim,
        )
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
    def __init__(self, d_model: int, dropout: float, expansion_factor: int = 4):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
        self.conv_1 = nn.Conv1d(
            d_model, d_model * expansion_factor, kernel_size=3, padding=1
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
    def __init__(self, model_config: AcousticModelConfig, data_path: str):
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


class LengthAdaptor(nn.Module):
    """Length Regulator"""

    def __init__(self, model_config: AcousticModel):
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
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_res = x_res.detach()
        log_duration_prediction = self.duration_predictor(x_res, src_mask,)
        x, _ = self.length_regulate(x, duration_target)
        return x, log_duration_prediction

    def upsample(
        self,
        x: torch.Tensor,
        x_res: torch.Tensor,
        src_mask: torch.Tensor,
        control: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_duration_prediction = self.duration_predictor(x_res, src_mask,)
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * control), min=0,
        )
        x, _ = self.length_regulate(x, duration_rounded)
        return x, duration_rounded


class BertAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, p_dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = ConformerMultiHeadedSelfAttention(
            d_model=d_model, num_heads=num_heads, dropout_p=p_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        style_pred: torch.Tensor,
        mask: torch.Tensor,
        attention_mask: torch.Tensor,
        encoding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_mask = attention_mask.view((attention_mask.shape[0], 1, 1, -1))
        res = x
        x = self.norm(x)
        x, attn = self.attn(
            query=x,
            key=style_pred,
            value=style_pred,
            mask=attention_mask,
            encoding=encoding,
        )
        x = x + res
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        return x, attn


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
