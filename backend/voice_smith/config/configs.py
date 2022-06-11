from dataclasses import dataclass, field
from typing import Any, Tuple, List
import torch


@dataclass
class STFTConfig:
    filter_length: int
    hop_length: int
    win_length: int
    n_mel_channels: int
    mel_fmin: int
    mel_fmax: int


@dataclass
class PreprocessingConfig:
    val_size: float = 0.05
    min_seconds: float = 0.5
    max_seconds: float = 10.0
    sampling_rate: int = 22050
    use_audio_normalization: bool = True
    workers: int = 11
    segment_size: int = 64
    stft: STFTConfig = STFTConfig(
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        mel_fmin=20,
        mel_fmax=11025,
    )


@dataclass
class AcousticTrainingOptimizerConfig:
    learning_rate: float
    weight_decay: float
    lr_decay: float
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 0.000000001
    grad_clip_thresh: float = 1.0
    warm_up_step: float = 4000
    anneal_steps: List[int] = field(default_factory=list)
    anneal_rate: float = 0.3


@dataclass
class AcousticFinetuningConfig:
    batch_size = 5
    grad_acc_step = 3
    train_steps = 30000
    log_step = 100
    synth_step = 250
    val_step = 4000
    save_step = 250
    freeze_bert_until = 0
    mcd_gen_max_samples = 400
    only_train_speaker_until = 5000
    optimizer_config: AcousticTrainingOptimizerConfig = AcousticTrainingOptimizerConfig(
        learning_rate=0.0002, weight_decay=0.1, lr_decay=0.99999
    )
    stft_lamb: int = 2.5


@dataclass
class AcousticPretrainingConfig:
    batch_size = 5
    grad_acc_step = 3
    train_steps = 500000
    log_step = 20
    synth_step = 250
    val_step = 4000
    save_step = 1000
    freeze_bert_until = 4000
    mcd_gen_max_samples = 400
    only_train_speaker_until = 0
    optimizer_config: AcousticTrainingOptimizerConfig = AcousticTrainingOptimizerConfig(
        learning_rate=0.051, weight_decay=0.1, lr_decay=0.99999
    )
    stft_lamb: int = 2.5


@dataclass
class ConformerConfig:
    n_layers: int
    n_heads: int
    n_hidden: int
    p_dropout: float
    kernel_size_conv_mod: int
    kernel_size_depthwise: int


@dataclass
class ReferenceEncoderConfig:
    bottleneck_size_p: int
    ref_enc_filters: List[int]
    ref_enc_size: int
    ref_enc_strides: List[int]
    ref_enc_pad: List[int]
    ref_enc_gru_size: int
    ref_attention_dropout: float
    token_num: int
    predictor_kernel_size: int


@dataclass
class StylePredictorConfig:
    n_head_layers: int


@dataclass
class VarianceAdaptorConfig:
    n_hidden: int
    kernel_size: int
    p_dropout: float
    n_bins: int


@dataclass
class AcousticModelConfig:
    speaker_embed_dim: int = 384
    style_embed_dim: int = 384
    encoder: ConformerConfig = ConformerConfig(
        n_layers=4,
        n_heads=6,
        n_hidden=384,
        p_dropout=0.1,
        kernel_size_conv_mod=7,
        kernel_size_depthwise=7,
    )
    decoder: ConformerConfig = ConformerConfig(
        n_layers=6,
        n_heads=6,
        n_hidden=384,
        p_dropout=0.1,
        kernel_size_conv_mod=11,
        kernel_size_depthwise=11,
    )
    reference_encoder: ReferenceEncoderConfig = ReferenceEncoderConfig(
        bottleneck_size_p=4,
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=3,
        ref_enc_strides=[1, 2, 1, 2, 1],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=32,
        ref_attention_dropout=0.2,
        token_num=32,
        predictor_kernel_size=5,
    )
    style_predictor: StylePredictorConfig = StylePredictorConfig(n_head_layers=2)
    variance_adaptor: VarianceAdaptorConfig = VarianceAdaptorConfig(
        n_hidden=384, kernel_size=5, p_dropout=0.5, n_bins=256
    )


@dataclass
class VocoderGeneratorConfig:
    noise_dim: int
    channel_size: int
    dilations: List[int]
    strides: List[int]
    lReLU_slope: float
    kpnet_conv_size: int


@dataclass
class VocoderMPDConfig:
    periods: List[int]
    kernel_size: int
    stride: int
    use_spectral_norm: bool
    lReLU_slope: float


@dataclass
class VocoderMRDConfig:
    resolutions: List[Tuple[int, int, int]]
    use_spectral_norm: bool
    lReLU_slope: float


@dataclass
class VocoderModelConfig:
    gen: VocoderGeneratorConfig = VocoderGeneratorConfig(
        noise_dim=64,
        channel_size=32,
        dilations=[1, 3, 9, 27],
        strides=[8, 8, 4],
        lReLU_slope=0.2,
        kpnet_conv_size=3,
    )
    mpd: VocoderMPDConfig = VocoderMPDConfig(
        periods=[2, 3, 5, 7, 11],
        kernel_size=5,
        stride=3,
        use_spectral_norm=False,
        lReLU_slope=0.2,
    )
    mrd: VocoderMRDConfig = VocoderMRDConfig(
        resolutions=[(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)],
        use_spectral_norm=False,
        lReLU_slope=0.2,
    )


@dataclass
class SampleSplittingRunConfig:
    workers: int
    device: torch.device

