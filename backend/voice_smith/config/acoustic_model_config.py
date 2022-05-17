from typing import Dict, Any

acoustic_model_config: Dict[str, Any] = {
  "speaker_embed_dim": 384,
  "style_embed_dim": 384,
  "encoder": {
    "n_layers": 4,
    "n_heads": 6,
    "n_hidden": 384,
    "p_dropout": 0.1,
    "kernel_size_conv_mod": 7,
    "kernel_size_depthwise": 7
  },
  "decoder": {
    "n_layers": 6,
    "n_heads": 6,
    "n_hidden": 384,
    "p_dropout": 0.1,
    "kernel_size_conv_mod": 11,
    "kernel_size_depthwise": 11
  },
  "reference_encoder": {
    "bottleneck_size_u": 256,
    "bottleneck_size_p": 4,
    "ref_enc_filters": [32, 32, 64, 64, 128, 128],
    "ref_enc_size": 3,
    "ref_enc_strides": [1, 2, 1, 2, 1], # '1' is to keep the sequence length
    "ref_enc_pad": [1, 1], # '1' is to keep the sequence length
    "ref_enc_gru_size": 32,
    "ref_attention_dropout": 0.2,
    "token_num": 32,
    "predictor_kernel_size": 5
  },
  "style_predictor": {
    "n_head_layers": 2
  },
  "variance_adaptor": {
    "n_hidden": 384,
    "kernel_size": 5,
    "p_dropout": 0.3,
    "n_bins": 256 
  }
}
