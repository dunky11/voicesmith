from typing import Dict, Any

acoustic_model_config: Dict[str, Any] = {
  "speaker_embed_dim": 256,
  "encoder": {
    "n_layers": 4,
    "n_heads": 4,
    "n_hidden": 256,
    "p_dropout": 0.1,
    "kernel_size_conv_mod": 7,
    "kernel_size_depthwise": 7
  },
  "variance_adaptor": {
    "n_hidden": 384,
    "kernel_size": 5,
    "p_dropout": 0.3,
    "n_bins": 256 
  }
}
