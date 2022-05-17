from voice_smith.model.style_predictor import BertStylePredictor
from voice_smith.config.acoustic_model_config import acoustic_model_config
from voice_smith.utils.tokenization import BertTokenizer
from torch.jit._trace import trace
from pathlib import Path
from voice_smith import ASSETS_PATH

if __name__ == "__main__":
    style_predictor = BertStylePredictor(
        output_dim=acoustic_model_config["style_embed_dim"],
    )
    tokenizer = BertTokenizer()
    encoding = tokenizer(
        ["This model will be exported to torchscript ..."],
    )
    style_predictor_torch = trace(
        style_predictor, (encoding["input_ids"], encoding["attention_mask"])
    )
    style_predictor_torch.save(ASSETS_PATH / "tiny_bert.pt")
