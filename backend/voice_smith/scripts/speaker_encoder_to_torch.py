from speechbrain.pretrained import EncoderClassifier
from torch.jit._trace import trace_module
from pathlib import Path
import torch
from voice_smith.config.globals import ASSETS_PATH

if __name__ == "__main__":
    classifier = EncoderClassifier.from_hparams(source=str(Path(ASSETS_PATH) / "ecapa_tdnn"))
    classifier.eval()
    classifier.device = torch.device("cpu")
    classifier.mods.embedding_model.cpu()
    input = torch.randn((6, 49440))
    relative_lens = torch.ones((6,))
    output_pre = classifier.encode_batch(input, relative_lens)
    classifier_torch = trace_module(
        classifier, {"encode_batch": (input, relative_lens)}
    )
    classifier_torch.save(Path(ASSETS_PATH) / "ecapa_tdnn.pt")
    output_post = classifier_torch.encode_batch(input, relative_lens)
    assert torch.allclose(output_pre, output_post)
