from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from pathlib import Path
from voice_smith import ASSETS_PATH


class BertStylePredictor(nn.Module):
    def __init__(self, n_layers=2, output_dim=256, lrelu_slope=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            ASSETS_PATH / "tiny_bert", torchscript=True
        )
        self.n_layers = n_layers
        self.lrelu_slope = lrelu_slope
        self.init_classification_head(output_dim)

    def init_classification_head(self, output_dim):
        self.ff_layers = nn.ModuleList(
            [
                nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
                for _ in range(self.n_layers - 1)
            ]
        )
        self.prediction_layer = nn.Linear(self.bert.config.hidden_size, output_dim)

    def freeze(self):
        for par in self.bert.parameters():
            par.requires_grad = False

    def unfreeze(self):
        for par in self.bert.parameters():
            par.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out, _ = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        for layer in self.ff_layers:
            out = layer(out)
            out = F.leaky_relu(out, self.lrelu_slope)
        out = self.prediction_layer(out)
        return out
