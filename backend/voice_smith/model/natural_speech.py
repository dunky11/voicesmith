import torch
from torch import nn
import time
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from voice_smith.utils.tools import initialize_embeddings
from voice_smith.model.acoustic_model import Conformer
from voice_smith.model.univnet import Generator as UnivNet
from voice_smith.model.acoustic_model import BertAttention, Conformer

LRELU_SLOPE = 0.3

class DurationPredictorBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, p_dropout=0.5):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.LeakyReLU(LRELU_SLOPE)
        self.ln = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = self.dropout(x)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, hidden_dim, n_blocks=3):
        super().__init__()
        self.duration_predictor_blocks = nn.ModuleList([
            DurationPredictorBlock(hidden_dim) for _ in range(n_blocks)
        ])
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, src_mask):
        for block in self.duration_predictor_blocks:
            x = block(x)
        x = self.linear(x)
        x = x.masked_fill(src_mask.unsqueeze(-1), 0.0)
        return x

def get_duration_matrices_non_parallel(x: torch.Tensor, n_frames: int):
    S = torch.zeros((x.shape[0], n_frames, x.shape[1])).to(x.device)
    E = torch.zeros((x.shape[0], n_frames, x.shape[1])).to(x.device)
    x = x.squeeze(2)
    x_summed = torch.cumsum(x, 1)
    for batch in range(S.shape[0]):
        for i in range(1, S.shape[1] + 1):
            for j in range(1, S.shape[2] + 1):
                S[batch, i - 1, j - 1] = i - x_summed[batch, j - 2]
                E[batch, i - 1, j - 1] = x_summed[batch, j - 1] - i
    return S, E

class DuratorMLP(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.act(x)
        return x


class Durator(nn.Module):
    def __init__(self, hidden_dim, q=4, p=2):
        super().__init__()
        self.duration_predictor = DurationPredictor(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, 8, kernel_size=3, padding=1)
        self.ln = nn.GroupNorm(1, 8)
        self.swish = nn.SiLU()
        self.mlp_q = DuratorMLP(10, q)
        self.mlp_p = DuratorMLP(10, p)
        self.proj_WH = nn.Linear(q * hidden_dim, hidden_dim)
        self.proj_WC = nn.Linear(q * p, hidden_dim)

    def get_duration_matrices(self, x: torch.Tensor, n_frames: int):
        """ https://arxiv.org/pdf/2205.04421.pdf Equation 6
        """
        x = x.squeeze(2)
        x_summed = torch.cumsum(x, 1)
        i_s = torch.arange(1, n_frames + 1, device=x.device).unsqueeze(0).unsqueeze(-1)
        x_summed_padded = torch.cat([torch.zeros(x.shape[0], 1, device=x.device), x_summed], 1)[:,:-1]
        x_summed, x_summed_padded = x_summed.unsqueeze(1), x_summed_padded.unsqueeze(1)
        S = i_s - x_summed_padded
        E = x_summed - i_s
        return S, E

    def project(self, x):
        """ Conv1D(Proj(H)) part of https://arxiv.org/pdf/2205.04421.pdf equation 7, 8
        """
        x = self.proj(x)
        x = x.permute((0, 2, 1))
        x = self.conv(x)
        x = self.ln(x)
        x = self.swish(x)
        return x


    def forward_train(self, H, src_mask, n_frames):
        d_n_1 = self.duration_predictor(H, src_mask)
        S, E = self.get_duration_matrices(d_n_1, n_frames)
        projected = self.project(H)

        d_n_test = torch.zeros((1, 2, 1))
        d_n_test[0, 0] = 2
        d_n_test[0, 1] = 1
        S_test, E_test = self.get_duration_matrices(d_n_test, 4)

        S = S.unsqueeze(-1)
        E = E.unsqueeze(-1)
        projected = projected.permute((0, 2, 1)).unsqueeze(1)
        projected = projected.expand((projected.shape[0], S.shape[1], projected.shape[2], projected.shape[3]))
        
        concat = torch.cat([
            S, 
            E,
            projected
        ], -1)

        C = self.mlp_p(concat)
        W = self.mlp_q(concat)
        # TODO masked fill with negative infinity
        # TODO check if correct dimension
        W = W.masked_fill_(src_mask.reshape((src_mask.shape[0], 1, -1, 1)), -1e9)
        print(W.shape)
        W = F.softmax(W, 2)
        W = W.permute((0, 3, 1, 2))
        O_left = torch.einsum("b q m n, b n h -> b m q h", W, H)
        O_left = self.proj_WH(O_left.reshape(O_left.shape[0], O_left.shape[1], -1))
        
        O_right = torch.einsum("b q m n, b m n p -> b m q p", W, C)
        O_right = self.proj_WC(O_right.reshape(O_right.shape[0], O_right.shape[1], -1))
        
        O = O_left + O_right
        return O

class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x

class Flow(nn.Module):
    def __init__(self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

class WN(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        self.hidden_channels =hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                        dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(
                x_in,
                g_l,
                n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:,:self.hidden_channels,:]
                x = (x + res_acts).masked_fill(x_mask, 0.0)
                output = output + res_skip_acts[:,self.hidden_channels:,:]
            else:
                output = output + res_skip_acts
        return output.masked_fill(x_mask, 0.0)

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)

class PosteriorEncoder(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x = self.pre(x).masked_fill(x_mask, 0.0)
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x).masked_fill(x_mask, 0.0)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)).masked_fill(x_mask, 0.0)
        return z, m, logs, x_mask

class ResidualCouplingLayer(nn.Module):
    def __init__(self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"

        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels]*2, 1)
        h = self.pre(x0).masked_fill(x_mask, 0.0)
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h).masked_fill(x_mask, 0.0)
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels]*2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs).masked_fill(x_mask, 0.0)
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1,2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs).masked_fill(x_mask, 0.0)
            x = torch.cat([x0, x1], 1)
            return x

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        return q, attn


class MemoryBank(nn.Module):
    def __init__(self, d_in, bank_size=1024, bank_hidden=192, n_head=2, p_dropout=0.1):
        super().__init__()
        assert bank_hidden % n_head == 0
        hidden_per_head = bank_hidden // n_head
        self.attention = MultiHeadAttention(n_head=n_head, d_model=d_in, d_k=hidden_per_head, d_v=hidden_per_head, dropout=p_dropout)
        self.memory_bank = nn.Parameter(
            initialize_embeddings((bank_size, bank_hidden))
        ).unsqueeze(0)

    def forward(self, x, mask):
        x = x.permute((0, 2, 1))
        memory_bank = self.memory_bank.expand(x.shape[0], self.memory_bank.shape[1], self.memory_bank.shape[2])
        x, _ = self.attention(x, memory_bank, memory_bank)
        x = x.permute((0, 2, 1))
        x = x.masked_fill(mask, 0.0)
        return x

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_1 = Conformer(
            dim=model_config["encoder"]["n_hidden"],
            n_layers=model_config["encoder"]["n_layers"] // 2,
            n_heads=model_config["encoder"]["n_heads"],
            embedding_dim=model_config["speaker_embed_dim"],
            p_dropout=model_config["encoder"]["p_dropout"],
            kernel_size_conv_mod=model_config["encoder"]["kernel_size_conv_mod"],
        )
        self.encoder_2 = Conformer(
            dim=model_config["encoder"]["n_hidden"],
            n_layers=model_config["encoder"]["n_layers"] // 2,
            n_heads=model_config["encoder"]["n_heads"],
            embedding_dim=model_config["speaker_embed_dim"],
            p_dropout=model_config["encoder"]["p_dropout"],
            kernel_size_conv_mod=model_config["encoder"]["kernel_size_conv_mod"],
        )
        self.bert_attention = BertAttention(
            d_model=model_config["encoder"]["n_hidden"],
            num_heads=model_config["encoder"]["n_heads"],
            p_dropout=model_config["encoder"]["p_dropout"],
        )


    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: torch.Tensor,
        embeddings: torch.Tensor,         
        style_embeds_pred: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        encoding = positional_encoding(
            self.emb_dim,
            max(x.shape[1], style_embeds_pred.shape[1]),
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
        return x

class NaturalSpeech(nn.Module):
    def __init__(self, 
        preprocess_config: Dict[str, Any],
        model_config: Dict[str, Any],
        n_speakers: int
    ):
        super().__init__()
        self.enc_p = TextEncoder()
        self.dec = UnivNet()
        self.enc_q = PosteriorEncoder(
            512,
            model_config["encoder"]["n_hidden"],
            model_config["encoder"]["n_hidden"],
            5,
            1,
            16,
            gin_channels=model_config["speaker_embed_dim"]
        )
        self.flow = Flow(model_config["encoder"]["n_hidden"], model_config["encoder"]["n_hidden"], 5, 1, 4, gin_channels=model_config["speaker_embed_dim"])
        self.speaker_embed = nn.Parameter(
            initialize_embeddings(
                (n_speakers, model_config["speaker_embed_dim"])
            )
        )
        self.bank = MemoryBank(model_config["encoder"]["n_hidden"])
        self.durator = Durator(model_config["encoder"]["n_hidden"])


    def get_embeddings(
        self, 
        token_idx: torch.Tensor, 
        speaker_ids: torch.Tensor, 
        src_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_embeddings = self.src_word_emb(token_idx)
        speaker_embeds = torch.index_select(self.speaker_embed, 0, speaker_ids).squeeze(
            1
        )
        token_embeddings = token_embeddings.masked_fill(src_mask.unsqueeze(-1), 0.0)
        return token_embeddings, speaker_embeds

    def forward_train(
        self,
        x: torch.Tensor,
        speakers: torch.Tensor,
        src_lens: torch.Tensor,
        specs: torch.Tensor,
        spec_lens: torch.Tensor,
        style_embeds_pred: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        src_mask = tools.get_mask_from_lengths(src_lens)
        spec_mask = tools.get_mask_from_lengths(spec_lens)
        x, embeddings = self.get_embeddings(x, speakers, src_mask)
        x = self.enc_p(
            src_mask=src_mask,
            embeddings=embeddings,         
            style_embeds_pred=style_embeds_pred,
            attention_mask=attention_mask
        )
        x = self.durator.forward_train(x, src_mask, n_frames)
        print(x.shape)


if __name__ == "__main__":
    from voice_smith.config.acoustic_model_config import acoustic_model_config as model_config
    from voice_smith.config.preprocess_config import preprocess_config
    from voice_smith.utils.model import get_param_num
    from voice_smith.utils.tokenization import BertTokenizer
    from pathlib import Path

    tokenizer = BertTokenizer("/home/media/main_volume/datasets/voice-smith/assets")
    model_config["encoder"]["n_hidden"] = 256
    model_config["speaker_embed_dim"] = 256
    model_config["encoder"]["n_heads"] = 4
    n_speakers = 100

    model = NaturalSpeech(
        preprocess_config=preprocess_config, 
        model_config=model_config, 
        n_speakers=n_speakers
    )
    

    x = torch.ones((5, 100))
    speakers = torch.ones(5)
    src_lens = torch.ones(5) * 100
    specs = torch.ones((5, 512, 150))
    spec_lens = torch.ones(5) * 150
    
    style_embeds_pred: torch.Tensor
    attention_mask: torch.Tensor


    print("Total parameter count: ", get_param_num(model))
    print("Total parameter count during inference: ", get_param_num(model) - get_param_num(model.enc_q))



