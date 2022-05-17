import torch
from torch import nn
import time
import torch.nn.functional as F

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
        print(S_test)
        print(E_test)

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

if __name__ == "__main__":
    from voice_smith.utils import tools
    src_mask = tools.get_mask_from_lengths(torch.cuda.LongTensor([8, 20]))

    x = torch.randn((2, 20, 256)).cuda()
    durator = Durator(256).cuda()
    start_time = time.time()
    y_pred = durator.forward_train(x, src_mask, 30)
    print(time.time() - start_time)

