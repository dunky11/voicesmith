# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.cuda
from numba import jit
from torch.autograd import Function
from numba import cuda
import math

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    I = tid

    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin

        # Wait for other threads in this block
        cuda.syncthreads()


# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(
    D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E
):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp(
                    (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma
                )
                E[k, i, j] = (
                    E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
                )

        # Wait for other threads in this block
        cuda.syncthreads()


# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](
            cuda.as_cuda_array(D.detach()),
            gamma.item(),
            bandwidth.item(),
            N,
            M,
            n_passes,
            cuda.as_cuda_array(R),
        )
        ctx.save_for_backward(D, R.clone(), gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1 : N + 1, 1 : M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](
            cuda.as_cuda_array(D_),
            cuda.as_cuda_array(R),
            1.0 / gamma.item(),
            bandwidth.item(),
            N,
            M,
            n_passes,
            cuda.as_cuda_array(E),
        )
        E = E[:, 1 : N + 1, 1 : M + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# ----------------------------------------------------------------------------------------------------------------------
#
# The following is the CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
# Credit goes to Kanru Hua.
# I've added support for batching and pruning.
#
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in range(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R


# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1 : N + 1, 1 : M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in range(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = (
                    E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
                )
    return E[:, 1 : N + 1, 1 : M + 1]


# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        R = torch.Tensor(compute_softdtw(D_, g_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_, b_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=1.0, bandwidth=None, dist_func=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

    def _get_func_dtw(self, dists):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        use_cuda = self.use_cuda

        if use_cuda and (
            dists.shape[1] > 1024 or dists.shape[2] > 1024
        ):  # We should be able to spawn enough threads in CUDA
            print(
                "SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)"
            )
            use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    def forward(self, dists):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """
        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(dists)
        return func_dtw(dists, self.gamma, self.bandwidth)


def get_log_likelihoods(x, means, stds):
    """
    args
    ----
    x: torch.Tensor of shape (b, c, t)
    means: torch.Tensor of shape (b, c, t)
    stds: torch.Tensor of shape (b, c, t)

    returns
    -------
    log likelihoods of shape (b, t)
    """
    assert x.shape == means.shape == stds.shape
    """initial_shape = x.shape
    x, means, stds = (
        x.permute((0, 2, 1)),
        means.permute((0, 2, 1)),
        stds.permute((0, 2, 1)),
    )
    x = x.reshape((x.shape[0] * x.shape[1], -1))
    means = means.reshape((means.shape[0] * means.shape[1], -1))
    stds = stds.reshape((stds.shape[0] * stds.shape[1], -1))"""
    dist = torch.distributions.Independent(torch.distributions.Normal(means, stds), 1)
    log_probs = dist.log_prob(x)
    return log_probs


def pairwise_log_likelihoods(x, means, stds, x_mask, stats_mask):
    """
    args
    ----
    x: torch.Tensor of shape (b, c, m)
    means: torch.Tensor of shape (b, c, n)
    stds: torch.Tensor of shape (b, c, n)
    x_mask: torch.Tensor of shape (b, 1, m) steps which will be masked
        correspond to 1 or True
    stats_mask: torch.Tensor of shape (b, 1, n) steps which will be masked
        correspond to 1 or True
    returns
    -------
    pairwise log likelihoods of shape (b, m, n), unmasked so it runs faster,
    you will have to mask the output yourself
    """
    assert x.shape[0] == means.shape[0] == stds.shape[0]
    assert x.shape[1] == means.shape[1] == stds.shape[1]
    assert means.shape[2] == stds.shape[2]
    b, c, m, n = x.shape[0], x.shape[1], x.shape[2], means.shape[2]
    # (b, c, m) => (b, c, m, 1)
    x = x.unsqueeze(-1)
    # (b, c, m, 1) => (b, c, m, n)
    x = x.expand(x.shape[0], x.shape[1], x.shape[2], n)
    # (b, c, n) => (b, c, 1, n)
    means, stds = means.unsqueeze(2), stds.unsqueeze(2)
    # (b, c, 1, n) => (b, c, m, n)
    means = means.expand(means.shape[0], means.shape[1], m, means.shape[3])
    stds = stds.expand(stds.shape[0], stds.shape[1], m, stds.shape[3])
    log_likelihoods = get_log_likelihoods(
        x.permute(0, 2, 3, 1), means.permute(0, 2, 3, 1), stds.permute(0, 2, 3, 1)
    )
    return log_likelihoods


def dtw_kl_divergence(
    x, means_same, stds_same, means_other, stds_other, mask_same, mask_other
):
    """Calculates the soft dynamic time warped KL divergence loss between the posterior
    and the prior as in https://arxiv.org/pdf/2205.04421.pdf equation 10 and 11.
    args
    ----
    x: torch.Tensor of shape (b, c, m)
    means_same: torch.Tensor of shape (b, c, m) Means of the isotropic gaussian
        x was sampled from
    stds_same: torch.Tensor of shape (b, c, m) Standard deviations of the isotropic
        gaussian x was sampled from
    means_other: torch.Tensor of shape (b, c, n) Means of the isotropic gaussian to evaluate
        the likelihood of f^-1(x) on
    stds_other: torch.Tensor of shape (b, c, n) Standard deviations of the isotropic gaussian to
        evaluate the likelihood of f^-1(x) on
    mask_same: torch.Tensor of shape (b, 1, m) Mask for the distribution x was sampled from.
        steps which will be masked should have a value of 1 or True.
    mask_other: torch.Tensor of shape (b, 1, n) Mask for the distribution f^-1(x) should be evaluated on
        steps which will be masked correspond to 1 or True
    """
    log_likelihoods_x = get_log_likelihoods(
        x.permute((0, 2, 1)),
        means_same.permute((0, 2, 1)),
        stds_same.permute((0, 2, 1)),
    )
    pairs = pairwise_log_likelihoods(
        x=x, means=means_other, stds=stds_other, x_mask=mask_same, stats_mask=mask_other
    )
    pairwise_kl = log_likelihoods_x.unsqueeze(-1) - pairs
    dtw = SoftDTW(use_cuda=pairwise_kl.is_cuda, gamma=0.01)
    # TODO can be parallelized
    losses = []
    for batch_idx in range(pairwise_kl.shape[0]):
        same_length = mask_same.shape[-1] - torch.sum(mask_same[batch_idx])
        other_length = mask_other.shape[-1] - torch.sum(mask_other[batch_idx])
        loss = dtw(
            pairwise_kl[batch_idx, :same_length, :other_length].unsqueeze(0).cuda()
        )
        losses.append(loss)
    loss = torch.mean(torch.cat(losses))
    return loss


if __name__ == "__main__":
    b, c, m, n = 10, 192, 30, 2

    """x = torch.ones(2, 10, 10, 10)
    loc = torch.zeros(2, 10, 10, 10)
    scale = torch.ones(2, 10, 10, 10) * 1

    print(get_log_likelihoods(x, loc, scale))

    x = torch.ones(2, 10, 10)
    loc = torch.zeros(2, 10, 10)
    scale = torch.ones(2, 10, 10) * 2

    print(get_log_likelihoods(x, loc, scale))

    x = torch.ones(2, 10)
    loc = torch.zeros(2, 10)
    scale = torch.ones(2, 10) * 3

    print(get_log_likelihoods(x, loc, scale))

    x = torch.FloatTensor(
        [
            [
                [1, 1],
                [1, 1],
            ]
        ]
    )
    means = torch.FloatTensor(
        [
            [
                [0, 0, 0],
                [0, 0, 0],
            ]
        ]
    )
    stds = torch.FloatTensor(
        [
            [
                [1, 1, 1],
                [1, 1, 1],
            ]
        ]
    )
    x_mask = torch.LongTensor(
        [
            [
                [0, 0],
            ]
        ]
    )
    stats_mask = torch.LongTensor(
        [
            [
                [0, 0, 0],
            ]
        ]
    )"""

    device = torch.device("cuda")

    b, c, m, n = 10, 256, 800, 100
    x = torch.ones(b, c, m).to(device) + 2
    means_same = torch.zeros_like(x).to(device)
    stds_same = torch.ones_like(x).to(device)
    means_other = torch.zeros(b, c, n).to(device)
    stds_other = torch.ones(b, c, n).to(device)
    x_mask = torch.zeros(b, 1, m).to(device).bool()
    stats_mask = torch.zeros(b, 1, n).to(device).bool()

    import time

    start_time = time.time()
    loss = dtw_kl_divergence(
        x=x,
        means_same=means_same,
        stds_same=stds_same,
        means_other=means_other,
        stds_other=stds_other,
        mask_same=x_mask,
        mask_other=stats_mask,
    )
    end_time = time.time() - start_time
    print(end_time)
    print(loss)
    """
    pairs = pairwise_log_likelihoods(
        x, means, stds, x_mask=x_mask, stats_mask=stats_mask
    )
    from scipy.stats import multivariate_normal
    y = multivariate_normal.logpdf(x=x[0, :, 0], mean=means[0, :, 0], cov=stds[0, :, 0] ** 2)
    print(y)
    print(pairs[0, 0, 0])
    m_idx, n_idx = 1, 1
    y = multivariate_normal.logpdf(x=x[0, :, m_idx], mean=means[0, :, n_idx], cov=stds[0, :, n_idx] ** 2)
    print(y)
    print(pairs[0, m_idx, n_idx])"""
