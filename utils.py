import copy
import torch.nn.functional as F
import torch
from torch import nn as nn, einsum, Tensor
import logging
from typing import Union, Tuple

import math
from torch.nn import BatchNorm1d, LayerNorm
import numpy as np
from numpy import ndarray
from common.utils.utils import exists


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def get_logger(filename, verbosity=1, name=None):
    """

    Args:
        filename:
        verbosity:
        name:

    Returns:

    """
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# class
class OffsetScale(nn.Module):
    r""""
    """
    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Affine transformation
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim=-2)


class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence ,(default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0, batch_first: bool = False) -> None:
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)
        self.batch_first = batch_first

    def forward(self, x: Tensor) -> Tensor:
        r"""Inputs of forward function

        Args:
            x: the sequence fed to the positional encoder model (required).

        Shape:
            x: [sequence length, batch size, embed dim] (batch_first=False) or
               [batch size, sequence length, embed dim] (batch_first=True)
            output: [sequence length, batch size, embed dim]
        """
        if self.batch_first:
            x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x).permute(1, 0, 2) if self.batch_first else self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = BatchNorm1d(d_model)

    def forward(self, x):
        seq_len = x.size(0)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(-1).expand(x.size()[:2])
        x = x + self.pos_embed(pos)
        return self.dropout(self.norm(x.permute(0, 2, 1)).permute(0, 2, 1))


def gaussian_kernel(mean, sigma):
    """probability density function of gaussian distribution.

    Args:
        mean: x - u
        sigma: \sigma in density function

    Shape:
        - mean: (N, N), symmetric matrix about distance
        - sigma: (N, 1) or a real number if fixed the sigma
    Returns:

    """
    normalize = 1 / (math.sqrt(2 * torch.pi) * sigma)
    # print('mean/sigma', mean / sigma, (mean / sigma).pow(2))
    return normalize * torch.exp(-0.5 * (mean / sigma).pow(2))


def prior_association(M: int, N: int, idx: int, sigma: float):
    """

    Args:
        M: M and N are two dimension of 2D matrix
        N: M and N are two dimension of 2D matrix
        idx: start point in M x N matrix
        sigma: control the peak of distribution. \sigma larger, peak smoother, otherwise jianrui. # TODO: replace jianrui.

    Returns:

    """
    center = torch.from_numpy(   # p denotes as (x-u)，u is the center of Gaussian kernel
        np.abs(
            np.mgrid[0: M, -idx: -idx + N][0] - np.mgrid[0: M, -idx: -idx + N][1])
    )
    # p = torch.clamp_max(p, 20)  # oherwise, large value in p cause 0 in gaussian kernel, and kl divergence be NaN
    gaussian = gaussian_kernel(center.float(), sigma)
    # print(gaussian)
    gaussian /= gaussian.sum(dim=-1).view(-1, 1)    # ensure row-sum is 1
    gaussian = torch.clamp_min(gaussian, 1e-16)
    return gaussian


def generate_prior(sigma: Tensor, label, T: int, L: int, layers: int, max_life: int = 125, window_size: int = 30):
    if exists(max_life):  # with max_life, rul label is 0-1
        prior = torch.stack(
            [torch.stack(
                [prior_association(T, L, L - label[i].cpu().detach().numpy() * max_life - window_size,
                                   # sigma[i, j]
                                   5
                                   )
                 for i in range(sigma.size(0))]  # mha_attn: (layers, batch_size, T, L)
            ) for _ in range(layers)]
        )

    else:  # without max_life, rul label is 0-100+
        prior = torch.stack(
            [prior_association(T,
                               L,
                               L - label[i].cpu().detach().numpy() - self.window_size,
                               # sigma
                               5
                               )
             for i in range(sigma.size(0))]
        )

    # prior = prior.repeat(mha_attn.size(0), 1, 1, 1)  # prior: (layers, batch_size, T, L)
    return prior