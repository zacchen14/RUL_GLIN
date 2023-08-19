import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def association_discrepancy(batch_P, batch_S):
    """

    Args:
        batch_P: (batch, T, d)
        batch_S: (batch, T, d)

    Returns:

    """
    return (1 / batch_P.size(0)) * sum(
        [
            layer_association_discrepancy(P, S)
            for P, S in zip(batch_P, batch_S)
        ]
    )


def layer_association_discrepancy(Pl, Sl):
    """

    Args:
        Pl: attention distribution (rows, columns)
        Sl: attention distribution (rows, columns)

    Returns:
        ad_vector: (T), where T is the number of attention map rows.
    """
    rowwise_kl = lambda row: (
            F.kl_div(Pl[row, :].log(), Sl[row, :]) + F.kl_div(Sl[row, :].log(), Pl[row, :])
    )
    ad_vector = torch.concat(
        [rowwise_kl(row).unsqueeze(0) for row in range(Pl.shape[0])]
    )
    return ad_vector


def js_div(p_output: Tensor, q_output: Tensor, get_softmax: bool = False):
    """function that measures JS divergence between target and output logits

    Args:
        p_output: p in KLDiv(p|q)
        q_output: q in KLDiv(p|q)
        get_softmax:

    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output) / 2).log()
    # return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2
    return (KLDivLoss(log_mean_output, p_output))

