import math
from typing import Tuple, Union
from torch import Tensor
import torch
from numpy import ndarray
from model.rulformer import RULformer
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
from common.utils.utils import exists


def calculate_score(prediction: Tensor, label: Tensor, max_life: Optional[int] = None):
    """ prediction and label should in [0, 1]!!

    Args:
        prediction: (instance, 1)
        label: (instance, 1)

    Returns:

    """
    output = 0.
    if exists(max_life):
        prediction = prediction * max_life
        label = label * max_life

    for i in range(prediction.size(0)):
        if prediction[i] <= label[i]:
            output += (math.exp((label[i] - prediction[i]) / 13) - 1)
        else:
            output += (math.exp((prediction[i] - label[i]) / 10) - 1)

    return output


def rul_evaluation(model, train_ds, test_ds, criterion, PAD_IDX: int = 0, max_life: Optional[int] = 125,
                   need_weights: bool = False, device=None, dtype=None):
    """ evaluation model in each training epoch.

    Args:
        model: model
        train_ds: train-dataset, (torch.Dataset class) use its full cycle to compare.
        test_ds: test-dataset, (torch.Dataset class)
        criterion: criterion that calculate loss
        PAD_IDX: PAD_IDX that pad variant-length full cycle to be length-consistent
        max_life:
        need_weights: return cross-attention per sample averaged along layers.
        device: device name
        dtype: data type

    Returns:
        loss: rul_evaluation loss
    """
    factory_kwargs = {'device': device, 'dtype': dtype}
    model.eval()

    engine_num = len(train_ds.full_cycle)
    # cycle (T, D)
    train_ds.full_cycle = [torch.flip(cycle, dims=[0]) for cycle in train_ds.full_cycle]
    engine = pad_sequence(train_ds.full_cycle, batch_first=True, padding_value=PAD_IDX).to(device)  # (N, L, C)
    engine = torch.flip(engine, dims=[1])

    prediction = []

    with torch.no_grad():
        for piece, _ in test_ds:
            piece = piece.unsqueeze(dim=0).repeat(engine_num, 1, 1).to(device)  # (N, L, C)

            src_mask, tgt_mask, src_padding_mask, _ = RULformer.create_mask(engine,
                                                                  piece,
                                                                  batch_first=True,
                                                                  **factory_kwargs)

            output, prior, empirical, _ = model(engine, piece,
                                                src_key_padding_mask=src_padding_mask,
                                                memory_key_padding_mask=src_padding_mask,
                                                need_ass_dis=False
                                                )
            output = torch.mean(output, dtype=dtype)    # aggregation

            prediction.append(output.cpu())

    prediction = torch.stack(prediction)

    if exists(max_life):
        loss = criterion(max_life * prediction, max_life * torch.tensor(test_ds.case_y)).detach().cpu().numpy()
        score_loss = calculate_score(prediction, torch.tensor(test_ds.case_y), max_life=max_life)
    else:
        loss = criterion(prediction, torch.tensor(test_ds.case_y)).detach().cpu().numpy()
        score_loss = calculate_score(prediction, torch.tensor(test_ds.case_y))

    return loss, score_loss
