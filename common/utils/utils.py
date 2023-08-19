import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Union, Tuple, Sequence, Any
from torch import Tensor
from einops import rearrange, repeat


# helper functions
def exists(val: Any) -> bool:
    return val is not None


def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)


def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def torch_pad_nan(arr, left=0, right=0, dim=0):
    """
        padding np.nan at the left or right side of arr.
    Args:
        arr: np.ndarray, be padded
        left:
        right:
        dim:

    Returns:

    """
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def take_per_row(A: np.ndarray, indx: np.ndarray, num_elem: int) -> np.ndarray:
    """
    Args:
        A: Shape(batch_size, time_stamps, features)
        indx: Shape(batch_size), the start point of each time_instance
        num_elem: int, length of TS piece

    Returns:
        object:

    - Example
    ''>>> A = np.random.randint(1, 10, (4, 4, 4))
    ''>>> index = np.arange(4, dtype=int)
    ''>>> print(A[index[:, None], index[:, None]])
    [[[6 6 2 2]]    # first row of A[0]
    [[7 1 8 1]]     # second row of A[1]
    [[4 1 6 9]]     # third row of A[2]
    [[1 1 9 4]]]    # forth row of A[3]

    ''>>> print(index[:, None] + np.arange(4))
    [[0 1 2 3]
    [1 2 3 4]
    [2 3 4 5]
    [3 4 5 6]]

    """
    all_indx = indx[:, None] + np.arange(num_elem)  # Shape: (batch_size, 1) -> (batch_size, num_elem)

    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


def centerize_vary_length_series(x):
    r"""centerize_vary_length_series

    Args:
        x:

    Returns:

    """
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def collate_fn(data: Tensor):
    """

    Args:
        data: Variable length time series

    Returns:

    """
    data.sort(key=lambda x: len(x), reverse=True)
    data = pad_sequence(data, batch_first=True, padding_value=0)
    return data


def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B * T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B * T,
        size=int(B * T * p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res


def name_with_datetime() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def init_dl_program(device_name: str, seed: Optional[float] = None, use_cudnn: bool = True,
                    deterministic: bool = False, benchmark: bool = False, use_tf32: bool = False,
                    max_threads: Optional[int] = None):
    """

    Args:
        device_name:
        seed:
        use_cudnn:
        deterministic:
        benchmark:
        use_tf32:
        max_threads:

    Returns:

    """
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)

    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)

    device_name = [device_name]

    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    return devices if len(devices) > 1 else devices[0]


def check_create_dir(dir_path):
    my_dir = Path(dir_path)
    if not my_dir.is_dir():
        os.makedirs(my_dir)


def check_file(dir_path):
    my_dir = Path(dir_path)
    try:
        _ = my_dir.resolve()
    except FileNotFoundError:
        "The data file does not exists. Please ensure run the dataloader.CMAPSS_Dataloader.py before main.py, " \
        "and put the data at right position. "


class _ConstantPadNd(nn.Module):
    __constants__ = ['padding', 'value']
    value: float
    padding: Sequence[int]

    def __init__(self, value: float) -> None:
        super(_ConstantPadNd, self).__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, 'constant', self.value)

    def extra_repr(self) -> str:
        return 'padding={}, value={}'.format(self.padding, self.value)


def padding_instance(data: np.ndarray, temporal_len: int, dim: int = 0,
                     padding_value: Union[str, int] = 0) -> np.ndarray:
    """padding the time window to the goal temporal_len.

    Args:
        data:
        temporal_len: Goal length
        dim: The padding dim
        padding_value:

    Returns:

    Shape:
        data: (t, d), where t denotes time window, d denotes feature dimension.
    """

    return np.pad(data, ((temporal_len - data.shape[0], 0), (0, 0)), 'constant', constant_values=padding_value)


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    Args:
        dim: the dimension to be padded (dimension of time in sequences) (optional)
    """
    def __init__(self, dim: int = 0) -> None:
        self.dim = dim

    def pad_collate(self, batch) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch: list of (tensor, label) (required)

        Returns:
            xs: a tensor of all examples in 'batch' after padding
            ys: a LongTensor of all labels in batch
        """
        # find longest sequence

        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        batch = map(lambda x:
                    (pad_tensor(x[0], pad=max_len, dim=self.dim), x[1]), batch)
        # stack all
        xs = torch.stack(map(lambda x: x[0], batch), dim=0)
        ys = torch.LongTensor(map(lambda x: x[1], batch))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)
