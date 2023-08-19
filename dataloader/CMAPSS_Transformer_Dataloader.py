import os
import random

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Dict
import torch
from pandas import DataFrame
from numpy import ndarray
from common.utils.utils import exists
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_rul_label(dataframe: DataFrame, max_life: Optional[int] = 125, smoothing: bool = False) -> DataFrame:
    r"""add RUL labels according to unit and cycle.

    Args:
        dataframe: The original dataframe. (required)
        max_life: if max_life is not None, the rul > 125 will be set as max_life (optional).
        smoothing: if smoothing == `True`, the rul > 125 will be added 0.1 per cycle (optional).

    Returns:
        The RUL-added dataframe at the last column.
    """
    for id_train in dataframe.index.unique():
        if exists(max_life) and smoothing is not True:
            dataframe.loc[id_train, 'RUL'] = -1 * dataframe.loc[id_train]['time'].apply(
                lambda x: x - dataframe.loc[id_train]['time'].max()
                if (x - dataframe.loc[id_train]['time'].max()) > -max_life else -max_life)
        elif exists(max_life) and smoothing is True:
            dataframe.loc[id_train, 'RUL'] = -1 * dataframe.loc[id_train]['time'].apply(
                lambda x: x - dataframe.loc[id_train]['time'].max()
                if (x - dataframe.loc[id_train]['time'].max()) > -max_life
                else (-125 + (x - dataframe.loc[id_train]['time'].max()) * 0.1))
        else:
            dataframe.loc[id_train, 'RUL'] = -1 * dataframe.loc[id_train]['time'].apply(
                lambda x: x - dataframe.loc[id_train]['time'].max())
    return dataframe


def load_train_dataset(file_path,
                       sub_dataset,
                       max_life: Optional[int] = None,
                       sensor_norm: Optional[str] = None,
                       smoothing: bool = False) -> Tuple[DataFrame, Optional[Dict]]:
    r"""load the train_set in the sub_dataset.

    Args:
        file_path: relative or absolute address of the directory `CMAPSS` (required).
        sub_dataset: sub_dataset name, FD001_prior to FD004 (required).
        max_life: piece-wise RUL, technically, 125. (optional)
        sensor_norm: if norm == ``minmax`` or ``z-score``, conduct minmax and z-score normalization among temporal dimension for each sensor,
        respectively. (optional)
        smoothing: label smoothing (optional)

    Returns:
        dataframe: the train_dataset
        statistic_num: the statistical property of the train_dataset
    """
    # get dataframe
    col_name = ['engine1', 'time', 'op_cond_1', 'op_cond_2', 'op_cond_3'] + ['sn_{}'.format(s + 1) for s in range(21)]
    dataframe = pd.read_csv(os.path.join(file_path, 'train_{0}.txt'.format(sub_dataset)),
                            header=None, names=col_name, delim_whitespace=True, index_col=0)

    # create RUL label
    dataframe = create_rul_label(dataframe, max_life=max_life, smoothing=smoothing)

    # drop stable features and operation features
    drop_features = ['time', 'op_cond_1', 'op_cond_2', 'op_cond_3', 'sn_1', 'sn_5', 'sn_6', 'sn_10', 'sn_16',
                     'sn_18', 'sn_19']
    dataframe.drop(drop_features, axis=1, inplace=True)

    # data normalization
    if exists(sensor_norm):  # dataframe-wise norm
        if sensor_norm == 'minmax':
            statistic_num = {'min': dataframe.iloc[:, :-1].min(), 'max': dataframe.iloc[:, :-1].max()}
        elif sensor_norm == 'z-score':
            statistic_num = {'mean': dataframe.iloc[:, :-1].mean(), 'std': dataframe.iloc[:, :-1].std()}
        else:
            raise TypeError('normalization method should be minmax or z-score, not {0}'.format(sensor_norm))
        return sensor_normalization(dataframe, method=sensor_norm, max_life=max_life, **statistic_num), statistic_num

    else:
        return dataframe, None


def load_test_dataset(file_path,
                      sub_dataset,
                      sensor_norm: Optional[str] = None,
                      max_life: Optional[int] = None,
                      min=None, max=None, mean=None, std=None) -> DataFrame:
    r"""

    Args:
        file_path: relative or absolute address of the directory `CMAPSS` (required).
        sub_dataset: sub_dataset name, FD001_prior to FD004 (required).
        max_life: piece-wise RUL, technically, 125. (optional)
        sensor_norm: if norm == ``minmax`` or ``z-score``, conduct minmax and z-score normalization among temporal dimension for each sensor,
        respectively. (optional)
        min:
        max:
        mean:
        std:

    Returns:

    """

    statistic_kwargs = {'min': min, 'max': max, 'mean': mean, 'std': std}

    col_name = ['engine1', 'time', 'op_cond_1', 'op_cond_2', 'op_cond_3'] + ['sn_{}'.format(s + 1) for s in range(21)]

    drop_features = ['time', 'op_cond_1', 'op_cond_2', 'op_cond_3', 'sn_1', 'sn_5', 'sn_6', 'sn_10', 'sn_16',
                     'sn_18', 'sn_19']

    dataframe = pd.read_csv(os.path.join(file_path, 'test_{0}.txt'.format(sub_dataset)),
                            header=None, names=col_name, delim_whitespace=True, index_col=0)
    rul_dataframe = pd.read_csv(os.path.join(file_path, 'RUL_{0}.txt'.format(sub_dataset)),
                                header=None, names=['RUL'], delim_whitespace=True)

    # create RUL label
    for id_test in dataframe.index.unique():
        if exists(max_life):
            dataframe.loc[id_test, 'RUL'] = -1 * dataframe.loc[id_test]['time'].apply(
                lambda x: x - dataframe.loc[id_test]['time'].max() - rul_dataframe.loc[id_test - 1].values[0]
                if (x - dataframe.loc[id_test]['time'].max() - rul_dataframe.loc[id_test - 1].values[
                    0]) > -max_life else -max_life)
        else:
            dataframe.loc[id_test, 'RUL'] = -1 * dataframe.loc[id_test]['time'].apply(
                lambda x: x - dataframe.loc[id_test]['time'].max() - rul_dataframe.loc[id_test - 1].values[0])

    # drop stable features and operation features
    dataframe.drop(drop_features, axis=1, inplace=True)

    # dataframe-wise normalization
    if exists(sensor_norm) and min is None and mean is None:    # min and mean are None, means test_set is normalized
        # by its ststistical property, otherwise by outer given one, generally from train_set.
        assert sensor_norm in {'minmax', 'z-score'}, 'norm should be minmax or z-score, not {}'.format(sensor_norm)
        if sensor_norm == 'minmax':
            statistic_num = {'min': dataframe.iloc[:, :-1].min(), 'max': dataframe.iloc[:, :-1].max()}
        elif sensor_norm == 'z-score':
            statistic_num = {'mean': dataframe.iloc[:, :-1].mean(), 'std': dataframe.iloc[:, :-1].std()}
        return sensor_normalization(dataframe, method=sensor_norm, max_life=max_life, **statistic_num)

    elif exists(sensor_norm) and (exists(min) or exists(mean)):
        return sensor_normalization(dataframe, method=sensor_norm, max_life=max_life, **statistic_kwargs)

    else:   # Do not normalization
        return dataframe


def slice_dataset_engine(dataframe: DataFrame,
                         window_size: int = 30,
                         engine_norm: str = None,
                         max_life: Optional[int] = None,
                         select_plat: Optional[int] = None) -> Tuple[list, list, list, list, dict]:
    r""" Slice the dataset according to unit and time window.
        It is for the downstreaming object `DataLoader` in pytorch.
        key: row slice, value: a slice that gives numpy index for the data that pertains to an engine1
        For instance, train_engine_slices = {..., '4': Slice(400, 480, ), '5': Slice(481, 560, ), ...}

    Shape:
        (n, input_channel=1, height=args.twlen, width=feature_num), where n is the number of samples.

    Args:
        dataframe: the dataset to be sliced by unit and time window. The index should be engine1 number (required).
        window_size: the size of time window, default: 30 (optional).
        engine_norm: engine-wise norm, data in each engine is normed separately by engine's statistical property (optional).
        max_life: piece-wise RUL.
        select_plat: if select_plat is not None, drop repeating rul=125 samples in each engine, and remain all samples if ``None``

    Returns:
        Tuple of src, tgt, y, case_x, case_y, where src refers to training samples, each full life cycle is a
    sample, tgt refers to piece while online predicting. and y refers to relative label. case_x, case_y refer to the
    last sample and relative label in each engine1, respectively. case_x, case_y is widely used in evaluation.

    """
    output = []
    cycle_length = []

    engines = dataframe.index.unique().values
    engine_slices = dict()
    for index, engine_num in enumerate(engines):
        row_name = dataframe.loc[engine_num].iloc[-1].name
        row_sl = dataframe.index.get_loc(row_name)
        engine_slices[engine_num] = row_sl

    x_piece, y_piece = [], []
    statistic_num = {}

    for _, engine_num in enumerate(tqdm(engines)):
        dataframe_engine = dataframe[engine_slices[engine_num]].copy()

        if exists(engine_norm):  # engine-wise norm
            statistic_num[engine_num] = {'min': dataframe_engine.iloc[:, :-1].min().values,
                                         'max': dataframe_engine.iloc[:, :-1].max().values,
                                         'mean': dataframe_engine.iloc[:, :-1].mean().values,
                                         'std': dataframe_engine.iloc[:, :-1].std().values}
            if not exists(max_life):
                dataframe_engine = sensor_normalization(dataframe_engine, method=engine_norm,
                                                        max_life=dataframe_engine.iloc[:, -1].max())
            else:
                dataframe_engine = sensor_normalization(dataframe_engine, method=engine_norm)

        engine_data = torch.tensor(np.array(dataframe_engine.iloc[:, :-1]), dtype=torch.float32)

        if window_size > engine_data.size(0):
            """print('The engine1 {0} is ignored, because the length ({1}) is smaller than time window ({2}).'.format(
                engine_num, engine_data.size(0), window_size))"""
            continue

        output.append(engine_data)
        # cycle_length.append(dataframe_engine.shape[0] - window_size + 1) # when we drop some samples, this line is wrong.

        if select_plat is not None:
            assert exists(max_life), 'max_life should not be `None` if drop_plat == `True`'
            for index in range(  # pick last 125 samples in each engine to reduce sample whose RUL is 125
                    np.max((window_size, dataframe_engine.shape[0] - max_life - select_plat)),
                    dataframe_engine.shape[0] + 1
            ):
                x_sample = np.array(dataframe_engine.iloc[index - window_size: index, :-1])
                y_sample = np.array(dataframe_engine.iloc[index - 1, -1])
                x_piece.append(torch.tensor(x_sample, dtype=torch.float32))
                y_piece.append(torch.tensor(y_sample, dtype=torch.float32))

            cycle_length.append(dataframe_engine.shape[0] + 1 - np.max((window_size,
                                                                        dataframe_engine.shape[0] - max_life - select_plat)))

        else:
            for index in range(
                    window_size, dataframe_engine.shape[0] + 1
            ):
                x_sample = np.array(dataframe_engine.iloc[index - window_size: index, :-1])
                y_sample = np.array(dataframe_engine.iloc[index - 1, -1])
                x_piece.append(torch.tensor(x_sample, dtype=torch.float32))
                y_piece.append(torch.tensor(y_sample, dtype=torch.float32))

            cycle_length.append(dataframe_engine.shape[0] - window_size + 1)

    if exists(engine_norm):
        min = np.mean([statistic_num[engine]['min'] for engine in engines], axis=0)
        max = np.mean([statistic_num[engine]['max'] for engine in engines], axis=0)
        mean = np.mean([statistic_num[engine]['mean'] for engine in engines], axis=0)
        std = np.mean([statistic_num[engine]['std'] for engine in engines], axis=0)

        statistic_num = {'min': min, 'max': max, 'mean': mean, 'std': std}  # to normalize test_data

    return output, x_piece, y_piece, cycle_length, statistic_num


def generate_evaluation_sample(dataframe: DataFrame, window_size: int = 30, engine_norm=None) -> \
        Tuple[list, list]:
    r""" Slice the dataset according to unit and time window.
        It is for the downstreaming object `DataLoader` in pytorch.
        key: row slice, value: a slice that gives numpy index for the data that pertains to an engine1
        For instance, train_engine_slices = {..., '4': Slice(400, 480, ), '5': Slice(481, 560, ), ...}

    Shape:
        (n, input_channel=1, height=args.twlen, width=feature_num), where n is the number of samples.

    Args:
        dataframe: The dataset to be sliced by unit and time window. The index should be engine1 number. (required)
        window_size:
        mode: ``train`` for training dataset, and ``test`` for test dataset. Default: ``train`` (optional).
        engine_norm

    Returns:
        Tuple of case_x, case_y, i.e. [tensor(..), tensor(..), ...], where
        src refers to training samples, each full life cycle is a sample, tgt refers to piece while online predicting.
        and y refers to relative label.
        case_x, case_y refer to the last sample and relative label in each engine1, respectively.
        case_x, case_y is widely used in evaluation.

    """
    engines = dataframe.index.unique().values
    engine_slices = dict()

    for index, engine_num in enumerate(engines):
        row_name = dataframe.loc[engine_num].iloc[-1].name
        row_sl = dataframe.index.get_loc(row_name)
        engine_slices[engine_num] = row_sl

    x_case = []
    y_case = []
    drop_num = []  # engine length < window size is dropped

    for _, engine_num in enumerate(tqdm(engines)):
        dataframe_engine = dataframe[engine_slices[engine_num]].copy()
        engine_data = torch.tensor(np.array(dataframe_engine.iloc[:, :-1]), dtype=torch.float32)

        if window_size > engine_data.size(0):
            drop_num.append(engine_num)
            continue

        x_sample = np.array(dataframe_engine.iloc[-window_size:, :-1])
        y_sample = np.array(dataframe_engine.iloc[-1, -1])
        x_case.append(torch.tensor(x_sample, dtype=torch.float32))
        y_case.append(torch.tensor(y_sample, dtype=torch.float32))

    print('dropped engines:{0}'.format(drop_num))
    return x_case, y_case


def sensor_normalization(data: DataFrame, method: str = 'minmax', max_life: Optional[int] = None,
                  min=None, max=None, mean=None, std=None) -> Union[DataFrame, ndarray]:
    r""" Normalize dataframe at column dimension. Normalization method includes: minmax, z-score

    Args:
        data: the data to be normalized, features in columns, and timestamps in rows. (required)
        method: normalization method, ``minmax`` or ``z-score``. (optional)
        max_life: if max_life exists, rul label is divided by max_life, else return original label. (optional)
        min: for test_dataframe, minimum is determined by train dataset. (optional)
        max: for test_dataframe, maximum is determined by train dataset. (optional)
        mean: for test_dataframe, mean is determined by train dataset. (optional)
        std: for test_dataframe, std is determined by train dataset. (optional)

    Returns:
        Normalized dataframe, same shape with input.
    """
    # TODO: consider how to deal with RUL label when z-score, and mean, std, min, max exists at the same time

    if method == 'z-score':
        if exists(mean) and exists(std):
            data.iloc[:, :-1] = (data.iloc[:, :-1] - mean) / std
        else:
            data.iloc[:, :-1] = (data.iloc[:, :-1] - data.iloc[:, :-1].mean()) / (data.iloc[:, :-1].std())

    elif method == 'minmax':
        if exists(min) and exists(max):
            data.iloc[:, :-1] = (data.iloc[:, :-1] - min) / (max - min)
            if exists(max_life):
                data.iloc[:, -1] = data.iloc[:, -1] / max_life
        else:
            data.iloc[:, :-1] = (data.iloc[:, :-1] - data.iloc[:, :-1].min()) / (
                    data.iloc[:, :-1].max() - data.iloc[:, :-1].min())
            if exists(max_life):
                data.iloc[:, -1] = data.iloc[:, -1] / max_life

    else:
        raise RuntimeError('Only support minmax and z-score normalization, rather than {0}'.format(method))

    return data


class CMAPSSTrainDataset(Dataset):
    r"""TODO: complete
        Args:
            readpath: file path directory,
            subdataset: FD001_prior - FD004 in CMAPSS dataset
            window_size: time window size (decoder input)
            norm: frame-wise norm
            engine_norm: engine-wise norm
            max_life: force sample with rul > 125 to rul = 125. If max_life != None, the rul label will be divided by
            max_life, ranging in [0, 1], otherwise, rul label remains original.
            drop_plat: a lot of 125 samples may cause gradient explosion. only pick the last 125 samples
    """

    def __init__(self, readpath: str, subdataset: str, window_size: int, norm=None, engine_norm=None,
                 max_life: Optional[int] = None, select_plat: Optional[int] = None, smoothing: bool = False,
                 task='end2end'):
        if norm is not None and engine_norm is not None:
            raise ValueError('engine_norm and norm cannot exist simultaneously')
        # if max_life is None and smoothing is not None:
        #     raise ValueError('smoothing is allowed when max_life exists')

        self.readpath = readpath
        self.subdataset = subdataset
        self.window_size = window_size
        self.norm = norm
        self.engine_norm = engine_norm
        self.max_life = max_life
        self.select_plat = select_plat
        self.smoothing = smoothing
        self.full_cycle, self.piece, self.label, self.engine_len, self.statistic_num = self.process(self)

    def visualization(self, engines: list, savepath: Optional[str] = None, showfig: bool = False) -> None:
        """ visualize multi-sensor data.

        Args:
            engines: engines to be visualized.
            savepath: if savepath is not None, the figure will be saved to the relative or absolute address.
            showfig: if showfig == `True`, fugure will show up.
        """

        if exists(savepath):
            if not Path(savepath).is_dir():
                os.makedirs(savepath, exist_ok=True)

        for engine_idx in engines:
            sensor_data = self.full_cycle[engine_idx]
            rul = self.label[np.sum(self.engine_len[:engine_idx]): np.sum(self.engine_len[:engine_idx + 1])] \
                if engine_idx != 0 else self.label[: self.engine_len[0]]
            rul = np.concatenate((np.ones(self.window_size), rul), axis=0)
            fig, ax1 = plt.subplots(1, 1)
            ax1.plot(sensor_data, linewidth=0.5)
            ax2 = ax1.twinx()
            ax2.plot(rul)
            ax1.set_xlabel('Cycle')
            ax1.set_ylabel('Sensor data')
            ax2.set_ylabel('RUL')
            ax1.grid(True)

            if showfig:
                plt.show()

            if exists(savepath):
                plt.savefig(os.path.join(savepath, '{0}.png'.format(engine_idx)))

            plt.close(fig)

    def __getitem__(self, idx):
        """i = None
        for i in range(1, len(self.engine_len)):
            if sum(self.engine_len[:i]) > idx:
                break
        i -= 1"""
        i = random.randint(0, len(self.full_cycle) - 1)
        """j = i
        while j == i:
            j = random.randint(0, len(self.full_cycle) - 1)"""
        return self.full_cycle[i], self.piece[idx], self.label[idx]
        # return self.full_cycle[i], self.full_cycle[j], self.piece[idx], self.label[idx]

    def __len__(self):
        assert len(self.piece) == len(self.label), 'the number of full_cycle, piece, and ' \
                                                                           'label should be same, but now are {0} {1}' \
                                                                           'respectively'.format(len(self.piece),
                                                                                                 len(self.label))
        return len(self.piece)

    @staticmethod
    def process(self):
        """slice dataframe by engine number (input of encoder), slice piece (input of decoder), and label.

        Returns:
            full_cycle (Tensors): full cycle 2D dataframe
            x (Tensors): tensor of 2D piece data sliced from full cycle
            y (Tensors): RUL label
            engine_len (ints): the number of pieces each engine produce
            statistic_num: statistical property of dataset, for processing test_dataframe
        """
        dataset, statistic_num = load_train_dataset(self.readpath, self.subdataset, max_life=self.max_life,
                                                    sensor_norm=self.norm, smoothing=self.smoothing)
        full_cycle, x, y, engine_len, engine_norm_statistic_num = slice_dataset_engine(dataset,
                                                                                       window_size=self.window_size,
                                                                                       max_life=self.max_life,
                                                                                       engine_norm=self.engine_norm,
                                                                                       select_plat=self.select_plat)

        if exists(statistic_num):
            return full_cycle, x, y, engine_len, statistic_num
        else:
            return full_cycle, x, y, engine_len, engine_norm_statistic_num


class CMAPSSTestDataset(Dataset):
    r"""TODO: complete
        Args:
            readpath: file path directory,
            subdataset: FD001_prior - FD004 in CMAPSS dataset
            window_size: time window size (decoder input)
            max_life: Teacher force sample with rul > 125 to rul = 125
    """

    def __init__(self, readpath: str, subdataset: str, window_size: int, max_life: Optional[int] = None,
                 norm=None,
                 min=None, max=None, mean=None, std=None):
        self.readpath = readpath
        self.subdataset = subdataset
        self.window_size = window_size
        self.max_life = max_life
        self.norm = norm

        statistic_kwargs = {'min': min, 'max': max, 'mean': mean, 'std': std}

        self.case_x, self.case_y = self.process(self, **statistic_kwargs)

    def __getitem__(self, idx):
        return self.case_x[idx], self.case_y[idx]

    def __len__(self):
        return len(self.case_x)

    """def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            print(self.n)
            output = self.__getitem__(self.n)
            self.n += 1
            return output
        else:
            raise StopIteration"""

    @staticmethod
    def process(self, min=None, max=None, mean=None, std=None):
        statistic_kwargs = {'min': min, 'max': max, 'mean': mean, 'std': std}

        test_dataset = load_test_dataset(self.readpath, self.subdataset, self.norm, max_life=self.max_life,
                                         **statistic_kwargs)  # dataset with rul

        case_x, case_y = generate_evaluation_sample(test_dataset, window_size=self.window_size)

        return case_x, case_y


class CMAPSSTestInstance(Dataset):
    r"""TODO: complete
        Args:
            readpath: file path directory,
            subdataset: FD001_prior - FD004 in CMAPSS dataset
            window_size: time window size (decoder input)
            max_life: Teacher force sample with rul > 125 to rul = 125
    """

    def __init__(self, readpath: str, subdataset: str, window_size: int, max_life: Optional[int] = None,
                 norm=None,
                 min=None, max=None, mean=None, std=None):
        self.readpath = readpath
        self.subdataset = subdataset
        self.window_size = window_size
        self.max_life = max_life
        self.norm = norm

        statistic_kwargs = {'min': min, 'max': max, 'mean': mean, 'std': std}

        self.case_x, self.case_y = self.process(self, **statistic_kwargs)

    def __getitem__(self, idx):
        return self.case_x[idx], self.case_y[idx]

    def __len__(self):
        return len(self.case_x)

    """def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            print(self.n)
            output = self.__getitem__(self.n)
            self.n += 1
            return output
        else:
            raise StopIteration"""

    @staticmethod
    def process(self, min=None, max=None, mean=None, std=None):
        statistic_kwargs = {'min': min, 'max': max, 'mean': mean, 'std': std}

        test_dataset = load_test_dataset(self.readpath, self.subdataset, self.norm, max_life=self.max_life,
                                         **statistic_kwargs)  # dataset with rul

        case_x, case_y = generate_evaluation_sample(test_dataset, window_size=self.window_size)

        return case_x, case_y