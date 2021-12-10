# -*- coding: utf-8 -*-
"""Core of delayed feedback experiments."""

import abc
from dataclasses import dataclass
import typing

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

import alib

keras = tf.keras
kb = tf.keras.backend
if typing.TYPE_CHECKING:
    from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import


@dataclass()
class ColumnConfig(object):
    # noinspection PyUnresolvedReferences
    """Configuration of columns.

    Args:
        click_ts: column name of click timestamp (int64).
        convert_ts: column name of conversion timestamp (Int64, with pd.NA for no conversion).
        features: mapping, column_name -> (shape, dtype, categorical_size, embedding_size).
        other_embedding_size: embedding size for columns that are not list in features.
    """
    # Need to set in __init__
    click_ts: 'str'
    convert_ts: 'str'
    features: 'Dict[str, Tuple[tuple, np.dtype, int, int]]'
    other_embedding_size: 'int'
    # Preset column names
    label = 'cc__label'
    prediction = 'cc__prediction'
    prophet = 'cc__prophet'
    appear_ts = 'cc__appear_ts'
    time_e = 'cc__time_e'
    mtl_tid = 'cc__mtl_tid'
    mtl_probs = 'cc__mtl_probs'
    # Preset dtypes
    label_dtype = np.int8
    ts_dtype = np.int64
    prediction_dtype = np.float32
    time_period_dtype = np.float32
    mtl_tid_dtype = np.int32


class MethodABC(abc.ABC):
    """Abstract class for a method."""
    description: 'str'

    def is_in(self, method_list: 'Sequence[Type[MethodABC]]') -> 'bool':
        return any(isinstance(self, method) for method in method_list)


class Methods(object):
    """Collections of all the methods."""

    @dataclass()
    class Prophet(MethodABC):
        description: 'str' = 'Prophet'

    @dataclass()
    class Waiting(MethodABC):
        window_size: 'int'
        description: 'str' = 'Waiting'

    @dataclass()
    class PU(MethodABC):
        description: 'str' = 'PU'

    @dataclass()
    class FNW(MethodABC):
        description: 'str' = 'FNW'

    @dataclass()
    class FNC(MethodABC):
        description: 'str' = 'FNC'

    @dataclass()
    class FTP(MethodABC):
        # noinspection PyUnresolvedReferences
        """Follow the Prophet.

        Args:
            task_window_sizes: a list of the waiting window size for each task.
                The last one will be regarded as the maximum delay time.
            n_more_shared_layers: how many layers above the embeddings should be shared across tasks.
            fp_train_min_ts: the minimum timestamp for training the policy.
            ema_loss_init: Set to the (approximate) loss for the cvr model.
                It takes effect if gradients of the shared layers are enabled for policy.
            enable_gradients: a list of length n_task + 1.
                Whether gradients of the shared layers are enabled (1 / 0) for each task and the policy.
            description: string.
        """
        task_window_sizes: 'Sequence[int]'
        n_more_shared_layers: 'int'
        fp_train_min_ts: 'int'
        ema_loss_init: 'float'
        enable_gradients: 'Sequence[int]'
        description: 'str' = 'FTP'


class DataProvider(object):
    """Hold the underlying data and simulate the online streaming."""

    def __init__(self,
                 df: 'pd.DataFrame',
                 cc: 'ColumnConfig',
                 fast_index: 'Union[Tuple[int, int, int], Dict[Tuple[int, int], Tuple[int, int]], None]' = None,
                 ):
        """
        Args:
            df: pd.DataFrame with all necessary columns.
            cc: ColumnConfig.
            fast_index: use for faster indexing, which can be:
                Tuple (start_timestamp, stop_timestamp, step): generate fast_index according to it;
                Dict (start_timestamp, stop_timestamp) -> (start_index, stop_index): reuse this fast_index;
                None: disable.
        """
        self.df = df
        self.cc = cc
        self.fast_index: 'Union[Dict[Tuple[int, int], Tuple[int, int]], None]' = None
        if isinstance(fast_index, dict):
            self.fast_index = fast_index
        elif fast_index:
            indexes = {}
            ts_step = fast_index[2]
            for ts in tqdm(range(fast_index[0], fast_index[1], fast_index[2]), desc='Fast Index'):
                index = self._get_period_by(cc.click_ts, ts_start=ts, ts_end=ts + ts_step).index
                if len(index) == 0:
                    indexes[(ts, ts + ts_step,)] = (0, 0,)
                    continue
                assert len(index) == index[-1] - index[0] + 1
                indexes[(ts, ts + ts_step,)] = (index[0], index[-1] + 1,)
            self.fast_index = indexes

    def setup_write_back_col(self, col: 'str', fill_value: 'Any', dtype: 'Union[Type[np.generic], str]'):
        """Setup a column for writing back."""
        self.df[col] = fill_value
        self.df[col] = self.df[col].astype(dtype)

    @classmethod
    def from_path(cls, path: 'str', cc: 'ColumnConfig') -> 'DataProvider':
        """Construct DataProvider from a feather file path and the ColumnConfig."""
        df = pd.read_feather(path)
        return cls(df=df, cc=cc)

    def _get_period_by(self, col: 'str', ts_start: 'int', ts_end: 'int') -> 'pd.DataFrame':
        """Get a subset subjected to the timestamp of col between ts_start and ts_end.
        The DataFrame index should not be changed here.
        """
        if col == self.cc.click_ts and self.fast_index is not None:
            low, high = self.fast_index.get((ts_start, ts_end,), (0, -1,))
            if high >= low:
                return self.df.iloc[low:high, :].copy()
        df = self.df[self.df[col].notnull() & (self.df[col] >= ts_start) & (self.df[col] < ts_end)]
        return df.copy()

    def serving_data(self, click_ts_start: 'int', click_ts_end: 'int') -> 'pd.DataFrame':
        """Get a subset to simulate online serving.
        The DataFrame index should not be changed here.
        """
        cc = self.cc
        df_click = self._get_period_by(cc.click_ts, ts_start=click_ts_start, ts_end=click_ts_end)
        df_click[cc.label] = df_click[cc.convert_ts].notnull().astype(self.cc.label_dtype)
        return df_click

    def serving_write_back(self, index: 'np.ndarray', column_indexer: 'Any', values: 'np.ndarray'):
        """Write the predictions back into the DataFrame for future use."""
        self.df.loc[index, column_indexer] = values

    def indexing_data(self, index: 'np.ndarray', ts_now: 'Union[int, None]') -> 'pd.DataFrame':
        """Get a subset according to the index."""
        cc = self.cc
        df = self.df.loc[index].copy()
        if ts_now is not None:
            df[cc.label] = df[cc.convert_ts].le(ts_now).fillna(False).astype(self.cc.label_dtype)
        else:
            df[cc.label] = df[cc.convert_ts].notnull().astype(self.cc.label_dtype)
        return df

    def _sort_appear(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Sort the DataFrame according to the appear timestamp."""
        return df.sort_values(self.cc.appear_ts, ascending=True, ignore_index=True)

    def get_real(self, click_ts_start: 'int', click_ts_end: 'int') -> 'pd.DataFrame':
        """Get real training data of a period."""
        df_click = self.serving_data(click_ts_start=click_ts_start, click_ts_end=click_ts_end)
        return df_click

    def get_assign_negative(self, click_ts_start: 'int', click_ts_end: 'int',
                            ts_now: 'int', time_period_scale: 'Union[float, None]' = None) -> 'pd.DataFrame':
        """Get observed training data of a period. Samples without feedback yet are mark as negative."""
        cc = self.cc

        df_click = self._get_period_by(cc.click_ts, ts_start=click_ts_start, ts_end=click_ts_end)
        df_click[cc.label] = df_click[cc.convert_ts].le(ts_now).fillna(False).astype(cc.label_dtype)

        if time_period_scale is not None:
            time_e_ts = (ts_now - df_click[cc.click_ts])
            df_click[cc.time_e] = (time_e_ts / time_period_scale).astype(cc.time_period_dtype)

        return df_click

    def get_fake_negative(self, click_ts_start: 'int', click_ts_end: 'int') -> 'pd.DataFrame':
        """Get fake negative data for a period. Positive samples will be duplicated when received feedback."""
        cc = self.cc

        df_click = self._get_period_by(cc.click_ts, ts_start=click_ts_start, ts_end=click_ts_end)
        df_click[cc.label] = 0

        df_convert = self._get_period_by(cc.convert_ts, ts_start=click_ts_start, ts_end=click_ts_end)
        df_convert[cc.label] = 1

        df_concat = pd.concat([df_click, df_convert], ignore_index=True)
        df_concat[cc.label] = df_concat[cc.label].astype(cc.label_dtype)
        return alib.utils.shuffle_df(df_concat)[0]


class BaseAgent(abc.ABC):
    """Abstract class for an agent.
    An agent should define how to build a model and how to do predictions / training.
    """

    def __init__(self,
                 method: 'MethodABC',
                 data_provider: 'DataProvider'
                 ):
        self.method = method
        self.data_provider = data_provider

        self.cc = data_provider.cc
        self.model_info: 'Dict[str, Any]' = {}  # set in reset_model

    def init(self):
        self.reset_model(verbose=True)
        self.reset_data()

    def reset_model(self, verbose=False):
        keras.backend.clear_session()
        self.model_info.clear()
        self._build_model(verbose=verbose)

    def reset_data(self):
        cc = self.cc
        self.data_provider.setup_write_back_col(col=cc.prediction, fill_value=np.nan, dtype=cc.prediction_dtype)

    @abc.abstractmethod
    def _build_model(self, verbose: 'bool'):
        """ Build model and write information to self.model_info """
        pass

    @abc.abstractmethod
    def serve_and_train(self, click_ts_start: 'int', click_ts_end: 'int', need_train: 'bool', need_evaluate: 'bool'):
        """Perform a (simulated) streaming step:
        Serve on data [click_ts_start, click_ts_end). Write log if necessary.
        If need_evaluate, update metrics.
        If need_train, start training. Training data should be collected according to the method.
        """
        pass

    def get_evaluation_results(self,
                               click_ts_start: 'int',
                               click_ts_end: 'int',
                               prediction_col: 'Union[str, None]' = None,
                               ) -> 'Dict[str, Any]':
        cc = self.cc
        if prediction_col is None:
            prediction_col = cc.prediction
        df_eval = self.data_provider.serving_data(click_ts_start=click_ts_start, click_ts_end=click_ts_end)
        results = dict(zip(
            alib.evaluation.cxr_metrics,
            alib.evaluation.cxr_sklearn_metrics(
                y_true=df_eval[cc.label].values,
                y_pred=df_eval[prediction_col].values,
            ),
        ))
        return results


def run_streaming(agent: 'BaseAgent',
                  min_ts: 'int',
                  max_ts: 'int',
                  step_ts: 'int',
                  eval_ts: 'int',
                  ) -> 'Dict[str, Any]':
    """Streaming simulation."""
    agent.init()
    for ts_start in tqdm(range(min_ts, max_ts, step_ts)):
        ts_end = ts_start + step_ts
        need_evaluation = (ts_start >= eval_ts)
        agent.serve_and_train(click_ts_start=ts_start, click_ts_end=ts_end,
                              need_train=True, need_evaluate=need_evaluation)
    return agent.get_evaluation_results(click_ts_start=eval_ts, click_ts_end=max_ts)
