# -*- coding: utf-8 -*-
"""Utils."""

import datetime
import gzip
import hashlib
import logging
import lzma
import os.path
import pickle
import typing

import numpy as np
import pandas as pd
import tensorflow as tf

keras = tf.keras
kb = keras.backend
if typing.TYPE_CHECKING:
    from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import


# DataFrame


def df_to_dict(df: 'pd.DataFrame', cols: 'Sequence[str]') -> 'Dict[str, np.ndarray]':
    return {k: v.values for k, v in df[cols].to_dict('series').items()}


def hash_df(df: 'pd.DataFrame') -> 'str':
    return hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()


def shuffle_df(df: 'pd.DataFrame') -> 'Tuple[pd.DataFrame, np.ndarray]':
    index = np.arange(len(df))
    np.random.shuffle(index)
    return df.iloc[index], index


# Tensorflow


def tf_compatible_mul_no_nan(x: 'tf.Tensor', y: 'tf.Tensor') -> 'tf.Tensor':
    assert x.shape.is_compatible_with(y.shape), f'{x.shape} not compatible with {y.shape}'
    return tf.math.multiply_no_nan(x, y)


def tf_1_add(x: 'tf.Tensor') -> 'tf.Tensor':
    return tf.add(tf.ones_like(x), x)


def tf_1_sub(x: 'tf.Tensor') -> 'tf.Tensor':
    return tf.subtract(tf.ones_like(x), x)


# IO


def ts_string(form: 'str' = '%Y%m%d-%H%M%S', tz: 'int' = +8) -> 'str':
    utc = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    tzt = utc.astimezone(datetime.timezone(datetime.timedelta(hours=tz)))
    return tzt.strftime(form)


def available_path(path: 'str') -> 'str':
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(root + '_' + str(i) + ext):
        i += 1
    return root + '_' + str(i) + ext


def open_z(path: 'str', mode: 'str') -> 'IO':
    if path.endswith('.gz'):
        return gzip.open(path, mode=mode)  # pytype: disable=bad-return-type
    if path.endswith('.xz'):
        return lzma.open(path, mode=mode)  # pytype: disable=bad-return-type
    return open(path, mode=mode)


class PickleZ(object):
    """Pickle with auto compression based on file name."""

    @classmethod
    def load(cls, path: 'str') -> 'Any':
        with open_z(path, 'rb') as f:
            data = pickle.load(f)
        return data

    @classmethod
    def dump(cls, data: 'Any', path: 'str') -> 'None':
        with open_z(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# Logging


def log_line(*args, **kwargs) -> 'str':
    return ', '.join(f'{x}' for x in args) + ', ' + ', '.join(f'{k}={v}' for k, v in kwargs.items())


def logger_path_or_stream(name, path_or_stream) -> 'logging.Logger':
    logger = logging.Logger(name=name, level=logging.INFO)
    if isinstance(path_or_stream, str):
        handler = logging.FileHandler(path_or_stream, encoding='utf-8')
    else:
        handler = logging.StreamHandler(path_or_stream)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)
    return logger
