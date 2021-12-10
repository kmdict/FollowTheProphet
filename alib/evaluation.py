# -*- coding: utf-8 -*-
"""Library for evaluation."""

import abc
import typing

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics as sklearn_metrics

keras = tf.keras
kb = keras.backend
if typing.TYPE_CHECKING:
    from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import


class BiasGlobalMetrics(tf.keras.metrics.Metric):
    """Keras metrics for computing global bias."""

    def __init__(self, name='bias_global', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_mean: 'tf.Variable' = self.add_weight(name='true_mean', initializer='zeros', dtype=tf.float32)
        self.pred_mean: 'tf.Variable' = self.add_weight(name='pred_mean', initializer='zeros', dtype=tf.float32)
        self.count: 'tf.Variable' = self.add_weight(name='count', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):  # pylint: disable=unused-argument
        y_true = kb.cast(y_true, tf.float32)
        y_pred = kb.cast(y_pred, tf.float32)
        count = kb.cast(kb.shape(y_true)[0], tf.float32)

        self.true_mean.assign((self.true_mean * self.count + kb.mean(y_true) * count) / (self.count + count))
        self.pred_mean.assign((self.pred_mean * self.count + kb.mean(y_pred) * count) / (self.count + count))
        self.count.assign_add(count)

    def result(self) -> 'tf.Tensor':
        return (self.pred_mean - self.true_mean) / self.true_mean

    def reset_states(self):
        self.true_mean.assign(0.)
        self.pred_mean.assign(0.)
        self.count.assign(0.)


def bias_global(y_true: 'np.ndarray', y_pred: 'np.ndarray'):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return (np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true)


cxr_metrics = ('LogLoss', 'ROC-AUC', 'PR-AUC', 'Bias-Global')
n_cxr_metrics = len(cxr_metrics)


def cxr_keras_metrics(prefix: 'str' = '') -> 'Tuple[keras.metrics.Metric, ...]':
    return (
        keras.metrics.BinaryCrossentropy(name=prefix + 'LogLoss'),
        keras.metrics.AUC(curve='ROC', name=prefix + 'ROC-AUC'),
        keras.metrics.AUC(curve='PR', name=prefix + 'PR-AUC'),
        BiasGlobalMetrics(name=prefix + 'Bias-Global'),
    )


def cxr_sklearn_metrics(y_true: 'np.ndarray', y_pred: 'np.ndarray') -> 'Tuple[float, ...]':
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return (
        sklearn_metrics.log_loss(y_true=y_true, y_pred=y_pred, labels=[0, 1]),
        sklearn_metrics.roc_auc_score(y_true=y_true, y_score=y_pred, labels=[0, 1]),
        sklearn_metrics.average_precision_score(y_true=y_true, y_score=y_pred),
        bias_global(y_true=y_true, y_pred=y_pred),
    )


def get_errors(y_true: 'np.ndarray', y_pred: 'np.ndarray') -> 'float':
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    abs_error = np.mean(np.abs(y_true - y_pred))
    return float(abs_error)


class Metrics(abc.ABC):
    """Abstract class."""

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def result(self):
        pass


class EmptyMetrics(Metrics):
    """A pseudo metrics that always return a fixed value."""

    def __init__(self, ret: 'Any'):
        self.ret = ret

    def reset(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def result(self) -> 'Any':
        return self.ret


class StreamingWeightedMeanMetrics(Metrics):
    """A metrics for computing the average values in streaming."""

    def __init__(self, n_metrics: 'int'):
        self.count = 0
        self.mean = np.zeros(shape=(n_metrics,), dtype=np.float32)

    def reset(self):
        self.count = 0
        self.mean.fill(0.)

    def update(self, count: 'int', mean: 'np.ndarray'):
        mean = np.array(mean)
        self.mean = (self.count * self.mean + count * mean) / (self.count + count)
        self.count = self.count + count

    def result(self) -> 'np.ndarray':
        return self.mean


class StreamingAUCMetrics(Metrics):
    """A metrics for computing (approximate) ROC-AUC and PR-AUC in streaming."""

    def __init__(self, n_classes: 'int' = 2, scale: 'int' = 1000000):
        self.n_classes = n_classes
        self.scale = scale
        self.counts: 'Dict[int, np.ndarray]' = {i: np.zeros(shape=(scale + 1,), dtype=int) for i in range(n_classes)}

    def reset(self):
        for i in range(self.n_classes):
            self.counts[i].fill(0)

    def update(self, y_true: 'np.ndarray', y_pred: 'np.ndarray'):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        for i in range(self.n_classes):
            array = y_pred[y_true == i]
            array = np.rint(array * self.scale).astype(int)
            self.counts[i] = self.counts[i] + np.bincount(array, minlength=self.scale + 1)

    def result(self) -> 'np.ndarray':
        y_true = np.concatenate([np.full(shape=(self.scale + 1), fill_value=i) for i in range(self.n_classes)])
        y_pred = np.concatenate([(np.arange(self.scale + 1) / self.scale) for _ in range(self.n_classes)])
        weights = np.concatenate([self.counts[i] for i in range(self.n_classes)])
        index = (weights != 0)
        y_true = y_true[index]
        y_pred = y_pred[index]
        weights = weights[index]
        return np.array([
            sklearn_metrics.roc_auc_score(y_true=y_true, y_score=y_pred, sample_weight=weights, labels=[0, 1]),
            sklearn_metrics.average_precision_score(y_true=y_true, y_score=y_pred, sample_weight=weights),
        ])


class StreamingBiasMetrics(Metrics):
    """A metrics for computing field-level bias in streaming."""

    def __init__(self, size: 'int'):
        self.size = size
        self.counts = np.zeros(shape=(size,), dtype=np.float32)
        self.sum_true = np.zeros(shape=(size,), dtype=np.float32)
        self.sum_pred = np.zeros(shape=(size,), dtype=np.float32)

    def reset(self):
        self.sum_pred.fill(0.)
        self.sum_pred.fill(0.)

    def update(self, y_true: 'np.ndarray', y_pred: 'np.ndarray', index: 'Union[np.ndarray, None]'):
        if index is None:
            if self.size != 1:
                raise ValueError
            index = np.zeros_like(y_true, dtype=int)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        index = np.array(index)
        for i in np.unique(index):
            pos = (index == i)
            self.counts[i] += np.sum(pos)
            self.sum_true[i] += np.sum(y_true[pos])
            self.sum_pred[i] += np.sum(y_pred[pos])

    def result(self, min_counts=200, min_sum=5) -> 'float':
        pos = np.logical_and((self.counts >= min_counts), (self.sum_true >= min_sum))
        if np.sum(pos) == 0:
            return 0.
        counts = self.counts[pos]
        sum_true = self.sum_true[pos]
        sum_pred = self.sum_pred[pos]
        bias_rel = np.abs(sum_pred - sum_true) / sum_true
        bias_weighted = np.sum(bias_rel * counts) / np.sum(counts)
        return bias_weighted


def ece(y_true, y_pred, n_bins=100) -> 'float':
    gdf = pd.DataFrame(
        {'y_true': y_true, 'y_pred': y_pred, 'bin': np.floor(y_pred * n_bins).astype(np.int)}
    ).groupby('bin')
    bdf = gdf.mean()
    bdf['count'] = gdf.count()['y_true']
    return np.average(np.abs(bdf['y_true'] - bdf['y_pred']), weights=bdf['count'])
