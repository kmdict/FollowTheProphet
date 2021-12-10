# -*- coding: utf-8 -*-
"""Implement of delayed feedback methods."""

import abc
import collections
from dataclasses import dataclass
import typing

import numpy as np
import pandas as pd
import tensorflow as tf

import alib
import delay.core

keras = tf.keras
kb = keras.backend
Methods = delay.core.Methods
TF1ADD = alib.utils.tf_1_add
TF1SUB = alib.utils.tf_1_sub
TF_MUL = alib.utils.tf_compatible_mul_no_nan
if typing.TYPE_CHECKING:
    from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import


def day_hour_str(ts: 'int') -> 'str':
    return f'{ts // (24 * 3600)}d'.rjust(3) + f'{ts % (24 * 3600) // 3600}h'.rjust(3)


def pu_loss_from_logits(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
    pred_logits = y_pred
    # noinspection PyUnusedLocal
    y_pred = None
    loss_y1 = TF_MUL(y_true, tf.log_sigmoid(pred_logits) - tf.log_sigmoid(-pred_logits))
    loss_y0 = TF_MUL(TF1SUB(y_true), tf.log_sigmoid(-pred_logits))
    loss = kb.squeeze(- loss_y1 - loss_y0, -1)
    tf.assert_rank(loss, 1)
    return loss


def weighted_log_loss_from_logits(y_true: 'tf.Tensor', pred_logits: 'tf.Tensor', weights: 'tf.Tensor') -> 'tf.Tensor':
    log_loss = keras.losses.binary_crossentropy(y_true=y_true, y_pred=pred_logits, from_logits=True)
    loss = TF_MUL(weights, log_loss)
    tf.assert_rank(loss, 1)
    return loss


def fnw_loss_from_logits(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
    pred_logits = y_pred
    # noinspection PyUnusedLocal
    y_pred = None
    pred_probs_no_grad = kb.stop_gradient(kb.sigmoid(pred_logits))
    positive_weights = TF1ADD(pred_probs_no_grad)
    negative_weights = TF_MUL(TF1SUB(pred_probs_no_grad), TF1ADD(pred_probs_no_grad))
    union_weights = TF_MUL(y_true, positive_weights) + TF_MUL(TF1SUB(y_true), negative_weights)
    return weighted_log_loss_from_logits(y_true=y_true, pred_logits=pred_logits, weights=kb.squeeze(union_weights, -1))


fake_negative_calibration_layer_config: 'alib.model.LayersClassConfigType' = [
    (keras.layers.Lambda, dict(
        function=kb.clip,
        arguments=dict(min_value=0.0 + kb.epsilon(), max_value=0.5 - kb.epsilon()),
        name='clip'
    )),
    (keras.layers.Lambda, dict(
        function=lambda y: (y / (1. - y)),
        name='calibration'
    )),
]


def prophecy_abs_error(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
    y_true = tf.broadcast_to(y_true, kb.shape(y_pred))
    return kb.abs(y_pred - y_true)


def get_hot_log_loss_from_logits_fn(hot: 'tf.Tensor', weight: 'tf.Tensor', ema_var: 'Union[tf.Variable, None]'
                                    ) -> 'Callable[[tf.Tensor, tf.Tensor], tf.Tensor]':
    hot_no_grad = kb.stop_gradient(hot)
    tf.assert_rank(hot_no_grad, 1)
    weight_no_grad = kb.stop_gradient(weight)
    tf.assert_scalar(weight_no_grad)

    def hot_log_loss_from_logits(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
        loss_element_wise = weighted_log_loss_from_logits(y_true=y_true, pred_logits=y_pred, weights=hot_no_grad)
        tf.assert_rank(loss_element_wise, 1)
        count = kb.cast(kb.sum(hot_no_grad, keepdims=False), tf.float32)
        tf.assert_scalar(count)
        batch_size = kb.cast(kb.shape(y_pred)[0], tf.float32)
        tf.assert_scalar(batch_size)
        if ema_var is not None:
            loss_scalar = tf.math.divide_no_nan(kb.sum(loss_element_wise, axis=0, keepdims=False), count)
            tf.assert_scalar(loss_scalar)
            count_not_zero = tf.math.divide_no_nan(count, count)
            ema_decay = count_not_zero * 0.9 + TF1SUB(count_not_zero) * 1.
            ema_op = tf.assign(ema_var, ema_decay * ema_var + TF1SUB(ema_decay) * loss_scalar)
            with tf.control_dependencies([weight_no_grad, loss_element_wise, batch_size, count, ema_op]):
                weighted_loss = tf.math.divide_no_nan(weight_no_grad * loss_element_wise * batch_size, count)
        else:
            weighted_loss = tf.math.divide_no_nan(weight_no_grad * loss_element_wise * batch_size, count)
        return weighted_loss

    return hot_log_loss_from_logits


def get_hot_ce_loss_from_logits_fn(hot: 'tf.Tensor', mtl_probs: 'tf.Tensor',
                                   weight: 'tf.Tensor', ema_var: 'Union[tf.Variable, None]'
                                   ) -> 'Callable[[tf.Tensor, tf.Tensor], tf.Tensor]':
    hot_no_grad = kb.stop_gradient(hot)
    tf.assert_rank(hot_no_grad, 1)
    mtl_probs_no_grad = kb.stop_gradient(mtl_probs)
    tf.assert_rank(mtl_probs_no_grad, 2)
    weight_no_grad = kb.stop_gradient(weight)
    tf.assert_scalar(weight_no_grad)

    def hot_selector_loss_from_logits(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> 'tf.Tensor':
        loss_element_wise = keras.losses.sparse_categorical_crossentropy(
            kb.argmin(prophecy_abs_error(y_true=y_true, y_pred=mtl_probs_no_grad), axis=-1),
            y_pred,
            from_logits=True,
        )
        tf.assert_rank(loss_element_wise, 1)
        loss_element_wise = TF_MUL(loss_element_wise, hot_no_grad)
        count = kb.cast(kb.sum(hot_no_grad, keepdims=False), tf.float32)
        tf.assert_scalar(count)
        batch_size = kb.cast(kb.shape(y_pred)[0], tf.float32)
        tf.assert_scalar(batch_size)
        if ema_var is not None:
            loss_scalar = tf.math.divide_no_nan(kb.sum(loss_element_wise, axis=0, keepdims=False), count)
            tf.assert_scalar(loss_scalar)
            count_not_zero = tf.math.divide_no_nan(count, count)
            ema_decay = count_not_zero * 0.9 + TF1SUB(count_not_zero) * 1.
            ema_op = tf.assign(ema_var, ema_decay * ema_var + TF1SUB(ema_decay) * loss_scalar)
            with tf.control_dependencies([weight_no_grad, loss_element_wise, batch_size, count, ema_op]):
                weighted_loss = tf.math.divide_no_nan(weight_no_grad * loss_element_wise * batch_size, count)
        else:
            weighted_loss = tf.math.divide_no_nan(weight_no_grad * loss_element_wise * batch_size, count)
        return weighted_loss

    return hot_selector_loss_from_logits


class DNNBaseAgent(delay.core.BaseAgent, metaclass=abc.ABCMeta):
    """Base class for an agent with dnn."""

    def __init__(self,
                 method: 'delay.core.MethodABC',
                 data_provider: 'delay.core.DataProvider',
                 intermediate_layers_config_fn: 'Callable[[], alib.model.LayersClassConfigType]',
                 fit_predict_kwargs: 'Dict[str, Any]',
                 ):
        super().__init__(method, data_provider)
        self.intermediate_layers_config_fn = intermediate_layers_config_fn
        self.fit_predict_kwargs = fit_predict_kwargs

    @property
    def feature_inputs(self) -> 'List[keras.layers.Input]':
        return [ie.keras_input for ie in self.model_info['feature_ie_list']]

    def _dnn_embeddings(self, ie_list: 'Sequence[alib.model.InputExtension]', prefix: 'str') -> 'tf.Tensor':
        """Build embedding layers. Return concat_embedding. """
        embedding_layer_list = [
            ie.to_embedding_layer(name_pattern=prefix + 'emb_{input_name}') for ie in ie_list
        ]
        concat_embedding_layer = keras.layers.Concatenate(name=prefix + 'concat_embedding')
        embeddings = [
            layer(ie.keras_input) for layer, ie in zip(embedding_layer_list, ie_list)
        ]
        concat_embedding = concat_embedding_layer(embeddings)

        self.model_info[prefix + 'embedding_layer_list'] = embedding_layer_list
        self.model_info[prefix + 'concat_embedding_layer'] = concat_embedding_layer
        self.model_info[prefix + 'embeddings'] = embeddings
        self.model_info[prefix + 'concat_embedding'] = concat_embedding

        return concat_embedding


class SimpleDNNAgent(DNNBaseAgent, metaclass=abc.ABCMeta):
    """An agent with a simple DNN as model."""

    @property
    def training_extra_cols(self) -> 'List[str]':
        return self.model_info['training_extra_cols']

    def reset_data(self):
        super().reset_data()
        cc = self.cc
        if isinstance(self.method, Methods.Prophet):
            self.data_provider.setup_write_back_col(col=cc.prophet, fill_value=np.nan, dtype=cc.prediction_dtype)

    def _build_model(self, verbose: 'bool'):
        cc = self.cc
        method = self.method
        feature_ie_list = [
            alib.model.InputExtension(col, *col_args) for col, col_args in self.cc.features.items()
        ]
        self.model_info['feature_ie_list'] = feature_ie_list
        concat_embedding = self._dnn_embeddings(feature_ie_list, prefix='simple/')
        self.model_info['training_extra_cols'] = []

        # Layers for simple model
        simple_layers = alib.model.build_sequential(
            self.intermediate_layers_config_fn(),
            name_prefix='simple/',
            name='simple/net',
        )
        simple_logit_layer = keras.layers.Dense(units=1, activation=None, name='simple/logit')
        simple_out_sigmoid_layer = keras.layers.Lambda(keras.activations.sigmoid, name='simple/out_sigmoid')
        simple_out_squeeze_layer = keras.layers.Lambda(kb.squeeze, name='simple/out_squeeze', arguments=dict(axis=-1))

        # Simple Forward
        simple_net = simple_layers(concat_embedding)
        simple_logit = simple_logit_layer(simple_net)
        simple_out_sigmoid = simple_out_sigmoid_layer(simple_logit)
        if isinstance(method, Methods.FNC):
            simple_out_sigmoid = alib.model.build_sequential(
                fake_negative_calibration_layer_config,
                name_prefix='simple/fnc/',
                name='simple/fnc'
            )(simple_out_sigmoid)
        simple_out_squeeze = simple_out_squeeze_layer(simple_out_sigmoid)
        self.model_info['serving'] = keras.Model(self.feature_inputs, simple_out_squeeze)

        # Training
        extra_inputs = []
        if method.is_in([Methods.Prophet, Methods.Waiting, Methods.FNC]):
            loss = keras.losses.BinaryCrossentropy(name='LogLoss', from_logits=True)
        elif isinstance(method, Methods.PU):
            loss = alib.model.FnLoss(pu_loss_from_logits, name='PU_Loss')
        elif isinstance(method, Methods.FNW):
            loss = alib.model.FnLoss(fnw_loss_from_logits, name='FNW_Loss')
        else:
            raise NotImplementedError(method)
        training_model = keras.Model(self.feature_inputs + extra_inputs, simple_logit)
        training_model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=loss,
        )
        self.model_info['training'] = training_model

    def serve_and_train(self, click_ts_start: 'int', click_ts_end: 'int', need_train: 'bool', need_evaluate: 'bool'):
        cc = self.cc
        method = self.method

        if need_evaluate or isinstance(method, Methods.Prophet):  # Serving
            model: 'keras.Model' = self.model_info['serving']
            df_eval = self.data_provider.serving_data(click_ts_start=click_ts_start, click_ts_end=click_ts_end)
            if len(df_eval) == 0:
                pass
            else:
                x_eval = alib.utils.df_to_dict(df_eval, [*cc.features])
                y_pred = model.predict(x_eval, **self.fit_predict_kwargs)
                if isinstance(method, Methods.Prophet):  # Prophet write back
                    self.data_provider.serving_write_back(
                        index=df_eval.index.values, column_indexer=cc.prophet, values=y_pred
                    )
                if need_evaluate:
                    self.data_provider.serving_write_back(
                        index=df_eval.index.values, column_indexer=cc.prediction, values=y_pred
                    )

        if not need_train:
            return

        # Training
        model: 'keras.Model' = self.model_info['training']
        if isinstance(method, Methods.Prophet):
            df_train = self.data_provider.get_real(click_ts_start=click_ts_start, click_ts_end=click_ts_end)
        elif isinstance(method, Methods.Waiting):
            df_train = self.data_provider.get_assign_negative(click_ts_start=click_ts_start - method.window_size,
                                                              click_ts_end=click_ts_end - method.window_size,
                                                              ts_now=click_ts_end)
        elif method.is_in([Methods.PU, Methods.FNW, Methods.FNC]):
            df_train = self.data_provider.get_fake_negative(click_ts_start=click_ts_start, click_ts_end=click_ts_end)
        else:
            raise NotImplementedError(method)
        if len(df_train) == 0:
            return
        x_train = alib.utils.df_to_dict(df_train, [*cc.features, *self.training_extra_cols])
        y_train = df_train[cc.label].values
        model.fit(x_train, y_train, epochs=1, shuffle=False, **self.fit_predict_kwargs)


@dataclass()
class TasksArtifacts(object):
    """Save data on tasks."""
    ts_end: 'int'
    index: 'np.ndarray'
    outputs: 'np.ndarray'


class FTPAgent(DNNBaseAgent, metaclass=abc.ABCMeta):
    """Follow the Prophet."""

    def __init__(self,
                 method: 'delay.core.MethodABC',
                 data_provider: 'delay.core.DataProvider',
                 intermediate_layers_config_fn: 'Callable[[], alib.model.LayersClassConfigType]',
                 fit_predict_kwargs: 'Dict[str, Any]'
                 ):
        super().__init__(method, data_provider, intermediate_layers_config_fn, fit_predict_kwargs)
        self.tasks_artifacts_buffer: 'collections.deque[TasksArtifacts]' = collections.deque()

    def reset_data(self):
        super().reset_data()
        cc = self.cc
        self.data_provider.setup_write_back_col(col='Best_Pred', fill_value=np.nan, dtype=cc.prediction_dtype)
        self.data_provider.setup_write_back_col(col='Best_TID', fill_value=-1, dtype=np.int32)
        self.tasks_artifacts_buffer.clear()

    def _build_model(self, verbose: 'bool'):
        cc = self.cc
        method = self.method
        assert isinstance(method, Methods.FTP)

        feature_ie_list = [
            alib.model.InputExtension(col, *col_args) for col, col_args in self.cc.features.items()
        ]
        self.model_info['feature_ie_list'] = feature_ie_list
        concat_embedding = self._dnn_embeddings(feature_ie_list, prefix='MTL/')

        intermediate_config = self.intermediate_layers_config_fn()

        n_tasks = len(method.task_window_sizes)

        # More shared layers
        if method.n_more_shared_layers > 0:
            more_shared_layers = alib.model.build_sequential(
                intermediate_config[:method.n_more_shared_layers],
                name_prefix='MTL/more_shared/',
                name='MTL/more_shared_net',
            )
        else:
            more_shared_layers = keras.layers.Layer(
                name='MTL/more_shared_net',
            )
        shared_net = more_shared_layers(concat_embedding)

        # Following layers for each task
        task_to_logit_config = [
            *intermediate_config[method.n_more_shared_layers:],
            (keras.layers.Dense, dict(units=1, activation=None, name='logit')),
        ]
        tasks_logit_list = []
        for i in range(n_tasks):
            task_layer = alib.model.build_sequential(
                task_to_logit_config,
                name_prefix=f'MTL/t{i}/',
                name=f'MTL/t{i}_logit',
            )
            tasks_logit_list.append(task_layer(
                shared_net if method.enable_gradients[i] > 0 else kb.stop_gradient(shared_net)
            ))

        tasks_concat_serving_probs = keras.layers.Lambda(
            keras.activations.sigmoid, name='MTL/tasks_concat_serving_probs')(
            keras.layers.Concatenate(name='MTL/concat_serving_logits')(tasks_logit_list)
        )

        # Follow Prophet
        fp_layers = alib.model.build_sequential(
            [
                *intermediate_config[method.n_more_shared_layers:],
                (keras.layers.Dense, dict(units=n_tasks, activation=None, name='logits'),)
            ],
            name_prefix='FP/',
            name='FP/net',
        )
        fp_net = fp_layers(
            shared_net if method.enable_gradients[n_tasks] > 0 else kb.stop_gradient(shared_net)
        )

        # Serving
        fp_serving_softmax = keras.layers.Lambda(keras.activations.softmax, name='FP/serving_softmax')(fp_net)
        fp_serving_out = keras.layers.Dot(axes=-1, name='FP/serving_out_no_squeeze')([
            tasks_concat_serving_probs,
            fp_serving_softmax,
        ])
        serving_out = keras.layers.Concatenate(axis=-1, name='serving_out')(
            [tasks_concat_serving_probs, fp_serving_out])
        self.model_info['serving'] = keras.Model(
            self.feature_inputs,
            serving_out,
        )

        # Training
        # mtl_tid input to one-hot. Only use in training
        ie_mtl_tid = alib.model.InputExtension(
            name=cc.mtl_tid,
            shape=(),
            dtype=cc.mtl_tid_dtype,
            categorical_size=-1,
            default_embedding_size=n_tasks + 1,
        )
        mtl_tid_one_hot_layer = ie_mtl_tid.to_embedding_layer(name_pattern='one_hot_{input_name}')
        mtl_tid_one_hot = mtl_tid_one_hot_layer(ie_mtl_tid.keras_input)

        training_output_list = tasks_logit_list + [fp_net]
        ie_fp_history_probs = alib.model.InputExtension(
            name=cc.mtl_probs,
            shape=(n_tasks,),
            dtype=cc.prediction_dtype,
            categorical_size=0,
            default_embedding_size=None,
        )
        fp_history_probs = ie_fp_history_probs.keras_input
        ema_loss_vars = [kb.variable(method.ema_loss_init, dtype=tf.float32,
                                     name=f'ema_loss_{i}') for i in range(n_tasks + 1)]
        training_loss_list = [get_hot_log_loss_from_logits_fn(
            hot=mtl_tid_one_hot[:, i],
            weight=tf.constant(1.),
            ema_var=ema_loss_vars[i],
        ) for i in range(n_tasks)]

        training_loss_list.append(get_hot_ce_loss_from_logits_fn(
            hot=mtl_tid_one_hot[:, n_tasks],
            mtl_probs=fp_history_probs,
            weight=tf.constant(1.) if method.enable_gradients[n_tasks] == 0 else tf.div_no_nan(
                kb.mean(kb.stack(ema_loss_vars[:n_tasks])),
                ema_loss_vars[n_tasks]
            ),
            ema_var=ema_loss_vars[n_tasks],
        ))
        training_model = keras.Model(
            self.feature_inputs + [ie_mtl_tid.keras_input, ie_fp_history_probs.keras_input],
            training_output_list,
        )

        def fp_metrics(y_true, y_pred):  # pylint: disable=unused-argument
            return ema_loss_vars[n_tasks]

        training_model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=training_loss_list,
            metrics=[[] for _ in range(n_tasks)] + [[
                fp_metrics
            ]]
        )
        self.model_info['training'] = training_model

        kb.get_session().run(tf.global_variables_initializer())

    def serve_and_train(self, click_ts_start: 'int', click_ts_end: 'int', need_train: 'bool', need_evaluate: 'bool'):
        cc = self.cc
        method = self.method
        assert isinstance(method, Methods.FTP)
        n_tasks = len(method.task_window_sizes)

        # Serving
        model: 'keras.Model' = self.model_info['serving']
        df_eval = self.data_provider.serving_data(click_ts_start=click_ts_start, click_ts_end=click_ts_end)
        eval_count = len(df_eval)
        if eval_count == 0:
            pass
        else:
            x_eval = alib.utils.df_to_dict(df_eval, [*cc.features])
            pred_array = model.predict(x_eval, **self.fit_predict_kwargs)
            self.tasks_artifacts_buffer.append(TasksArtifacts(click_ts_end, df_eval.index.values, pred_array))
            if need_evaluate:
                y_pred = pred_array[:, -1]  # last prediction (fp) as final prediction
                self.data_provider.serving_write_back(
                    index=df_eval.index.values, column_indexer=cc.prediction, values=y_pred
                )
                pred_array = pred_array[:, :n_tasks]

                # MTL_Best
                y_prophet = df_eval[cc.prophet].values
                pae_array = np.abs(pred_array.transpose() - y_prophet).transpose()
                best_tid = np.argmin(pae_array, axis=-1)
                self.data_provider.serving_write_back(
                    index=df_eval.index.values, column_indexer='Best_TID', values=best_tid
                )
                y_pred_best = pred_array[np.arange(eval_count), best_tid]
                self.data_provider.serving_write_back(
                    index=df_eval.index.values, column_indexer='Best_Pred', values=y_pred_best
                )

        if not need_train:
            return

        # Training
        model: 'keras.Model' = self.model_info['training']
        df_train = []
        mtl_probs = []
        # MTL Data
        for i, win_size in enumerate(method.task_window_sizes):
            df = self.data_provider.get_assign_negative(click_ts_start=click_ts_start - win_size,
                                                        click_ts_end=click_ts_end - win_size,
                                                        ts_now=click_ts_end)
            df[cc.mtl_tid] = i
            df_train.append(df)
            mtl_probs.append(np.zeros(shape=(len(df), n_tasks), dtype=cc.prediction_dtype))
        # FP Data
        fp_ts_end = click_ts_end - method.task_window_sizes[-1]
        while fp_ts_end >= self.tasks_artifacts_buffer[0].ts_end:  # data is ready (full feedback)
            tasks_artifacts = self.tasks_artifacts_buffer.popleft()
            if fp_ts_end < method.fp_train_min_ts:  # to early data, MTL may not be well trained.
                continue
            fp_original_df = self.data_provider.indexing_data(index=tasks_artifacts.index, ts_now=click_ts_end)
            fp_original_df[cc.mtl_tid] = n_tasks
            df_train.append(fp_original_df)
            mtl_probs.append(tasks_artifacts.outputs[:, :n_tasks])

        df_train = pd.concat(df_train, ignore_index=True)
        if len(df_train) == 0:
            return
        df_train[cc.mtl_tid] = df_train[cc.mtl_tid].astype(cc.mtl_tid_dtype)
        mtl_probs = np.concatenate(mtl_probs, axis=0)
        assert len(df_train) == len(mtl_probs)
        df_train, index = alib.utils.shuffle_df(df_train)
        mtl_probs = mtl_probs[index]
        x_train = alib.utils.df_to_dict(df_train, [*cc.features, cc.mtl_tid])
        x_train[cc.mtl_probs] = mtl_probs
        y_train = [df_train[cc.label].values for _ in range(n_tasks)] + [df_train[cc.prophet].values]
        model.fit(x_train, y_train, epochs=1, shuffle=False, **self.fit_predict_kwargs)
