#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
import sys
import typing

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(".")
import delay
import delay.agents
import delay.core

tf.disable_eager_execution()
keras = tf.keras
if typing.TYPE_CHECKING:
    from typing import *

print('Python', platform.python_version())
print('Tensorflow', tf.VERSION)
print('Keras', keras.__version__)

# Configuration

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

H = 3600
D = 24 * H

FIT_PREDICT_KWARGS = dict(
    batch_size=10000,
    verbose=0,
    callbacks=None,
)

DATA_PATH = './data/TencentCVR_base.feather'

COLUMN_CONFIG = delay.core.ColumnConfig(
    click_ts='clickTime',
    convert_ts='conversionTime',
    features={
        # name: (shape, dtype, categorical_size + 1, embedding_size)
        'creativeID': ((), np.int32, 48836, 256),
        'positionID': ((), np.int16, 21488, 256),
        'connectionType': ((), np.int8, 5, 8),
        'telecomsOperator': ((), np.int8, 4, 8),
        'adID': ((), np.int16, 29726, 256),
        'camgaignID': ((), np.int16, 6583, 128),
        'advertiserID': ((), np.int16, 639, 32),
        'appID': ((), np.int16, 465, 32),
        'appPlatform': ((), np.int8, 2, 8),
        'appCategory': ((), np.int8, 28, 8),
        'sitesetID': ((), np.int8, 3, 8),
        'positionType': ((), np.int8, 6, 8),
        'age': ((), np.int8, 81, 16),
        'gender': ((), np.int8, 3, 8),
        'education': ((), np.int8, 8, 8),
        'marriageStatus': ((), np.int8, 4, 8),
        'haveBaby': ((), np.int8, 7, 8),
        'hometown': ((), np.int16, 365, 32),
        'residence': ((), np.int16, 405, 32),
    },
    other_embedding_size=8,
)

MIN_TS, MAX_TS, STEP_TS, EVAL_TS = 0, 12 * D, 1 * H, 7 * D


def intermediate_layers_config_fn():
    return [
        (keras.layers.Dense, dict(units=128,
                                  activation=keras.layers.LeakyReLU(),
                                  kernel_regularizer=keras.regularizers.L1L2(l2=1e-7),
                                  name='hidden_1')
         ),
        (keras.layers.Dense, dict(units=128,
                                  activation=keras.layers.LeakyReLU(),
                                  kernel_regularizer=keras.regularizers.L1L2(l2=1e-7),
                                  name='hidden_2')
         ),
    ]


# Load data

data_provider = delay.core.DataProvider(df=pd.read_feather(DATA_PATH), cc=COLUMN_CONFIG,
                                        fast_index=(MIN_TS, MAX_TS, STEP_TS))
print(len(data_provider.df), data_provider.df[COLUMN_CONFIG.convert_ts].notnull().mean())


# Experiments

def run_exp(method: 'delay.core.MethodABC'):
    if isinstance(method, delay.core.Methods.FTP):
        agent_class = delay.agents.FTPAgent
    else:
        agent_class = delay.agents.SimpleDNNAgent
    agent = agent_class(method=method,
                        data_provider=data_provider,
                        intermediate_layers_config_fn=intermediate_layers_config_fn,
                        fit_predict_kwargs=FIT_PREDICT_KWARGS)
    result = delay.core.run_streaming(agent, min_ts=MIN_TS, max_ts=MAX_TS, step_ts=STEP_TS, eval_ts=EVAL_TS)
    print(method.description, result)
    return {'method': method.description, **result}


result_df = pd.DataFrame()

result_df = result_df.append(run_exp(delay.core.Methods.Prophet(
    description='Prophet*',
)), ignore_index=True)

result_df = result_df.append(run_exp(delay.core.Methods.Waiting(
    window_size=4 * H,
    description='Waiting(4h)',
)), ignore_index=True)

result_df = result_df.append(run_exp(delay.core.Methods.Waiting(
    window_size=6 * H,
    description='Waiting(6h)',
)), ignore_index=True)

result_df = result_df.append(run_exp(delay.core.Methods.Waiting(
    window_size=12 * H,
    description='Waiting(12h)',
)), ignore_index=True)

result_df = result_df.append(run_exp(delay.core.Methods.Waiting(
    window_size=24 * H,
    description='Waiting(24h)',
)), ignore_index=True)

result_df = result_df.append(run_exp(delay.core.Methods.Waiting(
    window_size=48 * H,
    description='Waiting(48H)',
)), ignore_index=True)

result_df = result_df.append(run_exp(delay.core.Methods.FTP(
    task_window_sizes=[1 * H, 6 * H, 24 * H, 48 * H],
    n_more_shared_layers=1,
    fp_train_min_ts=1 * D,
    ema_loss_init=0.10,
    enable_gradients=[1, 1, 1, 1, 1],
    description='FTP',
)), ignore_index=True)

print(result_df)
