#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DATA_DIR = './data/CriteoCVR_raw/'

int_features = ['int_' + str(i) for i in range(1, 9)]
cate_features = ['cate_' + str(i) for i in range(1, 10)]

df = pd.read_csv(
    DATA_DIR + 'data.txt',
    sep='\t',
    names=['click_timestamp', 'convert_timestamp'] + int_features + cate_features,
    dtype={
        'click_timestamp': np.int64,
        'convert_timestamp': 'Int64',
        **{col: 'Int64' for col in int_features},
        **{col: 'category' for col in cate_features},
    }
)

df = df.drop(index=df[df['convert_timestamp'] - df['click_timestamp'] < 0].index).reset_index(drop=True)

df = df[df['click_timestamp'] < 60 * 24 * 3600].sort_values('click_timestamp').reset_index(drop=True)

df_int = df[int_features].copy().apply(lambda x: np.floor(np.square(np.log(x + 2)))).astype('Int64').astype('category')
df[int_features] = df_int

min_count = 10
df2 = df.copy()
for col in tqdm(int_features + cate_features):
    df2[col].cat.add_categories('_Others', inplace=True)
    counts = df2[col].value_counts(dropna=False)
    df2[col][df2[col].isin(counts[counts < min_count].index)] = '_Others'

for col in tqdm(int_features + cate_features):
    df2[col].cat.remove_unused_categories(inplace=True)
    df2[col] = df2[col].cat.codes + 1

df2.to_feather('./data/CriteoCVR_base.feather')
