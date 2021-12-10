#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DATA_DIR = './data/TencentCVR_raw/'

table_list = [
    'train',
    'ad',
    'app_categories',
    'position',
    'user_app_actions',
    'user',
    'user_installedapps',
]

dfs = dict()
for table in tqdm(table_list):
    dfs[table] = pd.read_csv(DATA_DIR + table + '.csv', dtype='Int64')

df_join = dfs['train'].copy()
df_join = df_join.merge(dfs['ad'], how='left', on='creativeID', validate='many_to_one')
df_join = df_join.merge(dfs['app_categories'], how='left', on='appID', validate='many_to_one')
df_join = df_join.merge(dfs['position'], how='left', on='positionID', validate='many_to_one')
df_join = df_join.merge(dfs['user'], how='left', on='userID', validate='many_to_one')
df_join = df_join.drop(columns=['userID'])

not_categorical = {
    'label': np.int8,
    'clickTime': np.int64,
    'conversionTime': 'Int64',
}

for col in tqdm(df_join):
    if col in not_categorical:
        df_join[col] = df_join[col].astype(not_categorical[col])
        continue
    df_join[col] = df_join[col].astype('category')

feature_cols = [
    'creativeID', 'positionID',
    'connectionType', 'telecomsOperator', 'adID', 'camgaignID',
    'advertiserID', 'appID', 'appPlatform', 'appCategory', 'sitesetID',
    'positionType', 'age', 'gender', 'education', 'marriageStatus',
    'haveBaby', 'hometown', 'residence'
]

for col in tqdm(feature_cols):
    df_join[col] = df_join[col].cat.codes

min_datetime = 17000000
df_join = df_join[df_join['clickTime'] >= min_datetime].reset_index(drop=True)
for col in ['clickTime', 'conversionTime']:
    dt = df_join[col].copy()
    dt = dt - min_datetime
    day = dt // 100_00_00
    hour = dt % 100_00_00 // 100_00
    minute = dt % 100_00 // 100
    second = dt % 100
    df_join[col] = day * 24 * 60 * 60 + hour * 60 * 60 + minute * 60 + second
df_join = df_join.sort_values('clickTime').reset_index(drop=True)
df_join = df_join[df_join['clickTime'] < 12 * 24 * 3600].reset_index(drop=True)

delay = df_join['conversionTime'] - df_join['clickTime']
df_2d = df_join.copy()
df_2d.loc[delay > 48 * 3600, 'conversionTime'] = pd.NA
df_2d = df_2d.drop(columns=['label'])
df_2d.to_feather('./data/TencentCVR_base.feather')
