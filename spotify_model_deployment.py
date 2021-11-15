#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:40:31 2021

@author: irfana
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn import pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split

data=pd.read_csv("spotify_merged.csv")

train, test = train_test_split(data, test_size = 0.2, random_state = 69)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)


#webscaling dataframe

web_app_scaling_df = test.copy()

# Replacing boolean (True, False) by int32 (1, 0)
web_app_scaling_df.replace([True, False], [1, 0], inplace=True)

# adding new skipped column and dropping skip_1, skip_2 and skip_3
web_app_scaling_df["skipped"] = web_app_scaling_df["skip_1"]*web_app_scaling_df["skip_2"]*web_app_scaling_df["skip_3"]
web_app_scaling_df.drop(["skip_1", "skip_2", "skip_3", "not_skipped"], axis=1, inplace=True)

# encoding the mode
web_app_scaling_df['mode'].replace({'major': 1, 'minor': 0 }, inplace=True)

# chaning the date to weekday and droping the date column
web_app_scaling_df["date"] = pd.to_datetime(web_app_scaling_df["date"])
web_app_scaling_df['week_day'] = web_app_scaling_df["date"].dt.dayofweek
web_app_scaling_df.drop("date", inplace=True, axis=1)

# encoding categorical columns
categorical_columns = ['context_type', 'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end']

for col in categorical_columns:
    # merging labels if they are less than threshold (< 0.001)
    counts = web_app_scaling_df[col].value_counts(normalize=True)
    labels_less_then_threshold = counts[counts < 0.001].index.to_list()
    where_to_replace = web_app_scaling_df[col].isin(labels_less_then_threshold).copy()
    web_app_scaling_df.loc[where_to_replace, col] = 'merged'

# setting one hot encoding for categorical columns (Nominal Columns)

One_Hot_Encoder = OneHotEncoder()
context_type = pd.DataFrame(One_Hot_Encoder.fit_transform(web_app_scaling_df[['context_type']]).toarray())
context_type.columns = One_Hot_Encoder.get_feature_names(['context_type'])
hist_user_behavior_reason_start = pd.DataFrame(One_Hot_Encoder.fit_transform(web_app_scaling_df[['hist_user_behavior_reason_start']]).toarray())
hist_user_behavior_reason_start.columns = One_Hot_Encoder.get_feature_names(['hub_reason_start']) # hub = hist_user_behavior
hist_user_behavior_reason_end = pd.DataFrame(One_Hot_Encoder.fit_transform(web_app_scaling_df[['hist_user_behavior_reason_end']]).toarray())
hist_user_behavior_reason_end.columns = One_Hot_Encoder.get_feature_names(['hub_reason_end'])  # hub = hist_user_behavior


# Concatenate dataframe --> session_track_data + context_type + hist_user_behavior_reason_start + hist_user_behavior_reason_end
web_app_scaling_df = pd.concat([web_app_scaling_df, context_type, hist_user_behavior_reason_start, hist_user_behavior_reason_end], axis = 1)
web_app_scaling_df.drop(["context_type", "hist_user_behavior_reason_start", "hist_user_behavior_reason_end", "track_id"],
                        axis = 1, inplace = True)


# drop all highly correlated variables.


web_app_scaling_df.drop(["short_pause_before_play", "long_pause_before_play"], axis=1, inplace=True)
web_app_scaling_df.drop(['beat_strength', 'danceability', 'dyn_range_mean'], axis=1, inplace=True)

web_app_scaling_df.drop(["session_id"], axis=1, inplace=True)
web_app_scaling_df.drop("skipped", axis=1, inplace=True)


web_app_scaling_df.drop(['hub_reason_start_playbtn'], axis=1, inplace=True)
web_app_scaling_df = web_app_scaling_df[train.columns.tolist()]

web_app_scaling_df.to_csv('web_app_scaling_df.csv')

