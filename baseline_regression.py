import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

%matplotlib inline

################################################################################
# Import data and train-test split
################################################################################

root_dir = os.getcwd()
ngames_df = pd.read_csv(os.path.join(root_dir, "data", "ngames_data_final.csv"), index_col=[0,1,2])
ctmc_df = pd.read_csv(os.path.join(root_dir, "data", "ctmc_data_final.csv"), index_col=[0,1,2])
data_final = ngames_df.join(ctmc_df)

X_train = data_final[(data_final.index.get_level_values(2)<2016) & (data_final['D1_Match']==True)].drop(['target_margin'], axis=1).fillna(0)
y_train = data_final[(data_final.index.get_level_values(2)<2016) & (data_final['D1_Match']==True)]['target_margin']
X_test = data_final[(data_final.index.get_level_values(2)>=2016) & (data_final['D1_Match']==True)].drop(['target_margin'], axis=1).fillna(0)
y_test = data_final[(data_final.index.get_level_values(2)>=2016) & (data_final['D1_Match']==True)]['target_margin']

################################################################################
# Important features
################################################################################

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt.score(X_train, y_train)
dt.score(X_test, y_test)

fi_summary = pd.DataFrame({'features': X_train.columns, 'importances': dt.feature_importances_}).set_index('features')
top_features = fi_summary[fi_summary['importances']>0.005].sort_values('importances', ascending=False)
top_features.plot(kind='bar', title="Feature Importances for Fully Grown Decision Tree", legend=False, figsize=(12,4))

################################################################################
# Baseline One-Feature Linear Regression
################################################################################

## Create baseline features for single feature baselines

# 5 game average margins diff
X_train['5_Games_HH_VA_mean_margin'] = (X_train['5_Games_HH_margin_mean'] - X_train['5_Games_VA_margin_mean']*-1)
X_test['5_Games_HH_VA_mean_margin'] = (X_test['5_Games_HH_margin_mean'] - X_test['5_Games_VA_margin_mean']*-1)

# Current season average margins diff
X_train['CurSeason_HH_VA_mean_margin'] = (X_train['CurSeason_HH_margin_mean'] - X_train['CurSeason_VA_margin_mean']*-1)
X_test['CurSeason_HH_VA_mean_margin'] = (X_test['CurSeason_HH_margin_mean'] - X_test['CurSeason_VA_margin_mean']*-1)

# CTMC diff
X_train['CTMC_diff'] = (X_train['CTMC_Rating_Home'] - X_train['CTMC_Rating_Away'])
X_test['CTMC_diff'] = (X_test['CTMC_Rating_Home'] - X_test['CTMC_Rating_Away'])

# Train and report baseline model with 5 game trailing metric
ols = LinearRegression()
ols.fit(X_train['5_Games_HH_VA_mean_margin'].values.reshape(-1,1), y_train)
ols.score(X_train['5_Games_HH_VA_mean_margin'].values.reshape(-1,1), y_train)
ols.score(X_test['5_Games_HH_VA_mean_margin'].values.reshape(-1,1), y_test)
preds = ols.predict(X_test['5_Games_HH_VA_mean_margin'].values.reshape(-1,1))
mean_squared_error(preds, y_test)

# Train and report baseline model with current season metric
ols = LinearRegression()
ols.fit(X_train['CurSeason_HH_VA_mean_margin'].values.reshape(-1,1), y_train)
ols.score(X_train['CurSeason_HH_VA_mean_margin'].values.reshape(-1,1), y_train)
ols.score(X_test['CurSeason_HH_VA_mean_margin'].values.reshape(-1,1), y_test)
preds = ols.predict(X_test['CurSeason_HH_VA_mean_margin'].values.reshape(-1,1))
mean_squared_error(preds, y_test)

# Train and report baseline model CTMC difference
ols = LinearRegression()
ols.fit(X_train['CTMC_diff'].values.reshape(-1,1), y_train)
ols.score(X_train['CTMC_diff'].values.reshape(-1,1), y_train)
ols.score(X_test['CTMC_diff'].values.reshape(-1,1), y_test)
preds = ols.predict(X_test['CTMC_diff'].values.reshape(-1,1))
mean_squared_error(preds, y_test)
