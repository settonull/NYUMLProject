import os
import itertools
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor

from construct_data import load_csvs

%matplotlib inline

################################################################################
# Import data and train-test split
################################################################################

root_dir = os.getcwd()
scores_dir = 'allScores'
data_dir = os.path.join(root_dir, "data")

ngames_df = pd.read_csv(os.path.join(data_dir, "ngames_data_final.csv"),
                        index_col=[0,1,2])
scores_ctmc_df = pd.read_csv(os.path.join(data_dir, "scores_ctmc_data_final.csv"),
                        index_col=[0,1,2,3])
pa_ctmc_df = pd.read_csv(os.path.join(data_dir, "pa_ctmc_data_final.csv"),
                        index_col=[0,1,2,3])
ra_ctmc_df = pd.read_csv(os.path.join(data_dir, "ra_ctmc_data_final.csv"),
                        index_col=[0,1,2,3])
ry_ctmc_df = pd.read_csv(os.path.join(data_dir, "ry_ctmc_data_final.csv"),
                        index_col=[0,1,2,3])
fd_ctmc_df = pd.read_csv(os.path.join(data_dir, "fd_ctmc_data_final.csv"),
                        index_col=[0,1,2,3])
glicko_df = pd.read_csv(os.path.join(data_dir, "glicko_data_final.csv"),
                        index_col=[0,1,2,3])
data_final = ngames_df.join(scores_ctmc_df)
data_final = data_final.join(pa_ctmc_df, rsuffix='_pa')
data_final = data_final.join(ra_ctmc_df, rsuffix='_ra')
data_final = data_final.join(ry_ctmc_df, rsuffix='_ry')
data_final = data_final.join(fd_ctmc_df, rsuffix='_fd')
data_final = data_final.join(glicko_df)

# Load data locations
scores_names = glob.glob(os.path.join(data_dir, scores_dir, "NCAAAllScores201?_Week.csv"))
scores_df = load_csvs(scores_names)
scores_df['teams_key'] = list(map(lambda x, y: str(x)+"-"+str(y),
                                    scores_df['HomeID'],scores_df['AwayID']))
scores_df.rename(index=str, columns={'Start': 'DateTime'}, inplace=True)
scores_df.set_index(['teams_key','DateTime','Season'], inplace=True)
data_final = data_final.join(scores_df['Week'])

X_train = data_final[(data_final.index.get_level_values(2)<2016) & \
    (data_final['D1_Match']==True) & (data_final['Week']>4)].drop(
                                    ['target_margin','Week'], axis=1).fillna(0)
y_train = data_final[(data_final.index.get_level_values(2)<2016) & \
    (data_final['D1_Match']==True) & (data_final['Week']>4)]['target_margin']
X_test = data_final[(data_final.index.get_level_values(2)>=2016) & \
    (data_final['D1_Match']==True) & (data_final['Week']>4)].drop(
                                    ['target_margin','Week'], axis=1).fillna(0)
y_test = data_final[(data_final.index.get_level_values(2)>=2016) & \
    (data_final['D1_Match']==True) & (data_final['Week']>4)]['target_margin']

################################################################################
# Important features
################################################################################

dt = DecisionTreeRegressor()
X_train_tree = X_train.drop(['CTMC_Rating_Home','CTMC_Rating_Away'], 1).copy()
dt.fit(X_train_tree, y_train)

fi_summary = pd.DataFrame({'features': X_train_tree.columns,
                            'importances': dt.feature_importances_}
                            ).set_index('features')
top_features = fi_summary[fi_summary['importances']>0.005].sort_values(
                                                'importances', ascending=False)
top_features.plot(kind='bar',
                    title="Feature Importances for Fully Grown Decision Tree",
                    legend=False, figsize=(12,4))

featurestouse = top_features.iloc[:10].index.values.tolist() +\
                    ['Glicko_Rating_Home', 'Glicko_Rating_Away',
                    'Glicko_Rating_Deviance_Home', 'Glicko_Rating_Deviance_Away',
                    'Glicko_Sigma_Home', 'Glicko_Sigma_Away',
                    'CTMC_Rating_Home', 'CTMC_Rating_Away',
                    'CTMC_Rating_Home_pa', 'CTMC_Rating_Away_pa',
                    'CTMC_Rating_Home_ra', 'CTMC_Rating_Away_ra',
                    'CTMC_Rating_Home_ry', 'CTMC_Rating_Away_ry',
                    'CTMC_Rating_Home_fd', 'CTMC_Rating_Away_fd']

################################################################################
# Baseline One-Feature Linear Regression
################################################################################

## Create baseline features for single feature baselines

# 5 game average margins diff
X_train['5_Games_HH_VA_mean_margin'] = (X_train['5_Games_HH_margin_mean'] \
                                        - X_train['5_Games_VA_margin_mean']*-1)
X_test['5_Games_HH_VA_mean_margin'] = (X_test['5_Games_HH_margin_mean'] \
                                        - X_test['5_Games_VA_margin_mean']*-1)

# Current season average margins diff
X_train['CurSeason_HH_VA_mean_margin'] = (X_train['CurSeason_HH_margin_mean'] \
                                    - X_train['CurSeason_VA_margin_mean']*-1)
X_test['CurSeason_HH_VA_mean_margin'] = (X_test['CurSeason_HH_margin_mean'] \
                                        - X_test['CurSeason_VA_margin_mean']*-1)

# CTMC diff
X_train['CTMC_diff'] = (X_train['CTMC_Rating_Home'] \
                        - X_train['CTMC_Rating_Away'])
X_test['CTMC_diff'] = (X_test['CTMC_Rating_Home'] \
                        - X_test['CTMC_Rating_Away'])

# CTMC pass attempt diff
X_train['CTMC_diff_pa'] = (X_train['CTMC_Rating_Home_pa'] \
                        - X_train['CTMC_Rating_Away_pa'])
X_test['CTMC_diff_pa'] = (X_test['CTMC_Rating_Home_pa'] \
                        - X_test['CTMC_Rating_Away_pa'])

# CTMC pass attempt diff
X_train['CTMC_diff_ra'] = (X_train['CTMC_Rating_Home_ra'] \
                        - X_train['CTMC_Rating_Away_ra'])
X_test['CTMC_diff_ra'] = (X_test['CTMC_Rating_Home_ra'] \
                        - X_test['CTMC_Rating_Away_ra'])

# Glicko diff
X_train['Glicko_diff'] = (X_train['Glicko_Rating_Home'] \
                        - X_train['Glicko_Rating_Away'])
X_test['Glicko_diff'] = (X_test['Glicko_Rating_Home'] \
                        - X_test['Glicko_Rating_Away'])


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

# Train and report baseline model CTMC difference
ols = LinearRegression()
ols.fit(X_train['CTMC_diff_pa'].values.reshape(-1,1), y_train)
ols.score(X_train['CTMC_diff_pa'].values.reshape(-1,1), y_train)
ols.score(X_test['CTMC_diff_pa'].values.reshape(-1,1), y_test)
preds = ols.predict(X_test['CTMC_diff_pa'].values.reshape(-1,1))
mean_squared_error(preds, y_test)

# Train and report baseline model CTMC difference
ols = LinearRegression()
ols.fit(X_train['CTMC_diff_ra'].values.reshape(-1,1), y_train)
ols.score(X_train['CTMC_diff_ra'].values.reshape(-1,1), y_train)
ols.score(X_test['CTMC_diff_ra'].values.reshape(-1,1), y_test)
preds = ols.predict(X_test['CTMC_diff_ra'].values.reshape(-1,1))
mean_squared_error(preds, y_test)

# Train and report baseline model CTMC difference
ols = LinearRegression()
ols.fit(X_train['Glicko_diff'].values.reshape(-1,1), y_train)
ols.score(X_train['Glicko_diff'].values.reshape(-1,1), y_train)
ols.score(X_test['Glicko_diff'].values.reshape(-1,1), y_test)
preds = ols.predict(X_test['Glicko_diff'].values.reshape(-1,1))
mean_squared_error(preds, y_test)

################################################################################
# Full Ridge
################################################################################

standardscaler = StandardScaler()
X_trainscaled = standardscaler.fit_transform(X_train[featurestouse])
X_testscaled = standardscaler.transform(X_test[featurestouse])

lambdas = [10**x for x in np.arange(-2,0, 0.1)]
valloss = []
trainloss = []
best_loss = np.inf
for lambda_ in lambdas:
    ridge = Pipeline([('estimator', KernelRidge(alpha=lambda_, kernel='rbf'))])
    ridge.fit(X_trainscaled, y_train)
    valpreds = ridge.predict(X_testscaled)
    valloss += [mean_squared_error(valpreds, y_test)]
    trainpreds = ridge.predict(X_trainscaled)
    trainloss += [mean_squared_error(trainpreds, y_train)]

    if valloss[-1] < best_loss:
        best_loss = valloss[-1]
        best_lambda_ = lambda_
        best_ridge = Pipeline([('estimator', KernelRidge(alpha=lambda_,
                                                        kernel='rbf'))])

results = pd.DataFrame({'validation_loss': valloss,
                        'test_loss': trainloss},
                        index=lambdas)

results.plot(title="Validation Loss: {}".format(
                                        results['validation_loss'].min()))


gammas = [10**x for x in np.arange(-4,-2, 0.1)]
valloss = []
trainloss = []
best_loss = np.inf
for gamma in gammas:
    ridge = Pipeline([('estimator', KernelRidge(alpha=0.1,
                                                kernel='rbf',
                                                gamma = gamma))])
    ridge.fit(X_trainscaled, y_train)
    valpreds = ridge.predict(X_testscaled)
    valloss += [mean_squared_error(valpreds, y_test)]
    trainpreds = ridge.predict(X_trainscaled)
    trainloss += [mean_squared_error(trainpreds, y_train)]

    if valloss[-1] < best_loss:
        best_loss = valloss[-1]
        best_gamma = gamma
        best_ridge = Pipeline([('estimator', KernelRidge(alpha=0.1,
                                                        kernel='rbf',
                                                        gamma = gamma))])

results = pd.DataFrame({'validation_loss': valloss,
                        'test_loss': trainloss},
                        index=gammas)

best_ridge.fit(X_trainscaled, y_train)
best_rsquared = best_ridge.score(X_testscaled, y_test)

results.plot(title="Validation Loss: {:.3f} R-Squared: {:.3f}".format(
results['validation_loss'].min(), best_rsquared))


lambdas = [10**x for x in np.arange(-3,1, 0.1)]
gammas = [10**x for x in np.arange(-3,-1, 0.1)]
configs = list(itertools.product(lambdas, gammas))
valloss = []
trainloss = []
best_loss = np.inf
for config in itertools.product(lambdas, gammas):
    ridge = Pipeline([('estimator', KernelRidge(alpha=config[0],
                                                kernel='rbf',
                                                gamma = config[1]))])
    ridge.fit(X_trainscaled, y_train)
    valpreds = ridge.predict(X_testscaled)
    valloss += [mean_squared_error(valpreds, y_test)]
    trainpreds = ridge.predict(X_trainscaled)
    trainloss += [mean_squared_error(trainpreds, y_train)]

    if valloss[-1] < best_loss:
        best_loss = valloss[-1]
        best_labda_ = config[0]
        best_gamma = config[1]
        best_ridge = Pipeline([('estimator', KernelRidge(alpha=best_lambda_,
                                                        kernel='rbf',
                                                        gamma = best_gamma))])
best_lambda_
best_gamma
results = pd.DataFrame({'validation_loss': valloss,
                        'test_loss': trainloss},
                        index=configs)

best_ridge.fit(X_trainscaled, y_train)
best_rsquared = best_ridge.score(X_testscaled, y_test)

results.plot(title="Validation Loss: {:.3f} R-Squared: {:.3f}".format(
results['validation_loss'].min(), best_rsquared))





maxdepths = np.arange(2,20)
valloss = []
trainloss = []
for maxdepth in maxdepths:
    boost = Pipeline([('estimator',
                        GradientBoostingRegressor(max_depth=maxdepth,
                                                min_samples_leaf=4,
                                                subsample=0.8,
                                                n_estimators=3000,
                                                learning_rate=0.0001))])
    boost.fit(X_trainscaled, y_train)
    valpreds = boost.predict(X_testscaled)
    valloss += [mean_squared_error(valpreds, y_test)]
    trainpreds = boost.predict(X_trainscaled)
    trainloss += [mean_squared_error(trainpreds, y_train)]

results = pd.DataFrame({'validation_loss': valloss,
                        'test_loss': trainloss},
                        index=maxdepths)

results.plot(title="Validation Loss: {}".format(
                                        results['validation_loss'].min()))


samplesizes = np.arange(0.1, 1, 0.1)
valloss = []
trainloss = []
for samplesize in samplesizes:
    boost = Pipeline([('estimator',
                        GradientBoostingRegressor(max_depth=20,
                                                min_samples_leaf=4,
                                                subsample=samplesize,
                                                n_estimators=3000,
                                                learning_rate=0.0001))])
    boost.fit(X_trainscaled, y_train)
    valpreds = boost.predict(X_testscaled)
    valloss += [mean_squared_error(valpreds, y_test)]
    trainpreds = boost.predict(X_trainscaled)
    trainloss += [mean_squared_error(trainpreds, y_train)]

results = pd.DataFrame({'validation_loss': valloss,
                        'test_loss': trainloss},
                        index=maxfeatures)

results.plot(title="Validation Loss: {}".format(
                                        results['validation_loss'].min()))
