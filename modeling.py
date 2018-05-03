import os
import itertools
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

%matplotlib inline

################################################################################
# Import data
################################################################################

root_dir = os.getcwd()
ctmc_dir = 'ctmc'
glicko_dir = 'glicko'
ngames_dir = 'ngames'
ultimate_dir = 'ultimate'
scores_pe_dir = 'scores_pe'
bcs_dir = 'BCS'
conf_dir = 'conferences'
data_dir = os.path.join(root_dir, "data")

##### IJH Data
file = os.path.join(data_dir, ultimate_dir, "ultimate_2.csv")
scores_pe_df = pd.read_csv(file)
scores_pe_df = scores_pe_df.set_index(['HomeID', 'VisID', 'Season', 'Week'])
scores_pe_df['target_margin'] = scores_pe_df['HomeFinal'] - scores_pe_df['VisFinal']
scores_pe_df = scores_pe_df[['HomeElo', 'HomeEloProb','HomeLuck','HomePrevLuck',
                            'HomePythPct','HomePythWins','HomeWinPct','VisElo',
                            'VisEloProb','VisLuck', 'VisPrevLuck','VisPythPct',
                            'VisPythWins','VisWinPct','SpreadElo','HomeConf_NotMajor',
                            'VisConf_NotMajor','target_margin']]

####### Create Baseline Feature
inseason_sum_home_team = scores_pe_df.reset_index().\
                                      groupby(['HomeID','Season','Week']).\
                                      agg('max').groupby(level=[0,1]).\
                                      cumsum()['target_margin']
inseason_count_home_team = scores_pe_df.reset_index().\
                                        groupby(['HomeID','Season','Week']).\
                                        agg('count').\
                                        groupby(level=[0,1]).\
                                        cumsum()['target_margin']
inseason_sum_vis_team = scores_pe_df.reset_index().\
                                     groupby(['VisID','Season','Week']).\
                                     agg('max').\
                                     groupby(level=[0,1]).\
                                     cumsum()['target_margin']
inseason_count_vis_team = scores_pe_df.reset_index().\
                                       groupby(['VisID','Season','Week']).\
                                       agg('count').\
                                       groupby(level=[0,1]).\
                                       cumsum()['target_margin']

inseason_home_mean_margin = inseason_sum_home_team/inseason_count_home_team
inseason_home_mean_margin.name = 'CurSeason_HH_margin_mean'
inseason_home_mean_margin = inseason_home_mean_margin.reset_index()
inseason_vis_mean_margin = inseason_sum_vis_team/inseason_count_vis_team
inseason_vis_mean_margin.name = 'CurSeason_VA_margin_mean'
inseason_vis_mean_margin = inseason_vis_mean_margin.reset_index()

inseason_home_mean_margin['Week'] = inseason_home_mean_margin['Week'].shift(1)
inseason_vis_mean_margin['Week'] = inseason_vis_mean_margin['Week'].shift(1)

scores_pe_df = scores_pe_df.reset_index().merge(inseason_home_mean_margin,
                                                left_on=['HomeID','Season','Week'],
                                                right_on=['HomeID','Season','Week'])
scores_pe_df = scores_pe_df.merge(inseason_vis_mean_margin,
                                  left_on=['VisID','Season','Week'],
                                  right_on=['VisID','Season','Week'])

scores_pe_df = scores_pe_df.set_index(['HomeID', 'VisID', 'Season', 'Week'])

######## SWC Data
file = os.path.join(data_dir, ctmc_dir, "scores_ctmc_ultimate_data_final.csv")
scores_ctmc_df = pd.read_csv(file,index_col=[0,1,2,3]).astype(float)
scores_ctmc_df.index.names = ['HomeID', 'VisID', 'Season', 'Week']
file = os.path.join(data_dir, glicko_dir, "glicko_ultimate_data_final.csv")
glicko_df = pd.read_csv(file, index_col=[0,1,2,3]).astype(float)
glicko_df.index.names = ['HomeID', 'VisID', 'Season', 'Week']

# Join SC Data
data_final = scores_pe_df.join(scores_ctmc_df)
data_final = data_final.join(glicko_df)

##### CDR Data
file = os.path.join(data_dir, bcs_dir, "BCS-SOS.csv")
bcs_df = pd.read_csv(file)
bcs_df = bcs_df.set_index(['HomeID', 'VisID', 'Season', 'Week'])

# Join CDR Data
data_final = data_final.join(bcs_df)

# Join Conference Data
file = os.path.join(data_dir, conf_dir, "mergedConferences.csv")
conf_df = pd.read_csv(file)
data_final = data_final.reset_index().merge(conf_df,
                                            left_on=['HomeID', 'Season'],
                                            right_on=['ID','Year'],
                                            suffixes=('','Home'))
data_final = data_final.reset_index().merge(conf_df,
                                            left_on=['VisID', 'Season'],
                                            right_on=['ID','Year'],
                                            suffixes=('','Vis'))
data_final = data_final.set_index(['HomeID', 'VisID', 'Season', 'Week'])
data_final = data_final.drop(['ID','Year','IDVis','index'],1)

# Impute HomeConf Data
data_final['HomeConf_NotMajor'] = data_final['HomeConf_NotMajor'].fillna(0)

################################################################################
# Train - Val - Test Splits
################################################################################

X_train = data_final[(data_final.index.get_level_values(2)<2016) & \
                     (data_final['Conf']!='NotMajor') &
                     (data_final.index.get_level_values(3)>4)].\
                                       drop(['target_margin'], axis=1).\
                                       fillna(data_final.mean())
y_train = data_final[(data_final.index.get_level_values(2)<2016) & \
                     (data_final['Conf']!='NotMajor') & \
                     (data_final.index.get_level_values(3)>4)]['target_margin']

X_val = data_final[(data_final.index.get_level_values(2)==2016) & \
                   (data_final['Conf']!='NotMajor') & \
                   (data_final.index.get_level_values(3)>4)].\
                                     drop(['target_margin'], axis=1).\
                                     fillna(data_final.mean())
y_val = data_final[(data_final.index.get_level_values(2)==2016) & \
                   (data_final['Conf']!='NotMajor') & \
                   (data_final.index.get_level_values(3)>4)]['target_margin']

X_test = data_final[(data_final.index.get_level_values(2)==2017) & \
                    (data_final['Conf']!='NotMajor') & \
                    (data_final.index.get_level_values(3)>4)].\
                                      drop(['target_margin'], axis=1).\
                                      fillna(data_final.mean())
y_test = data_final[(data_final.index.get_level_values(2)==2017) & \
                    (data_final['Conf']!='NotMajor') & \
                    (data_final.index.get_level_values(3)>4)]['target_margin']

################################################################################
# Final Feature Selection
################################################################################

base_featurestouse = ['CurSeason_HH_VA_mean_margin']

swc_featurestouse = ['Glicko_Rating_Home', 'Glicko_Rating_Away',
                 'Glicko_Rating_Deviance_Home', 'Glicko_Rating_Deviance_Away',
                 'Glicko_Sigma_Home', 'Glicko_Sigma_Away',
                 'CTMC_Rating_Home', 'CTMC_Rating_Away']

ijh_featurestouse = ['HomeElo', 'HomeEloProb','HomeLuck','HomePrevLuck',
                    'HomePythPct','HomePythWins','HomeWinPct','VisElo',
                    'VisEloProb','VisLuck', 'VisPrevLuck','VisPythPct',
                    'VisPythWins','VisWinPct','SpreadElo','HomeConf_NotMajor',
                    'VisConf_NotMajor']

cdr_featurestouse = bcs_df.columns.values.tolist()

featurestouse = base_featurestouse + swc_featurestouse + ijh_featurestouse + cdr_featurestouse

################################################################################
# Baseline Model
################################################################################

# Current season average margins diff
X_train['CurSeason_HH_VA_mean_margin'] = (X_train['CurSeason_HH_margin_mean'] \
                                    - X_train['CurSeason_VA_margin_mean']*-1)
X_val['CurSeason_HH_VA_mean_margin'] = (X_val['CurSeason_HH_margin_mean'] \
                                        - X_val['CurSeason_VA_margin_mean']*-1)
X_test['CurSeason_HH_VA_mean_margin'] = (X_test['CurSeason_HH_margin_mean'] \
                                        - X_test['CurSeason_VA_margin_mean']*-1)

# Train and report baseline model with current season metric
ols = LinearRegression()
ols.fit(X_train['CurSeason_HH_VA_mean_margin'].values.reshape(-1,1), y_train)
ols.score(X_train['CurSeason_HH_VA_mean_margin'].values.reshape(-1,1), y_train)
ols.score(X_val['CurSeason_HH_VA_mean_margin'].values.reshape(-1,1), y_val)
preds = ols.predict(X_val['CurSeason_HH_VA_mean_margin'].values.reshape(-1,1))
mean_squared_error(preds, y_val)

################################################################################
# Full Model
################################################################################

####### Scale data select features
standardscaler = StandardScaler()
X_trainscaled = standardscaler.fit_transform(X_train[featurestouse])
X_valscaled = standardscaler.transform(X_val[featurestouse])
X_testscaled = standardscaler.transform(X_test[featurestouse])

####### Setup grid search
def do_grid_search(X_train, y_train, X_val, y_val):
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.concatenate((y_train, y_val))
    val_fold = [-1]*len(X_train) + [0]*len(X_val) #0 corresponds to validation

    param_grid = [{'alpha': [10**x for x in np.arange(-4,-2, 0.1)],
                   'gamma': [10**x for x in np.arange(-5,-3, 0.1)]}]
    estimator = KernelRidge(kernel='rbf')
    grid = GridSearchCV(estimator,
                        param_grid,
                        return_train_score=True,
                        cv = PredefinedSplit(test_fold=val_fold),
                        refit = True,
                        scoring = make_scorer(mean_squared_error,
                                              greater_is_better = False))
    grid.fit(X_train_val, y_train_val)
    df = pd.DataFrame(grid.cv_results_)

    cols_to_keep = ["param_alpha", "param_gamma",
                    "mean_test_score","mean_train_score"]
    df_toshow = df[cols_to_keep].fillna('-')
    df_toshow = df_toshow.sort_values(by=["mean_test_score"])
    return grid, df_toshow

####### Grid search and results
grid, df_toshow = do_grid_search(X_trainscaled, y_train, X_valscaled, y_val)
df_toshow.sort_values('mean_test_score', ascending=False)

####### Plot parameter space
results = pd.pivot_table(df_toshow,
                         index='param_gamma',
                         columns='param_alpha',
                         values='mean_test_score')
fig, ax = plt.subplots(1,1, figsize=(15,10))
sns.heatmap(results, annot=True, cmap='seismic', fmt="0.1f", ax=ax)

####### Best Model MSE and R-Squared
best_alpha = grid.best_params_['alpha']
best_gamma = grid.best_params_['gamma']

best_ridge = KernelRidge(kernel='rbf', alpha=best_alpha, gamma=best_gamma)
best_ridge.fit(X_trainscaled, y_train)
best_ridge.score(X_valscaled, y_val)
preds = best_ridge.predict(X_valscaled)
mean_squared_error(preds, y_val)
