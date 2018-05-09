import os
import itertools
import glob
import datetime

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
from sklearn.model_selection import train_test_split

%matplotlib inline

################################################################################
# Import data
################################################################################

root_dir = os.getcwd()
ctmc_dir = 'ctmc'
glicko_dir = 'glicko'
curseason_dir = 'curseason'
scorepreds_dir = 'scorepreds'
snooz_dir = 'snoozle'
bcs_dir = 'BCS'
conf_dir = 'conferences'
data_dir = os.path.join(root_dir, "data")

##### IJH Data
file = os.path.join(data_dir, snooz_dir, "snoozle_ijh.csv")
snooz_df = pd.read_csv(file)
snooz_df = snooz_df.set_index(['HomeID', 'VisID', 'Season', 'Week'])
snooz_df['target_margin'] = snooz_df['HomeFinal'] - snooz_df['VisFinal']
snooz_df = snooz_df[['HomeElo', 'HomeEloProb','HomeLuck','HomePrevLuck',
                            'HomePythPct','HomePythWins','HomeWinPct','VisElo',
                            'VisEloProb','VisLuck', 'VisPrevLuck','VisPythPct',
                            'VisPythWins','VisWinPct','SpreadElo',
                            'target_margin','HomeFinal','VisFinal']]
snooz_df = snooz_df.drop_duplicates()

######## SWC Data
file = os.path.join(data_dir, ctmc_dir, "score_ctmc_snoozle.csv")
scores_ctmc_df = pd.read_csv(file,index_col=[0,1,2,3]).drop_duplicates()
scores_ctmc_df.index.names = ['HomeID', 'VisID', 'Season', 'Week']
file = os.path.join(data_dir, glicko_dir, "glicko_snoozle.csv")
glicko_df = pd.read_csv(file, index_col=[0,1,2,3]).drop_duplicates()
glicko_df.index.names = ['HomeID', 'VisID', 'Season', 'Week']
file = os.path.join(data_dir, curseason_dir, "curseason.csv")
curseason_df = pd.read_csv(file, index_col=[0,1,2,3]).drop_duplicates()

# Join SC Data
data_final = snooz_df.join(scores_ctmc_df, how='left')
data_final = data_final.join(glicko_df)
data_final = data_final.join(curseason_df)

##### CDR Data
file = os.path.join(data_dir, bcs_dir, "BCS-SOS.csv")
bcs_df = pd.read_csv(file).drop_duplicates()
bcs_df = bcs_df.set_index(['HomeID', 'VisID', 'Season', 'Week'])

# Join CDR Data
data_final = data_final.join(bcs_df)

# Join Conference Data
file = os.path.join(data_dir, conf_dir, "mergedConferences.csv")
conf_df = pd.read_csv(file).drop_duplicates()
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
data_final['HomeConf_NotMajor'] = np.where(data_final['Conf'] == 'NotMajor', 1, 0)
data_final['VisConf_NotMajor'] = np.where(data_final['ConfVis'] == 'NotMajor', 1, 0)

################################################################################
# Train - Val - Test Splits
################################################################################

X_train = data_final[(data_final.index.get_level_values(2)<2016) & \
                     (data_final['Conf']!='NotMajor') &
                     (data_final.index.get_level_values(3)>4)].\
                                       drop(['target_margin'], axis=1).\
                                       fillna(data_final.mean())

X_val = data_final[(data_final.index.get_level_values(2)==2016) & \
                   (data_final['Conf']!='NotMajor') & \
                   (data_final.index.get_level_values(3)>4)].\
                                     drop(['target_margin'], axis=1).\
                                     fillna(data_final.mean())

X_test = data_final[(data_final.index.get_level_values(2)==2017) & \
                    (data_final['Conf']!='NotMajor') & \
                    (data_final.index.get_level_values(3)>4)].\
                                      drop(['target_margin'], axis=1).\
                                      fillna(data_final.mean())

################################################################################
# Final Feature Selection
################################################################################

base_featurestouse = [col for col in data_final.columns if \
                      col.find('InSeason') > -1]

swc_featurestouse = ['Glicko_Rating_Home', 'Glicko_Rating_Away',
                     'Glicko_Rating_Deviance_Home', 'Glicko_Rating_Deviance_Away',
                     'Glicko_Sigma_Home', 'Glicko_Sigma_Away',
                     'CTMC_Rating_Home', 'CTMC_Rating_Away',
                     'CurSeason_HH_VA_mean_margin']

ijh_featurestouse = ['HomeElo', 'HomeEloProb','HomeLuck','HomePrevLuck',
                    'HomePythPct','HomePythWins','HomeWinPct','VisElo',
                    'VisEloProb','VisLuck', 'VisPrevLuck','VisPythPct',
                    'VisPythWins','VisWinPct','SpreadElo','HomeConf_NotMajor',
                    'VisConf_NotMajor']

cdr_featurestouse = bcs_df.columns.values.tolist()

# featurestouse = base_featurestouse + swc_featurestouse + ijh_featurestouse + cdr_featurestouse
featurestouse = swc_featurestouse + ijh_featurestouse + cdr_featurestouse

################################################################################
# Standardize data
################################################################################

# Current season average margins diff
X_train['CurSeason_HH_VA_mean_margin'] = (X_train['SpreadHomeInSeasonAvg'] \
                                    - X_train['SpreadVisInSeasonAvg']*-1)
X_val['CurSeason_HH_VA_mean_margin'] = (X_val['SpreadHomeInSeasonAvg'] \
                                        - X_val['SpreadVisInSeasonAvg']*-1)
X_test['CurSeason_HH_VA_mean_margin'] = (X_test['SpreadHomeInSeasonAvg'] \
                                        - X_test['SpreadVisInSeasonAvg']*-1)

####### Scale data select features
standardscaler = StandardScaler()
X_trainscaled = standardscaler.fit_transform(X_train[featurestouse])
X_valscaled = standardscaler.transform(X_val[featurestouse])
X_testscaled = standardscaler.transform(X_test[featurestouse])

################################################################################
# Score Model Features
################################################################################

y_train_hscore = data_final[(data_final.index.get_level_values(2)<2016) & \
                     (data_final['Conf']!='NotMajor') & \
                     (data_final.index.get_level_values(3)>4)]['HomeFinal']
y_train_vscore = data_final[(data_final.index.get_level_values(2)<2016) & \
                     (data_final['Conf']!='NotMajor') & \
                     (data_final.index.get_level_values(3)>4)]['VisFinal']


y_val_hscore = data_final[(data_final.index.get_level_values(2)==2016) & \
                   (data_final['Conf']!='NotMajor') & \
                   (data_final.index.get_level_values(3)>4)]['HomeFinal']
y_val_vscore = data_final[(data_final.index.get_level_values(2)==2016) & \
                   (data_final['Conf']!='NotMajor') & \
                   (data_final.index.get_level_values(3)>4)]['VisFinal']


y_test_hscore = data_final[(data_final.index.get_level_values(2)==2017) & \
                    (data_final['Conf']!='NotMajor') & \
                    (data_final.index.get_level_values(3)>4)]['HomeFinal']
y_test_vscore = data_final[(data_final.index.get_level_values(2)==2017) & \
                    (data_final['Conf']!='NotMajor') & \
                    (data_final.index.get_level_values(3)>4)]['VisFinal']

####### Setup grid search
def do_grid_search(X_train, y_train, X_val, y_val):
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.concatenate((y_train, y_val))
    val_fold = [-1]*len(X_train) + [0]*len(X_val) #0 corresponds to validation

    param_grid = [{'alpha': [10**x for x in np.arange(-5,-1, 0.5)],
                   'gamma': [10**x for x in np.arange(-5,-1, 0.5)]}]
    estimator = KernelRidge(kernel='rbf')
    grid = GridSearchCV(estimator,
                        param_grid,
                        return_train_score=True,
                        cv = PredefinedSplit(test_fold=val_fold),
                        refit = False,
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
grid, df_toshow = do_grid_search(X_trainscaled, y_train_hscore, X_valscaled, y_val_hscore)
# df_toshow.sort_values('mean_test_score', ascending=False)

####### Best Model MSE and R-Squared
best_alpha = grid.best_params_['alpha']
best_gamma = grid.best_params_['gamma']

best_ridge_hscore = KernelRidge(kernel='rbf', alpha=best_alpha, gamma=best_gamma)
best_ridge_hscore.fit(X_trainscaled, y_train_hscore)
best_ridge_hscore.score(X_valscaled, y_val_hscore)
preds = best_ridge_hscore.predict(X_valscaled)
mean_squared_error(preds, y_val_hscore)

hscore_train_feats = best_ridge_hscore.predict(X_trainscaled)
hscore_val_feats = best_ridge_hscore.predict(X_valscaled)
hscore_test_feats = best_ridge_hscore.predict(X_testscaled)

X_train['HomePredFinal'] = hscore_train_feats
X_val['HomePredFinal'] = hscore_val_feats
X_test['HomePredFinal'] = hscore_test_feats

grid, df_toshow = do_grid_search(X_trainscaled, y_train_vscore, X_valscaled, y_val_vscore)

####### Best Model MSE and R-Squared
best_alpha = grid.best_params_['alpha']
best_gamma = grid.best_params_['gamma']

best_ridge_vscore = KernelRidge(kernel='rbf', alpha=best_alpha, gamma=best_gamma)
best_ridge_vscore.fit(X_trainscaled, y_train_vscore)
best_ridge_vscore.score(X_valscaled, y_val_vscore)
preds = best_ridge_vscore.predict(X_valscaled)
mean_squared_error(preds, y_val_vscore)

vscore_train_feats = best_ridge_vscore.predict(X_trainscaled)
vscore_val_feats = best_ridge_vscore.predict(X_valscaled)
vscore_test_feats = best_ridge_hscore.predict(X_testscaled)

X_train['VisPredFinal'] = vscore_train_feats
X_val['VisPredFinal'] = vscore_val_feats
X_test['VisPredFinal'] = vscore_test_feats

data_final = pd.concat([X_train, X_val, X_test])
data_final = data_final[['HomePredFinal','VisPredFinal']]

file = os.path.join(root_dir, data_dir, 'scorepreds', 'scorepreds.csv')
data_final.to_csv(file)
