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
                            'VisConf_NotMajor','target_margin','HomeFinal','VisFinal']]
scores_pe_df = scores_pe_df.drop_duplicates()

######## SWC Data
file = os.path.join(data_dir, ctmc_dir, "scores_ctmc_ultimate_data_final.csv")
scores_ctmc_df = pd.read_csv(file,index_col=[0,1,2,3]).drop_duplicates()
scores_ctmc_df.index.names = ['HomeID', 'VisID', 'Season', 'Week']
file = os.path.join(data_dir, glicko_dir, "glicko_ultimate_data_final.csv")
glicko_df = pd.read_csv(file, index_col=[0,1,2,3]).drop_duplicates()
glicko_df.index.names = ['HomeID', 'VisID', 'Season', 'Week']
file = os.path.join(data_dir, curseason_dir, "curseason.csv")
curseason_df = pd.read_csv(file, index_col=[0,1,2,3]).drop_duplicates()
file = os.path.join(data_dir, scorepreds_dir, "scorepreds.csv")
scorepreds_df = pd.read_csv(file, index_col=[0,1,2,3]).drop_duplicates()

# Join SC Data
data_final = scores_pe_df.join(scores_ctmc_df, how='left')
data_final = data_final.join(glicko_df)
data_final = data_final.join(curseason_df)
data_final = data_final.join(scorepreds_df, how='left')

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

base_featurestouse = [col for col in data_final.columns if \
                      col.find('InSeason') > -1]

swc_featurestouse = ['Glicko_Rating_Home', 'Glicko_Rating_Away',
                     'Glicko_Rating_Deviance_Home', 'Glicko_Rating_Deviance_Away',
                     'Glicko_Sigma_Home', 'Glicko_Sigma_Away',
                     'CTMC_Rating_Home', 'CTMC_Rating_Away','HomePredFinal',
                     'VisPredFinal','CurSeason_HH_VA_mean_margin']

ijh_featurestouse = ['HomeElo', 'HomeEloProb','HomeLuck','HomePrevLuck',
                    'HomePythPct','HomePythWins','HomeWinPct','VisElo',
                    'VisEloProb','VisLuck', 'VisPrevLuck','VisPythPct',
                    'VisPythWins','VisWinPct','SpreadElo','HomeConf_NotMajor',
                    'VisConf_NotMajor']

cdr_featurestouse = bcs_df.columns.values.tolist()

# featurestouse = base_featurestouse + swc_featurestouse + ijh_featurestouse + cdr_featurestouse
featurestouse = swc_featurestouse + ijh_featurestouse + cdr_featurestouse

################################################################################
# Baseline Model
################################################################################

# Current season average margins diff
X_train['CurSeason_HH_VA_mean_margin'] = (X_train['SpreadHomeInSeasonAvg'] \
                                    - X_train['SpreadVisInSeasonAvg']*-1)
X_val['CurSeason_HH_VA_mean_margin'] = (X_val['SpreadHomeInSeasonAvg'] \
                                        - X_val['SpreadVisInSeasonAvg']*-1)
X_test['CurSeason_HH_VA_mean_margin'] = (X_test['SpreadHomeInSeasonAvg'] \
                                        - X_test['SpreadVisInSeasonAvg']*-1)

# Train and report baseline model with current season metric
ols = LinearRegression()
ols.fit(X_train['CurSeason_HH_VA_mean_margin'].values.reshape(-1,1), y_train)
ols.score(X_train['CurSeason_HH_VA_mean_margin'].values.reshape(-1,1), y_train)
ols.score(X_val['CurSeason_HH_VA_mean_margin'].values.reshape(-1,1), y_val)
preds = ols.predict(X_val['CurSeason_HH_VA_mean_margin'].values.reshape(-1,1))
mean_squared_error(preds, y_val)

################################################################################
# Standardize data
################################################################################

####### Scale data select features
standardscaler = StandardScaler()
X_trainscaled = standardscaler.fit_transform(X_train[featurestouse])
X_valscaled = standardscaler.transform(X_val[featurestouse])
X_testscaled = standardscaler.transform(X_test[featurestouse])

################################################################################
# Full Model
################################################################################

####### Setup grid search
def do_grid_search(X_train, y_train, X_val, y_val):
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.concatenate((y_train, y_val))
    val_fold = [-1]*len(X_train) + [0]*len(X_val) #0 corresponds to validation

    param_grid = [{'alpha': [10**x for x in np.arange(-3,-1, 0.25)],
                   'gamma': [10**x for x in np.arange(-6,-1, 0.25)]}]
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

################################################################################
# Odds Model
################################################################################

def load_csvs(file_names):
    """Loads and concatentates csv's from a directory"""
    df = pd.DataFrame()
    for each_file in file_names:
        new_df = pd.read_csv(each_file)
        df = pd.concat([df, new_df])
    return df

def join_data(scores_df, stats_df, odds_df):
    """
    Creates a unique key for each game using the date the game was played
    and the home and away abbreviated names (Not all data sets have a HomeID
    and AwayID)
    """

    # Add dates to join on
    scores_df['Year'] = scores_df['WeekDate'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").year)
    scores_df['Month'] = scores_df['WeekDate'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").month)
    scores_df['Day'] = scores_df['WeekDate'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").day)
    stats_df['Year'] = stats_df['Start'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").year)
    stats_df['Month'] = stats_df['Start'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").month)
    stats_df['Day'] = stats_df['Start'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day)
    odds_df['Year'] = odds_df['DATE(date)'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").year)
    odds_df['Month'] = odds_df['DATE(date)'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").month)
    odds_df['Day'] = odds_df['DATE(date)'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").day)

    # Join Data
    data = scores_df.merge(
        stats_df.drop(['Season', 'Start', 'Week'], axis=1),
        left_on = ['Year', 'Month', 'Day', 'Home', 'Visiter'],
        right_on = ['Year', 'Month', 'Day', 'Home', 'Away'])
    data = data.merge(odds_df.drop(['DATE(date)', 'HomeScore', 'AwayScore'],
                                    axis=1),
        left_on = ['Year', 'Month', 'Day', 'Home', 'Visiter'],
        right_on = ['Year', 'Month', 'Day', 'Home', 'Away'])

    # Target feature
    data['target_margin'] = data['HomeFinal'] - data['VisFinal']

    # Other features
    data['D1_Match'] = [True if not pd.isnull(x) else False for \
                        x in data['Spread_Mirage']]

    return data

# Load data locations
scores_dir = 'scores_pe'
stats_dir = 'stats'
odds_dir = 'odds'

scores_names = glob.glob(os.path.join(root_dir, data_dir, scores_dir, "scores_pythElo201?.csv"))
stats_names =  glob.glob(os.path.join(root_dir, data_dir, stats_dir, "ncaastats201?.csv"))
odds_names = [os.path.join(root_dir, data_dir, odds_dir, "NCAAF_Odds.csv")]

# Import data and join
scores_df = load_csvs(scores_names)
stats_df = load_csvs(stats_names)
odds_df = load_csvs(odds_names)
data = join_data(scores_df, stats_df, odds_df)

spreads = data.set_index(['HomeID','VisID','Season','Week']).filter(regex="Spread_")
m = spreads.mean(axis=1)
for i, col in enumerate(spreads):
    # using i allows for duplicate columns
    # inplace *may* not always work here, so IMO the next line is preferred
    # df.iloc[:, i].fillna(m, inplace=True)
    spreads.iloc[:, i] = spreads.iloc[:, i].fillna(m)
# spreads['target_margin'] = data['target_margin']
spreads.dropna(axis=0, inplace=True)
spreads = spreads.join(pd.DataFrame(data.set_index(['HomeID','VisID','Season','Week'])['target_margin']))

# Join Conference Data
file = os.path.join(data_dir, conf_dir, "mergedConferences.csv")
conf_df = pd.read_csv(file).drop_duplicates()
spreads= spreads.reset_index().merge(conf_df,
                                            left_on=['HomeID', 'Season'],
                                            right_on=['ID','Year'],
                                            suffixes=('','Home'))
spreads = spreads.reset_index().merge(conf_df,
                                            left_on=['VisID', 'Season'],
                                            right_on=['ID','Year'],
                                            suffixes=('','Vis'))
spreads['Week'] = spreads['Week'].astype(int)
spreads['Week'] = np.where(spreads['Season']==2016, spreads['Week'] - 1, spreads['Week'])
spreads = spreads.set_index(['HomeID', 'VisID', 'Season', 'Week'])
spreads = spreads.drop(['ID','Year','IDVis','index','Team','TeamVis','ConfVis','Year','YearVis'],1)

X_train_odds = spreads[(spreads.index.get_level_values(2)<2016) & \
                     (spreads['Conf']!='NotMajor') &
                     (spreads.index.get_level_values(3)>4)].\
                                       drop(['target_margin'], axis=1).\
                                       fillna(spreads.mean()).drop('Conf',1)
y_train_odds = spreads[(spreads.index.get_level_values(2)<2016) & \
                     (spreads['Conf']!='NotMajor') & \
                     (spreads.index.get_level_values(3)>4)]['target_margin']

X_val_odds = spreads[(spreads.index.get_level_values(2)==2016) & \
                   (spreads['Conf']!='NotMajor') & \
                   (spreads.index.get_level_values(3)>4)].\
                                     drop(['target_margin'], axis=1).\
                                     fillna(spreads.mean()).drop('Conf',1)
y_val_odds = spreads[(spreads.index.get_level_values(2)==2016) & \
                   (spreads['Conf']!='NotMajor') & \
                   (spreads.index.get_level_values(3)>4)]['target_margin']

lm = LinearRegression().fit(X_train_odds, y_train_odds)
predictions = lm.predict(X_val_odds)
print(mean_squared_error(y_val_odds, predictions))
lm.score(X_val_odds, y_val_odds)

####### Our Model for Comprison
X_train_odds_comp = X_train.loc[X_train_odds.index]
X_val_odds_comp = X_val.loc[X_val_odds.index]

####### Scale data select features
standardscaler = StandardScaler()
X_trainscaled_odds_comp = standardscaler.fit_transform(X_train_odds_comp[featurestouse])
X_valscaled_odds_comp = standardscaler.transform(X_val_odds_comp[featurestouse])

####### Setup grid search
def do_grid_search(X_train, y_train, X_val, y_val):
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.concatenate((y_train, y_val))
    val_fold = [-1]*len(X_train) + [0]*len(X_val) #0 corresponds to validation

    param_grid = [{'alpha': [10**x for x in np.arange(-3,-1, 0.25)],
                   'gamma': [10**x for x in np.arange(-6,-1, 0.25)]}]
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
grid_odds, df_toshow_odds = do_grid_search(X_trainscaled_odds_comp, y_train_odds, X_valscaled_odds_comp, y_val_odds)

####### Best Model MSE and R-Squared
best_alpha_odds = grid_odds.best_params_['alpha']
best_gamma_odds = grid_odds.best_params_['gamma']

best_ridge_odds = KernelRidge(kernel='rbf', alpha=best_alpha_odds, gamma=best_gamma_odds)
best_ridge_odds.fit(X_trainscaled_odds_comp, y_train_odds)
best_ridge_odds.score(X_valscaled_odds_comp, y_val_odds)
preds_odds = best_ridge_odds.predict(X_valscaled_odds_comp)
mean_squared_error(preds_odds, y_val_odds)
