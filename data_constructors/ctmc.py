import numpy as np
import pandas as pd
import os
from collections import defaultdict

def unique_teams_faced(df):
    """
    Counts the number of unique teams faced for each team in the input
    dataframe.  This count is used to filter which teams are scores by the
    CTMC function.
    """

    # Matrix of games played with values of the opponents ID
    games = pd.pivot_table(df, index='HomeID', columns='AwayID',
                            aggfunc='count')[df.columns[0]]
    games /= games
    games = games.fillna(0)
    home_games = games * games.columns.values
    away_games = games.T * games.T.columns.values

    # Combines the list of opponents from each team as a home player and away
    # player, then counts the unique opponents
    games_dict = defaultdict()
    for i, row in home_games.iterrows():
        games_dict[i] = list(row.unique())
    for i, row in away_games.iterrows():
        if i in games_dict:
            games_dict[i] += list(row.unique())
        else:
            games_dict[i] = list(row.unique())
    for key in games_dict:
        games_dict[key] = len(set(games_dict[key])) - 1

    return pd.DataFrame.from_dict(games_dict, orient='index')


def compare(a, b, larger_is_better):
    """Greater than or less then comparison based on which is better"""

    if larger_is_better:
        return a > b
    else:
        return a < b

def build_transition_matrix(df, home_metric_label, away_metric_label,
                            larger_is_better):
    """
    Computes the transition matrix based on the column names supplied
    """

    # Determine who is the "better" and "worse" team based on the metric and
    # sum metrics for better and worse teams
    df['betterID'] = list(map(lambda home_id, away_id, home_metric, away_metric:
        home_id if compare(home_metric, away_metric, larger_is_better) else \
        away_id, df['HomeID'], df['AwayID'],
        df[home_metric_label], df[away_metric_label]))
    df['betterMetric'] = list(map(lambda home_id, away_id, home_metric, away_metric:
        home_metric if compare(home_metric, away_metric, larger_is_better) else \
        away_metric, df['HomeID'], df['AwayID'],
        df[home_metric_label], df[away_metric_label]))

    df['worseID'] = list(map(lambda home_id, away_id, better_id: home_id if \
                            better_id == away_id else away_id,
                            df['HomeID'], df['AwayID'], df['betterID']))
    df['worseMetric'] = list(map(lambda home_id, home_score, away_id, \
                        away_score, better_id:
                        home_score if better_id == away_id else \
                        away_score, df['HomeID'], df[home_metric_label], \
                        df['AwayID'], df[away_metric_label], df['betterID']))

    # Combine the better and worse data
    better_df = df[['betterID', 'worseID', 'betterMetric']]
    worse_df = df[['worseID', 'betterID', 'worseMetric']]
    better_df.columns = ['TeamID', 'OpponentID', 'Metric']
    worse_df.columns = ['TeamID', 'OpponentID', 'Metric']
    both_df = pd.concat([better_df, worse_df], axis=0)
    both_df = both_df.groupby(['TeamID', 'OpponentID']).agg('sum')

    # Create and return the transition matrix
    return pd.pivot_table(both_df, index=['OpponentID'], columns=['TeamID'],
                            fill_value=1) #changed to one for ultimate

def unique_matches_filter(df, min_matches=2):
    """
    Computes the number of unique matches that each team has participated in
    and filters the input dataframe based on the minimum required matches.
    """
    teams = unique_teams_faced(df)
    teams = teams[teams >= min_matches].dropna()
    teams_mask = list(map(lambda home_id, away_id:
        True if home_id in teams.index.values or away_id in \
        teams.index.values else False,
        df['HomeID'], df['AwayID']))
    df = df[teams_mask]
    return df

def CTMC(df, metrics=('HomeFinal', 'VisFinal'), larger_is_better = True,
        unique_matches_required = 2):
    """Computes the rankings of all teams based on the given metrics."""

    # Remove teams with fewer than the required number of matches
    df = unique_matches_filter(df, unique_matches_required)

    # Create the transition matrix and solve the system of equations to equate
    # the transition out rate with the transition in-rate.  This gives the
    # steady state probabilities.
    transition_df = build_transition_matrix(df, metrics[0], metrics[1], larger_is_better)
    expectation_df = transition_df['Metric'].T
    out_rates = transition_df.sum(axis=1)

    # Set diagonal to the out rate
    for i in range(expectation_df.shape[0]):
        for j in range(expectation_df.shape[1]):
            if i == j:
                 expectation_df.iloc[i,j] = -out_rates.iloc[i]

    expectation_df.iloc[0,:] = 1
    ratings = np.linalg.solve(expectation_df, [1] + [0]*(expectation_df.shape[0] - 1))

    return pd.DataFrame({'TeamID': transition_df.index.values, 'Ratings': ratings}).set_index('TeamID')

def ctmc_summarize(df, metrics=('HomeFinal', 'VisFinal'), larger_is_better = True,
                   unique_matches_required = 2, min_weeks=4):
    """
    For each week greater than min_weeks in each season, computes the CTMC
    ratings for all teams meeting the minimum number of matches.
    """

    # Loop through seasons and weeks to create full history of ratings by team
    results = pd.DataFrame()
    for season in df['Season'].unique():
        for week in df[df['Season']==season]['Week'].unique():
            if week > min_weeks:
                ratings = CTMC(df[(df['Season']==season) & (df['Week']<week)].copy(),
                                metrics, larger_is_better, unique_matches_required)
                ratings.reset_index(inplace=True)
                ratings.columns = ['TeamID','CTMC_Rating']
                ratings['Season'] = season
                ratings['Week'] = week
                results = pd.concat([results, ratings], axis=0, ignore_index=True)

    # Join the ratings to the original schedule of games
    df = df.merge(results, left_on=['Season','Week','HomeID'],
                            right_on=['Season','Week','TeamID'],
                            suffixes=('','_Home'))
    df.drop('TeamID', 1, inplace=True)

    df = df.merge(results, left_on=['Season','Week','AwayID'],
                            right_on=['Season','Week','TeamID'],
                            suffixes=('','_Away'))
    df.drop('TeamID', 1, inplace=True)

    # Create key and set index to join with n_game summaries dataset.
    df.set_index(['HomeID', 'AwayID', 'Season', 'Week'], inplace=True)
    df = df[['CTMC_Rating', 'CTMC_Rating_Away']]
    df.columns = ['CTMC_Rating_Home', 'CTMC_Rating_Away']

    return df
