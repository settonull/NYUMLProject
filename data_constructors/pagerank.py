import numpy as np
import pandas as pd
import os
from collections import defaultdict

cur_dir = os.getcwd()
data = pd.read_csv(os.path.join(cur_dir, "data", "allScores","NCAAAllScores2013_Week.csv"), index_col=False)


def compare(a, b, larger_is_better):
    """Greater than or less then comparison based on which is better"""

    if larger_is_better:
        return a > b
    else:
        return a < b

def build_transition_matrix(df, home_metric_label, away_metric_label, larger_is_better):
    """
    Computes the transition matrix based on the column names supplied
    """

    # Determine who is the "better" and "worse" team based on the metric and
    # sum metrics for better and worse teams
    df['betterID'] = list(map(lambda home_id, away_id, home_metric, away_metric:
        home_id if compare(home_metric, away_metric, larger_is_better) else away_id, df['HomeID'], df['AwayID'],
        df[home_metric_label], df[away_metric_label]))
    df['betterMetric'] = list(map(lambda home_id, away_id, home_metric, away_metric:
        home_metric if compare(home_metric, away_metric, larger_is_better) else away_metric, df['HomeID'], df['AwayID'],
        df[home_metric_label], df[away_metric_label]))

    df['worseID'] = list(map(lambda home_id, away_id, better_id: home_id if better_id == away_id else away_id,
        df['HomeID'], df['AwayID'], df['betterID']))
    df['worseMetric'] = list(map(lambda home_id, home_score, away_id, away_score, better_id:
        home_score if better_id == away_id else away_score, df['HomeID'],
        df[home_metric_label], df['AwayID'], df[away_metric_label], df['betterID']))

    df['betterDiff'] = df['betterMetric'] - df['worseMetric']
    df['worseDiff'] = 0

    # Combine the better and worse data
    better_df = df[['betterID', 'worseID', 'betterMetric']]
    worse_df = df[['worseID', 'betterID', 'worseMetric']]
    better_df.columns = ['TeamID', 'OpponentID', 'Metric']
    worse_df.columns = ['TeamID', 'OpponentID', 'Metric']
    both_df = pd.concat([better_df, worse_df], axis=0)
    both_df = both_df.groupby(['TeamID', 'OpponentID']).agg('sum')

    # Create and return the transition matrix
    return pd.pivot_table(both_df, index=['OpponentID'], columns=['TeamID'], fill_value=0)['Metric']


def steady_state(transition_matrix, damping=0.85):
    nteams = transition_matrix.shape[0]
    identity = np.eye(nteams)
    prob_transition_matrix = (transition_matrix/transition_matrix.sum(axis=0)).fillna(0).T
    partial_steady_state = np.linalg.inv(identity - damping*prob_transition_matrix)
    damping_vector = (1 - damping)/nteams * np.ones(nteams)
    steady_state = partial_steady_state.dot(damping_vector)
    return pd.DataFrame(steady_state, index = prob_transition_matrix.index.values)

def steady_state_iter(transition_matrix, damping=0.85):
    nteams = transition_matrix.shape[0]
    teamdist = np.repeat(1/nteams, nteams)
    prob_transition_matrix = (transition_matrix/transition_matrix.sum(axis=0)).fillna(0).T
    for i in range(10000):
        teamdist = damping*prob_transition_matrix.dot(teamdist) + ((1-damping)/nteams)*np.ones(nteams)
    teamdist = pd.DataFrame(teamdist, index = prob_transition_matrix.index.values)
    return teamdist

def pagerank(df, metrics, larger_is_better, damping=0.85):
    transition_matrix = build_transition_matrix(df, metrics[0], metrics[1], larger_is_better)
    return steady_state(transition_matrix, damping=damping)

def pagerank_summarize(df, metrics=('HomeFinal', 'VisFinal'), larger_is_better = True,
                        damping=0.85, min_weeks=4):
    """
    For each week greater than min_weeks in each season, computes the PageRank
    ratings for all teams meeting the minimum number of matches.
    """

    # Loop through seasons and weeks to create full history of ratings by team
    results = pd.DataFrame()
    for season in df['Season'].unique():
        for week in df[df['Season']==season]['Week'].unique():
            if week > min_weeks:
                ratings = pagerank(df[(df['Season']==season) & (df['Week']<=week)].copy(),
                                    metrics, larger_is_better, damping)
                ratings.reset_index(inplace=True)
                ratings.columns = ['TeamID','PageRank_Rating']
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
    df['teams_key'] = list(map(lambda x, y: str(x)+"-"+str(y), df['HomeID'], df['AwayID']))
    df.set_index(['teams_key', 'DateTime', 'Season'], inplace=True)
    df = df[['PageRank_Rating', 'PageRank_Rating_Away']]
    df.columns = ['PageRank_Rating_Home', 'PageRank_Rating_Away']

    return df
