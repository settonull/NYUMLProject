import pandas as pd

def ngames_summaries(dataframes, n_games):
    """
    Uses a list of dataframes to construct stats (mean, std, min, quartiles,
    max) for 1 game up to n_games, each as new fetures.
    """
    df = pd.DataFrame()
    for frame in dataframes:
        frame = frame.sort_values('DateTime', ascending=False)
        frame = frame.drop(['DateTime','Season'], 1)
        for i in range(n_games):
            n_samples = frame.shape[0]
            sub_frame = frame.iloc[:min(i,n_samples)+1, :]
            sub_frame_summary = sub_frame.describe()
            sub_frame_summary = sub_frame_summary.unstack()
            sub_frame_summary.index = [str(i+1) + '_Games_' + '_'.join(
                index).strip() for index in sub_frame_summary.index.values]
            df = pd.concat([df, pd.DataFrame(sub_frame_summary).T], axis=1)

    return df

def in_season_summary(dataframes, season):
    """
    Uses a list of dataframes to construct stats (mean, std, min, quartiles,
    max) for the current season, each as new fetures.
    """
    df = pd.DataFrame()
    for frame in dataframes:
        frame = frame[frame['Season']==season]
        frame = frame.drop(['DateTime','Season'], 1)
        frame_summary = frame.describe()
        frame_summary = frame_summary.unstack()
        frame_summary.index = ['CurSeason_' + '_'.join(
            index).strip() for index in frame_summary.index.values]
        df = pd.concat([df, pd.DataFrame(frame_summary).T], axis=1)

    return df

def ngames_summarize(dataframe, home_features, away_features, gen_features):
    """
    For each game, creates a ngames_summary for the home and away teams past
    performances as both homenteams and visiting teams.  Currently ignores
    general features that are not specific to the home or away team.
    """

    df = pd.DataFrame()
    for idx, game in dataframe.iterrows():
        # Current game info
        homeID = game['HomeID']
        awayID = game['AwayID']
        cg_year = game['Year']
        cg_week = game['Week']
        cg_margin = game['margin']
        cg_datetime = game['DateTime']
        cg_season = game['Season']

        # Build historical game stats for home and away teams separately
        ht_h_games = dataframe[(dataframe['HomeID']==homeID) & \
                                (dataframe['DateTime'] < cg_datetime) \
                                ][home_features + ['VisFinal','margin',
                                                    'Season','DateTime']]
        ht_h_games.columns = ['HH_' + x if x not in ['Season','DateTime'] else \
                                x for x in ht_h_games.columns]
        ht_a_games = dataframe[(dataframe['AwayID']==homeID) & \
                                (dataframe['DateTime'] < cg_datetime) \
                                ][away_features + ['VisFinal','margin',
                                'Season','DateTime']]
        ht_a_games.columns = ['HA_' + x if x not in ['Season','DateTime'] else \
                                x for x in ht_a_games.columns]
        vt_h_games = dataframe[(dataframe['HomeID']==awayID) & \
                                (dataframe['DateTime'] < cg_datetime) \
                                ][home_features + ['HomeFinal','margin',
                                'Season','DateTime']]
        vt_h_games.columns = ['VH_' + x if x not in ['Season','DateTime'] else \
                                x for x in vt_h_games.columns]
        vt_a_games = dataframe[(dataframe['AwayID']==awayID) & \
                                (dataframe['DateTime'] < cg_datetime) \
                                ][away_features + ['HomeFinal','margin',
                                'Season','DateTime']]
        vt_a_games.columns = ['VA_' + x if x not in ['Season','DateTime'] else \
                                x for x in vt_a_games.columns]

        # Apply summary functions to historical data
        n_games = 4 # All results need to have the same shape. Might need to build handling for < 10 games
        n_game_summaries_df = ngames_summaries([ht_h_games, ht_a_games,
                                                vt_h_games, vt_a_games],
                                                n_games)
        in_season_summaries_df = in_season_summary([ht_h_games, ht_a_games,
                                                    vt_h_games, vt_a_games],
                                                    cg_season)
        summaries_df = pd.concat([n_game_summaries_df,
                                    in_season_summaries_df], axis=1)

        # Combine historical game stats for the current game
        summaries_df['target_margin'] = cg_margin
        summaries_df['HomeID'] = homeID
        summaries_df['AwayID'] = awayID
        summaries_df['Season'] = cg_season
        summaries_df['Week'] = cg_week
        summaries_df = summaries_df.set_index(['HomeID', 'AwayID',
                                                'Season', 'Week'])
        df = pd.concat([df, summaries_df])

    return df
