import pandas as pd
import os

def in_season_summary(dataframe):
    # Restrict ultimate dataset to stat columns only
    stats_cols = [col for col in dataframe.columns if col.find('Pyth') == -1 and\
                  col.find('Elo') == -1 and col.find('Conf') == -1 and \
                  col.find('game_id') == -1 and col.find('conf') == -1 and \
                  col.find('Abb') == -1 and col.find('Name') == -1 and \
                  col.find('Luck') == -1 and col.find('Start') == -1 and \
                  col.find('School') == -1 and col.find('Nick') == -1]
    dataframe = dataframe[stats_cols]

    # Compute averages components
    inseason_sum_home_team = dataframe.groupby(['HomeID','Season','Week']) \
                                      .agg('max').groupby(level=[0,1]) \
                                      .cumsum() \
                                      .drop('VisID', 1)
    inseason_count_home_team = dataframe.groupby(['HomeID','Season','Week']) \
                                        .agg('count').groupby(level=[0,1]) \
                                        .cumsum() \
                                        .drop('VisID', 1)
    inseason_sum_vis_team = dataframe.groupby(['VisID','Season','Week']) \
                                     .agg('max').groupby(level=[0,1]) \
                                     .cumsum() \
                                     .drop('HomeID', 1)
    inseason_count_vis_team = dataframe.groupby(['VisID','Season','Week']) \
                                       .agg('count').groupby(level=[0,1]) \
                                       .cumsum() \
                                       .drop('HomeID', 1)

    # Compute averages and rename columns
    inseason_home_mean = inseason_sum_home_team/inseason_count_home_team
    inseason_home_mean.columns = [col+'HomeInSeasonAvg' for col in inseason_home_mean.columns]
    inseason_home_mean = inseason_home_mean.reset_index()
    inseason_vis_mean = inseason_sum_vis_team/inseason_count_vis_team
    inseason_vis_mean.columns = [col+'VisInSeasonAvg' for col in inseason_vis_mean.columns]
    inseason_vis_mean = inseason_vis_mean.reset_index()

    # Shift weeks to generate t-1 data
    inseason_home_mean_grouped = inseason_home_mean.groupby(['HomeID','Season'],as_index=True).shift(1)
    inseason_home_mean_grouped = pd.concat([inseason_home_mean[['HomeID','Season']], inseason_home_mean_grouped], axis=1)
    inseason_home_mean_grouped['Week'] = inseason_home_mean['Week']
    inseason_home_mean = inseason_home_mean_grouped
    inseason_vis_mean_grouped = inseason_vis_mean.groupby(['VisID','Season'],as_index=True).shift(1)
    inseason_vis_mean_grouped = pd.concat([inseason_vis_mean[['VisID','Season']], inseason_vis_mean_grouped], axis=1)
    inseason_vis_mean_grouped['Week'] = inseason_vis_mean['Week']
    inseason_vis_mean = inseason_vis_mean_grouped

    # Join to actual game structure
    result = dataframe[['HomeID', 'VisID', 'Season', 'Week']]
    result = result.merge(inseason_home_mean,
                          left_on=['HomeID','Season','Week'],
                          right_on=['HomeID','Season','Week'])
    result = result.merge(inseason_vis_mean,
                          left_on=['VisID','Season','Week'],
                          right_on=['VisID','Season','Week'])
    result = result.set_index(['HomeID', 'VisID', 'Season', 'Week'])

    return result

if __name__=='__main__':

    cur_dir = os.getcwd()
    data_dir = 'data'
    ultimate_dir = 'ultimate'
    ultimate_file = 'ultimate_2.csv'
    curseason_dir = 'curseason'
    curseason_file = 'curseason.csv'

    file = os.path.join(cur_dir, data_dir, ultimate_dir, ultimate_file)
    dataframe = pd.read_csv(file)
    inseason_df = in_season_summary(dataframe)

    file = os.path.join(cur_dir, data_dir, curseason_dir, curseason_file)
    inseason_df.to_csv(file)
