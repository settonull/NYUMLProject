import pandas as pd
import numpy as np
import datetime
import argparse
import glob
import sys
import os

from data_constructors.ctmc import ctmc_summarize
from data_constructors.n_games import ngames_summarize

################################################################################
# Data import and joining functions
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
    scores_df['Year'] = scores_df['Start'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").year)
    scores_df['Month'] = scores_df['Start'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").month)
    scores_df['Day'] = scores_df['Start'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day)
    scores_df['Hour'] = scores_df['Start'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
    scores_df['DateTime'] = scores_df['Start'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    stats_df['Year'] = stats_df['Start'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").year)
    stats_df['Month'] = stats_df['Start'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").month)
    stats_df['Day'] = stats_df['Start'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day)
    odds_df['Year'] = odds_df['DATE(date)'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").year)
    odds_df['Month'] = odds_df['DATE(date)'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").month)
    odds_df['Day'] = odds_df['DATE(date)'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").day)

    # Join Data
    data = scores_df.drop(['Start'], axis=1).merge(
        stats_df.drop(['Season', 'Start', 'Week'], axis=1),
        left_on = ['Year', 'Month', 'Day', 'Home', 'Away'],
        right_on = ['Year', 'Month', 'Day', 'Home', 'Away'])
    data = data.merge(odds_df.drop(['DATE(date)', 'HomeScore', 'AwayScore'], axis=1),
        left_on = ['Year', 'Month', 'Day', 'Home', 'Away'],
        right_on = ['Year', 'Month', 'Day', 'Home', 'Away'])

    # Target feature
    data['margin'] = data['HomeFinal'] - data['VisFinal']

    # Other features
    data['D1_Match'] = [True if not pd.isnull(x) else False for x in data['Spread_Mirage']]

    return data

################################################################################
# Build final data sets
################################################################################

def main(scores_dir, stats_dir, odds_dir, root_dir, home_metric, away_metric,
         unique_matches_required, larger_is_better, min_weeks, ctmc_only):
    """Construct the final data set"""

    # Load data locations
    scores_names = glob.glob(os.path.join(root_dir, scores_dir, "NCAAAllScores201?_Week.csv"))
    stats_names =  glob.glob(os.path.join(root_dir, stats_dir, "ncaastats201?.csv"))
    odds_names = [os.path.join(root_dir, odds_dir, "NCAAF_Odds.csv")]

    # Import data and join
    scores_df = load_csvs(scores_names)
    stats_df = load_csvs(stats_names)
    odds_df = load_csvs(odds_names)
    data = join_data(scores_df, stats_df, odds_df)

    # Feature groups
    home_features = ['Home1Q', 'Home2Q', 'Home3Q', 'Home4Q', 'HomeOT', 'HomeFinal', 'HPts',
                     'HFD', 'HFum', 'HFumL', 'HPA', 'HPI', 'HRA', 'HRY', 'HomeML_Mirage',
                     'HomeSpreadOdds_Mirage', 'HomeML_Pinnacle', 'HomeSpreadOdds_Pinnacle',
                     'HomeML_Sportsbet', 'HomeSpreadOdds_Sportsbet', 'HomeML_Westgate',
                     'HomeSpreadOdds_Westgate', 'HomeML_Station', 'HomeSpreadOdds_Station',
                     'HomeML_SIA', 'HomeSpreadOdds_SIA', 'HomeML_SBG', 'HomeSpreadOdds_SBG',
                     'HomeML_BetUS', 'HomeSpreadOdds_BetUS']
    away_features = ['Vis1Q', 'Vis2Q', 'Vis3Q', 'Vis4Q', 'VisOT', 'VisFinal', 'APts',
                     'AFD', 'AFum', 'AFumL', 'APA', 'API', 'ARA', 'ARY', 'AwayML_Mirage',
                     'AwaySpreadOdds_Mirage', 'AwayML_Pinnacle', 'AwaySpreadOdds_Pinnacle',
                     'AwayML_Sportsbet', 'AwaySpreadOdds_Sportsbet', 'AwayML_Westgate',
                     'AwaySpreadOdds_Westgate', 'AwayML_Station', 'AwaySpreadOdds_Station',
                     'AwayML_SIA', 'AwaySpreadOdds_SIA', 'AwayML_SBG', 'AwaySpreadOdds_SBG',
                     'AwayML_BetUS', 'AwaySpreadOdds_BetUS']
    gen_features = [x for x in data.columns if x not in home_features + away_features]

    # Build n-games summary data set and export to root directory
    if not ctmc_only :
        ngames = ngames_summarize(data, home_features, away_features, gen_features)
        ngames.to_csv(root_dir+"/ngames_data_final.csv")

    # Build CTMC scores data set and export to root directory
    ratings = ctmc_summarize(data,
                            metrics=(home_metric, away_metric),
                             larger_is_better = larger_is_better,
                             unique_matches_required = unique_matches_required,
                             min_weeks = min_weeks)
    ratings.to_csv(root_dir+"/ctmc_data_final.csv")

if __name__=="__main__":
    """
    Usage:

    To generate the ctmc_data_final.csv:

    python construct_data.py -sc "allScores" -st "stats" -od "odds" -r "/Users/stephencarrow/Documents/DS-GA 1003 Machine Learning/Project/NYUMLProject/data"

    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-sc", "--scores-directory", required=True,
    	help="directory of the scores files")
    ap.add_argument("-st", "--stats-directory", required=True,
    	help="directory of the stats files")
    ap.add_argument("-od", "--odds-directory", required=True,
    	help="directory of the odds files")
    ap.add_argument("-r", "--root-directory", required=True,
    	help="root directory of the project")
    ap.add_argument("-hm", "--home-metric", default='HomeFinal', required=False,
    	help="the home team metric to use for CTMC ratings")
    ap.add_argument("-am", "--away-metric", default='AwayFinal', required=False,
    	help="the away team metric to use for CTMC ratings")
    ap.add_argument("-um", "--unique-matches-required", default=2, required=False,
    	help="the minimum number of opponents a team must face to use for CTMC ratings")
    ap.add_argument("-lb", "--larger-is-better", default=True, required=False,
    	help="is a larger value better in the CTMC ratings")
    ap.add_argument("-mw", "--min-weeks", default=4, required=False,
    	help="minimum weeks required to compute the CTMC ratings")
    ap.add_argument("-ctmc", "--ctmc-only", default=True, required=False,
    	help="only compute the CTMC ratings")
    args = vars(ap.parse_args())

    # Execute final data set construction
    main(args["scores_directory"],
         args["stats_directory"],
         args["odds_directory"],
         args["root_directory"],
         args["home_metric"],
         args["away_metric"],
         args["unique_matches_required"],
         args["larger_is_better"],
         args["min_weeks"],
         args["ctmc_only"])
