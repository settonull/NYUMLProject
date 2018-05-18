import os

import numpy as np
import pandas as pd

"""
Implementation of the Gicko2 rating system found here:
http://www.glicko.net/glicko/glicko2.pdf
"""

def gtransform(phi):
    return 1/np.sqrt(1 + 3*(phi**2)/(np.pi**2))

def etransform(mu_diff, phi):
    return 1/(1 + np.exp(-gtransform(phi)*(mu_diff)))

def glicko_to_glicko2(rating, ratingdeviance):
    mu = (rating - 1500)/173.7178
    phi = ratingdeviance/173.7178
    return mu, phi

def glicko2_to_glicko(mu, phi):
    rating = 173.7178*mu + 1500
    ratingdeviance = 173.7178*phi
    return rating, ratingdeviance

def compare(a, b, larger_is_better):
    """Greater than or less then comparison based on which is better"""

    if larger_is_better:
        return a > b
    else:
        return a < b

def get_opponent_outcomes(df, home_metric_label='HomeFinal',
                            away_metric_label='VisFinal',
                            larger_is_better=True):

    # Determine who is the "better" and "worse" team based on the metric and
    # sum metrics for better and worse teams
    df['betterID'] = list(map(lambda home_id, away_id, home_metric,away_metric:
        home_id if compare(home_metric, away_metric, larger_is_better) else \
                            away_id, df['HomeID'], df['VisID'],
        df[home_metric_label], df[away_metric_label]))
    df['betterMetric'] = 1

    df['worseID'] = list(map(lambda home_id, away_id, better_id: home_id if \
                                better_id == away_id else away_id,
        df['HomeID'], df['VisID'], df['betterID']))
    df['worseMetric'] = 0

    # Combine the better and worse data
    better_df = df[['betterID', 'worseID', 'betterMetric']]
    worse_df = df[['worseID', 'betterID', 'worseMetric']]
    better_df.columns = ['TeamID', 'OpponentID', 'Metric']
    worse_df.columns = ['TeamID', 'OpponentID', 'Metric']
    both_df = pd.concat([better_df, worse_df], axis=0)

    getopponentids = lambda x: [teamID for teamID in x.OpponentID]
    team_opponents = both_df.groupby('TeamID').apply(getopponentids)
    getoutcomes = lambda x: [outcome for outcome in x.Metric]
    team_outcomes = both_df.groupby('TeamID').apply(getoutcomes)

    return team_opponents, team_outcomes

def objective(x, delta, phi, v, a, t):
    num = np.exp(x)*(delta**2 - phi**2 - v - np.exp(x))
    den = 2*(phi**2 + v + np.exp(x))**2
    frac = (x - a)/t**2
    return num/den - frac

def update_stats(team_opponents, team_outcomes, stats, tau, tolerance=1e-6):
    newsigmas = []
    newphis = []
    newmus = []

    for teamid in stats.index.values:
        if teamid in team_opponents and len(team_opponents[teamid]) > 0:
            opponentids = team_opponents[teamid]
            ####### Compute Variance Update
            opponentsphi = stats['phi'][opponentids]
            mudiff = stats['mu'][teamid] - stats['mu'][opponentids]
            gtransformed = gtransform(opponentsphi)
            gtransformedquared = gtransformed**2
            etransformed = etransform(mudiff, opponentsphi)
            variance = 1/np.sum(gtransformedquared
                                *etransformed
                                *(1-etransformed))

            ####### Compute Rating Improvement
            delta = np.array(team_outcomes[teamid]) - etransformed
            delta = variance * np.sum(gtransformed*delta)

            ####### Compute New Sigma
            phi = stats['phi'][teamid]
            mu = stats['mu'][teamid]
            a = np.log(stats['sigma'][teamid]**2)
            A = a

            if delta**2 > phi**2 + variance:
                B = np.log(delta**2 - phi**2 - variance)
            else:
                k = 1
                while objective(a - k*tau, delta, phi, variance, a, tau) < 0:
                    k += 1
                B = a - k*tau

            fa = objective(A, delta, phi, variance, a, tau)
            fb = objective(B, delta, phi, variance, a, tau)
            while np.absolute(B - A) > tolerance:
                C = A + (A - B)*fa/(fb - fa)
                fc = objective(C, delta, phi, variance, a, tau)

                if fc*fb < 0:
                    A = B
                    fa = fb
                else:
                    fa /= 2

                B = C
                fb = fc

            newsigma = np.exp(A/2)

            ####### Compute New Phi and Mu
            phistar = np.sqrt(phi**2 + newsigma**2)
            newphi = 1/np.sqrt(1/phistar**2 + 1/variance)
            newmu = mu + newphi**2 * np.sum(gtransformed
                                            * (np.array(team_outcomes[teamid])
                                                - etransformed))
        else:
            newsigma = stats['sigma'][teamid]
            newphi = np.sqrt(stats['phi'][teamid]**2 + newsigma**2)
            newmu = stats['mu'][teamid]

        ####### Aggregate New Stats
        newsigmas += [newsigma]
        newphis += [newphi]
        newmus += [newmu]

    stats['phi'] = newphis
    stats['sigma'] = newsigmas
    stats['mu'] = newmus

    return stats

def glicko2(data, unique_teamids, glicko_stats):
    mu, phi = glicko_to_glicko2(glicko_stats['ratings'],
                                glicko_stats['ratingsdeviance'])
    sigma = glicko_stats['sigma']

    stats = pd.DataFrame({'mu': mu, 'phi': phi, 'sigma': sigma},
                        index=unique_teamids)
    team_opponents, team_outcomes = get_opponent_outcomes(data)
    newstats = update_stats(team_opponents, team_outcomes, stats,
                            tau=0.5, tolerance=1e-6)

    ratings, ratingsdeviance = glicko2_to_glicko(newstats['mu'],
                                                newstats['phi'])
    newsigma = newstats['sigma']

    glicko_stats['ratings'] = ratings
    glicko_stats['ratingsdeviance'] = ratingsdeviance
    glicko_stats['sigma'] = newsigma

    return glicko_stats

def glicko2_summarize(df, min_weeks=4, use_prior=False):
    """
    For each week greater than min_weeks in each season, computes the CTMC
    ratings for all teams meeting the minimum number of matches.
    """

    # Loop through seasons and weeks to create full history of ratings by team
    results = pd.DataFrame()
    for season in df['Season'].sort_values().unique():
        for week in df[df['Season']==season]['Week'].sort_values().unique():
            if week > min_weeks:
                if week == min_weeks + 1:
                    season_df = df[df['Season']==season].copy()
                    uniqueteamids = pd.concat([season_df['VisID'],
                                                season_df['HomeID']]).unique()
                    if use_prior == True and season > df['Season'].min():
                        ratings = np.repeat(1500, len(uniqueteamids))
                        ratingsdeviance = np.repeat(350, len(uniqueteamids))
                        sigma = np.repeat(0.06, len(uniqueteamids))
                        glicko_stats = pd.DataFrame({'ratings': ratings,
                                                    'ratingsdeviance': ratingsdeviance,
                                                    'sigma': sigma}, index=uniqueteamids)
                        prior = results[results['Season']==season-1]
                        prior_id_mask = [True if id in uniqueteamids else False for id in prior['TeamID']]
                        prior = prior[prior_id_mask]
                        prior = prior.sort_values('Week').groupby('TeamID').tail(1)
                        prior = prior.drop('Week',1)
                        prior = prior.set_index('TeamID')
                        glicko_stats.loc[prior.index, 'ratings'] = prior['Glicko_Rating'] - (prior['Glicko_Rating'] - 1500)/2
                        glicko_stats.loc[prior.index, 'ratingsdeviance'] = prior['Glicko_Rating_Deviance'] - (prior['Glicko_Rating_Deviance'] - 350)/2
                        glicko_stats.loc[prior.index, 'sigma'] = prior['Glicko_Sigma'] - (prior['Glicko_Sigma'] - 0.06)/2
                    else:
                        ratings = np.repeat(1500, len(uniqueteamids))
                        ratingsdeviance = np.repeat(350, len(uniqueteamids))
                        sigma = np.repeat(0.06, len(uniqueteamids))
                        glicko_stats = pd.DataFrame({'ratings': ratings,
                                                    'ratingsdeviance': ratingsdeviance,
                                                    'sigma': sigma}, index=uniqueteamids)

                week_df = df[(df['Season']==season) & (df['Week']<week)].copy()
                glicko_stats = glicko2(week_df, uniqueteamids, glicko_stats)


                glicko_results = glicko_stats.reset_index()
                print(glicko_results.head(), season)
                glicko_results.columns = ['TeamID','Glicko_Rating',
                                            'Glicko_Rating_Deviance',
                                            'Glicko_Sigma']
                glicko_results['Season'] = season
                glicko_results['Week'] = week
                results = pd.concat([results, glicko_results], axis=0,
                                        ignore_index=True)

    # Join the ratings to the original schedule of games
    df = df.merge(results, left_on=['Season','Week','HomeID'],
                            right_on=['Season','Week','TeamID'],
                            suffixes=('','_Home'))
    df.drop('TeamID', 1, inplace=True)

    df = df.merge(results, left_on=['Season','Week','VisID'],
                            right_on=['Season','Week','TeamID'],
                            suffixes=('','_Away'))
    df.drop('TeamID', 1, inplace=True)

    # Create key and set index to join with n_game summaries dataset.
    df.set_index(['HomeID', 'VisID', 'Season', 'Week'], inplace=True)
    df = df[['Glicko_Rating', 'Glicko_Rating_Deviance', 'Glicko_Sigma',
            'Glicko_Rating_Away', 'Glicko_Rating_Deviance_Away',
            'Glicko_Sigma_Away']]
    df.columns = ['Glicko_Rating_Home', 'Glicko_Rating_Deviance_Home',
                'Glicko_Sigma_Home', 'Glicko_Rating_Away',
                'Glicko_Rating_Deviance_Away', 'Glicko_Sigma_Away']

    return df

if __name__=='__main__':

    cur_dir = os.getcwd()
    data_dir = 'data'
    snooz_dir = 'snoozle'
    snooz_file = 'snoozle-combined.csv'
    glicko_dir = 'glicko'
    glicko_file = 'glicko_snoozle_prior.csv'

    file = os.path.join(cur_dir, data_dir, snooz_dir, snooz_file)
    dataframe = pd.read_csv(file)
    glicko_df = glicko2_summarize(dataframe, min_weeks=4, use_prior=True)

    file = os.path.join(cur_dir, data_dir, glicko_dir, glicko_file)
    glicko_df.to_csv(file)
