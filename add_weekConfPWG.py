import numpy as np
import pandas as pd
from datetime import *

#######################################################################################

def fix_dates(d, wrong): #Fix dates that are one day off
    if d in wrong:
        return d - timedelta(days=1) #return to yesterday
    else:
        return d

def fix_week(year): #Fix the week column in allscores_201X data
    
    sched_path = 'data/schedules/schedule{}.csv'.format(year) #sched path
    sched_data = pd.read_csv(sched_path) #get schedule data
    sched_data['WeekDate'] = pd.to_datetime(sched_data['Date']).dt.date #Get week from WeekDate col
    sched = sched_data[['Week','WeekDate']].drop_duplicates() #Keep 1 entry for Week and date
    
    scores_path = 'data/scores_wrong/NCAAAllScores{}.csv'.format(year) #allscores path
    scores_data = pd.read_csv(scores_path) #allscores
    scores_data['Start'] = pd.to_datetime(scores_data['Start']) #convert to date-time
    scores_data['WrongDate'] = scores_data['Start'].dt.date #Put date in own col to work with
    scores_data.drop(['Week'], axis=1, inplace=True) #Remove bad Week col
    scores_data.drop(scores_data.loc[scores_data['WrongDate'] > (sched_data['WeekDate'].max() + timedelta(days=1))].index, axis=0, inplace=True) #Remove all-star games and any other games after final according to sched data
       
    wrong_dates = np.setdiff1d(scores_data['WrongDate'].drop_duplicates().tolist(),sched['WeekDate'].drop_duplicates().tolist())#Find wrong dates
    
    scores_data['WeekDate'] = scores_data['WrongDate'].apply(lambda x: fix_dates(x, wrong_dates)) #Fix wrong dates
    scores_data.drop('WrongDate', axis=1, inplace=True) #Drop prev Date col
    scores_clean = pd.merge(scores_data, sched[['WeekDate','Week']], on='WeekDate',how='left') #Merge df's to get proper Week
    
    return scores_clean

#######################################################################################

def ptsWinsGames(teams, ind, prev_weeks): #Find pts scores, wins, games up to given week, for teams in a given game
    
    game = pd.Series([0.0 for x in range(len(ind))], index=ind) #pre-allocate array to return
    
    for i, team in enumerate(teams): #repeat for home and vis
        pos = 'Home' if i == 0 else 'Vis'
        prev_game = prev_weeks.loc[(prev_weeks['HomeID'] == team) | (prev_weeks['VisID'] == team)].tail(1) #find last appearance
        if prev_game.shape[0] == 0: #skip if team has not yet played
            continue
        elif int(prev_game['HomeID']) == team: #if prev appearance was home game
            prev_pos, prev_opp = 'Home','Vis' 
        else: #if prev appearance was vis game
            prev_pos, prev_opp = 'Vis', 'Home' 
            
        game['{}PtsF'.format(pos)] = int(prev_game['{}PtsF'.format(prev_pos)]) + int(prev_game['{}Final'.format(prev_pos)]) #add prev scores pts to prev game performance
        game['{}PtsA'.format(pos)] = int(prev_game['{}PtsA'.format(prev_pos)]) + int(prev_game['{}Final'.format(prev_opp)]) #add prev scores pts to prev game performance
        
        if prev_pos == 'Home': #add wins
            game['{}Wins'.format(pos)] = int(prev_game['{}Wins'.format(prev_pos)]) + 1 if int(prev_game['Spread']) > 0 else int(prev_game['{}Wins'.format(prev_pos)])
        else:
            game['{}Wins'.format(pos)] = int(prev_game['{}Wins'.format(prev_pos)]) + 1 if int(prev_game['Spread']) < 0 else int(prev_game['{}Wins'.format(prev_pos)])
        
        game['{}Games'.format(pos)] = int(prev_game['{}Games'.format(prev_pos)]) + 1 #add 1 to prev games played
        game['{}WinPct'.format(pos)] = int(game['{}Wins'.format(pos)])/int(game['{}Games'.format(pos)]) #get new win pct
        
        game['{}PtsFPG'.format(pos)] = game['{}PtsF'.format(pos)]/game['{}Games'.format(pos)]
        game['{}PtsAPG'.format(pos)] = game['{}PtsA'.format(pos)]/game['{}Games'.format(pos)]
    
    game['HomeDiffOD'] = game['HomePtsFPG'] - game['VisPtsAPG']
    game['VisDiffOD'] = game['VisPtsFPG'] - game['HomePtsAPG']
    game['DiffPtsFPG'] = game['HomePtsFPG'] - game['VisPtsFPG']
    game['DiffPtsAPG'] = game['HomePtsAPG'] - game['VisPtsAPG']
    
    return game

def perGame(teams, ind, prev_weeks):
    
    game = pd.Series([0.0 for x in range(len(ind))], index=ind) #pre-allocate array to return
    
    for i, team in enumerate(teams): #repeat for home and vis
        pos = 'Home' if i == 0 else 'Vis'
        prev_games = prev_weeks.loc[(prev_weeks['HomeID'] == team)|(prev_weeks['VisID'] == team)]
        games = prev_games.shape[0]
        if games == 0:
            continue
            
        ry = int(prev_games.loc[prev_games['HomeID']==team,'HomeRushY'].sum()+prev_games.loc[prev_games['VisID']==team,'VisRushY'].sum())
        ra = int(prev_games.loc[prev_games['HomeID']==team,'HomeRushA'].sum()+prev_games.loc[prev_games['VisID']==team,'VisRushA'].sum())
        game['{}RushYPG'.format(pos)] = ry/games
        game['{}RushAPG'.format(pos)] = ra/games
        game['{}RushYPA'.format(pos)] = ry/ra if ra != 0 else 0.0
        
        py = int(prev_games.loc[prev_games['HomeID']==team,'HomePassY'].sum()+prev_games.loc[prev_games['VisID']==team,'VisPassY'].sum())
        pa = int(prev_games.loc[prev_games['HomeID']==team,'HomePassY'].sum()+prev_games.loc[prev_games['VisID']==team,'VisPassY'].sum())
        pc = int(prev_games.loc[prev_games['HomeID']==team,'HomePassC'].sum()+prev_games.loc[prev_games['VisID']==team,'VisPassC'].sum())
        game['{}PassYPG'.format(pos)] =  py/games
        game['{}PassAPG'.format(pos)] = pa/games
        game['{}PassYPA'.format(pos)] = py/pa if pa != 0 else 0.0
        game['{}PassYPC'.format(pos)] = py/pc if pc != 0 else 0.0
        
        game['{}PnlPG'.format(pos)] = int(prev_games.loc[prev_games['HomeID']==team,'HomePnl'].sum()+prev_games.loc[prev_games['VisID']==team,'VisPnl'].sum())/games
        game['{}PnlYPG'.format(pos)] = int(prev_games.loc[prev_games['HomeID']==team,'HomePnlY'].sum()+prev_games.loc[prev_games['VisID']==team,'VisPnlY'].sum())/games
        game['{}FmbPG'.format(pos)] = int(prev_games.loc[prev_games['HomeID']==team,'HomeFmb'].sum()+prev_games.loc[prev_games['VisID']==team,'VisFmb'].sum())/games
        game['{}IntPG'.format(pos)] = int(prev_games.loc[prev_games['HomeID']==team,'HomeInt'].sum()+prev_games.loc[prev_games['VisID']==team,'VisInt'].sum())/games
    
    return game

def add_ptsWinsGames(scores): #initialize df for year

    scores = scores.assign(HomePtsF=0.0, HomePtsA=0.0, HomeWins=0.0, HomeGames=0.0, HomeWinPct=0.0, HomePtsFPG=0.0, HomePtsAPG=0.0,
                           VisPtsF=0.0, VisPtsA=0.0, VisWins=0.0, VisGames=0.0, VisWinPct=0.0, VisPtsFPG=0.0, VisPtsAPG=0.0,
                           HomeDiffOD=0.0, VisDiffOD=0.0, DiffPtsFPG=0.0, DiffPtsPAG=0.0,
                           HomeRushYPG=0.0, HomeRushAPG=0.0, HomeRushYPA=0.0,
                           HomePassYPG=0.0, HomePassAPG=0.0, HomePassYPA=0.0, HomePassYPC=0.0,
                           HomePnlPG=0.0, HomePnlYPG=0.0, HomeFmbPG=0.0, HomeIntPG=0.0,
                           VisRushYPG=0.0, VisRushAPG=0.0, VisRushYPA=0.0,
                           VisPassYPG=0.0, VisPassAPG=0.0, VisPassYPA=0.0, VisPassYPC=0.0,
                           VisPnlPG=0, VisPnlYPG=0.0, VisFmbPG=0.0, VisIntPG=0.0) #init features in df
    pwg_ind = ['HomePtsF','HomePtsA','HomeWins','HomeGames','HomeWinPct','HomePtsFPG','HomePtsAPG','HomeDiffOD',
               'VisPtsF','VisPtsA','VisWins','VisGames','VisWinPct','VisPtsFPG','VisPtsAPG','VisDiffOD',
               'DiffPtsFPG','DiffPtsAPG'] 
    pg_ind = ['HomeRushYPG', 'HomeRushAPG', 'HomeRushYPA',
              'HomePassYPG', 'HomePassAPG', 'HomePassYPA', 'HomePassYPC',
              'HomePnlPG', 'HomePnlYPG', 'HomeFmbPG', 'HomeIntPG',
              'VisRushYPG', 'VisRushAPG', 'VisRushYPA',
              'VisPassYPG', 'VisPassAPG', 'VisPassYPA', 'VisPassYPC',
              'VisPnlPG', 'VisPnlYPG','VisFmbPG', 'VisIntPG'] 
    
    scores['Spread'] = scores['HomeFinal'] - scores['VisFinal']  #add Spread
    scores_list = [group for _, group in scores.groupby('Week')] #split df by week
    
    for i, score in enumerate(scores_list[1:]): #iterate through weeks, ignore Week1
        i += 1
        prev_weeks = pd.concat(scores_list[:i]) #get df of prev weeks
        scores_list[i][pwg_ind] = scores_list[i].apply(lambda x: ptsWinsGames((x['HomeID'],x['VisID']),pwg_ind,prev_weeks), axis = 1)
        scores_list[i][pg_ind] = scores_list[i].apply(lambda x: perGame((x['HomeID'],x['VisID']),pg_ind, prev_weeks), axis = 1) 
    scores = pd.concat(scores_list) #rebuild df
    
    return scores

#######################################################################################

def indices(date):
    return pd.Series([date.year, date.month, date.day], index=['Year','Month','Day'])


def add_indices(data, ind):
    data[ind] = pd.to_datetime(data[ind]).dt.date
    data[['Year','Month','Day']] = data[ind].apply(lambda x: indices(x))
    data.drop(ind, axis=1, inplace=True)
    return data

#######################################################################################

def merge_snoozle(scores, year):
    snoozle_odds = pd.read_csv('data/snoozle/odds_fixed/odds{}.csv'.format(year))
    snoozle_stats = pd.read_csv('data/snoozle/stats_fixed/stats{}.csv'.format(year))
    
    scores = pd.merge(scores, snoozle_odds.drop(['Home','Visiter'], axis=1), on=['HomeID', 'VisID', 'Month', 'Day', 'Year'], how='left')
    scores = pd.merge(scores, snoozle_stats.drop(['Home','Visiter'], axis=1), on=['HomeID', 'VisID', 'Month', 'Day', 'Year'], how='left')
    return scores

#######################################################################################

def merge_odds(scores):
    odds = pd.read_csv('data/oddsAdjusted.csv')
    
    scores = pd.merge(scores, odds, on=['Year','Month','Day','Home','Visiter'], how='left')
    return scores

#######################################################################################

for year in [2013, 2014, 2015, 2016]:
    print('Starting Season: {}'.format(year))
    scores = fix_week(year)
    scores = add_indices(scores, 'Start')
    scores = add_conference(year, scores, conferences, teams_info)
    scores = merge_snoozle(scores, year)
    scores = merge_odds(scores)
    scores = add_ptsWinsGames(scores)
    scores.to_csv('data/scores_wcp/scores_wcp{}.csv'.format(year), index=False)
    print('Finishing Season: {}'.format(year))