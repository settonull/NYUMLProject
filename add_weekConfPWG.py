import numpy as np
import pandas as pd
from datetime import *

#######################################################################################

def fix_dates(d, wrong): #Fix dates that are one day off
    if d in wrong:
        return d - timedelta(days=1) #return to yesterday
    else:
        return d

def fixWeek(year): #Fix the week column in allscores_201X data
    
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
   
conferences = ['SunBelt','SEC','PAC12','MountainWest','MidAmerican','Independent','ConferenceUSA','Big10','Big12','AAC','ACC'] #FBS confs
teams_info = pd.read_csv('data/cfbTeams.csv', encoding='latin-1').fillna('NotMe')[['id','SportsRefName','SportsRefAlt']] #TeamName and id
teams_info['Team'] = teams_info.apply(lambda x: x['SportsRefName'] if x['SportsRefAlt']=='NotMe' else x['SportsRefAlt'], axis=1) #Fix inconsistincies btwn SportsRefName and SportsRefAlt
teams_info.drop(['SportsRefName','SportsRefAlt'],axis=1,inplace=True) #Drop bad cols

def findConference(teams, conferences, ind): #Find conf for given team
    game = pd.Series(['',''],index=ind) #Array to return
    
    for i, team in enumerate(teams): #Iterate through Home, Vis
        pos = 'Home' if i ==0 else 'Vis'
        conf = conferences.loc[conferences['id']==team,'Conference'] #Find matching conf using ID
        if conf.shape[0] == 0: #If conf not found. notMajor team
            game['{}Conf'.format(pos)] = 'NotD1'
        else:
            game['{}Conf'.format(pos)] = conf.values[0]
    return game

def addConference(year, scores, conferences, teams_info): #Add Conference info for allscores
    teams_list = [0 for x in range(len(conferences))] #unique list of teams

    for i,conf in enumerate(conferences): #connect team and conference for given year
        conf_team = pd.read_csv('data/conferences/{}_{}.csv'.format(conf,year),header=1).iloc[:,0] #conference data
        teams_list[i] = pd.DataFrame({'Conference':[conf for x in range(conf_team.shape[0])],'Team':conf_team}) #Apply conference to teams
    teams = pd.concat(teams_list).reset_index(drop=True) #to dataFrame
    
    teams.loc[teams['Team']=='San Jose State','Team'] = 'SJSU' #Fix special case
    
    teams_conf = pd.merge(teams, teams_info, on='Team',how='left') #Join team and conf data
    
    ind = ['HomeConf','VisConf']
    scores = scores.assign(HomeConf=0,VisConf=0)
    scores[ind] = scores.apply(lambda x: findConference((x['HomeID'],x['VisID']), teams_conf, ind), axis=1) #Add conf to allScores
    scores = scores.join(pd.get_dummies(scores[ind]))
    return scores

#######################################################################################

def ptsWinsGames(teams, ind, scores): #Find pts scores, wins, games up to given week, for teams in a given game
    
    game = pd.Series([0.0 for x in range(len(ind))], index=ind) #pre-allocate array to return
    
    for i, team in enumerate(teams): #repeat for home and vis
        pos = 'Home' if i == 0 else 'Vis'
        prev_game = scores.loc[(scores['HomeID'] == team) | (scores['VisID'] == team)].tail(1) #find last appearance
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
        
    return game

def add_ptsWinsGames(scores): #initialize df for year

    scores = scores.assign(HomePtsF=0,HomePtsA=0,HomeWins=0,HomeGames=0,HomeWinPct=0,VisPtsF=0,VisPtsA=0,VisWins=0,VisGames=0,VisWinPct=0) #init features in df
    ind = ['HomePtsF','HomePtsA', 'HomeWins','HomeGames','HomeWinPct','VisPtsF','VisPtsA', 'VisWins', 'VisGames','VisWinPct'] #features to add
    scores['Spread'] = scores['HomeFinal'] - scores['VisFinal']  #add Spread
    scores_list = [group for _, group in scores.groupby('Week')] #split df by week
    
    for i, score in enumerate(scores_list[1:]): #iterate through weeks, ignore Week1
        i += 1
        prev_weeks = pd.concat(scores_list[:i]) #get df of prev weeks
        scores_list[i][ind] = scores_list[i].apply(lambda x: ptsWinsGames((x['HomeID'],x['VisID']),ind,prev_weeks), axis = 1) #get pts,wins,games for every game in week 
    scores = pd.concat(scores_list) #rebuild df
    
    return scores

#######################################################################################

for d in [2013, 2014, 2015, 2016]:
    print('Starting Season: {}'.format(d))
    scores = fixWeek(d)
    scores = add_ptsWinsGames(scores)
    scores = addConference(d, scores, conferences, teams_info)
    scores.to_csv('data/scores_wc/scores_wcp{}.csv'.format(d), index=False)
    print('Finishing Season: {}'.format(d))