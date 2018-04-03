import numpy as np
import pandas as pd
from datetime import *

#######################################################################################

def fix_dates(d, wrong):
    if d in wrong:
        return d - timedelta(days=1)
    else:
        return d

def fixWeek(year):
    
    sched_path = 'data/schedules/schedule{}.csv'.format(year)
    sched_data = pd.read_csv(sched_path)
    sched_data['WeekDate'] = pd.to_datetime(sched_data['Date']).dt.date
    sched = sched_data[['Week','WeekDate']].drop_duplicates()
    
    scores_path = 'data/scores_wrong/NCAAAllScores{}.csv'.format(year)
    scores_data = pd.read_csv(scores_path)
    scores_data['Start'] = pd.to_datetime(scores_data['Start'])
    scores_data['WrongDate'] = scores_data['Start'].dt.date
    scores_data.drop(['Week'], axis=1, inplace=True)
    scores_data.drop(scores_data.loc[scores_data['WrongDate'] > (sched_data['WeekDate'].max() + timedelta(days=1))].index, axis=0, inplace=True)
    scores_data['Spread'] = scores_data['HomeFinal'] - scores_data['VisFinal']   
    
    wrong_dates = np.setdiff1d(scores_data['WrongDate'].drop_duplicates().tolist(),sched['WeekDate'].drop_duplicates().tolist())
    
    scores_data['WeekDate'] = scores_data['WrongDate'].apply(lambda x: fix_dates(x, wrong_dates))
    scores_data.drop('WrongDate', axis=1, inplace=True)
    scores_clean = pd.merge(scores_data, sched[['WeekDate','Week']], on='WeekDate',how='left')
    return scores_clean
    #scores_clean.to_csv('data/scores/scores{}.csv'.format(year), index=False)
    
####################################################################################### 
   
conferences = ['SunBelt','SEC','PAC12','MountainWest','MidAmerican','Independent','ConferenceUSA','Big10','Big12','AAC','ACC']
teams_info = pd.read_csv('data/cfbTeams.csv', encoding='latin-1').fillna('NotMe')[['id','SportsRefName','SportsRefAlt']]
teams_info['Team'] = teams_info.apply(lambda x: x['SportsRefName'] if x['SportsRefAlt']=='NotMe' else x['SportsRefAlt'], axis=1)
teams_info.drop(['SportsRefName','SportsRefAlt'],axis=1,inplace=True)

def findConference(teams, conferences, ind):
    game = pd.Series(['',''],index=ind)
    
    for i, team in enumerate(teams):
        pos = 'Home' if i ==0 else 'Vis'
        conf = conferences.loc[conferences['id']==team,'Conference']
        if conf.shape[0] == 0:
            game['{}Conf'.format(pos)] = 'NotMajor'
        else:
            game['{}Conf'.format(pos)] = conf.values[0]
    return game

def addConference(year, scores, conferences, teams_info):
    teams_list = [0 for x in range(len(conferences))]

    for i,conf in enumerate(conferences):
        conf_team = pd.read_csv('data/conferences/{}_{}.csv'.format(conf,year),header=1).iloc[:,0]
        teams_list[i] = pd.DataFrame({'Conference':[conf for x in range(conf_team.shape[0])],'Team':conf_team})
    teams = pd.concat(teams_list).reset_index(drop=True)
    
    teams.loc[teams['Team']=='San Jose State','Team'] = 'SJSU'
    teams_conf = pd.merge(teams, teams_info, on='Team',how='left')
    
    ind = ['HomeConf','VisConf']
    scores = scores.assign(HomeConf=0,VisConf=0)
    scores[ind] = scores.apply(lambda x: findConference((x['HomeID'],x['VisID']), teams_conf, ind), axis=1)
    return scores

for d in [2013, 2014, 2015, 2016]:
    print('Starting Season: {}'.format(d))
    scores = fixWeek(d)
    scores = addConference(d, scores, conferences, teams_info)
    scores.to_csv('data/scores/scores{}.csv'.format(d), index=False)
    print('Finishing Season: {}'.format(d))