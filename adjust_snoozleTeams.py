import numpy as np
import pandas as pd
from datetime import *

def indices(date, ind):
    return pd.Series([date.year, date.month, date.day], index=ind)


def add_indices(data, ind):
    data[ind] = pd.to_datetime(data[ind]).dt.date
    data[['Year','Month','Day']] = data[ind].apply(lambda x: indices(x, ['Year','Month','Day']))
    data.drop(ind, axis=1, inplace=True)
    return data

#######################################################################################

teams_ref = pd.read_csv('data/cfbTeams.csv', encoding='latin-1').fillna('N/A') #reference teams and ids
teams_ref['Team'] = teams_ref.apply(lambda x: x['SportsRefName'] if x['SportsRefAlt']=='N/A' else x['SportsRefAlt'], axis=1)

snoozleList = [0, 0, 0, 0] #get all team data
for i, year in enumerate(range(2013,2017)):
    snoozleList[i] = pd.read_csv('data/snoozle/odds/odds{}.csv'.format(year))
    
snoozle = pd.concat(snoozleList)
snoozle_teams_m = np.union1d(snoozle['Home'].unique(), snoozle['Visiter'].unique()) #unique snoozle teams

snoozle_fix = pd.DataFrame([ #teams to fix manually
    ('Cal', 'California'),('Cent Michigan', 'Central Michigan'),('E Michigan', 'Eastern Michigan'),
    ('ECU', 'East Carolina'),('FAU', 'Florida Atlantic'),('FIU', 'Florida International'),
    ('FSU', 'Florida State'),('Ga Southern', 'Georgia Southern'),('LA Tech', 'Louisiana Tech'),
    ('LA-Lafayette', 'Louisiana-Lafayette'),('Mid Tennessee', 'Middle Tennessee State'),('Miss St', 'Mississippi State'),
    ('N Illinois', 'Northern Illinois'),('New Mexico St', 'New Mexico State'),('North Carolina St', 'North Carolina State'),
    ('North Dakota St', 'North Dakota State'),('OSU', 'Ohio State'),('San Jose State', 'SJSU'),
    ('UConn', 'Connecticut'),('UMass', 'Massachusetts'),('UNC', 'North Carolina'),
    ('USF', 'South Florida'),('UVA', 'Virginia'),('VT', 'Virginia Tech'),('W Kentucky', 'Western Kentucky'),
    ('W Michigan', 'Western Michigan'),('Washington St', 'Washington State')], columns=['SnoozleTeam','FixedTeam'])

#######################################################################################

snoozle_teams_f = [(np.nan,np.nan,np.nan, np.nan) for i in range(snoozle_teams_m.shape[0])] #final df

for i, team in enumerate(snoozle_teams_m): #iterate through snoozle teams
    team_ID = teams_ref.loc[(teams_ref['SportsRefName']==team)|(teams_ref['SportsRefAlt']==team)|(teams_ref['Team']==team)| #find id
                            (teams_ref['DonBestName']==team)|(teams_ref['DonBestAlt']==team)|(teams_ref['DonBestAlt2']==team)]
    if team_ID.shape[0] == 0: #manual fixes
        
        team_ID_m = snoozle_fix.loc[snoozle_fix['SnoozleTeam']==team]['FixedTeam'].iloc[0]
        team_ID = teams_ref.loc[(teams_ref['SportsRefName']==team_ID_m)|(teams_ref['SportsRefAlt']==team_ID_m)|
                                (teams_ref['Team']==team_ID_m)|(teams_ref['DonBestName']==team_ID_m)|
                                (teams_ref['DonBestAlt']==team_ID_m)|(teams_ref['DonBestAlt2']==team_ID_m)]  
        snoozle_teams_f[i] = (team, team, int(team_ID['id']), int(team_ID['id']))
    else:
        snoozle_teams_f[i] = (team, team, int(team_ID['id']), int(team_ID['id']))

snoozle_teams = pd.DataFrame(snoozle_teams_f, columns=['Home','Visiter','HomeID','VisID'])

#######################################################################################

for year in range(2013,2017):
    odds = pd.read_csv('data/snoozle/odds/odds{}.csv'.format(year))
    odds = add_indices(odds, 'Date')
    odds = pd.merge(odds, snoozle_teams[['Home','HomeID']], on='Home',how='left')
    odds = pd.merge(odds, snoozle_teams[['Visiter','VisID']], on='Visiter',how='left')
    odds = odds.rename({'Spread':'SpreadWag','OverUnder':'OverUnderWag'}, axis=1)
    odds.to_csv('data/snoozle/odds_fixed/odds{}.csv'.format(year), index=False)
    
    stats = pd.read_csv('data/snoozle/stats/stats{}.csv'.format(year))
    stats = add_indices(stats, 'Date')
    stats = pd.merge(stats, snoozle_teams[['Home','HomeID']], on='Home',how='left')
    stats = pd.merge(stats, snoozle_teams[['Visiter','VisID']], on='Visiter',how='left')
    stats.drop(['HomeScore','VisScore'], axis=1, inplace=True)
    stats.to_csv('data/snoozle/stats_fixed/stats{}.csv'.format(year), index=False)
                       
                      