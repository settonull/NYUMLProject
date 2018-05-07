import numpy as np
import pandas as pd
from datetime import *
import csv
import unidecode
import datetime

team_lookup = {}

def add_lookup(name, id_num):
    if name == '': return
    name = unidecode.unidecode(name)
    if name not in team_lookup:
        team_lookup[name] = id_num
    #else:
        #if id_num != team_lookup[name]:
        #    print("Collision:", name, id_num, team_lookup[name])

def adjustName(team_name):
    team_name = team_name.strip()
    if team_name in snoozle_mappings:
        team_name = snoozle_mappings[team_name]
    return team_name

def lookup_id(team_name):
    if (team_name not in team_lookup):
        print("can't find team:", team_name)
        return -1
    else:
        return team_lookup[team_name]

def fix_date(strDate):
    if '/' in strDate:
        dmy = strDate.split('/')
        return [dmy[2], dmy[1], dmy[0]]
    if '-' in strDate:
        return strDate.split('-')

    return ['UNKN', 'UNKN', 'UNKN']

#id,abbrev,location,name,SportsRefName,SportsRefAlt,DonBestName,DonBestAlt,DonBestAlt2
with open('data/cfbTEAMS.csv', 'r', encoding='latin-1') as ofile:
    teamr = csv.reader(ofile, delimiter=',')
    next(teamr) #skip the header
    for row in teamr:
        team_id = row[0]
        add_lookup(row[1], team_id)
        add_lookup(row[3], team_id )
        add_lookup(row[4], team_id )
        add_lookup(row[5], team_id )
        add_lookup(row[6], team_id )
        add_lookup(row[7], team_id )
        add_lookup(row[8], team_id )


snoozle_mappings = {
'Abil Christian':'Abilene Christian',
'Alabama St.' : 'Alabama State',
'Albany (N.Y.)' : 'Albany',
'AR-Pine Bluff' : 'Arkansas-Pine Bluff',
'Cal':'California',
'Cent Arkansas':'Central Arkansas',
'Cent Michigan':'Central Michigan',
'Central Conn. St.':'Central Connecticut',
'Charleston So':'Charleston Southern',
'East Tennessee St':'East Tennessee St.',
'E Illinois':'Eastern Illinois',
'E Kentucky':'Eastern Kentucky',
'E Michigan':'Eastern Michigan',
'E Washington':'Eastern Washington',
'Ga Southern':'Georgia Southern',
'Grambling St':'Grambling State',
'LA-Lafayette':'Louisiana-Lafayette',
'LA Tech':'Louisiana Tech',
'McNeese':'McNeese State',
'Mid Tennessee':'Middle Tennessee State',
'Miss St':'Mississippi State',
'N Arizona':'Northern Arizona',
'N Colorado':'Northern Colorado',
'New Mexico St':'New Mexico State',
'Nicholls':'Nicholls State',
'N Illinois':'Northern Illinois',
'No Carolina A+T':'North Carolina A&T',
'North Carolina St':'North Carolina State',
'North Dakota St':'North Dakota State',
'Northwestern St':'Northwestern State',
'Sam Houston':'Sam Houston State',
'S Carolina St':'South Carolina State',
'SF Austin':'Stephen F. Austin',
'S Illinois':'Southern Illinois',
'UConn':'Connecticut',
'UMass':'Massachusetts',
'UT Martin':'Tennessee Martin',
'Washington St':'Washington State',
'W Carolina':'Western Carolina',
'W Illinois':'Western Illinois',
'W Kentucky':'Western Kentucky',
'W Michigan':'Western Michigan'}

conference_lookup = {}

#load converences
with open('data/conferences/complete_conf.csv', 'r') as ofile:
    confr = csv.reader(ofile, delimiter=',')
    next(confr) #skip the header
    for row in confr:
        team_id = lookup_id(row[0])
        if not (team_id) == -1:
            if team_id not in conference_lookup:
                conference_lookup[team_id] = {}
            conference_lookup[team_id][row[1]] = row[2]
        else:
            print("coulding find id for", row[0])

def lookup_conf(team_id, year):
    if (team_id in conference_lookup):
        team_data = conference_lookup[team_id]
        if year in team_data:
            return team_data[year]

    print("Didn't find", team_id, year)
    return "NON-D1"

outfile = open('data/snoozle/snoozle-combined.csv', 'w', newline='')
owriter = csv.writer(outfile, delimiter=',')

owriter.writerow(['Date','Vis_Team_Name','v_rushing_yards','v_rushing_attempts','v_passing_yards','v_passing_attempts','v_passing_completions','v_penalties','v_penalty_yards','v_fumbles_lost','v_interceptions_thrown','v_1st_Downs','v_3rd_Down_Attempts','v_3rd_Down_Conversions','v_4th_Down_Attempts','v_4th_Down_conversions','v_Time_of_Possession','VisFinal','Home_Team_Name','h_rushing_yards','h_rushing_attempts','h_passing_yards','h_passing_attempts','h_passing_completions','h_penalties','h_penalty_yards','h_fumbles_lost','h_interceptions_thrown','h_1st_Downs','h_3rd_Down_Attempts','h_3rd_Down_Conversions','h_4th_Down_Attempts','h_4th_Down_conversions','h_Time_of_Possession','HomeFinal', 'Season', 'Year', 'Month','Day', 'Week', 'VisID', 'HomeID', 'VisConf', 'HomeConf', 'conference_game'])

for season in range(2001,2018):

    week = 1
    lastThursday = None

    with open('data/snoozle/stats/snoozle-{}.csv'.format(season), 'r') as ofile:
        statsr = csv.reader(ofile, delimiter=',')
        next(statsr)  # skip the header
        for row in statsr:
            row[1] = adjustName(row[1])
            row[18] = adjustName(row[18])
            vis_id = lookup_id(row[1])
            home_id = lookup_id(row[18])

            date = fix_date(row[0])
            dt = datetime.date(year = int(date[0]), month= int(date[1]), day = int(date[2]))
            if not lastThursday: #first game of a season
                lastThursday = dt + timedelta(days = (3 - dt.weekday())) #3 = thrusday
            else:
                if dt >= (lastThursday + timedelta(days=7)):
                    week = week +1
                    lastThursday = lastThursday +  timedelta(days=7)
            row.append(season)
            row = row + date
            row.append(week)
            row.append(vis_id)
            row.append(home_id)

            vis_conf = lookup_conf(vis_id, str(season))
            home_conf = lookup_conf(home_id, str(season))
            isInConf = (vis_conf != 'NON-D1') | (home_conf != 'NON-D1')
            row.append(vis_conf)
            row.append(home_conf)
            row.append(int(isInConf))

            owriter.writerow(row)

print("done")
exit(0)








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

snoozle_teams = pd.DataFrame(snoozle_teams_f, columns=['Home Team',' Vis Team','HomeID','VisID'])



#######################################################################################

for year in range(2001,2002):
    '''
    odds = pd.read_csv('data/snoozle/odds/odds{}.csv'.format(year))
    odds = add_indices(odds, 'Date')
    odds = pd.merge(odds, snoozle_teams[['Home Team','HomeID']], on='Home Team',how='left')
    odds = pd.merge(odds, snoozle_teams[['Visiter','VisID']], on='Visiter',how='left')
    odds = odds.rename({'Spread':'SpreadWag','OverUnder':'OverUnderWag'}, axis=1)
    odds.to_csv('data/snoozle/odds_fixed/odds{}.csv'.format(year), index=False)
    '''

    stats = pd.read_csv('data/snoozle/stats/snoozle-{}.csv'.format(year))
    stats = add_indices(stats, 'Date')
    stats = pd.merge(stats, snoozle_teams[['Home Team','HomeID']], on='Home Team',how='left')
    stats = pd.merge(stats, snoozle_teams[[' Vis Team','VisID']], on=' Vis Team',how='left')
    stats.to_csv('data/snoozle/stats_fixed/stats{}.csv'.format(year), index=False)
                       
                      
