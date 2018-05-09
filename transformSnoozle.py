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

def lookup_conf(team_id, team_name, year):
    if (team_id in conference_lookup):
        team_data = conference_lookup[team_id]
        if year in team_data:
            return team_data[year]

    print("No conference for", team_name, 'id:', team_id, 'year:', year)

    if not (team_id) == -1:
        if team_id not in conference_lookup:
            conference_lookup[team_id] = {}
        conference_lookup[team_id][year] = "NON-D1"

    return "NON-D1"

outfile = open('data/snoozle/snoozle-combined.csv', 'w', newline='')
owriter = csv.writer(outfile, delimiter=',')

owriter.writerow(['Date','Vis_Team_Name','v_rushing_yards','v_rushing_attempts','v_passing_yards','v_passing_attempts','v_passing_completions','v_penalties','v_penalty_yards','v_fumbles_lost','v_interceptions_thrown','v_1st_Downs','v_3rd_Down_Attempts','v_3rd_Down_Conversions','v_4th_Down_Attempts','v_4th_Down_conversions','v_Time_of_Possession','VisFinal','Home_Team_Name','h_rushing_yards','h_rushing_attempts','h_passing_yards','h_passing_attempts','h_passing_completions','h_penalties','h_penalty_yards','h_fumbles_lost','h_interceptions_thrown','h_1st_Downs','h_3rd_Down_Attempts','h_3rd_Down_Conversions','h_4th_Down_Attempts','h_4th_Down_conversions','h_Time_of_Possession','HomeFinal', 'Season', 'Year', 'Month','Day', 'Week', 'VisID', 'HomeID', 'VisConf', 'HomeConf', 'conference_game', 'HomeOdds'])

for season in range(2001,2018):

    #for Week calculations
    week = 1
    lastThursday = None

    #load an odds lookup table, create lists since some team pairs play twice
    odds_lookup = {}
    with open('data/snoozle/odds/odds-{}.csv'.format(season), 'r') as ofile:
        oddsr = csv.reader(ofile, delimiter=',')
        next(oddsr)  # skip the header
        for row in oddsr:
            key = adjustName(row[1]) + '-' + adjustName(row[2])
            if key in odds_lookup:
                odds_lookup[key].append(row[3])
            else:
                odds_lookup[key] = [row[3]]


    #for removing duplicate entries in snoozle data
    seen_list =[]
    
    #hack to handle 2 week start
    if season >= 2016:
        week = 0
        
    with open('data/snoozle/stats/snoozle-{}.csv'.format(season), 'r') as ofile:
        statsr = csv.reader(ofile, delimiter=',')
        next(statsr)  # skip the header
        for row in statsr:
            key = row[1] + '-' + row[2] + '-' + row[17] + '-'+ row[18] + '-' + row[19] + '-' + row[34]
            if key in seen_list:
                continue
            seen_list.append(key)

            row[1] = adjustName(row[1])
            row[18] = adjustName(row[18])
            vis_id = lookup_id(row[1])
            home_id = lookup_id(row[18])


            date = fix_date(row[0])
            dt = datetime.date(year = int(date[0]), month= int(date[1]), day = int(date[2]))
            if not lastThursday: #first game of a season
                lastThursday = dt + timedelta(days = (1 - dt.weekday())) #1 = tuesday
            else:
                if dt >= (lastThursday + timedelta(days=7)):
                    week = week +1
                    lastThursday = lastThursday +  timedelta(days=7)
            row.append(season)
            row = row + date
            row.append(week if week != 0 else 1)
            row.append(vis_id)
            row.append(home_id)

            vis_conf = lookup_conf(vis_id, row[1], str(season))
            home_conf = lookup_conf(home_id, row[18], str(season))
            isInConf = (vis_conf != 'NON-D1') | (home_conf != 'NON-D1')
            row.append(vis_conf)
            row.append(home_conf)
            row.append(int(isInConf))
            key = row[1] + '-' + row[18]
            odds = 0
            if key in odds_lookup:
                if len(odds_lookup[key]) == 0:
                    print("odd, found empty odds", key)
                else:
                    odds = odds_lookup[key].pop(0)
            row.append(odds)

            owriter.writerow(row)

print("done")
