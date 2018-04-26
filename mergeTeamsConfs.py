import numpy as np
import pandas as pd

ultimate = pd.read_csv('data/ultimate/ultimate_cfb_dataset.csv')

########################################################################################
#MOVE THIS SOMEWHERE ELS
########################################################################################

def featureFixer(prefix, feature, split):
    words = feature.split(split)
    new_name = prefix
    for w in range(1, len(words)):
        new_name = '{}{}'.format(new_name, words[w].capitalize())
    return new_name

new_names = {}

for feat in ultimate:
    if ('home_' in feat)|('h_' in feat):
        new_names[feat] = featureFixer('Home', feat, '_')
    elif ('away_' in feat)|('v_' in feat):
        new_names[feat] = featureFixer('Vis', feat, '_')
      
    
ultimate2 = ultimate.copy()
ultimate2 = ultimate2.rename(new_names, axis=1)

########################################################################################

ultimates = [ult for _,ult in ultimate2.groupby('Season')]

teams_ref = pd.read_csv('data/cfbTeams.csv', encoding='latin-1').fillna('N/A') #reference teams and ids
teams_ref['Team'] = teams_ref.apply(lambda x: x['SportsRefName'] if x['SportsRefAlt']=='N/A' else x['SportsRefAlt'], axis=1)

conferences = pd.read_csv('data/conferences/complete_conf.csv')

########################################################################################

def mergeRefAndConf(ref_teams, conf_teams):
    ref_conf = [(np.nan, np.nan, np.nan) for i in range(conf_teams.shape[0])]
    
    special_cases = pd.DataFrame([('Louisiana','Louisiana-Lafayette'),
                                   ('San Jose State', 'SJSU')], columns=['Conf','Ref'])
    
    for i, teamConf in enumerate(conf_teams): 
        team, conf = teamConf
        ID = ref_teams.loc[(ref_teams['SportsRefName']==team)|(ref_teams['SportsRefAlt']==team)|(ref_teams['Team']==team)| 
                                (ref_teams['DonBestName']==team)|(ref_teams['DonBestAlt']==team)|(ref_teams['DonBestAlt2']==team),'id']
        if ID.shape[0] == 0: #manual fixes
            team2 = special_cases.loc[special_cases['Conf']==team,'Ref']
            if team2.shape[0] == 0:
                ref_conf[i] = (team, 'Missing', 'Missing')
            else:
                team2 = team2.as_matrix()[0]
                ID2 = ref_teams.loc[(ref_teams['SportsRefName']==team2)|(ref_teams['SportsRefAlt']==team2)|(ref_teams['Team']==team2)| 
                                (ref_teams['DonBestName']==team2)|(teams_ref['DonBestAlt']==team2)|(ref_teams['DonBestAlt2']==team2),'id']
                ref_conf[i] = (team2, conf, ID2.as_matrix()[0])
        else:
            ref_conf[i] = (team, conf, ID.as_matrix()[0])

    return pd.DataFrame(ref_conf, columns=['Team','Conf','ID'])


def getConfPerYear(year, ult_teams, ref_teams, conf_teams):
    teams = pd.DataFrame({'ID':np.union1d(ult_teams['HomeID'].unique(), ult_teams['VisID'].unique())})
    ref_conf = mergeRefAndConf(ref_teams, conf_teams)
    teams = pd.merge(teams, ref_conf, on=['ID'], how='left').fillna('NotMajor')
    teams['Year'] = year
    return teams

########################################################################################

new_conf = [0 for x in range(2001,2018)]
for i, year, ult in zip(range(0,(2018-2001)), range(2001,2018), ultimates):
    conf_teams = conferences.loc[conferences['Year']==year,['Team','Conf']].as_matrix()
    new_conf[i] = getConfPerYear(year, ult[['HomeID','VisID']], teams_ref, conf_teams)

conferences2 = pd.concat(new_conf)

conferences2.to_csv('data/conferences/mergedConferences.csv', index=False)