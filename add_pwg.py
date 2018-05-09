import numpy as np
import pandas as pd

def get_Teams(scores): #list of teams and conferences for given year
    teams = np.union1d(scores['HomeID'].unique(), scores['VisID'].unique()) #teams
    stats = pd.DataFrame({'Team':teams}) 
    
    conf_year = []
    for pos in ['Home','Vis']: #add conferences
        if '{}Conf_NotMajor'.format(pos) in scores.columns:
            temp = scores[['{}ID'.format(pos),'{}Conf_NotMajor'.format(pos)]]
            temp = temp.rename(index=str, columns = {'{}ID'.format(pos):'Team'})
            temp.drop_duplicates(inplace=True)
            stats = pd.merge(stats, temp, on='Team', how='left')
            conf_year.append(pos)
            
    if len(conf_year) == 2:
        stats['Conf'] = stats['HomeConf_NotMajor'].fillna(stats['VisConf_NotMajor']) #merge conf features
        stats.drop(['VisConf_NotMajor','HomeConf_NotMajor'], axis=1, inplace=True) #drop old conf features
    elif 'Home' in conf_year:
        stats['Conf'] = stats['HomeConf_NotMajor']
        stats.drop(['HomeConf_NotMajor'], axis=1, inplace=True)
    elif 'Vis' in conf_year:
        stats['Conf'] = stats['VisConf_NotMajor']
        stats.drop(['VisConf_NotMajor'], axis=1, inplace=True)
    else:
        stats['Conf'] = 0
        
    stats.set_index('Team', inplace=True) #set team as index
    
    return stats

#######################################################################################

def endYear_Pyth(team, ind, final_game, exp):
    stats = pd.Series([0.0 for x in range(len(ind))], index=ind) #pre-allocate array to return
    
    final_pos = 'Home' if int(final_game['HomeID']) == team else 'Vis' #find final pos
    final_opp = 'Vis' if final_pos == 'Home' else 'Home'
    
    stats['PtsF'] = int(final_game['{}PtsF'.format(final_pos)]) + int(final_game['{}Final'.format(final_pos)]) #final pts scores
    stats['PtsA'] = int(final_game['{}PtsA'.format(final_pos)]) + int(final_game['{}Final'.format(final_opp)]) #final pts let-up
    
    stats['Games'] = final_game['{}Games'.format(final_pos)] + 1 #final games played
    if final_pos == 'Home': #final wins
            stats['Wins'] = int(final_game['{}Wins'.format(final_pos)]) + 1 if int(final_game['Spread']) > 0 else int(final_game['{}Wins'.format(final_pos)])
    else:
            stats['Wins'] = int(final_game['{}Wins'.format(final_pos)]) + 1 if int(final_game['Spread']) < 0 else int(final_game['{}Wins'.format(final_pos)])
    stats['WinPct'] = int(stats['Wins'])/int(stats['Games']) #final win pct
       
    stats['PtsFPG'] = stats['PtsF']/stats['Games']
    stats['PtsAPG'] = stats['PtsA']/stats['Games']
        
    stats['PythPct'] =  int(stats['PtsF'])**exp/(int(stats['PtsF'])**exp + int(stats['PtsA'])**exp) #final pyth pct
    stats['PythWins'] = float(stats['PythPct']) * int(stats['Games']) #final pyth wins
    pyth_win_diff = int(stats['Wins']) - int(stats['PythWins']) #final diff in wins
            
    if pyth_win_diff > 1: #final luck
        stats['Luck'] = 1
    elif pyth_win_diff < -1:
        stats['Luck'] = -1
    else:
        stats['Luck'] = 0
        
    return stats

def getFinal_pyth(scores, exp): #wrapper for getting final stats
    ind = ['Games', 'Wins', 
           'WinPct','PythPct', 'PythWins', 'Luck',
           'PtsF', 'PtsA','PtsFPG', 'PtsAPG']
    
    stats = get_Teams(scores) #get teams and conferences
    stats = stats.assign(Games=0, Wins=0, WinPct=0, PtsF=0, PtsA=0, PythPct=0, PythWins=0, Luck=0) #allocate new features
    
    stats[ind] = stats.index.to_series().apply(lambda x: endYear_Pyth(x, ind, scores.loc[(scores['HomeID'] == x) | (scores['VisID'] == x)].tail(1), exp)) #fill features
    return stats

def prevLuck(teams, ind, stats): #final luck for teams in given matchup
    teams_luck = pd.Series([0.0 for x in range(len(ind))], index=ind) #pre-allocate array
    
    for i, team in enumerate(teams): #iterate through home and vis
        pos = 'Home' if i == 0 else 'Vis' #get pos
        if team not in stats.index: #if team didn't play, they have neutral luck
            teams_luck['{}PrevLuck'.format(pos)] = 0
        else:
            teams_luck['{}PrevLuck'.format(pos)] = float(stats.loc[team, 'Luck']) #get luck from final stats
    return teams_luck

#######################################################################################

def pythWins(teams, ind, exp):
    game = pd.Series([0.0 for x in range(len(ind))], index=ind)
    
    for i, team in enumerate(teams):
        pos = 'Home' if i == 0 else 'Vis'
        if (team[0] != 0) & (team[1] != 0):
            game['{}PythPct'.format(pos)] =  (team[0]**exp)/(team[0]**exp + team[1]**exp) #pyth pct
            game['{}PythWins'.format(pos)] = game['{}PythPct'.format(pos)] * team[2] #pyth wins
            pyth_win_diff = team[3] - game['{}PythWins'.format(pos)] #diff in wins
            
            if pyth_win_diff > 1:
                game['{}Luck'.format(pos)] = 1
            elif pyth_win_diff < -1:
                game['{}Luck'.format(pos)] = -1
            else:
                game['{}Luck'.format(pos)] = 0
    return game

def add_pythWins(scores, exp, prev_scores=None):
    
    ind = ['HomePythPct','HomePythWins','HomeLuck','VisPythPct','VisPythWins','VisLuck'] #features to add
    scores = scores.assign(HomePythPct=0,HomePythWins=0,HomeLuck=0,HomePrevLuck=0,VisPythPct=0,VisPythWins=0,VisLuck=0,VisPrevLuck=0)
    
    if prev_scores is not None: #prev luck
        prev_stats = getFinal_pyth(prev_scores, exp) #get 
        scores[['HomePrevLuck','VisPrevLuck']] = scores.apply(lambda x: prevLuck((x['HomeID'],x['VisID']),['HomePrevLuck','VisPrevLuck'],prev_stats), axis=1) #update prev-luck
    
    scores[ind] = scores.apply(lambda x: pythWins([(x['HomePtsF'],x['HomePtsA'],x['HomeGames'],x['HomeWins']),(x['VisPtsF'],x['VisPtsA'],x['VisGames'],x['VisWins'])],ind,exp),axis=1)
    
    return scores

#######################################################################################

def endYear_Elo(team, ind, final_game, k): 
    update = pd.Series([0], index=ind) #pre-allocate array

    elo_diff = final_game['HomeElo'] - final_game['VisElo'] #diff in Elo ratings

    if int(final_game['HomeID']) == team: #get pos, result and prob of winning
        final_pos = 'Home'
        result = 1 if int(final_game['Spread']) > 0 else 0
        elo_prob = 1/(10**(-elo_diff/400)+1)
    else:
        final_pos = 'Vis'   
        result = 1 if int(final_game['Spread']) < 0 else 0
        elo_prob = 1/(10**(elo_diff/400)+1)

    pos_elo = int(final_game['{}Elo'.format(final_pos)]) #get final elo
    elo_w = final_game['HomeElo'] if int(final_game['Spread']) > 0 else final_game['VisElo'] #find winner and loser
    elo_l = final_game['HomeElo'] if int(final_game['Spread']) < 0 else final_game['VisElo']
    mv = np.log(abs(final_game['Spread']) + 1) * (2.2/((elo_w-elo_l)*0.001 + 2.2)) * k #cacl margin of victory
    
    update['Elo'] = pos_elo + mv * (result - elo_prob) #adjust elo
    
    return update

def getFinal_elo(scores, k): #wrapper for end of year Elo
    ind = ['Elo'] 
    
    stats = get_Teams(scores) #df of teams who played in given year
    stats = stats.assign(Elo=0)
    
    stats[ind] = stats.index.to_series().apply(lambda x: endYear_Elo(x, ind, scores.loc[(scores['HomeID'] == x) | (scores['VisID'] == x)].tail(1), k))
    
    confs = stats.groupby('Conf').mean() #conference means
    return stats, confs

def revertElo(teams, prev_elo, ind, stats, conf_means): #revert Elo for new year
    teams_elo = pd.Series(prev_elo, index=ind) #pre-allocate array
    
    for i, team in enumerate(teams): #iterate through both teams
        pos = 'Home' if i == 0 else 'Vis'
        if team in stats.index: #if team played prev year
            conf = stats.loc[team, 'Conf'] #get conf 
            teams_elo['{}Elo'.format(pos)] = stats.loc[team,'Elo'] - (1/3) * (stats.loc[team,'Elo'] - int(conf_means.loc[conf, 'Elo']))
        
    return teams_elo

#######################################################################################

def elo(teams, prev_elo, ind, k, scores): #update Elo for game
    game = pd.Series(prev_elo, index=ind) #pre-allocate array
    
    for i, team in enumerate(teams): #iterate through teams in given game
        pos = 'Home' if i == 0 else 'Vis'
        prev_game = scores.loc[(scores['HomeID'] == team) | (scores['VisID'] == team)].tail(1) #find prev game played
        if prev_game.shape[0] == 0: #skip teams that haven't played before
            continue
            
        elo_diff = prev_game['HomeElo'] - prev_game['VisElo'] #diff in ratings
            
        if int(prev_game['HomeID']) == team: #get prev pos and result, prob of winning
            prev_pos = 'Home' 
            result = 1 if int(prev_game['Spread']) > 0 else 0
            elo_prob = 1/(10**(-elo_diff/400)+1)
        else:
            prev_pos = 'Vis'   
            result = 1 if int(prev_game['Spread']) < 0 else 0
            elo_prob = 1/(10**(elo_diff/400)+1)
            
        elo_w = prev_game['HomeElo'] if int(prev_game['Spread']) > 0 else prev_game['VisElo'] #get prev winner
        elo_l = prev_game['HomeElo'] if int(prev_game['Spread']) < 0 else prev_game['VisElo']
        mv = np.log(abs(prev_game['Spread']) + 1) * (2.2/((elo_w-elo_l)*0.001 + 2.2)) * k #get margin of victory
        
        game['{}Elo'.format(pos)] = int(prev_game['{}Elo'.format(prev_pos)]) + mv * (result - elo_prob) #update
              
    return game

def add_elo(scores, k, prev_scores=None):
    
    ind = ['HomeElo','VisElo'] #new features
    
    scores = scores.assign(HomeElo=1500,SpreadElo=0,VisElo=1500)
    
    if prev_scores is not None: #Adjust elo for prev year
        prev_stats, conf_means = getFinal_elo(prev_scores, k)
        scores[ind] = scores.apply(lambda x: revertElo((x['HomeID'],x['VisID']),[x['HomeElo'],x['VisElo']],ind,prev_stats,conf_means), axis=1)
    else: #properly set elo for first year
        scores.loc[scores['HomeConf']=='NotMajor','HomeElo'] = 1300
        scores.loc[scores['VisConf']=='NotMajor','VisElo'] = 1300
    
    scores_list = [group for _, group in scores.groupby('Week')] #groupby week 
    
    for i, score in enumerate(scores_list[1:]): #iterate through weeks
        i += 1
        prev_weeks = pd.concat(scores_list[:i])
        scores_list[i][ind] = scores_list[i].apply(lambda x: elo((x['HomeID'],x['VisID']),[x['HomeElo'],x['VisElo']],ind,k, prev_weeks), axis = 1)
    scores = pd.concat(scores_list) #rebuild df
    
    elo_diff = scores['HomeElo'] - scores['VisElo']
    scores['SpreadElo'] = (elo_diff)/25 + 2.6 #predicted spread
    scores['HomeEloProb'] = 1/(10**(-elo_diff/400)+1)
    scores['VisEloProb'] = 1/(10**(elo_diff/400)+1)
    
    return scores

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

def add_ptsWinsGames(scores): #initialize df for year

    scores = scores.assign(HomePtsF=0.0, HomePtsA=0.0, HomeWins=0.0, HomeGames=0.0, HomeWinPct=0.0, HomePtsFPG=0.0, HomePtsAPG=0.0,
                           VisPtsF=0.0, VisPtsA=0.0, VisWins=0.0, VisGames=0.0, VisWinPct=0.0, VisPtsFPG=0.0, VisPtsAPG=0.0,
                           HomeDiffOD=0.0, VisDiffOD=0.0, DiffPtsFPG=0.0, DiffPtsAPG=0.0,) #init features in df
    pwg_ind = ['HomePtsF','HomePtsA','HomeWins','HomeGames','HomeWinPct','HomePtsFPG','HomePtsAPG','HomeDiffOD',
               'VisPtsF','VisPtsA','VisWins','VisGames','VisWinPct','VisPtsFPG','VisPtsAPG','VisDiffOD',
               'DiffPtsFPG','DiffPtsAPG'] 
    
    scores['Spread'] = scores['HomeFinal'] - scores['VisFinal']  #add Spread
    scores_list = [group for _, group in scores.groupby('Week')] #split df by week
    
    for i, score in enumerate(scores_list[1:]): #iterate through weeks, ignore Week1
        i += 1
        prev_weeks = pd.concat(scores_list[:i]) #get df of prev weeks
        scores_list[i][pwg_ind] = scores_list[i].apply(lambda x: ptsWinsGames((x['HomeID'],x['VisID']),pwg_ind,prev_weeks), axis = 1)
    scores = pd.concat(scores_list) #rebuild df
    
    return scores


#######################################################################################

conf = pd.read_csv('data/conferences/mergedConferences.csv')

conf['HomeID'] = conf['ID']
conf['VisID'] = conf['ID']
conf['Season'] = conf['Year']
conf['HomeConf'] = conf['Conf']
conf['VisConf'] = conf['Conf']
confs = [cnf for _,cnf in conf.groupby('Season')]

#######################################################################################

ultimate = pd.read_csv('data/snoozle/snoozle-combined.csv')
ultimate['HomeFinal'] = abs(ultimate['HomeFinal'])
ultimate['VisFinal'] = abs(ultimate['VisFinal'])
ultimate['HomeConf'] = ultimate['HomeConf'].apply(lambda x: 'NotMajor' if x == 'NON-D1' else x)
ultimate['VisConf'] = ultimate['VisConf'].apply(lambda x: 'NotMajor' if x == 'NON-D1' else x)
ultimate[['HomeConf_NotMajor', 'VisConf_NotMajor']] = pd.get_dummies(ultimate[['HomeConf','VisConf']])[['HomeConf_NotMajor','VisConf_NotMajor']]

for col in ultimate:
    if 'v_' in col:
        ultimate = ultimate.rename({col:'Vis{}'.format(col.split('v_')[1].title().replace('_', ''))}, axis=1)
    elif 'h_' in col:
        ultimate = ultimate.rename({col:'Home{}'.format(col.split('h_')[1].title().replace('_', ''))}, axis=1)
    elif '_' in col:
        ultimate = ultimate.rename({col:'{}'.format(col.title().replace('_',''))}, axis=1)

ultimates = [ultSsn for _,ultSsn in ultimate.groupby('Season')]

ultimates2 = [0 for i in range(len(ultimates))]
  
for year, ult in enumerate(ultimates):
    print('Year: {}'.format(year))
    ultimates2[year] = add_ptsWinsGames(ult)
    print('Adding Pyth and Elo')
    if year == 0:
        ultimates2[year] = add_pythWins(ultimates2[year], 2.37)
        ultimates2[year] = add_elo(ultimates2[year], 20)
    else:
        ultimates2[year] = add_pythWins(ultimates2[year], 2.37, ultimates2[year-1])
        ultimates2[year] = add_elo(ultimates2[year], 20, ultimates2[year-1])
        
ultimate_complete = pd.concat(ultimates2)

print('finished')
ultimate_complete.to_csv('data/snoozle/snoozle_ijh.csv', index=False)
    
    
    