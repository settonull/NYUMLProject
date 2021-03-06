import numpy as np
import pandas as pd

def get_Teams(scores): #list of teams and conferences for given year
    teams = np.union1d(scores['HomeID'].unique(), scores['VisID'].unique()) #teams
    stats = pd.DataFrame({'Team':teams}) 

    for pos in ['Home','Vis']: #add conferences
        temp = scores[['{}ID'.format(pos),'{}Conf_NotD1'.format(pos)]]
        temp = temp.rename(index=str, columns = {'{}ID'.format(pos):'Team'})
        stats = pd.merge(stats, temp, on='Team', how='left')

    stats = stats.drop_duplicates().reset_index(drop=True) #drop extra rows from bad merge
    stats['Conf'] = stats['HomeConf_NotD1'].fillna(stats['VisConf_NotD1']) #merge conf features
    stats.drop(['VisConf_NotD1','HomeConf_NotD1'], axis=1, inplace=True) #drop old conf features
    
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
        scores.loc[scores['HomeConf']=='NotD1','HomeElo'] = 1300
        scores.loc[scores['VisConf']=='NotD1','VisElo'] = 1300
    
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

data_list = [0,0,0,0]

for j, year in enumerate(range(2013, 2017)):
    print('Starting year: {}'.format(year))
    data_list[j] = pd.read_csv('data/scores_wcp/scores_wcp{}.csv'.format(year))
    if year == 2013:
        data_list[j] = add_pythWins(data_list[j], 2.37)
        data_list[j] = add_elo(data_list[j], 20)
    else:
        data_list[j] = add_pythWins(data_list[j], 2.37, data_list[j-1])
        data_list[j] = add_elo(data_list[j], 20, data_list[j-1])
        
for j, year in enumerate(range(2013, 2017)):
    print('Printing year: {}'.format(year))
    data_list[j].to_csv('data/scores_pe/scores_pythElo{}.csv'.format(year), index=False)