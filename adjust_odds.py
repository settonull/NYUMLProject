import numpy as np
import pandas as pd
from datetime import *

#######################################################################################

def indices(date, ind):
    return pd.Series([date.year, date.month, date.day], index=ind)

def add_indices(data, ind):
    data[ind] = pd.to_datetime(data[ind]).dt.date
    data[['Year','Month','Day']] = data[ind].apply(lambda x: indices(x, ['Year','Month','Day']))
    data.drop(ind, axis=1, inplace=True)
    return data

#######################################################################################

odds = pd.read_csv('data/NCAAF_Odds.csv')
odds = add_indices(odds, 'DATE(date)')
odds = odds.rename({'Away':'Visiter'}, axis=1)
odds.drop(['HomeScore','AwayScore'],axis=1, inplace=True)

for col in odds:
    changed = False
    if '_' in col:
        new_col = col.replace('_', '')
        odds = odds.rename({col:new_col}, axis=1)
        changed = True
    if 'Away' in col:
        if changed:
            new_col2 = new_col.replace('Away', 'Vis')
            odds = odds.rename({new_col:new_col2}, axis=1)
        else:
            new_col = col.replace('Away', 'Vis')
            odds = odds.rename({col:new_col}, axis=1)
            
odds.to_csv('data/oddsAdjusted.csv', index=False)        
           