import numpy as np
import pandas as pd
from datetime import *

def fix_dates(d, wrong):
    if d in wrong:
        return d - timedelta(days=1)
    else:
        return d

def fix_week(year):
    
    sched_path = 'data/schedules/schedule{}.csv'.format(year)
    sched_data = pd.read_csv(sched_path)
    sched_data['WeekDate'] = pd.to_datetime(sched_data['Date']).dt.date
    sched = sched_data[['Week','WeekDate']].drop_duplicates()
    
    scores_path = 'data/allScores_Wrong/NCAAAllScores{}.csv'.format(year)
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
    scores_clean.to_csv('data/allScores/NCAAAllScores{}_Week.csv'.format(year), index=False)
    

for d in [2013, 2014, 2015, 2016]:
    print('Starting Season: {}'.format(d))
    fix_week(d)
    print('Finishing Season: {}'.format(d))