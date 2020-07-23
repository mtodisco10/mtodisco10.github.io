---
title: "Google Cloud & NCAA ML Competition 2019-Men's"
date: 2019-03-01
tags: [machine learning, kaggle]
# header:
#  overlay_image: "/images/amari_rex.png"
excerpt: "Machine Learning, Kaggle"
mathjax: "true"
---

# Google Cloud & NCAA ML Competition 2019-Men's


```python
#Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
%matplotlib inline  

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import log_loss
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')
```

    /anaconda2/lib/python2.7/site-packages/lightgbm/__init__.py:46: UserWarning: Starting from version 2.2.1, the library file in distribution wheels for macOS is built by the Apple Clang (Xcode_8.3.3) compiler.
    This means that in case of installing LightGBM from PyPI via the ``pip install lightgbm`` command, you don't need to install the gcc compiler anymore.
    Instead of that, you need to install the OpenMP library, which is required for running LightGBM on the system with the Apple Clang compiler.
    You can install the OpenMP library by the following command: ``brew install libomp``.
      "You can install the OpenMP library by the following command: ``brew install libomp``.", UserWarning)


### Read in CSV Data


```python
massey_df = pd.read_csv('2019DataFiles/MasseyOrdinals_thru_2019_day_128.csv')
teams = pd.read_csv('2019DataFiles/Teams.csv')
reg_season_results_df = pd.read_csv('2019DataFiles/RegularSeasonCompactResults.csv')
reg_season_stats_df = pd.read_csv('2019DataFiles/RegularSeasonDetailedResults.csv')
tourney_results_df = pd.read_csv('2019DataFiles/NCAATourneyCompactResults.csv')
conf_results_df = pd.read_csv('2019DataFiles/ConferenceTourneyGames.csv')
rpi_df = pd.read_csv('2019DataFiles/2019Ranks.csv')
seeds_df = pd.read_csv('2019DataFiles/NCAATourneySeeds.csv')
sample_submission_df = pd.read_csv('2019DataFiles/SampleSubmissionStage2.csv')
```

### Strength of Schedule, RPI, Road Success by Team
- External data:
    - https://extra.ncaa.org/solutions/rpi/SitePages/Home.aspx
    - http://warrennolan.com/basketball/2019/rpi-live

### Record by Season per Team
Build a dataframe with regular season counts of total wins, losses, home wins, and neutral wins


```python
def create_reg_season_record_df(df):
    #Create 4 dataframes - wins, losses, home wins, nuetral wins
    wins_df = df.groupby(['WTeamID','Season'], as_index=False)['WScore'].count()
    loss_df = df.groupby(['LTeamID','Season'], as_index=False)['LScore'].count()
    home_wins_df = df[df.WLoc == 'H'].groupby(['WTeamID','Season'], 
                                              as_index=False)['WScore'].count()
    
    #Merge the dataframes
    reg_season_record_df = wins_df.merge(loss_df, how = "outer", \
                         left_on =['WTeamID','Season'], right_on = ['LTeamID','Season'])
    
    reg_season_record_df = reg_season_record_df.merge(home_wins_df, how = "outer", \
                           left_on =['WTeamID','Season'], right_on = ['WTeamID','Season'])
    #Rename columns
    reg_season_record_df.columns = ['TeamID','Season','Wins','LTeamID','Losses','Home_wins']

    #Fill in NA columns with 0 and change the column data type to an integer
    reg_season_record_df['TeamID'] = reg_season_record_df['TeamID'].fillna(reg_season_record_df['LTeamID']).astype(int)
    reg_season_record_df[['Wins','LTeamID','Losses','Home_wins']] = \
        reg_season_record_df[['Wins','LTeamID','Losses','Home_wins']].fillna(0).astype(int)

    #The LTeamID column is not needed so drop it
    reg_season_record_df = reg_season_record_df.drop('LTeamID', axis = 1)
    
    reg_season_record_df = reg_season_record_df.loc[reg_season_record_df.Season >= 2003]
    
    return reg_season_record_df
    
reg_season_record_df = create_reg_season_record_df(reg_season_results_df)
```

### Rankings by Season per Team


```python
system_lst = ['POM','RTH','SAG','WOL','WLK']
rank_day_lst = [126,127,128]
def assign_tm_ranks(system_lst, day_of_rank):
    ranking_lst = []
    #Subset data for Ranking System and the day of rank
    for system in system_lst:
        tm_rank = massey_df.loc[(massey_df.SystemName == system) & \
                         (massey_df.RankingDayNum.isin(day_of_rank))][['TeamID','Season','OrdinalRank']]
        
        tm_rank.rename(columns={'OrdinalRank': system}, inplace=True)
        
        ranking_lst.append(tm_rank)
    
    full_tm_rank = reduce(lambda x, y: pd.merge(x, y, how='left', on=['TeamID','Season']), ranking_lst)
    
    reg_season_rank_df = reg_season_record_df.merge(full_tm_rank,
                                                    how='left',
                                                    on=['TeamID','Season'])
    return reg_season_rank_df

reg_season_rank_df = assign_tm_ranks(system_lst, rank_day_lst).fillna(350).astype(int)
```

### Quality Wins by Season per Team
   - based on opponents RPI, construct features to count the number of wins against the top 25, 50, top 100, and top 200


```python
#Merge the RPI df with the regular season results df
top_wins_df = reg_season_results_df.merge(rpi_df, how = 'inner', \
                            left_on = ['Season','LTeamID'], right_on = ['Season', 'TeamID'])

#Create columns to mark the category of ranking that the losing opponent fell into
top_wins_df['Wins1_25'] = top_wins_df['RPI'].apply(lambda x: 1 if x <=25 else 0)
top_wins_df['Wins26_50'] = top_wins_df['RPI'].apply(lambda x: 1 if x <=50 else 0)
top_wins_df['Wins51_100'] = top_wins_df['RPI'].apply(lambda x: 1 if x <=100 and x > 50 else 0)
top_wins_df['Wins101_200'] = top_wins_df['RPI'].apply(lambda x: 1 if x <=200 and x > 100 else 0)

#Aggregate the sum of wins by opponent ranking category
top_wins_df = top_wins_df.groupby(['Season','WTeamID'], 
                                  as_index=False)[['Wins1_25','Wins26_50','Wins51_100','Wins101_200']].sum()

#Rename the Team Column
top_wins_df.rename(columns={'WTeamID':'TeamID'}, inplace=True)
```

### Team Personal Regular Season Stats
Create a dataframe with features that represent season long per game averages of the team's own statistics.  These include: points scored, offensive rebounds, defensive rebounds, field goals made, field goals attempted, 3 point field goals made, 3 point field goals attempted, free throws made, free throws attempted, turnovers, steals, blocks, personal fouls, assists


```python
#Subset the columns of the regular season stats dataframe for the columns we want based on the games the team won
own_wins_stats_df = reg_season_stats_df[['Season','WTeamID','WScore','WFGM','WFGA','WFGM3',\
                                         'WFGA3','WFTM','WFTA','WOR','WDR','WAst','WTO','WStl','WBlk','WPF']]

#Rename the columns
own_wins_stats_df.columns = ['Season','TeamID','O_Score','O_FGM','O_FGA','O_FGM3','O_FGA3','O_FTM','O_FTA',\
                            'O_OR','O_DR','O_Ast','O_TO','O_Stl','O_Blk','O_PF']

#Subset the columns of the regular season stats dataframe for the columns we want based on the games the team loss
own_loss_stats_df = reg_season_stats_df[['Season','LTeamID','LScore','LFGM','LFGA','LFGM3',\
                                         'LFGA3','LFTM','LFTA','LOR','LDR','LAst','LTO','LStl','LBlk','LPF']]

#Rename the columns so that they match the columns in the wins df
own_loss_stats_df.columns = ['Season','TeamID','O_Score','O_FGM','O_FGA','O_FGM3','O_FGA3','O_FTM','O_FTA',\
                            'O_OR','O_DR','O_Ast','O_TO','O_Stl','O_Blk','O_PF']

#Append the loss df to the win df
own_stats_df = own_wins_stats_df.append(own_loss_stats_df)

#Take the average of the stats, grouped by season and team
own_stats_agg = own_stats_df.groupby(['Season','TeamID'], as_index=False).mean()
```

### Team Oppenent Regular Season Stats
  - same as above, but for the team's opponent


```python
#Subset the columns of the regular season stats dataframe for the columns we want based on the games the team won, but taking the stats of the losing team
opp_wins_stats_df = reg_season_stats_df[['Season','WTeamID','LScore','LFGM','LFGA','LFGM3',\
                                         'LFGA3','LFTM','LFTA','LOR','LDR','LAst','LTO','LStl','LBlk','LPF']]

#Rename the columns
opp_wins_stats_df.columns = ['Season','TeamID','D_Score','D_FGM','D_FGA','D_FGM3','D_FGA3','D_FTM','D_FTA',\
                            'D_OR','D_DR','D_Ast','D_TO','D_Stl','D_Blk','D_PF']

#Subset the columns of the regular season stats dataframe for the columns we want based on the games the team loss, but taking the stats of the winning team
opp_loss_stats_df = reg_season_stats_df[['Season','LTeamID','WScore','WFGM','WFGA','WFGM3',\
                                         'WFGA3','WFTM','WFTA','WOR','WDR','WAst','WTO','WStl','WBlk','WPF']]

#Rename the columns so that they match the columns in the wins df
opp_loss_stats_df.columns = ['Season','TeamID','D_Score','D_FGM','D_FGA','D_FGM3','D_FGA3','D_FTM','D_FTA',\
                            'D_OR','D_DR','D_Ast','D_TO','D_Stl','D_Blk','D_PF']

#Append the loss df to the win df
opp_stats_df = opp_wins_stats_df.append(opp_loss_stats_df)

#Take the average of the stats, grouped by season and team
opp_stats_agg = opp_stats_df.groupby(['Season','TeamID'], as_index=False).mean()
```

### Combining Team Personal & Team Opponent Regular Season Stats


```python
#Merge the average personal team stats with the average opponent team stats
reg_season_stats_agg = own_stats_agg.merge(opp_stats_agg, how='left', \
                                           left_on=['Season','TeamID'], right_on=['Season','TeamID'])
```

### Combining Quality Wins, Ranking, Average Season Stats, and Strength of Schedule
- Join the 4 dataframes to make 1 "Resume" dataframe


```python
#Merging the ranking dataframe with the season stats dataframe
resume_df = reg_season_stats_agg.merge(reg_season_rank_df, how = 'left', \
                                      left_on=['Season','TeamID'], right_on=['Season','TeamID'])

#Merging the Quality Wins dataframe with the previous two
resume_df = resume_df.merge(top_wins_df, how ='left', left_on=['Season','TeamID'], right_on=['Season','TeamID'])

#Filling any NA fields with 0, because some teams did not have quality wins
resume_df = resume_df.fillna(0)

#Converting the quality wins columns to integers
resume_df['Wins1_25'] = resume_df['Wins1_25'].astype(int)
resume_df['Wins26_50'] = resume_df['Wins26_50'].astype(int)
resume_df['Wins51_100'] = resume_df['Wins51_100'].astype(int)
resume_df['Wins101_200'] = resume_df['Wins101_200'].astype(int)

#Subsetting the data to after the 2003 season because that is when they two ranking systems we are using began
resume_df = resume_df.loc[resume_df['Season'] > 2003]

#Merging the Strength of Schedule dataframe with the previous 3
resume_df = resume_df.merge(rpi_df, how = 'left', \
               left_on = ['Season','TeamID'], right_on = ['Season','TeamID']).drop(['Team'], axis = 1)

#Filling in any missing data from the Strength of Schedule dataframe with the worst rankings and converting them to integers
resume_df = resume_df.fillna(200).astype(int)
```

### NCAA Tourney Results
   - We need to use past NCAA tournament wins and losses as the training target
   - The code below will transform the tournament results data and combine it with the resume data to set the framework for the model to train on


```python
#Create function to determine the team with the lower ID
def lower_id_func(row):
    if row['WTeamID'] < row['LTeamID']:
        return int(row['WTeamID'])
    else:
        return int(row['LTeamID'])

#Create function to determine whether or not the lower ID team won
def lower_won_func(row):
    if row['WTeamID'] < row['LTeamID']:
        return 1
    else:
        return 0
```


```python
#Create a column with the lower ID team as team_1
tourney_results_df['team_1'] = tourney_results_df.apply(lower_id_func, axis = 1)

#Create a column with the higher ID team as team_2
tourney_results_df['team_2'] = tourney_results_df.apply(lambda x: int(x['LTeamID']) if int(x['WTeamID']) == x['team_1']\
                                                        else x['WTeamID'], axis = 1)

#Create a column that shows whether or not team_1 won
tourney_results_df['Lower_Won'] = tourney_results_df.apply(lower_won_func, axis = 1)

tourney_results_df = tourney_results_df[['Season','team_1','team_2','Lower_Won']]

#Merge in resume data with the tournament results data with team_1 as the key
tourney_results_df = tourney_results_df.merge(resume_df, how = 'inner',\
                                              left_on=['Season','team_1'], \
                                              right_on=['Season','TeamID'])

#Drop the TeamID column
tourney_results_df = tourney_results_df.drop('TeamID', axis = 1)

#Merge in resume data with the tournament results data with team_2 as the key
tourney_results_df = tourney_results_df.merge(resume_df, how = 'inner',\
                                              left_on=['Season','team_2'], \
                                              right_on=['Season','TeamID'])

#Drop the TeamID column
tourney_results_df = tourney_results_df.drop('TeamID', axis = 1)

#Rename the columns to get rid of the suffixes using list comprehension
column_lst = [i.replace('_x','_team_1') for i in list(tourney_results_df.columns)]
column_lst = [i.replace('_y','_team_2') for i in column_lst]

tourney_results_df.columns = column_lst
```

### NCAA Tournament Seeds
   - The seeds data need very little manipulating with the exception of removing the region code from the seed number


```python
#Extract the seed number (withouth the region letters) from the seed column
seeds_df['Seed'] = seeds_df['Seed'].apply(lambda x: int(x[1:]) if len(x) == 3 else int(x[1:3])) 

#Merge the seeds dataframe with the tournament results with team_1 as the key
tourney_results_df = tourney_results_df.merge(seeds_df, how = 'left', \
                         left_on = ['Season','team_1'], right_on = ['Season','TeamID']).drop('TeamID', axis = 1)
#Rename the columns
tourney_results_df.rename(columns={'Seed':'Seed_team_1'}, inplace=True)

#Merge the seeds dataframe with the tournament results with team_2 as the key
tourney_results_df = tourney_results_df.merge(seeds_df, how = 'left', \
                         left_on = ['Season','team_2'], right_on = ['Season','TeamID']).drop('TeamID', axis = 1)

#Rename the columns
tourney_results_df.rename(columns={'Seed':'Seed_team_2'}, inplace=True)
```

### Conference Tournament Game Results
- In an effort to lengthen the training data, we will include the results from conference tournament games


```python
#Create a column with the lower ID team as team_1
conf_results_df['team_1'] = conf_results_df.apply(lower_id_func, axis =1)

#Create a column with the higher ID team as team_2
conf_results_df['team_2'] = conf_results_df.apply(lambda x: int(x['LTeamID']) if int(x['WTeamID']) == x['team_1']\
                                                        else x['WTeamID'], axis = 1)

#Create a column that shows whether or not team_1 won
conf_results_df['Lower_Won'] = conf_results_df.apply(lower_won_func, axis = 1)

conf_results_df = conf_results_df[['Season','team_1','team_2','Lower_Won']]

#Merge in resume data with the conference tournament results data with team_1 as the key
conf_tourney_df = conf_results_df.merge(resume_df, how = 'inner', left_on = ['Season','team_1'], \
                      right_on = ['Season','TeamID']).drop('TeamID', axis =1)

#Merge in resume data with the conference tournament results data with team_2 as the key
conf_tourney_df = conf_tourney_df.merge(resume_df, how = 'inner', left_on = ['Season','team_2'], \
                      right_on = ['Season','TeamID']).drop('TeamID', axis =1)

#Rename the columns to get rid of the suffixes using list comprehension
column_lst = [i.replace('_x','_team_1') for i in list(conf_tourney_df.columns)]
column_lst = [i.replace('_y','_team_2') for i in column_lst]

conf_tourney_df.columns = column_lst

#Merge the seeds dataframe with the conference tournament results with team_1 as the key
conf_tourney_df = conf_tourney_df.merge(seeds_df, how = 'inner', \
                         left_on = ['Season','team_1'], right_on = ['Season','TeamID']).drop('TeamID', axis = 1)

#Rename column
conf_tourney_df.rename(columns={'Seed':'Seed_team_1'}, inplace=True)

#Merge the seeds dataframe with the conference tournament results with team_2 as the key
conf_tourney_df = conf_tourney_df.merge(seeds_df, how = 'inner', \
                         left_on = ['Season','team_2'], right_on = ['Season','TeamID']).drop('TeamID', axis = 1)

#Rename column
conf_tourney_df.rename(columns={'Seed':'Seed_team_2'}, inplace=True)

#Append the conference tournament games to the NCAA tournament games
tourney_results_df = tourney_results_df.append(conf_tourney_df)
```

### Resume Comparison
   - Create a dataframe that compares the resume of two teams playing against each other in the tournaments
   - The features will be combined into differences by subtracting team_1 values from team_2 values


```python
#Subset the fields that are not metrics
resume_comparison_df = tourney_results_df[['Season','team_1','team_2','Lower_Won']]

#Create a unique list of metrics that we want to compare
metric_lst = [x[0:x.find('_team_1')] for x in tourney_results_df.columns if '_team_1' in x]

#For each metric, calculate the difference and add it as a field to the resume comparison df
for metric in metric_lst:
    resume_comparison_df[metric + '_diff'] = tourney_results_df[metric + '_team_2'] - tourney_results_df[metric + '_team_1']
```

### Extract the dependent variable from the independent variables
   - The Lower_Won column is the target variable, which we are looking to predict based on the other numeric variables in the resume_comparison dataframe


```python
#Dependent Variables
X = resume_comparison_df.drop(['Season','team_1','team_2','Lower_Won'], axis = 1)

#Target
y = resume_comparison_df.Lower_Won
```

### Scale the input features


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Split the data into Training and Testing Sets
 - This is done in order to test the performance of models without overfitting


```python
#1/3 of the data will be used to test, while 2/3 of it will be used to train the model
#A random state is set for reproducibility

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.33, random_state = 42)
```

### Instantiate, Fit, and Test Models

 - Several different models were trained and tested here including a Random Forest with a gini split, SVM, Nueral Network, and XGBoost
 - Random Forest with an entropy split showed the most promising results during the training period and was ultimately used for the winning prediction.  So that is the only model shown here.

#### Random Forest (Entropy)


```python
#Instantiate the model
model_entropy = RandomForestClassifier(n_estimators = 3000, 
                                       random_state = 1, 
                                       oob_score = True, 
                                       criterion='entropy')

model_scaled = RandomForestClassifier(n_estimators = 3000, 
                                       random_state = 1, 
                                       oob_score = True, 
                                       criterion='entropy')

#Fit the model on the training data
model_scaled.fit(X_train, y_train)

#Make predictions based on the testing data
pred_rfentropy = pd.Series([x[1] for x in list(model_scaled.predict_proba(X_test))])

#Show the log loss of predicted vs actual on the testing data
#The log loss was compared with other trained models.  I kept the two models with the lowest log loss. 
log_loss(np.array(y_test.astype(float)), np.array(pred_rfentropy.astype(float)))
```




    0.5814644980329275




```python
#Show a df of the most least feautures
pd.DataFrame(sorted(zip(model_scaled.feature_importances_, X.columns), reverse=True), \
             columns = ['Importance','Metric']).tail(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Importance</th>
      <th>Metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>0.015693</td>
      <td>O_FTM_diff</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.015599</td>
      <td>O_OR_diff</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.015442</td>
      <td>O_PF_diff</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.015334</td>
      <td>D_OR_diff</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.015144</td>
      <td>O_Blk_diff</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.014938</td>
      <td>O_TO_diff</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.014604</td>
      <td>O_DR_diff</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.014409</td>
      <td>D_Ast_diff</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.014338</td>
      <td>D_FTM_diff</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.014069</td>
      <td>D_FGM_diff</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.013129</td>
      <td>D_FGM3_diff</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.012933</td>
      <td>D_PF_diff</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.012566</td>
      <td>O_FGM3_diff</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.010027</td>
      <td>D_Stl_diff</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0.008341</td>
      <td>D_Blk_diff</td>
    </tr>
  </tbody>
</table>
</div>



### Feature Importance
   - which of the feautures in the trained model are most important to predict y?


```python
#Show a df of the most important feautures
pd.DataFrame(sorted(zip(model_scaled.feature_importances_, X.columns), reverse=True), \
             columns = ['Importance','Metric']).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Importance</th>
      <th>Metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.059533</td>
      <td>RPI_diff</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.043554</td>
      <td>Wins_diff</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.041963</td>
      <td>Seed_diff</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.041304</td>
      <td>WLK_diff</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.040228</td>
      <td>POM_diff</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.039930</td>
      <td>RTH_diff</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.039293</td>
      <td>Wins26_50_diff</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.037880</td>
      <td>WOL_diff</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.037089</td>
      <td>SoS_diff</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.036987</td>
      <td>SAG_diff</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Plot the top 15 most important feautures in the model
feat_imp = pd.Series(model_scaled.feature_importances_, index=X.columns)
feat_imp = feat_imp.nlargest(15)
feat_imp.plot(kind='barh')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1147b7c90>




![png](output_40_1.png)


- Not suprisingly, the ranking variables are most important
- However, there isn't a large drop-off when looking at the other inputs

### Fit to entire dataset


```python
model_entropy.fit(X, y)
model_scaled.fit(X_scaled, y)

#Make predictions based on the testing data
#pred_rfentropy = pd.Series([x[1] for x in list(model_entropy.predict_proba(X_test))])
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=3000, n_jobs=1,
                oob_score=True, random_state=1, verbose=0, warm_start=False)



### 2019 NCAA Tournament Data
   - Now that the model has been fit and selected, predictions can be made on the new 2019 data
   - The Sample Submission File (provided by Kaggle) is used to build out the data to predict on 
   - This is a very similar process to the steps taken above for the training data


```python
#From the Sample Submission File, split out the year from the ID column
year = sample_submission_df.ID.str.split('_').apply(lambda x: x[0]).astype(int)

#Extract the first team
team_1 = sample_submission_df.ID.str.split('_').apply(lambda x: x[1]).astype(int)

#Extract the second team
team_2 = sample_submission_df.ID.str.split('_').apply(lambda x: x[2]).astype(int)

#Put the 3 series in a data frame
new_sample_submission_df = pd.DataFrame({"ID": sample_submission_df.ID,
                                        "year": year,
                                        "team_1": team_1,
                                         "team_2": team_2})

#Merge the new_sample_submission_df with the resume_df by team and season to get the predictive variables
#First by Team 1
submission_df = new_sample_submission_df.merge(resume_df, how = 'left', left_on = ['team_1', 'year'],\
                                                right_on = ['TeamID','Season'])
#Then by Team 2
submission_df = submission_df.merge(resume_df, how = 'left', left_on = ['team_2', 'Season'],\
                                                right_on = ['TeamID','Season'])

#Drop the unneccessary columns and suffixes created from the merges
submission_df = submission_df.drop(['TeamID_x','TeamID_y'], axis =1)
column_lst = [i.replace('_x','_team_1') for i in list(submission_df.columns)]
column_lst = [i.replace('_y','_team_2') for i in column_lst]
submission_df.columns = column_lst

#Merge in the seeding for the 2018 Tournament and rename columns
submission_df = submission_df.merge(seeds_df, how = 'left', \
                                    left_on = ['Season','team_1'], \
                                    right_on =['Season', 'TeamID']).drop('TeamID', axis=1)

submission_df.rename(columns={'Seed':'Seed_team_1'}, inplace=True)

submission_df = submission_df.merge(seeds_df, how = 'left', \
                                    left_on = ['Season','team_2'], \
                                    right_on =['Season', 'TeamID']).drop('TeamID', axis=1)

submission_df.rename(columns={'Seed':'Seed_team_2'}, inplace=True)
```


```python
#Build a resume comparison data frame by subtracting team_1 values from team_2 values
submission_final_df = submission_df[['ID','team_1','team_2','Season']]

metric_lst = [x[0:x.find('_team_1')] for x in submission_df.columns if '_team_1' in x]

#For each metric, calculate the difference and add it as a field to the resume comparison df
for metric in metric_lst:
    submission_final_df[metric + '_diff'] = submission_df[metric + '_team_2'] - submission_df[metric + '_team_1']
```

### Prediction
   - With the 2019 input data prepared, we can predict the probability that team 1 will win (Lower_Won)


```python
#Drop the columns that will not be used for predictions
X_test = submission_final_df.drop(['ID','team_1','team_2','Season'], axis = 1)
X_test_scaled = scaler.fit_transform(X_test)

#Fit models
#model_entropy_fit = model_entropy.fit(X_test, y)
#model_entropy_scaled.fit(X_test)
```


```python
#Create a column for team_1 probability to win prediction, 
#using the 2nd element in the predict_proba array for each game
#This is probability Lower_Won = 1 
submission_final_df['Pred'] = pd.Series([x[1] for x in list(model_entropy.predict_proba(X_test))])
submission_final_df['Pred_scaled'] = pd.Series([x[1] for x in list(model_scaled.predict_proba(X_test_scaled))])
```

### Manual Adjustments
 - Dean Wade is hurt so knock down Kansas State


```python
prediction_df = submission_final_df[['ID', 'team_1', 'team_2', 'Pred', 'Pred_scaled']]
```


```python
prediction_df[['Pred','Pred_scaled']] = prediction_df.apply(lambda row: row[['Pred','Pred_scaled']] + 0.1 \
                                            if (row['team_2'] == '1243') \
                                                             & (row['Pred'] < 0.95) \
                                                             & (row['Pred_scaled'] < 0.95) \
                                            else row[['Pred','Pred_scaled']], axis = 1)

prediction_df[['Pred','Pred_scaled']] = prediction_df.apply(lambda row: row[['Pred','Pred_scaled']] - 0.1 \
                                            if (row['team_1'] == '1243') \
                                                            & (row['Pred'] > 0.1) \
                                                            & (row['Pred_scaled'] > 0.1)
                                            else row[['Pred','Pred_scaled']], axis = 1)
```


```python
prediction_df.Pred.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x114805f10>




![png](output_53_1.png)


### First four 1 seeds


```python
#DUKE
prediction_df.loc[(prediction_df.ID == '2019_1181_1295') | (prediction_df.ID == '2019_1181_1300')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>team_1</th>
      <th>team_2</th>
      <th>Pred</th>
      <th>Pred_scaled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>592</th>
      <td>2019_1181_1295</td>
      <td>1181</td>
      <td>1295</td>
      <td>0.966667</td>
      <td>0.968667</td>
    </tr>
    <tr>
      <th>594</th>
      <td>2019_1181_1300</td>
      <td>1181</td>
      <td>1300</td>
      <td>0.961000</td>
      <td>0.963667</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Virginia
prediction_df.loc[(prediction_df.ID == '2019_1205_1438')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>team_1</th>
      <th>team_2</th>
      <th>Pred</th>
      <th>Pred_scaled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>841</th>
      <td>2019_1205_1438</td>
      <td>1205</td>
      <td>1438</td>
      <td>0.131333</td>
      <td>0.148333</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Gonzaga
prediction_df.loc[(prediction_df.ID == '2019_1192_1211')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>team_1</th>
      <th>team_2</th>
      <th>Pred</th>
      <th>Pred_scaled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>629</th>
      <td>2019_1192_1211</td>
      <td>1192</td>
      <td>1211</td>
      <td>0.030667</td>
      <td>0.032</td>
    </tr>
  </tbody>
</table>
</div>




```python
#North Carolina
prediction_df.loc[(prediction_df.ID == '2019_1233_1314')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>team_1</th>
      <th>team_2</th>
      <th>Pred</th>
      <th>Pred_scaled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1025</th>
      <td>2019_1233_1314</td>
      <td>1233</td>
      <td>1314</td>
      <td>0.014</td>
      <td>0.018333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Make Adjustment for Virginia.  God Help them if they lose in the first round again.
# All the other teams look ok to me
prediction_df[['Pred','Pred_scaled']] = prediction_df.apply(lambda row: .03 \
                                            if (row['ID'] == '2019_1205_1438') \
                                            else row[['Pred','Pred_scaled']], axis = 1)

prediction_df.loc[(prediction_df.ID == '2019_1205_1438')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>team_1</th>
      <th>team_2</th>
      <th>Pred</th>
      <th>Pred_scaled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>841</th>
      <td>2019_1205_1438</td>
      <td>1205</td>
      <td>1438</td>
      <td>0.03</td>
      <td>0.03</td>
    </tr>
  </tbody>
</table>
</div>



### Prepare Data for Final Submission


```python
prediction_df = prediction_df[['ID','Pred','Pred_scaled']]
prediction_df['blended'] = (prediction_df['Pred'] + prediction_df['Pred_scaled']) / 2
```


```python
#Export Submission Files
#1st submission
prediction_df[['ID','Pred']].to_csv('2019Predictions/pred1.csv', index=False)

pred_scaled = prediction_df[['ID','Pred_scaled']].rename(columns={'Pred_scaled':'Pred'})
pred_scaled.to_csv('2019Predictions/pred_scaled.csv', index=False)
    
pred_blended = prediction_df[['ID','blended']].rename(columns={'blended':'Pred'})
pred_blended.to_csv('2019Predictions/pred_blended.csv', index=False)
```

## Export for Easy Team Comparison


```python
prediction_df['Team1_ID'] = prediction_df.ID.str.split('_', expand=True)[1]
prediction_df['Team2_ID'] = prediction_df.ID.str.split('_', expand=True)[2]
prediction_df[['Team1_ID','Team2_ID']] = prediction_df[['Team1_ID','Team2_ID']].astype(int)
```


```python
tm_comparison = prediction_df.merge(teams[['TeamID','TeamName']],
                                    how='left',
                                    left_on='Team1_ID',
                                    right_on='TeamID')
tm_comparison = tm_comparison.merge(teams[['TeamID','TeamName']],
                                    how='left',
                                    left_on='Team2_ID',
                                    right_on='TeamID')

tm_comparison = tm_comparison.drop(['TeamID_x', 'TeamID_y'], axis =1)

tm_comparison.rename(columns={'TeamName_x': 'TeamName_1', 'TeamName_y':'TeamName_2'}, inplace=True)

tm_comparison.to_csv('2019Predictions/bracket_preds_scaled.csv', index=False)
```


```python
tm_comparison.loc[tm_comparison.TeamName_1 == 'Old Dominion']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Pred</th>
      <th>Pred_scaled</th>
      <th>blended</th>
      <th>Team1_ID</th>
      <th>Team2_ID</th>
      <th>TeamName_1</th>
      <th>TeamName_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2002</th>
      <td>2019_1330_1332</td>
      <td>0.324333</td>
      <td>0.438000</td>
      <td>0.381167</td>
      <td>1330</td>
      <td>1332</td>
      <td>Old Dominion</td>
      <td>Oregon</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>2019_1330_1341</td>
      <td>0.801667</td>
      <td>0.725667</td>
      <td>0.763667</td>
      <td>1330</td>
      <td>1341</td>
      <td>Old Dominion</td>
      <td>Prairie View</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>2019_1330_1345</td>
      <td>0.115000</td>
      <td>0.312667</td>
      <td>0.213833</td>
      <td>1330</td>
      <td>1345</td>
      <td>Old Dominion</td>
      <td>Purdue</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>2019_1330_1371</td>
      <td>0.371000</td>
      <td>0.558333</td>
      <td>0.464667</td>
      <td>1330</td>
      <td>1371</td>
      <td>Old Dominion</td>
      <td>Seton Hall</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>2019_1330_1385</td>
      <td>0.413667</td>
      <td>0.590000</td>
      <td>0.501833</td>
      <td>1330</td>
      <td>1385</td>
      <td>Old Dominion</td>
      <td>St John's</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>2019_1330_1387</td>
      <td>0.544667</td>
      <td>0.628000</td>
      <td>0.586333</td>
      <td>1330</td>
      <td>1387</td>
      <td>Old Dominion</td>
      <td>St Louis</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>2019_1330_1388</td>
      <td>0.383333</td>
      <td>0.554333</td>
      <td>0.468833</td>
      <td>1330</td>
      <td>1388</td>
      <td>Old Dominion</td>
      <td>St Mary's CA</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>2019_1330_1393</td>
      <td>0.335000</td>
      <td>0.478667</td>
      <td>0.406833</td>
      <td>1330</td>
      <td>1393</td>
      <td>Old Dominion</td>
      <td>Syracuse</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>2019_1330_1396</td>
      <td>0.457667</td>
      <td>0.569333</td>
      <td>0.513500</td>
      <td>1330</td>
      <td>1396</td>
      <td>Old Dominion</td>
      <td>Temple</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>2019_1330_1397</td>
      <td>0.193333</td>
      <td>0.295333</td>
      <td>0.244333</td>
      <td>1330</td>
      <td>1397</td>
      <td>Old Dominion</td>
      <td>Tennessee</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>2019_1330_1403</td>
      <td>0.180667</td>
      <td>0.317333</td>
      <td>0.249000</td>
      <td>1330</td>
      <td>1403</td>
      <td>Old Dominion</td>
      <td>Texas Tech</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>2019_1330_1414</td>
      <td>0.503333</td>
      <td>0.551667</td>
      <td>0.527500</td>
      <td>1330</td>
      <td>1414</td>
      <td>Old Dominion</td>
      <td>UC Irvine</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>2019_1330_1416</td>
      <td>0.279667</td>
      <td>0.504667</td>
      <td>0.392167</td>
      <td>1330</td>
      <td>1416</td>
      <td>Old Dominion</td>
      <td>UCF</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>2019_1330_1429</td>
      <td>0.442667</td>
      <td>0.558667</td>
      <td>0.500667</td>
      <td>1330</td>
      <td>1429</td>
      <td>Old Dominion</td>
      <td>Utah St</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>2019_1330_1433</td>
      <td>0.448000</td>
      <td>0.609000</td>
      <td>0.528500</td>
      <td>1330</td>
      <td>1433</td>
      <td>Old Dominion</td>
      <td>VA Commonwealth</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>2019_1330_1436</td>
      <td>0.596667</td>
      <td>0.602333</td>
      <td>0.599500</td>
      <td>1330</td>
      <td>1436</td>
      <td>Old Dominion</td>
      <td>Vermont</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>2019_1330_1437</td>
      <td>0.228333</td>
      <td>0.439667</td>
      <td>0.334000</td>
      <td>1330</td>
      <td>1437</td>
      <td>Old Dominion</td>
      <td>Villanova</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>2019_1330_1438</td>
      <td>0.126000</td>
      <td>0.148667</td>
      <td>0.137333</td>
      <td>1330</td>
      <td>1438</td>
      <td>Old Dominion</td>
      <td>Virginia</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>2019_1330_1439</td>
      <td>0.144000</td>
      <td>0.277333</td>
      <td>0.210667</td>
      <td>1330</td>
      <td>1439</td>
      <td>Old Dominion</td>
      <td>Virginia Tech</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>2019_1330_1449</td>
      <td>0.407667</td>
      <td>0.551333</td>
      <td>0.479500</td>
      <td>1330</td>
      <td>1449</td>
      <td>Old Dominion</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>2019_1330_1458</td>
      <td>0.167333</td>
      <td>0.322667</td>
      <td>0.245000</td>
      <td>1330</td>
      <td>1458</td>
      <td>Old Dominion</td>
      <td>Wisconsin</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>2019_1330_1459</td>
      <td>0.329000</td>
      <td>0.491667</td>
      <td>0.410333</td>
      <td>1330</td>
      <td>1459</td>
      <td>Old Dominion</td>
      <td>Wofford</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>2019_1330_1463</td>
      <td>0.580333</td>
      <td>0.632000</td>
      <td>0.606167</td>
      <td>1330</td>
      <td>1463</td>
      <td>Old Dominion</td>
      <td>Yale</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
