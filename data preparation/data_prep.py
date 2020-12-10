import numpy as np
import pandas as pd
import skill_level
import sqlite3


def prep_team_data(df, country):

    '''
    Cleans match result data
    :param df: Dataframe of raw match results
    :param country: List of countries to keep in the dataset
    :return: DataFrame of clean match results
    '''

    try:

        df.drop(df.columns.difference(['Team', 'Result','For','Aga','Diff','HTf',
                                       'HTa','Opposition', 'Ground', 'Match Date']), 1, inplace=True)

        # Clean opposition field
        df['Opposition'] = df['Opposition'].str.split(" ", n=1, expand=True)[1]

        # Ensure team and opposition names match
        u = {'United States of America': 'USA'}
        df['Team'] = df['Team'].map(u).fillna(df['Team'])

        # Remove one copy of each result
        df = df.drop_duplicates(subset=['Result', 'For', 'Aga', 'Diff', 'Ground', 'Match Date'], keep='first')

        # Take only wins and draws to remove duplicates
        df = df[df.Result != 'lost']

        # Remove rows where opposition is blank
        df['Opposition'].replace('', np.nan, inplace=True)
        df = df.dropna(subset=['Opposition'])

        # Remove rows where games havent been played yet
        df['Result'].replace('-', np.nan, inplace=True)
        df.dropna(subset=['Result'], inplace=True)

        # Only use countries from list
        df = df[df['Team'].isin(country) & df['Opposition'].isin(country)]

        df['Match Date'] = pd.to_datetime(df['Match Date']).dt.date

    except Exception as e:
        print(e)
        print("Team data preparation failed")

    return df


def add_rankings(match_results_clean, rankings):

    '''
    Adds WR rankings to match results
    :param match_results_clean: DataFrame of clean match results
    :param rankings: DataFrame of WR rankings
    :return: A DataFrame of combined match results and their associated team WR rankings
    '''

    # Ensure Match Date format is the same between team_ratings and all_rankings
    rankings['Match Date']=rankings['Match Date'].apply(lambda x : pd.to_datetime(x, format = '%d/%m/%Y'))
    match_results_clean['Match Date'] = pd.to_datetime(match_results_clean['Match Date'], format='%Y/%m/%d')

    # Merge team and opposition rankings
    match_results_rankings = pd.merge(match_results_clean,rankings, left_on=['Team','Match Date'],
                                      right_on=['Team','Match Date'], how='left')

    match_results_rankings.drop_duplicates(inplace=True)
    match_results_rankings = pd.merge(match_results_rankings,rankings, left_on=['Opposition','Match Date'],
                                      right_on=['Team','Match Date'], how='left')

    match_results_rankings.rename({"pts_x":'Team ranking pts',"pts_y":'Opposition ranking pts',"Team_x":'Team'},
                                  axis=1, inplace=True)
    match_results_rankings.drop_duplicates(subset=['Team', 'Opposition', 'Match Date'], inplace=True)

    match_results_rankings = match_results_rankings[['Team', 'Result','For','Aga','Diff',
                                                     'Opposition','Match Date','Team ranking pts',
                                                     'Opposition ranking pts']]

    # World Rugby rankings started in Oct 2003 so remove blank rows / matches before Oct 2003
    match_results_rankings.dropna(subset=['Team ranking pts'], inplace=True)

    return match_results_rankings


def add_lost_results(df):

    '''
    Switches winning team and losing team in match results
    :param df: DataFrame of match results
    :return: DataFrame of match results with team1 and team2 inverted i.e. winning team becomes losing team
    '''

    try:
        # Prepare ratings data by adding in lost results (i.e. duplicate wins and invert data)
        tmp = pd.DataFrame()
        tmp['For'] = df['Aga']
        tmp['Aga'] = df['For']
        tmp['Team skill'] = df['Opp skill']
        tmp['Opp skill'] = df['Team skill']
        tmp['Team sigma'] = df['Opp sigma']
        tmp['Opp sigma'] = df['Team sigma']
        tmp['Team ranking pts'] = df['Opposition ranking pts']
        tmp['Opposition ranking pts'] = df['Team ranking pts']
        tmp['Win prob'] = (1-df['Win prob'])
        tmp['Match Quality'] = df['Match Quality']
        tmp['Team'] = df['Opposition']
        tmp['Opposition'] = df['Team']
        tmp['Match Date'] = df['Match Date']
        tmp['Diff'] = df['Diff']
        tmp['Result'] = "lost"

        data_for_model = pd.concat([df, tmp], sort=False, ignore_index=True)

        return data_for_model

    except Exception as e:

        print(e)
        print("Failed adding lost results")


# Get latest team ranking and skill level data

def get_stats(df, country):

    '''
    Get latest rankings and skills as at last played match
    :param df: DataFrame of latest match results, WR rankings and skill levels
    :param country: List of countries to include
    :return: DataFrame of latest ranking and skill level by team
    '''


    df = df.sort_values(['Match Date'], ascending=True)
    stats = pd.DataFrame()
    for i in country:
        tmp = (df.loc[(df['Team'] == i)].tail(1))
        stats = pd.concat([stats,tmp])

    return stats[['Team','Match Date','Team ranking pts',
                  'Team skill','Team sigma']]

conn = sqlite3.connect('match_results.db')
cursor = conn.cursor()

country = ['New Zealand', 'South Africa', 'England', 'Wales',
           'Scotland', 'Australia', 'France', 'Argentina',
           'Ireland', 'Fiji', 'Italy', 'Samoa', 'Japan',
           'Canada', 'Tonga', 'USA',
           'Georgia', 'Russia', 'Romania']

# Clean match result data
try:
    raw_data = pd.read_sql_query('SELECT * from raw_data',conn)
    match_results_clean = prep_team_data(raw_data, country)
    print('Team data prepared')

except Exception as e:
    print(e)
    print("Failed on match result data prep")

# Add World Rugby Rankings
try:
    rankings = pd.read_sql_query('SELECT * from rankings', conn)
    match_results_clean = match_results_clean.drop_duplicates(subset=['Team', 'Result','For','Aga','Diff','HTf',
                                                                      'HTa','Opposition','Ground','Match Date'])

    match_results_with_rankings = add_rankings(match_results_clean, rankings)
    # match_results_with_rankings.to_sql('match_results_with_rankings', conn, if_exists='replace')
    print("World Rugby rankings added")

except Exception as e:
    print(e)
    print("Failed adding World Rugby Rankings")

# Calculate team skill levels
try:
    matches_rankings_skill = skill_level.rate_teams(match_results_with_rankings)
    print("Team skill assessment complete")

except Exception as e:
    print(e)
    print("Failed on team skill level")

# Add in lost results for model
try:
    data_for_model = add_lost_results(matches_rankings_skill)
    # data_for_model.to_csv('model_training_data.csv')
    matches_rankings_skill.to_sql('model_training_data', conn, if_exists='replace')
    print("Lost results added back to data")
    print("Model training data prepared")

except Exception as e:
    print(e)
    print("Failed at model data preparation")

# Get latest rankings and skills as at last played match
try:
    latest_stats = get_stats(data_for_model, country)
    latest_stats.to_sql('latest_stats', conn, if_exists='replace')
except Exception as e:
    print(e)
    print("Failed retrieving latest team stats")








