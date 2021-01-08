import pandas as pd
new_data = pd.read_spss('~/Downloads/Muscle US Z scores pattern recognition project 17062020_NB.sav')
# SPSS origin is the start of the Gregorian calendar, Unix origin is 1970, so subtract the difference
new_data['Correct_timestamp'] = new_data['Study_date'] - 12219379200
new_data['Correct_timestamp'].fillna(0,inplace=True)
new_data.at[469,'Correct_timestamp'] = 0
new_data['s'] = pd.to_datetime(new_data['Correct_timestamp'],unit='s', origin='unix')


