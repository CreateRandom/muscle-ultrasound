import pandas as pd
import os

from sklearn.model_selection import train_test_split
label_frame = pd.read_csv('/home/klux/Thesis_2/data/myositis/im_muscle_chart.csv')

# recode diagnosis into binary
label_frame['Diagnosis_bin'] = label_frame['Diagnosis'].replace(
    {'I' : True, 'N': False, 'P': True, 'D': True})

patients = list(set(label_frame['PatientID']))

print(patients)

labels = [label_frame[label_frame['PatientID'] == patient].iloc[0]['Diagnosis_bin'] for patient in patients]
label_map = dict(zip(patients,labels))

print(labels)
train, temp = train_test_split(patients, train_size = 0.8, stratify=labels)
sublabels = [label_map[k] for k in temp]
val, test = train_test_split(temp, train_size = 0.5, stratify=sublabels)

train = label_frame[label_frame['PatientID'].isin(train)]
val = label_frame[label_frame['PatientID'].isin(val)]
test = label_frame[label_frame['PatientID'].isin(test)]

print(train['Diagnosis_bin'].value_counts())
print(val['Diagnosis_bin'].value_counts())
print(test['Diagnosis_bin'].value_counts())

os.makedirs('sets', exist_ok = True)

train.to_csv('/home/klux/Thesis_2/data/myositis/train.csv')
val.to_csv('/home/klux/Thesis_2/data/myositis/val.csv')
test.to_csv('/home/klux/Thesis_2/data/myositis/test.csv')