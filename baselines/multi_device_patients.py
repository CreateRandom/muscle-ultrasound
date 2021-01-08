import pandas as pd

from loading.loaders import get_data_for_spec
from utils.experiment_utils import get_default_set_spec_dict


def export_multi_device_patients():
    '''Find and export all patients whose muscles have been recorded on two different devices. This is used in the
        EI_transformation notebook to train a regression model mapping between brightness values.'''
    set_spec_dict = get_default_set_spec_dict()
    all_patients = []
    for key, value in set_spec_dict.items():
        if key.startswith('Multiple'):
            patients = get_data_for_spec(value, loader_type='bag', attribute_to_filter=None,
                                         muscles_to_use=None)
            all_patients.extend(patients)

    match_table = []
    for patient in all_patients:
        p, e = get_matched_records(patient)
        if not p:
            continue
        match_table.append(align_records(p, e))
    # A dataframe that contains matched brightness values (i.e. EI_e for Esaote and EI_p for Philips)
    total = pd.concat(match_table)
    total.reset_index(inplace=True, drop=True)

    total.to_pickle('final_models/multi_patients_aligned.pkl')

    return total


def get_record_frame(patient):
    info = []
    for i, record in enumerate(patient.records):
        info.append({**record.meta_info, **{'i': i}})

    record_frame = pd.DataFrame(info)

    return record_frame


def get_matched_records(patient):
    record_frame = get_record_frame(patient)

    if not 'ESAOTE_6100' in record_frame['DeviceInfo'].values or not 'Philips Medical Systems_iU22' in record_frame['DeviceInfo'].values:
        return None, None

    esaote_records = record_frame[record_frame['DeviceInfo'] == 'ESAOTE_6100'].sort_values('RecordingDate', ascending=True)
    esaote_record = esaote_records.iloc[0]
    esaote_record = patient.records[esaote_record['i']]
    philips_records = record_frame[record_frame['DeviceInfo'] == 'Philips Medical Systems_iU22'].sort_values('RecordingDate',
                                                                                           ascending=False)
    philips_record = philips_records.iloc[0]
    philips_record = patient.records[philips_record['i']]

    return philips_record, esaote_record


def get_muscle_frame(record):
    muscles_to_encode = [{'Muscle': v1, 'Side': v2, 'EI': v3, 'EIZ': v4} for v1, v2, v3, v4 in
                         zip(record.meta_info['Muscles_list'], record.meta_info['Sides'], record.meta_info['EIs'], record.meta_info['EIZ'])]

    return pd.DataFrame(muscles_to_encode)


def align_records(record_a, record_b):
    date_diff = record_b.meta_info['RecordingDate'] - record_a.meta_info['RecordingDate']
    muscle_frame_a = get_muscle_frame(record_a)

    muscle_frame_b = get_muscle_frame(record_b)

    x = muscle_frame_a.merge(muscle_frame_b, on=['Muscle', 'Side'], suffixes=('_p', '_e'))
    x = x.dropna(subset=['EI_p', 'EI_e'])
    x['DateDiff'] = date_diff
    return x