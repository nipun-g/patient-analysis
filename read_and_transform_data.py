
import pandas as pd
import os
import sys
import random
import numpy as np


class TransformData:

    def __init__(self):
        self.project_path = os.path.dirname(sys.modules['__main__'].__file__)

    def read_data(self):

        # Read the raw data from available csv files

        bill_id = pd.read_csv(self.project_path + '/raw_data/bill_id.csv')
        bill_amount = pd.read_csv(self.project_path + '/raw_data/bill_amount.csv')
        demographics = pd.read_csv(self.project_path + '/raw_data/demographics.csv')
        clinical_data = pd.read_csv(self.project_path + '/raw_data/clinical_data.csv')

        return bill_id, bill_amount, demographics, clinical_data

    def clean_up_data(self, demographics, clinical_data):

        # Clean up in medical_history_3. Change 'Yes' to 1, 'No' to 0
        clinical_data['medical_history_3'] = clinical_data['medical_history_3'].replace(['Yes'], 1)
        clinical_data['medical_history_3'] = clinical_data['medical_history_3'].replace(['No'], 0)
        clinical_data['medical_history_3'] = clinical_data['medical_history_3'].astype(float)

        # Change m to Male, f to Female. Then Male = 1, female = 0
        demographics['gender'] = demographics['gender'].replace(['m'], 'Male')
        demographics['gender'] = demographics['gender'].replace(['f'], 'Female')
        demographics.loc[demographics['gender'] == 'Male', 'gender'] = 1
        demographics.loc[demographics['gender'] == 'Female', 'gender'] = 0
        demographics['gender'] = demographics['gender'].astype(float)

        # Change race value "India" to "Indian", "chinese" to "Chinese"
        demographics.loc[demographics['race'] == 'India','race'] = 'Indian'
        demographics.loc[demographics['race'] == 'chinese', 'race'] = 'Chinese'

        return demographics, clinical_data

    def join_data(self):

        # Transform and join the data to get de-normalised (flattened) data ready for analysis

        bill_id, bill_amount, demographics, clinical_data = self.read_data()
        demographics, clinical_data = self.clean_up_data(demographics, clinical_data)
        clinical_data = clinical_data.rename(columns={'id': 'patient_id'})

        bill_info = pd.merge(bill_id, bill_amount, how='left',on=['bill_id'])

        bills_of_patients = bill_info.groupby(['patient_id','date_of_admission'], as_index=False)[['amount']].sum()

        raw_data = pd.merge(clinical_data, bills_of_patients, how='left', on=['patient_id','date_of_admission'])

        raw_data = pd.merge(raw_data, demographics, how='left',on=['patient_id'])

        return raw_data

    def check_nulls(self):

        # To check nulls in various features

        raw_data = self.join_data()
        df_nulls = raw_data.isnull().sum()

        return df_nulls

    def fill_missing_values(self):

        raw_data = self.join_data()

        cols_missing_vals = ['medical_history_2','medical_history_5']

        for col_name in cols_missing_vals:
            df_description = raw_data[col_name].describe()
            number_of_nulls = sum(raw_data[col_name].isnull())

            mean_of_data = float(df_description.ix['mean'])
            number_of_ones = round(mean_of_data * number_of_nulls)

            list_of_1 = [1] * int(number_of_ones)
            missing_vals_list = [0] * int(number_of_nulls-number_of_ones)
            missing_vals_list.extend(list_of_1)

            random.shuffle(missing_vals_list)

            raw_data.loc[raw_data[col_name].isnull(), col_name] = missing_vals_list

        return raw_data


# td = TransformData()
# td.fill_missing_values()
