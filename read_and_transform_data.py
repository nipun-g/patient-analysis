
import pandas as pd
import os
import sys


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

    def clean_up_data(self, clinical_data):

        # Clean up the data in DF clinical_data. Change 'Yes' to 1, 'No' to 0

        clinical_data['medical_history_3'] = clinical_data['medical_history_3'].replace(['Yes'], 1)
        clinical_data['medical_history_3'] = clinical_data['medical_history_3'].replace(['No'], 0)

        return clinical_data

    def join_data(self):

        # Transform and join the data to get de-normalised (flattened) data ready for analysis

        bill_id, bill_amount, demographics, clinical_data = self.read_data()
        clinical_data = self.clean_up_data(clinical_data)
        clinical_data = clinical_data.rename(columns={'id': 'patient_id'})

        bill_info = pd.merge(bill_id, bill_amount, how='left',on=['bill_id'])

        bills_of_patients = bill_info.groupby(['patient_id','date_of_admission'], as_index=False)[['amount']].sum()

        raw_data = pd.merge(clinical_data, bills_of_patients, how='left', on=['patient_id','date_of_admission'])

        raw_data = pd.merge(raw_data, demographics, how='left',on=['patient_id'])

        return raw_data

