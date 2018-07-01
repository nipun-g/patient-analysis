import pandas as pd
from read_and_transform_data import TransformData

class FeatureEngineering:

    def __init__(self):

        td = TransformData()
        self.df_data = td.fill_missing_values()

    def feature_age(self):

        # Create age as a feature

        self.df_data['date_of_admission'] = pd.to_datetime(self.df_data['date_of_admission'], format='%Y-%m-%d')
        self.df_data['date_of_birth'] = pd.to_datetime(self.df_data['date_of_birth'], format='%Y-%m-%d')

        diff_in_years = (self.df_data['date_of_admission'] - self.df_data['date_of_birth'])/365

        age = diff_in_years.dt.days
        self.df_data['age'] = age

    def feature_days_admitted(self):

        # Create number of days admitted as a feature

        self.df_data['date_of_discharge'] = pd.to_datetime(self.df_data['date_of_discharge'], format='%Y-%m-%d')

        diff_in_days = self.df_data['date_of_discharge'] - self.df_data['date_of_admission']

        days_admitted = diff_in_days.dt.days
        self.df_data['days_admitted'] = days_admitted

    def feature_total_medical_history(self):

        # Create total number of medical histories of a patient

        col_list = ['medical_history_1', 'medical_history_2', 'medical_history_3', 'medical_history_4',
                    'medical_history_5', 'medical_history_6', 'medical_history_7']
        self.df_data['total_med_history'] = self.df_data[col_list].sum(axis=1)

    def feature_total_preop_medication(self):

        # Create total number of preop medications of a patient

        col_list = ['preop_medication_1', 'preop_medication_2', 'preop_medication_3', 'preop_medication_4',
                    'preop_medication_5', 'preop_medication_6']
        self.df_data['total_preop_med'] = self.df_data[col_list].sum(axis=1)

    def feature_total_symptoms(self):

        # Create total number of symptoms of a patient

        col_list = ['symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5']
        self.df_data['total_symptoms'] = self.df_data[col_list].sum(axis=1)

    def feature_lab_results_ratios(self):

        # Create ratios of lab results of a patient

        self.df_data['lab_ratio_1_2'] = self.df_data['lab_result_1'] / self.df_data['lab_result_2']
        self.df_data['lab_ratio_1_3'] = self.df_data['lab_result_1'] / self.df_data['lab_result_3']
        self.df_data['lab_ratio_2_3'] = self.df_data['lab_result_2'] / self.df_data['lab_result_3']
