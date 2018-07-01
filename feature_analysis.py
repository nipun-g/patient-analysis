import pandas as pd
import seaborn as sns
from read_and_transform_data import TransformData
import matplotlib.pyplot as plt


class FeatureAnalysis:

    def __init__(self):
        self.get_data = TransformData()
        self.raw_data = self.get_data.join_data()
        self.df_data = self.get_data.fill_missing_values()

    def feature_corrs(self):

        # Select all the numerical columns of the raw data
        numerical_raw_data = self.raw_data[['medical_history_1', 'medical_history_2', 'medical_history_3',
                                            'medical_history_4', 'medical_history_5', 'medical_history_6',
                                            'medical_history_7', 'preop_medication_1', 'preop_medication_2',
                                            'preop_medication_3', 'preop_medication_4', 'preop_medication_5',
                                            'preop_medication_6', 'symptom_1', 'symptom_2', 'symptom_3',
                                            'symptom_4', 'symptom_5', 'lab_result_1', 'lab_result_2',
                                            'lab_result_3', 'weight', 'height', 'amount', 'gender']]

        feature_corr = numerical_raw_data.corr()

        corr_medical_history_2 = feature_corr['medical_history_2']
        corr_medical_history_2[corr_medical_history_2 == 1] = 0 # change the self correlation from 1 to 0

        corr_medical_history_5 = feature_corr['medical_history_5']
        corr_medical_history_5[corr_medical_history_5 == 1] = 0 # change the self correlation from 1 to 0

        min_max_corr_med_history_2 = [corr_medical_history_2.min(), corr_medical_history_2.max()]
        min_max_corr_med_history_5 = [corr_medical_history_5.min(), corr_medical_history_5.max()]

        return min_max_corr_med_history_2, min_max_corr_med_history_5

    def race_var_with_med_history(self):

        # Explore the relation between "race" variable and "medical_history_2"

        # g2 = sns.barplot(x="race", y="medical_history_2", data=self.raw_data)
        # g2 = g2.set_ylabel("medical_history_2")

        df_aux2 = self.raw_data[['race','medical_history_2']]
        df_race_med_history_2 = df_aux2.groupby('race', as_index=False)['medical_history_2'].agg(['sum','count','mean']).reset_index()

        # Explore the relation between "race" variable and "medical_history_5"

        # g5 = sns.barplot(x="race", y="medical_history_5", data=self.raw_data)
        # g5 = g5.set_ylabel("medical_history_5")

        df_aux5 = self.raw_data[['race','medical_history_5']]
        df_race_med_history_5 = df_aux5.groupby('race', as_index=False)['medical_history_5'].agg(['sum','count','mean']).reset_index()

        return df_race_med_history_2, df_race_med_history_5

    def resident_status_with_med_history(self):
        # Explore the relation between "resident status" variable and "medical_history_2"

        # g2 = sns.barplot(x="resident_status", y="medical_history_2", data=self.raw_data)
        # g2 = g2.set_ylabel("medical_history_2")

        df_aux2 = self.raw_data[['resident_status', 'medical_history_2']]
        df_res_status_med_history_2 = df_aux2.groupby('resident_status', as_index=False)['medical_history_2'].agg(
            ['sum', 'count', 'mean']).reset_index()

        # Explore the relation between "resident_status" variable and "medical_history_5"

        # g5 = sns.barplot(x="resident_status", y="medical_history_5", data=self.raw_data)
        # g5 = g5.set_ylabel("medical_history_5")

        df_aux5 = self.raw_data[['resident_status', 'medical_history_5']]
        df_res_status_med_history_5 = df_aux5.groupby('resident_status', as_index=False)['medical_history_5'].agg(
            ['sum', 'count', 'mean']).reset_index()

        return df_res_status_med_history_2, df_res_status_med_history_5

    def mod_feature_corrs(self):

        # Select all the numerical columns of the raw data
        numerical_raw_data = self.df_data[['medical_history_1', 'medical_history_2', 'medical_history_3',
                                            'medical_history_4', 'medical_history_5', 'medical_history_6',
                                            'medical_history_7', 'preop_medication_1', 'preop_medication_2',
                                            'preop_medication_3', 'preop_medication_4', 'preop_medication_5',
                                            'preop_medication_6', 'symptom_1', 'symptom_2', 'symptom_3',
                                            'symptom_4', 'symptom_5', 'lab_result_1', 'lab_result_2',
                                            'lab_result_3', 'weight', 'height', 'amount', 'gender']]

        feature_corrs = numerical_raw_data.astype(float).corr()

        return feature_corrs


