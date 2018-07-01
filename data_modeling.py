import pandas as pd
from feature_engineering import FeatureEngineering
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import statsmodels.api as sm

class DataModeling:

    def __init__(self):

        fe = FeatureEngineering()
        fe.feature_age()
        fe.feature_days_admitted()
        fe.feature_total_medical_history()
        fe.feature_total_preop_medication()
        fe.feature_total_symptoms()
        fe.feature_lab_results_ratios()

        df_data = self.__drop_cols(fe.df_data)
        self.df_data = self.__get_dummy_vars(df_data)

    def __drop_cols(self, df_data):

        # Drop columns that are not used in data modeling

        columns = ['patient_id', 'date_of_admission', 'date_of_discharge', 'date_of_birth']
        df_data = df_data.drop(columns, axis=1)

        return df_data

    def __get_dummy_vars(self, df_data):

        # Get dummy variables from all categorical features

        df_data = pd.get_dummies(df_data, columns=['race'])
        df_data = pd.get_dummies(df_data, columns=['resident_status'])

        return df_data


    def linear_regression(self):

        # Run linear regression model

        df_data = self.df_data.drop(['race_Others','resident_status_Singaporean'], axis=1)  # To avoid multi-collinearity trap

        y = df_data['amount']
        X = df_data.drop(['amount'],axis=1)

        X_normalized = (X - X.mean()) / (2 * X.std())

        X_normalized = sm.add_constant(X_normalized)
        model = sm.OLS(y,X_normalized).fit()
        print(model.summary())

    def random_forests(self):

        # Run random forests model

        df_data = self.df_data.drop(['race_Others', 'resident_status_Singaporean'],axis=1)  # To avoid multi-collinearity trap

        y = df_data['amount']
        X = df_data.drop(['amount'],axis=1)

        rfr = RandomForestRegressor(n_estimators=1000, random_state=2)
        rfr.fit(X, y)
        y_pred = rfr.predict(X)

        errors = abs(y_pred - y)
        mape = 100 * np.mean(errors / y)
        accuracy = 100 - mape

        # Get numerical feature importances
        importances = list(rfr.feature_importances_)

        # List of tuples
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(list(X.columns), importances)]

        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

        feature_importances = pd.DataFrame(feature_importances)

        print('Accuracy of model: ' + str(accuracy) + ' %')
        print('Feature Importances: ' + str(feature_importances))
