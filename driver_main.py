from read_and_transform_data import TransformData
from feature_analysis import FeatureAnalysis
from data_modeling import DataModeling

# Run the following code snippets to analyse data

# See the raw joined data
td = TransformData()
raw_data = td.join_data()


# Check the missing data
td = TransformData()
df_nulls = td.check_nulls()


# See the correlation between raw features
fa = FeatureAnalysis()
min_max_corr_med_history_2, min_max_corr_med_history_5 = fa.feature_corrs()


# Sum, count, mean of medical history 2 & 5 w.r.t race
fa = FeatureAnalysis()
df_race_med_history_2, df_race_med_history_5 = fa.race_var_with_med_history()


# Sum, count, mean of medical history 2 & 5 w.r.t resident_status
fa = FeatureAnalysis()
df_res_status_med_history_2, df_res_status_med_history_5 = fa.resident_status_with_med_history()


# Fill in the missing values
td = TransformData()
df_data = td.fill_missing_values()


# See correlations among numerical variables - for feature analysis
fa = FeatureAnalysis()
feature_corrs = fa.mod_feature_corrs()


# Run linear regression algorithm
dm = DataModeling()
dm.linear_regression()


# Run random forests algorithm
dm = DataModeling()
dm.random_forests()

