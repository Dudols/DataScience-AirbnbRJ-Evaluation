import pandas as pd
import pathlib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import joblib


def lims(col):
    '''Calculates the superior and inferior limit to identify outliers based on the third and first quartile, respectfully, times the amplitude which is the difference between the quartiles.

    Parameters:
        col (str): pd.DataFrame[column]

    Returns:
        inferior_limit (float): first quartile times amplitude 
        superior_limit (float): third quartile times amplitude
    '''
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    amplitude = q3 - q1
    superior_limit = q3 + 1.5 * amplitude
    inferior_limit = q1 - 1.5 * amplitude
    return inferior_limit, superior_limit

def remove_outliers(df, col_name):
    '''Removes the outliers from the DataFrame taking in consideration the limits defined in the lims function

    Parameters:
        df (str): pd.Dataframe
        col_name (str): pd.DataFrame column name as string 

    Returns:
        df (pd.DataFrame): pd.DataFrame without outliers
        rows_removed (int): number of rows removed from the DataFrame
    '''
    row_count = df.shape[0]
    inferior_limit, superior_limit = lims(df[col_name])
    df = df.loc[(df[col_name] >= inferior_limit) & (df[col_name] <= superior_limit), :]
    row_count_new = df.shape[0]
    rows_removed = row_count - row_count_new
    return df, rows_removed


# Importing data adding year and month column separetedly
months = {
    "jan": 1,
    "fev": 2,
    "mar": 3,
    "abr": 4,
    "mai": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8,
    "set": 9,
    "out": 10,
    "nov": 11,
    "dez": 12
}

path_datasets = pathlib.Path('dataset')
data = []

for dataset in path_datasets.iterdir():
    year = dataset.name[-8:]
    year = int(year.replace('.csv', ''))
    month_name = dataset.name[:3]
    month = months[month_name]
    df = pd.read_csv(path_datasets / dataset.name)
    df['year'] = year
    df['month'] = month
    data.append(df)

airbnb_df = pd.concat(data)

# Setting collumns that will be used as features for the machine learning model.
ml_features = ['host_is_superhost', 'host_identity_verified', 'latitude', 'longitude',
               'accommodates', 'bathrooms', 'bedrooms', 'beds', 'extra_people',
               'instant_bookable', 'year', 'month', 'property_type', 'room_type',
               'bed_type', 'cancellation_policy', 'price']

airbnb_df = airbnb_df[ml_features]

# Removing missing values
airbnb_df = airbnb_df.dropna()

# Converting data to 32 bit
airbnb_df['price'] = airbnb_df['price'].str.replace('$', '')
airbnb_df['price'] = airbnb_df['price'].str.replace(',', '')
airbnb_df['price'] = airbnb_df['price'].astype(np.float32)
airbnb_df['extra_people'] = airbnb_df['extra_people'].str.replace('$', '')
airbnb_df['extra_people'] = airbnb_df['extra_people'].str.replace(',', '')
airbnb_df['extra_people'] = airbnb_df['extra_people'].astype(np.float32)
airbnb_df[['bedrooms', 'beds']] = airbnb_df[['bedrooms', 'beds']].astype(np.int32)
float64_cols = list(airbnb_df.select_dtypes(include='float64'))
airbnb_df[float64_cols] = airbnb_df[float64_cols].astype(np.float32)
int64_cols = list(airbnb_df.select_dtypes(include='int64'))
airbnb_df[int64_cols] = airbnb_df[int64_cols].astype(np.int32)

# Processing features
airbnb_df, rows_removed = remove_outliers(airbnb_df, 'price')
airbnb_df, rows_removed = remove_outliers(airbnb_df, 'extra_people')

property_types = airbnb_df['property_type'].value_counts()
min_value = int((airbnb_df['property_type'] == 'Other').sum())
grouped_columns = []

for property_type in property_types.index:
    if property_types[property_type] < min_value:
        grouped_columns.append(property_type)
for type_removed in grouped_columns:
    airbnb_df.loc[airbnb_df['property_type'] == type_removed, 'property_type'] = 'Other'

cancellation_policys = airbnb_df['cancellation_policy'].value_counts()
cancellation_policys_kept = ['flexible', 'moderate', 'strict']
grouped_categories = []

for cancellation_policy in cancellation_policys.index:
    if cancellation_policy not in cancellation_policys_kept:
        grouped_categories.append(cancellation_policy)
for type_removed in grouped_categories:
    if 'flexible' in type_removed:
        airbnb_df.loc[airbnb_df['cancellation_policy'] == type_removed, 'cancellation_policy'] = 'flexible'
    if 'moderate' in type_removed:
        airbnb_df.loc[airbnb_df['cancellation_policy'] == type_removed, 'cancellation_policy'] = 'moderate'
    if 'strict' in type_removed:
        airbnb_df.loc[airbnb_df['cancellation_policy'] == type_removed, 'cancellation_policy'] = 'strict'

# Encoding True or False features and creating dummy variables for categoric features
tf_features = ['host_is_superhost', 'host_identity_verified', 'instant_bookable']
categoric_features = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']

for col in tf_features:
    airbnb_df.loc[airbnb_df[col]=='t', col] = 1
    airbnb_df.loc[airbnb_df[col]=='f', col] = 0

airbnb_df = pd.get_dummies(data = airbnb_df, columns=categoric_features, dtype=(np.int32))

# Set machine learning model
et_model = ExtraTreesRegressor()

# Set variables for ml model
y = airbnb_df['price']
X = airbnb_df.drop('price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
et_model.fit(X_train, y_train)

# Test model
prediction = et_model.predict(X_test)

# Save clean database
X['price'] = y
X.to_csv(r'clean_data_airbnb.csv')

# Save trained ml model
joblib.dump(et_model, 'PredictingModel.joblib')