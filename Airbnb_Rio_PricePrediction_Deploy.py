import pandas as pd
import streamlit as st
import joblib

# Columns list in order as of the ml model was trained.
x_columns = ['host_is_superhost', 'host_identity_verified', 'latitude', 'longitude',
       'accommodates', 'bathrooms', 'bedrooms', 'beds', 'extra_people',
       'instant_bookable', 'year', 'month', 'property_type_Apartment',
       'property_type_Bed and breakfast', 'property_type_Condominium',
       'property_type_Guest suite', 'property_type_Guesthouse',
       'property_type_Hostel', 'property_type_House', 'property_type_Loft',
       'property_type_Other', 'property_type_Serviced apartment',
       'room_type_Entire home/apt', 'room_type_Hotel room',
       'room_type_Private room', 'room_type_Shared room', 'bed_type_Airbed',
       'bed_type_Couch', 'bed_type_Futon', 'bed_type_Pull-out Sofa',
       'bed_type_Real Bed', 'cancellation_policy_flexible',
       'cancellation_policy_moderate', 'cancellation_policy_strict']

# Separate different types of features
x_numeric = {'latitude': 0, 'longitude': 0, 'accommodates': 0, 'bathrooms': 0, 'bedrooms': 0, 'beds': 0, 
             'extra_people': 0, 'year': 0, 'month': 0}

x_tf = {'host_is_superhost': 0, 'host_identity_verified': 0, 'instant_bookable': 0}

x_lists = {'property_type': ['Apartment', 'Bed and breakfast', 'Condominium', 'Guest suite',
            'Guesthouse', 'Hostel', 'House', 'Loft', 'Other', 'Serviced apartment'],
            'room_type': ['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'],
            'bed_type': ['Airbed', 'Couch', 'Futon', 'Pull-out Sofa', 'Real Bed'],
            'cancellation_policy': ['flexible', 'moderate', 'strict']
            }

# Create buttons for all types of features with the correct processing for each
aux_dict = {}
for item in x_lists:
    for value in x_lists[item]:
        aux_dict[f'{item}_{value}'] = 0

for item in x_numeric:
    if item == 'latitude' or item == 'longitude':
        value = st.number_input(f'{item}', step=0.001, value = 0.0, format='%.3f')
        st.write('Tip: check Google Maps for neighbourhood approximation on latitude and longitude')
    elif item == 'extra_people':
        value = st.number_input(f'{item}', step=0.01, value = 0.0)
    else:
        value = st.number_input(f'{item}', step=1, value=0)
    x_numeric[item] = value

for item in x_tf:
    value = st.selectbox(f'{item}', ('Yes', 'No'))
    if value == 'Yes':
        x_tf[item] = 1
    else:
        x_tf[item] = 0

for item in x_lists:
    value = st.selectbox(f'{item}', x_lists[item])
    aux_dict[f'{item}_{value}'] = 1

# Set button for initializing prediction
predict = st.button('Predict Accomodation Price')
if predict:
    aux_dict.update(x_numeric)
    aux_dict.update(x_tf)
    x_values = pd.DataFrame(aux_dict, index=[0])
    x_values = x_values[x_columns]
    PredictionModel = joblib.load('PredictingModel.joblib')
    price = PredictionModel.predict(x_values)
    st.write(f'The price on average is R${price[0]}')