import pandas as pd
import streamlit as st
import joblib


x_numeric = {'latitude': 0, 'longitude': 0, 'accommodates': 0, 'bathrooms': 0, 'bedrooms': 0, 'beds': 0, 
             'extra_people': 0, 'year': 0, 'month': 0}

x_tf = {'host_is_superhost': 0, 'host_identity_verified': 0, 'instant_bookable': 0}

x_lists = {'property_type': ['Apartment', 'Bed and breakfast', 'Condominium', 'Guest suite', 'Guesthouse', 'Hostel', 'House', 'Loft', 'Other', 'Serviced apartment'],
            'room_type': ['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'],
            'bed type': ['Airbed', 'Couch', 'Futon', 'Pull-out Sofa', 'Real Bed'],
            'cancelation_policy': ['flexible', 'moderate', 'strict']
            }

aux_dict = {}
for item in x_lists:
    for value in x_lists[item]:
        aux_dict[f'{item}_{value}'] = 0

for item in x_numeric:
    if item == 'latitude' or item == 'longitude':
        value = st.number_input(f'{item}', step=0.00001, value = 0.0, format='%.5f')
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
    aux_dict[f'{item}_{value}'] = value

predict = st.button('Predict Accomodation Price')

if predict:
    aux_dict.update(x_numeric)
    aux_dict.update(x_tf)
    x_values = pd.DataFrame(aux_dict, index=[0])
    PredictionModel = joblib.load('PredictingModel.joblib')
    price = PredictionModel.predict(x_values)
    st.write(f'${price[0]}')