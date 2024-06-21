import pandas as pd
import streamlit as st
import joblib


x_numeric = {'host_total_listings_count': 0,'latitude': 0, 'longitude': 0, 'accommodates': 0, 'bathrooms': 0, 'bedrooms': 0, 'beds': 0, 
             'extra_people': 0, 'minimum_nights': 0, 'number_of_reviews': 0, 'year': 0, 'month': 0, 'n_amenities': 0}

x_tf = {'host_is_superhost': 0, 'host_identity_verified': 0, 'instant_bookable': 0}

x_lists = {'property_type': ['Apartment', 'Bed and breakfast', 'Condominium', 'Guest suite', 'Guesthouse', 'Hostel', 'House', 'Loft', 'Other', 'Serviced apartment'],
            'room_type': ['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'],
            'bed type': ['Airbed', 'Couch', 'Futon', 'Pull-out Sofa', 'Real Bed'],
            'cancelation_policy': ['flexible', 'moderate', 'strict']
            }


# PredictionModel = joblib.load('PredictingModel.joblib')