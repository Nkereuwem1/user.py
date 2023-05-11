import pandas as pd
import numpy as np
import streamlit as st

drugs = pd.read_csv("./dataset/Drug_Consumption.csv")

# drop unnecessary columns
drugs = drugs.drop(['ID', 'Choc', 'Semer'], axis=1)

# encode categorical features
def encode_feature(df, column):
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    return df

drugs = encode_feature(drugs, 'Gender')
stimulants = ['Alcohol', 'Amyl', 'Amphet', 'Benzos', 'Caff', 'Cannabis', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'VSA']
for column in stimulants:
    drugs = encode_feature(drugs, column)

# combine cocaine and crack cocaine usage into one feature
cocaine_df = drugs.copy()
cocaine_df['coke_user'] = cocaine_df['Coke'].apply(lambda x: 0.5 if x not in [0,1] else 0)
cocaine_df['crack_user'] = cocaine_df['Coke'].apply(lambda x: 0.5 if x not in [0,1] else 0)
cocaine_df['both_user'] = cocaine_df[['coke_user', 'crack_user']].iloc[:].sum(axis=1)
cocaine_df['Cocaine_User'] = cocaine_df['both_user'].apply(lambda x: 1 if x > 0 else 0)
cocaine_df = cocaine_df.drop(['coke_user', 'crack_user', 'both_user' ], axis=1)

# create function for preprocessing inputs
def preprocess_input(df):
    df = df.copy()
    df = encode_feature(df, 'Gender')
    for column in stimulants:
    return df

# create function for making predictions
def predict(df):
    # preprocess input data
    df = preprocess_input(df)
    # make predictions
    cocaine_pred = np.random.choice([0,1])
    meth_pred = np.random.choice([0,1])
    heroin_pred = np.random.choice([0,1])
    nico_pred = np.random.choice([0,1])
    # return prediction results
    results = {
        'Cocaine': cocaine_pred,
        'Methamphetamine': meth_pred,
        'Heroin': heroin_pred,
        'Nicotine': nico_pred
    }
    return results

# create Streamlit app
def app():
    st.title('Drug Use Prediction App')
    st.write('Enter your information to check if you are likely to use cocaine, methamphetamine, heroin, or nicotine.')

    # create input fields
    age = st.slider('Age', 18, 65, 25)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    alcohol = st.slider('Alcohol Use', 0, 6, 0)

    # create input dictionary
    results = {
        'Cocaine': cocaine_pred,
        'Methamphetamine': meth_pred,
        'Heroin': heroin_pred,
        'Nicotine': nico_pred
    }
    input_dict = {
        'Age': age,
        'Gender': gender,
        'Alcohol': alcohol
    }

    # make prediction
    prediction = predict(pd.DataFrame(input_dict, index=[0]))

    # show prediction results
    st.write('Stimulant Use Prediction Results:')
    for drug, value in results.items():
        if value == 1:
            st.write(f'You are likely to use {drug}.')
        else:
            st.write(f'You are not likely to use {drug}.')
