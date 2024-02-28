import streamlit as st
import pandas as pd
import pickle

st.write("""
# Advertising Sales Prediction App

This app predicts the **Sales** for type of advertising stratergy!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    tv = st.sidebar.slider('TV', 0.7, 296.0, 10.0)
    radio = st.sidebar.slider('Radio', 0.0, 50.0, 15.0)
    newspaper = st.sidebar.slider('Newspaper', 0.3, 114.0, 15.0)
    data = {'TV': tv,
            'Radio': radio,
            'Newspaper': newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("AdvertisingSales_model.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
