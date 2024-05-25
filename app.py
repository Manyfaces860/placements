import streamlit as st
import pickle
import pandas as pd

scaler = pickle.load(open('scaler.pkl' , 'rb'))

model = pickle.load(open('model.pkl' , 'rb'))

st.header('Placement Estimator')

cgpa = st.number_input('CGPA')
iq = st.number_input('IQ')

if st.button('tell me'):
    df = pd.DataFrame({
    'cgpa': [cgpa],
    'iq': [iq]
    })
    df = scaler.transform(df)
    value = model.predict(df)
    if value == 0:
        st.title('Not placed')
    else:
        st.title('You Will be Placed!')
    



