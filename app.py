import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("./placement.csv")
df = df.drop("Unnamed: 0", axis=1)
X = df.iloc[:, 0:2]
Y = df.iloc[:, -1]
x_train , x_test , y_train , y_test = train_test_split(X, Y , test_size=0.1,train_size=0.9)
scaler = StandardScaler()
scaler.fit_transform(x_train)

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
    



