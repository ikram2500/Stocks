import numpy as np 
import pandas as pd 
import yfinance as yf 
from keras.models import load_model 
import streamlit as st 
import matplotlib.pyplot as plt 

model = load_model('D:\DataScience\DS\Stocks\Stock Predictions Model.keras')

st.header('Stock Markete Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scler = MinMaxScaler(feature_range=(0,1))

pass_100_days = data_train.tail(100)    
data_test = pd.concat((pass_100_days, data_test), ignore_index=True)
data_test_scale = scler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data_test['Close'].rolling(window=50).mean()
fig1 = plt.figure(figsize=(12,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data_test['Close'], 'g', label='Price')

plt.show()
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data_test['Close'].rolling(window=100).mean()
fig2 = plt.figure(figsize=(12,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b' , label='MA100')
plt.plot(data_test['Close'], 'g', label='Price')
plt.show()
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA50 vs MA100')
ma_200_days = data_test['Close'].rolling(window= 200).mean()
fig3 = plt.figure(figsize=(12,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(ma_200_days, 'y', label='MA200')
plt.plot(data_test['Close'], 'g', label='Price')
plt.show()
plt.legend()
st.pyplot(fig3)


X = []
y = []

for i in range(100, data_test_scale.shape[0]):
    X.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

X, y = np.array(X), np.array(y)

predict = model.predict(X)

scale = 1/scler.scale_

predict = predict*scale
y = y*scale 

st.subheader('Predictions')

st.subheader('Price vs Predictions')

fig4 = plt.figure(figsize=(12,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label='Predicted Price')
plt.xlabel('time')
plt.ylabel('Price')
plt.show()
plt.legend()
st.pyplot(fig4)
