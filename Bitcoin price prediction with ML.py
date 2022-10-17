#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yfinance


# In[10]:


pip install AutoTs


# In[1]:


pip install datetime


# In[4]:


pip install plotly


# In[1]:


import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
today=date.today()
d1=today.strftime("%Y-%m-%d")
end_date=d1
d2=date.today() - timedelta(days=730)
d2=d2.strftime("%Y-%m-%d")
start_date= d2

data=yf.download('BTC-USD',start=start_date, end=end_date, progress=False)
data["Date"]= data.index
data= data[["Date","Open","High","Low","Close","Adj Close","Volume"]]
data.reset_index(drop=True, inplace= True)
print(data.head())


# In[2]:


data.shape


# In[3]:



import plotly.graph_objects as go
figure=go.Figure(data=[go.Candlestick(x=data["Date"],open=data["Open"],high=data["High"],low=data["Low"],close=data["Close"])])
figure.update_layout(title="Bitcoin price analysis",xaxis_rangeslider_visible=False)
figure.show()


# In[4]:


correlation=data.corr()
print(correlation["Close"].sort_values(ascending=False))


# In[ ]:


from autots import AutoTs
model= AutoTs(forecast_length=30,frequency='infer',ensemble='simple')
model=model.fit(data, date_col='Date',value_col='Close',id_col=None)
prediction=model.predict()
forecast= prediction.forecast
print(forecast)


# In[ ]:




