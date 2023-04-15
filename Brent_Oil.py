import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

#!pip install prophet


import streamlit as st

from PIL import Image

# command to run on command prompt 'streamlit run Brent_Oil.py'



df = pd.read_csv('Brent Oil Price_MacroV2.csv')

#st.set_page_config(page_title='Oil Price Forecasting', layout="wide")
st.set_page_config(page_title="Oil Price Forecasting", layout="wide", page_icon="':money_with_wings:'", initial_sidebar_state="expanded")
st.title('Oil Price Forecasting using Exogenous variables')
# Add some descriptive text
st.write("This app uses 3 models to forecast the future prices of crude oil based on 2 exogenous variables: Personal Consumption Expenditure (PCE) & Gross Domestic Product (GDP)")

#image = Image.open('Oil_Price_Trend_.jpg')
##image_ = Image.open('Oil barrel world.jpg')
#
#st.header("Brent Oil Price Trend")
#st.subheader("Historical Price vs. events")
##st.write("Historical Price vs. events")
##st.text("Historical Price vs. events")
#st.image(image, use_column_width=True)
#
#image = Image.open('Oil_Price_Trend.jpg')

#Import Images
image1 = Image.open('Oil_Price_Trend1.jpg')
image2 = Image.open('Oil barrel dollar.jpg')


# Display the images
col1, col2 = st.columns(2)

with col1:
    st.image(image1, use_column_width=True)

with col2:
    st.image(image2, use_column_width=True)





## LINE CHART
#st.line_chart(df['oil_price'])
#st.line_chart(df['GDP'])
#st.line_chart(df['PCE'])

# SLIDER
#ts_in = st.slider('date', min_value=0, max_value=22, step=1, value=1)





# Upload data
data = pd.read_csv("Brent Oil Price_MacroV2.csv")
data["date"] = pd.to_datetime(data["date"])
data.set_index("date", inplace=True)


# COLUMNS
# Show descriptive statistics
left_column, right_column = st.columns(2)

with left_column:
    st.subheader('Data Table')
    st.write(data)

with right_column:
    st.subheader('Descriptive Statistics')
    stats = data.describe()
    st.write(stats)




#Select target columns
target_column = st.selectbox('Select target column', data.columns)
#input_variables = st.selectbox('Select date column', data.columns)


# COLUMNS
#Line chart for target variable & exogenous variables
left_column, right_column = st.columns(2)

with left_column:
    st.write("Output Variable:")
    st.line_chart(data[target_column])

with right_column:
    st.write("Input Variables:")
    st.line_chart(data[['PCE', 'GDP']])


# Visualize the variable interaction
st.write('## Variable Interaction')  
# COLUMNS
left_column, center_column, right_column = st.columns(3)
with left_column:
    fig, ax = plt.subplots()
    sns.regplot(x='PCE', y='oil_price', data=df, ax=ax)
    ax.set_title('PCE vs Oil Price')
    ax.set_xlabel('Personal Consumption Expenditures (PCE)')
    ax.set_ylabel('Price of Oil (per barrel)')
    st.pyplot(fig)

with center_column:
    fig, ax = plt.subplots()
    sns.regplot(x='GDP', y='oil_price', data=data, ax=ax)
    ax.set_title('GDP vs Oil Price')
    ax.set_xlabel('Gross Domestic Product (GDP)')
    ax.set_ylabel('Price of Oil (per barrel)')
    st.pyplot(fig)

with right_column:
    fig, ax = plt.subplots()
    sns.regplot(x='GDP', y='PCE', data=data, ax=ax)
    ax.set_title('GDP vs PCE')
    ax.set_xlabel('Gross Domestic Product (GDP)')
    ax.set_ylabel('Personal Consumption Expenditures (PCE)')
    st.pyplot(fig)



data1 = data.dropna()

cols = ['oil_price','PCE','GDP']

for col in cols:
    data1[col] = data1[col].astype(int)
data1.info()

# MULTIPLECHOICE: Selection of forecasting model
model_name = st.sidebar.selectbox('Select a Forecasting Model', ["ARIMAX", "SARIMAX", "PROPHET", "ENSEMBLE"])

if model_name == "SARIMAX":
    # Define the exogenous variables
    exog = data1[['PCE', 'GDP']]
    
    # Fit SARIMAX model with exogenous variables
    model = sm.tsa.statespace.SARIMAX(data1['oil_price'], exog=exog, order=(1, 0, 0), seasonal_order=(1, 1, 1, 12),enforce_invertibility=False)
    results = model.fit()
    
    exog_forecast = data[278:][['PCE', 'GDP']]
    
    #Forecasr
    fcast = results.predict(len(data1),len(data1)+20,exog=exog_forecast).rename('SARIMAX(1,0,0)(1,1,1,12) Forecast')
    
    # Plot forecast
    st.title('SARIMAX FORECAST')
#    st.line_chart(fcast)
    fig_forecast = px.line(fcast, title='Oil Price Forecast')
    fig_forecast.update_traces(line=dict(color='white'))
    fig_forecast.update_layout(title_font_size=30)
    st.plotly_chart(fig_forecast)
    
#    st.write('Forecast Components')
#    # Decompose predictions into trend and seasonal components
#    decomposition = seasonal_decompose(fcast, model='additive')
#    
#    # Plot original data, trend, and seasonality
#    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
#    axs[0].plot(data)
#    axs[0].set_title('Original Data')
#    axs[1].plot(decomposition.trend)
#    axs[1].set_title('Trend Component')
#    axs[2].plot(decomposition.seasonal)
#    axs[2].set_title('Seasonal Component')
#    plt.tight_layout()
#    plt.show()



elif model_name == "ARIMAX":

      # Define the exogenous variables
    exog = data1[['PCE', 'GDP']]
    
    # Fit ARIMAX model with exogenous variables
    model = ARIMA(data1['oil_price'], exog=exog, order=(1, 1, 1))
    results = model.fit()
    
    exog_forecast = data[278:][['PCE', 'GDP']]
    
    #Forecasr
    fcast = results.predict(len(data1),len(data1)+20,exog=exog_forecast).rename('ARIMAX(1,1,1) Forecast')
    
    # Plot forecast
    st.title('ARIMAX FORECAST')
#    st.line_chart(fcast)
    fig_forecast = px.line(fcast, title='Oil Price Forecast')
    fig_forecast.update_traces(line=dict(color='white'))
    fig_forecast.update_layout(title_font_size=30)
    st.plotly_chart(fig_forecast)



elif model_name == "PROPHET":
    data = pd.read_csv("Brent Oil Price_MacroV2.csv")
    data["date"] = pd.to_datetime(data["date"])
    # Creating Prophet model
    m = Prophet()
    
    # Adding exogenous variables
    m.add_regressor('PCE')
    m.add_regressor('GDP')
    
    # Renaming columns to fit Prophet's requirements
    data = data.rename(columns={'date': 'ds', 'oil_price': 'y', 'PCE': 'PCE', 'GDP': 'GDP'})
    
    # Fitting the model
    m.fit(data) 
    
    # Creating future dataframe with exogenous variables for forecast
    future_df = data[['PCE', 'GDP']].tail(21).reset_index(drop=True)
    future_df['ds'] = pd.date_range(start='2023-03-01', end='2024-11-01', freq='MS')
    
    # Making predictions
    forecast = m.predict(future_df)
    # Extracting relevant columns from forecast
#    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # Plot forecast
    st.title('PROPHET FORECAST')
    #st.line_chart(forecast)
#    st.line_chart(forecast[['yhat']])
    fig_forecast = px.line(forecast, x='ds', y='yhat', title='Oil Price Forecast')
    fig_forecast.update_traces(line=dict(color='white'))
    fig_forecast.update_layout(title_font_size=30)
    st.plotly_chart(fig_forecast)
                                    
#    fig = m.plot(forecast)
#    fig = m.plot_components(forecast)
    st.write('Forecast Components')
    components = m.plot_components(forecast)
    st.write(components)
    
        
#    # Show correlations between variables
#    st.subheader('Correlations')
#    corr = data.corr()
#    fig = px.imshow(corr, color_continuous_scale='RdYlGn', title='Correlation Matrix')
#    fig.update_layout(width=600, height=600)
#    st.plotly_chart(fig)
    

    # Show scatter plot of PCE and GDP against oil prices
    st.subheader('Exogenous Variables')
    fig = px.scatter(df, x='PCE', y='GDP', color='oil_price', title='Oil Price vs. PCE and GDP')
    fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(title_font_size=30)
    st.plotly_chart(fig)
    

elif model_name == "ENSEMBLE":
    data = pd.read_csv("Brent Oil Price_Ensemble FCST.csv")
    data["date"] = pd.to_datetime(data["date"])
   
    st.title('ENSEMBLE FORECAST')
    #st.line_chart(forecast)
#    st.line_chart(forecast[['yhat']])
    fig_forecast = px.line(data, x='date', y='oil_price', title='Oil Price Forecast')
    fig_forecast.update_traces(line=dict(color='white'))
    fig_forecast.update_layout(title_font_size=30)
    st.plotly_chart(fig_forecast)








