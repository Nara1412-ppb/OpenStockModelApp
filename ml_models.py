import pandas as pd
import numpy as np
import pandas_datareader as web
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os





def Tata_motors_stock():
    df = web.DataReader('TATAMOTORS.NS', data_source='yahoo',start=datetime.now()-timedelta(days=3*365),end=datetime.now()).reset_index()
    df1  = web.DataReader('^NSEI',data_source='yahoo',start=datetime.now()-timedelta(days=3*365),end=datetime.now()).reset_index()
    df['Nifty_Index'] = df1['Open']
    #df.isnull().sum()
    df.fillna(method ='ffill' , inplace= True)
    df['Date'] = pd.to_datetime(df.Date)
    y = df.filter(["Open",'Nifty_Index'])[1:].reset_index()
    x = df.drop(["Open",'Nifty_Index'],axis = 1)[:-1]
    New_data = {}
    New_data.update(x.filter(["Close","Volume"]))
    New_data.update(y.filter(['Nifty_Index',"Open"]))
    data = pd.DataFrame(New_data)
    Q1 = data['Volume'].quantile(.30)
    Q3 = data['Volume'].quantile(.70)
    IQR = Q3-Q1
    upper_limit = Q3 + 1.5*IQR
    lower_limit = Q1 - 1.5*IQR
    data1 = data[(data['Volume']>lower_limit) & (data['Volume']<upper_limit)]
    q3 = data1['Nifty_Index'].quantile(.75)
    q1 = data1['Nifty_Index'].quantile(.30)
    iqr = q3-q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    Data_model_use = data1[(data1['Nifty_Index']>lower) & (data1['Nifty_Index']<upper)]
    Y = Data_model_use['Open']
    X = Data_model_use.drop(['Open'],axis = 1)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)
    classifier = LinearRegression()
    classifier.fit(X_train,Y_train)
    try:
        if not os.path.isdir('Job_files'):
            os.mkdir('Job_files')
        model = joblib.load('Job_files/Tata_Stock_model.joblib')
    except Exception as e:
        if not os.path.isdir('Job_files'):
            os.mkdir('Job_files')
        joblib.dump(classifier,'Job_files/Tata_Stock_model.joblib')
        model = joblib.load('Job_files/Tata_Stock_model.joblib')
    return model
def Axis_stock():
    data = web.DataReader('AXISBANK.NS', data_source='yahoo',start = datetime.now() - timedelta(days = 3*365), end = datetime.now()).reset_index()
    data['Date'] = pd.to_datetime(data.Date)
    data.set_index('Date',inplace=True)
    nifty = web.DataReader('^NSEI', data_source='yahoo',start = datetime.now() - timedelta(days = 3*365), end = datetime.now()).reset_index()
    nifty['Date'] = pd.to_datetime(nifty.Date)
    df1 = nifty[['Date','Open']]
    df1 =df1.rename(columns = {'Open':'Nifty_Index'},inplace=False)
    df1.set_index('Date',inplace=True)
    df2 = data.join(df1)
    df2.fillna(method='ffill',inplace = True)
    df3 = df2.filter(['Open','Nifty_Index'])[1:].reset_index()
    df3 = df3.drop(['Date'], axis=1)
    df4 = df2.filter(['Close','Volume'])[:-1].reset_index()
    df4 = df4.drop(['Date'],axis = 1)
    new_data = {}
    new_data.update(df4)
    new_data.update(df3)
    data_frame = pd.DataFrame(new_data)
    data_frame
    q1 = data_frame['Volume'].quantile(.25)
    q2 = data_frame['Volume'].quantile(.69)
    iqr = q2-q1
    upper_limit = q2+1.5*iqr
    lower_limit = q1-1.5*iqr
    data_frame = data_frame[(data_frame['Volume']>lower_limit)&(data_frame['Volume']<upper_limit)]
    q3 = data_frame['Nifty_Index'].quantile(.25)
    q4 = data_frame['Nifty_Index'].quantile(.75)
    IQR = q4-q3
    upper = q4+1.5*IQR
    lower = q3-1.5*IQR
    data_frame = data_frame[(data_frame['Nifty_Index']>lower)&(data_frame['Nifty_Index']<upper)]
    Y = data_frame['Open']
    X = data_frame.drop(['Open'],axis= 1)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,random_state = 0)
    classifier = LinearRegression()
    classifier.fit(X_train,Y_train)
    try:
        if not os.path.isdir('Job_files'):
            os.mkdir('Job_files')
        model = joblib.load('Job_files/Axis_stock_model.joblib')
    except Exception as e:
        if not os.path.isdir('Job_files'):
            os.mkdir('Job_files')
        joblib.dump(classifier,'Job_files/Axis_stock_model.joblib')
        model = joblib.load('Job_files/Axis_stock_model.joblib')
    return model
stock_methods = {'TATAMOTORS':Tata_motors_stock(),'AXISBANK':Axis_stock()}
def stocks():
    stock_methods = {'TATAMOTORS':Tata_motors_stock(),'AXISBANK':Axis_stock()}
    return stock_methods.keys()
def get_stocks(stock):
    return stock_methods[stock]