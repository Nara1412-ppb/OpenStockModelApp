from datetime import datetime
import ml_models
from datetime import datetime,timedelta
import pandas_datareader as web
import pandas as pd
import numpy as np
def predict_test(symbol,start,end):
    data = web.DataReader(symbol, data_source='yahoo',start = start, end = end).reset_index()
    data['Date'] = pd.to_datetime(data.Date)
    data.set_index('Date',inplace=True)
    nifty = web.DataReader('^NSEI', data_source='yahoo',start = start, end = end).reset_index()
    nifty['Date'] = pd.to_datetime(nifty.Date)
    df1 = nifty[['Date','Open']]
    df1 =df1.rename(columns = {'Open':'Nifty_Index'},inplace=False)
    df1.set_index('Date',inplace=True)
    df2 = data.join(df1)
    df2.fillna(method='ffill',inplace = True)
    df3 = df2.filter(['Open','Nifty_Index'])[1:].reset_index()
    df4 = df2.filter(['Close','Volume'])[:-1].reset_index()
    new_data = {}
    new_data.update(df4)
    new_data.update(df3)
    i = len(new_data['Close'])- 1
    data_frame = pd.DataFrame(new_data)
    close = float(data_frame['Close'][i])
    volume = float(data_frame['Volume'][i])
    nifty = float(data_frame['Nifty_Index'][i])
    input1 =(close,volume,nifty)
    input = np.asarray(input1).reshape(1,-1)
    return input



def predict_stock(stock):
    symbol = str(stock)+'.NS'    
    model = ml_models.get_stocks(stock)
    start = datetime.now()-timedelta(days=2)
    end = datetime.now()
    input = predict_test(symbol,start,end)
    prediction = model.predict(input)
    return prediction

