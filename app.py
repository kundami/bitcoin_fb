
import os
import json
import numpy as np
import pandas as pd
import pickle
import quandl
from fbprophet import Prophet
import sys
from flask import Flask, render_template, request, jsonify
import flask
app = Flask(__name__)

import plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
#py.init_notebook_mode(connected=True)

#py.init_notebook_mode(connected=True)

def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df


btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')
btc_usd_price_kraken.tail()

# Chart the BTC pricing data
#btc_trace = go.Scatter(x=btc_usd_price_kraken.index, y=btc_usd_price_kraken['Weighted Price'])
#py.iplot([btc_trace])

#Function to make repeated calls to Prophet for future values of all X values
#Open	High	Low	Close	Volume (BTC)	Volume (Currency)	Weighted Price
def run_fbp(reg_data,field_str,look_ahead):
    reg_data['y'] = np.log(reg_data[field_str])
    reg_data['ds'] = reg_data.index
    fb_reg_data = reg_data.loc[:,['ds','y']]
    fb_reg_data = fb_reg_data.reset_index(drop=True)
    fb_reg_data.dropna(inplace=True)
    m = Prophet()
    m.fit(fb_reg_data)
    future = m.make_future_dataframe(periods=look_ahead)
    forecast = m.predict(future)
    predicted  = forecast[['yhat', 'yhat_lower', 'yhat_upper']].applymap(np.exp)
    return predicted

def plot_fb(predict,actual,columns,title):

# lot_data(predicted, actual, future, cols,title)
    trace1 = go.Scatter(
        x=actual.index,
        y=actual,
        name='Actual Close'
    )
    #Lower_closing  Upper_closing
    trace2 = go.Scatter(
        x=actual.index,
        y=predict['Upper_closing'],
        name='Predicted Close Upper Bound',
        fill=None,
        mode='lines',
        line=dict(color='rgb(143,19,131)'),        
       
    )
    trace3 = go.Scatter(
        x=actual.index,
        y=predict['Lower_closing'],
        name='Predicted Close Lower Bound',
        fill='tonexty',
        mode='lines',
        line=dict(color='rgb(143,19,131)')
    )

    data = [trace1, trace2, trace3]
    layout = go.Layout(
        title='Actual Vs Predicted using FBProphet',
        yaxis=dict(
            title='Closing Price'
        )
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


#Start Regression
reg_data = btc_usd_price_kraken['20170814':'20180222'].loc[:,['Open','High', 'Low', 'Close', 'Volume (BTC)' , 'Volume (Currency)', 'Weighted Price']]

def date_parse(date_string):
    date_list = date_string.split('-')
    date_index_0 = datetime.date(2016, 1, 1)
    date = datetime.date(int(date_list[0]),int(date_list[1]),int(date_list[2]))
    diff = date - date_index_0
    return diff.days

cols = ["Mean", "Lower_closing", "Upper_closing"]

predicted_open = run_fbp(reg_data,'Open',28)
predicted_close = run_fbp(reg_data,'High',28)
predicted_high = run_fbp(reg_data,'Low',28)
predicted_low = run_fbp(reg_data,'Close',28)


predicted_close.columns = cols
actual = reg_data['Close']
#actual = actual[::-1]
actual.name = "Actual  Close" 
title = "Closing price distribution of bitcoin"
plot_fb(predicted_close, actual, cols, title)

#predicted_close.head()

#actual.head()

@app.route('/', methods = ["GET", "POST"])
def test():
    btc_data_new = reg_data
    date_index = 639
    if flask.request.method == 'POST':
        test_list = request.form.getlist("chk_box")
        date_index = date_parse(request.form.get('date'))
        print("I am here")
        print(date_index)
        print(test_list)
        btc_data_new = btc_data_new[test_list]
        close_var = "var{}(t)".format(list(btc_data_new.columns.values).index('close')+1)
    
    # Convert the figures to JSON
    graphJSON_fb = json.dumps(plot_fb(predicted_close, actual, cols,title), cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('test.html', graphJSON_fb=graphJSON_fb)



if __name__ == '__main__':
   app.run(debug=True)
