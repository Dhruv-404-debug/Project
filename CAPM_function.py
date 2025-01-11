import datetime
import dateutil
import plotly.express as px
import numpy as np
import pandas as pd
import pandas_ta as pta
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go

# Function to plot interactive chart
def interactive_plot(df):
    fig = go.Figure()
    # Add each column as a separate line
    for i in df.columns[1:]:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df[i],
                mode='lines',
                name=i,
                line=dict(width=2)  # Set line width for better visuals
            )
        )

    # Update layout for improved visuals
    fig.update_layout(
        title="Stock Prices Over Time",  # Chart title
        xaxis_title="Date",  # X-axis label
        yaxis_title="Stock Price (USD)",  # Y-axis label
        xaxis=dict(showgrid=True),  # Show gridlines for clarity
        yaxis=dict(showgrid=True, zeroline=False),
        width=700,  # Adjust width
        height=500,  # Adjust height
        template="plotly_dark",  # Set dark theme
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

# Function to normalize the price based on the initial price
 
def normalize(df_2):
    df = df_2.copy()
    for i in df.columns[1:]:
        df[i] = df[i]/df[i][0]
    return df

# function to calculate daily return
 
def daily_return(df):
    df_daily_return = df.copy()
    for i in df.columns[1:]:
        for j in range(1,len(df)):
            df_daily_return[i][j] = ( (df[i][j] - df[i][j-1]) / df[i][j-1])*100
        df_daily_return[i][0]= 0
    return df_daily_return 

# Function to Calculate beta
 
def calculate_beta(stock_daily_return,stock):
    rm = stock_daily_return['SP500'].mean()*252

    b, a= np.polyfit(stock_daily_return['SP500'],stock_daily_return[stock],1) 

    return b,a 


def plotly_table(dataframe):
    headerColor = '#3E3E3E'  # Dark grey for header
    rowEvenColor = '#f8fafd'  # Light blue-grey for even rows
    rowOddColor = '#e1efff'  # Slightly darker blue-grey for odd rows

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>"],  # Define the two columns
            line_color='darkslategray',
            fill_color=headerColor,  # Header background color
            align=['left','center'],  # Center-align header text
            font=dict(color='white', size=16),  # White text with medium font size
            height=30  # Adjust header height
        ),
        cells=dict(
            values=[dataframe['Metric'], dataframe['Value']],  # First column: Metric, Second column: Value
            fill_color=[[rowOddColor, rowEvenColor] * (len(dataframe) // 2 + 1)],  # Alternate row colors
            align=['left','right'],  # Left-align for better readability
            line_color='darkslategray',
            font=dict(color="black", size=14),  # Black text with smaller font size
        )
    )])

    # Adjust the layout for cleaner presentation
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),  # Tighten margins
    )

    return fig




def filter_data(dataframe, num_period):
    if num_period == '1mo':
        date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(months=-1)
    elif num_period == '5d':
        date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(days=-5)
    elif num_period == '6mo':
        date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(months=-6)
    elif num_period == '1y':
        date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(years=-1)
    elif num_period == '5y':
        date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(years=-5)
    elif num_period == 'ytd':
        date = datetime.datetime(dataframe.index[-1].year, 1, 1)
    else:
        date = dataframe.index[0]
    
    # Filter the DataFrame
    return dataframe.reset_index()[dataframe.reset_index()['Date'] > date]

"""
def close_chart(dataframe, num_period=False):
    if num_period:
        dataframe = filter_data(dataframe, num_period)
    
    fig = go.Figure()

    # Add traces for 'Open', 'Close', 'High', and 'Low'
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'], 
            y=dataframe['Open'], 
            mode='lines', 
            name='Open', 
            line=dict(width=2, color='#5ab7ff')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'], 
            y=dataframe['Close'], 
            mode='lines', 
            name='Close', 
            line=dict(width=2, color='black')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'], 
            y=dataframe['High'], 
            mode='lines', 
            name='High', 
            line=dict(width=2, color='#0078ff')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'], 
            y=dataframe['Low'], 
            mode='lines', 
            name='Low', 
            line=dict(width=2, color='red')
        )
    )

    # Update x-axis with range slider
    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
    template="plotly_dark",  # Use the dark theme
    plot_bgcolor="black",  # Set the plot background color
    paper_bgcolor="black",  # Set the paper (outer) background color
    font=dict(
        color="white",  # Set font color to white for contrast
        size=12
    ),
    xaxis=dict(
        showgrid=True,  # Show gridlines
        gridcolor="gray",  # Gridline color
        zerolinecolor="white"  # Zero line color
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="gray",
        zerolinecolor="white"
    ),
    legend=dict(
        title="Indicators",
        font=dict(color="white"),  # Legend font color
        bgcolor="black",  # Legend background color
        bordercolor="white",  # Legend border color
        borderwidth=1  # Legend border width
    )
)


    # Update layout for better appearance
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='black',
        paper_bgcolor='#e1efff',
        legend=dict(yanchor="top", xanchor="right")
    )

    return fig
"""
def close_chart(dataframe, num_period=False):
    if num_period:
        dataframe = filter_data(dataframe, num_period)
    
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Open'], mode='lines', name='Open', line=dict(width=2, color='#1f77b4')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Close'], mode='lines', name='Close', line=dict(width=2, color='#ff7f0e')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['High'], mode='lines', name='High', line=dict(width=2, color='#2ca02c')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Low'], mode='lines', name='Low', line=dict(width=2, color='#d62728')))

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=500,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=12),
        margin=dict(l=0, r=20, t=20, b=0),
        xaxis=dict(showgrid=True, gridcolor="gray", zerolinecolor="white"),
        yaxis=dict(showgrid=True, gridcolor="gray", zerolinecolor="white"),
        legend=dict(title="Indicators", font=dict(color="white"), bgcolor="black", bordercolor="white", borderwidth=1)
    )
    return fig

"""
def candlestick(dataframe, num_period):
    dataframe = filter_data(dataframe, num_period)
    
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=dataframe['Date'],
        open=dataframe['Open'],
        high=dataframe['High'],
        low=dataframe['Low'],
        close=dataframe['Close']
    ))

    fig.update_layout(
        showlegend=False,
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='black',
        paper_bgcolor='#e1efff'
    )
    fig.update_layout(
    template="plotly_dark",  # Use the dark theme
    plot_bgcolor="black",  # Set the plot background color
    paper_bgcolor="black",  # Set the paper (outer) background color
    font=dict(
        color="white",  # Set font color to white for contrast
        size=12
    ),
    xaxis=dict(
        showgrid=True,  # Show gridlines
        gridcolor="gray",  # Gridline color
        zerolinecolor="white"  # Zero line color
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="gray",
        zerolinecolor="white"
    ),
    legend=dict(
        title="Indicators",
        font=dict(color="white"),  # Legend font color
        bgcolor="black",  # Legend background color
        bordercolor="white",  # Legend border color
        borderwidth=1  # Legend border width
    )
)

    # After adding traces for your chart
    fig.update_traces(line=dict(color="blue"), selector=dict(name="Open"))
    fig.update_traces(line=dict(color="green"), selector=dict(name="Close"))
    fig.update_traces(line=dict(color="orange"), selector=dict(name="High"))
    fig.update_traces(line=dict(color="red"), selector=dict(name="Low"))

    return fig
"""

def candlestick(dataframe, num_period):
    dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()

    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=dataframe['Date'],
        open=dataframe['Open'], high=dataframe['High'],
        low=dataframe['Low'], close=dataframe['Close'],
        name="Candlestick"
    ))

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=500,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=12),
        margin=dict(l=0, r=20, t=20, b=0),
        xaxis=dict(showgrid=True, gridcolor="gray", zerolinecolor="white"),
        yaxis=dict(showgrid=True, gridcolor="gray", zerolinecolor="white"),
        legend=dict(title="Indicators", font=dict(color="white"), bgcolor="black", bordercolor="white", borderwidth=1)
    )
    return fig

"""
def RSI(dataframe, num_period):
    # Calculate RSI using pandas_ta
    dataframe['RSI'] = pta.rsi(dataframe['Close'])
    
    # Filter the data based on the specified period
    dataframe = filter_data(dataframe, num_period)
    
    # Initialize the figure
    fig = go.Figure()

    # Add the RSI line
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['RSI'],
            name='RSI',
            marker_color='orange',
            line=dict(width=2, color='orange')
        )
    )

    # Add the overbought line at RSI 70
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=[70] * len(dataframe),
            name='Overbought',
            marker_color='red',
            line=dict(width=2, color='red', dash='dash')
        )
    )

    # Add the oversold line at RSI 30
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=[30] * len(dataframe),
            name='Oversold',
            marker_color='#79da84',
            line=dict(width=2, color='#79da84', dash='dash')
        )
    )

    # Update layout settings
    fig.update_layout(
        yaxis_range=[0, 100],
        height=200,
        plot_bgcolor='black',
        paper_bgcolor='#e1efff',
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
"""
def RSI(dataframe, num_period):
    dataframe['RSI'] = pta.rsi(dataframe['Close'])
    dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()

    # Add RSI and levels
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['RSI'], name='RSI', line=dict(width=2, color='orange')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=[70] * len(dataframe), name='Overbought', line=dict(width=2, color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=[30] * len(dataframe), name='Oversold', line=dict(width=2, color='green', dash='dash')))

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=200,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=12),
        margin=dict(l=0, r=0, t=0, b=0),
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="gray", zerolinecolor="white"),
        xaxis=dict(showgrid=True, gridcolor="gray", zerolinecolor="white"),
        legend=dict(orientation="h", yanchor="top", xanchor="right", x=1)
    )
    return fig

"""
def Moving_average(dataframe, num_period):
    # Calculate the 50-period Simple Moving Average (SMA)
    dataframe['SMA_50'] = pta.sma(dataframe['Close'], 50)

    # Filter the data based on the specified period
    dataframe = filter_data(dataframe, num_period)

    # Initialize the figure
    fig = go.Figure()

    # Add traces for Open, Close, High, Low, and SMA_50
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['Open'],
            mode='lines',
            name='Open',
            line=dict(width=2, color='#5ab7ff')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['Close'],
            mode='lines',
            name='Close',
            line=dict(width=2, color='black')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['High'],
            mode='lines',
            name='High',
            line=dict(width=2, color='#0078ff')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['Low'],
            mode='lines',
            name='Low',
            line=dict(width=2, color='red')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(width=2, color='purple')
        )
    )

    # Add a range slider and update layout settings
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='black',
        paper_bgcolor='#e1efff',
        legend=dict(
            yanchor="top",
            xanchor="right"
        )
    )

    fig.update_layout(
    template="plotly_dark",  # Use the dark theme
    plot_bgcolor="black",  # Set the plot background color
    paper_bgcolor="black",  # Set the paper (outer) background color
    font=dict(
        color="white",  # Set font color to white for contrast
        size=12
    ),
    xaxis=dict(
        showgrid=True,  # Show gridlines
        gridcolor="gray",  # Gridline color
        zerolinecolor="white"  # Zero line color
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="gray",
        zerolinecolor="white"
    ),
    legend=dict(
        title="Indicators",
        font=dict(color="white"),  # Legend font color
        bgcolor="black",  # Legend background color
        bordercolor="white",  # Legend border color
        borderwidth=1  # Legend border width
    )
)


    return fig
"""

def Moving_average(dataframe, num_period):
    dataframe['SMA_50'] = pta.sma(dataframe['Close'], 50)
    dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Open'], mode='lines', name='Open', line=dict(width=2, color='#1f77b4')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Close'], mode='lines', name='Close', line=dict(width=2, color='#ff7f0e')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['High'], mode='lines', name='High', line=dict(width=2, color='#2ca02c')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Low'], mode='lines', name='Low', line=dict(width=2, color='#d62728')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['SMA_50'], mode='lines', name='SMA 50', line=dict(width=2, color='purple')))

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=500,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=12),
        margin=dict(l=0, r=20, t=20, b=0),
        xaxis=dict(showgrid=True, gridcolor="gray", zerolinecolor="white"),
        yaxis=dict(showgrid=True, gridcolor="gray", zerolinecolor="white"),
        legend=dict(title="Indicators", font=dict(color="white"), bgcolor="black", bordercolor="white", borderwidth=1)
    )
    return fig

"""
def MACD(dataframe, num_period):
    # Calculate MACD, MACD Signal, and MACD Histogram
    macd = pta.macd(dataframe['Close']).iloc[:, 0]
    macd_signal = pta.macd(dataframe['Close']).iloc[:, 1]
    macd_hist = pta.macd(dataframe['Close']).iloc[:, 2]

    # Add MACD, Signal, and Histogram to the DataFrame
    dataframe['MACD'] = macd
    dataframe['MACD Signal'] = macd_signal
    dataframe['MACD Hist'] = macd_hist

    # Filter the data based on the specified period
    dataframe = filter_data(dataframe, num_period)

    # Initialize the figure
    fig = go.Figure()

    # Add MACD line
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['MACD'],
            name='MACD',
            marker_color='orange',
            line=dict(width=2, color='orange')
        )
    )

    # Add MACD Signal line
    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['MACD Signal'],
            name='MACD Signal',
            marker_color='red',
            line=dict(width=2, color='red', dash='dash')
        )
    )

    # Add MACD Histogram as bars
    colors = ['red' if val < 0 else 'green' for val in macd_hist]
    fig.add_trace(
        go.Bar(
            x=dataframe['Date'],
            y=dataframe['MACD Hist'],
            name='MACD Histogram',
            marker_color=colors
        )
    )

    # Update layout settings
    fig.update_layout(
        height=200,
        plot_bgcolor='black',
        paper_bgcolor='#e1efff',
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_layout(
    template="plotly_dark",  # Use the dark theme
    plot_bgcolor="black",  # Set the plot background color
    paper_bgcolor="black",  # Set the paper (outer) background color
    font=dict(
        color="white",  # Set font color to white for contrast
        size=12
    ),
    xaxis=dict(
        showgrid=True,  # Show gridlines
        gridcolor="gray",  # Gridline color
        zerolinecolor="white"  # Zero line color
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="gray",
        zerolinecolor="white"
    ),
    legend=dict(
        title="Indicators",
        font=dict(color="white"),  # Legend font color
        bgcolor="black",  # Legend background color
        bordercolor="white",  # Legend border color
        borderwidth=1  # Legend border width
    )
)


    return fig

"""

def MACD(dataframe, num_period):
    macd = pta.macd(dataframe['Close']).iloc[:, 0]
    macd_signal = pta.macd(dataframe['Close']).iloc[:, 1]
    macd_hist = pta.macd(dataframe['Close']).iloc[:, 2]

    dataframe['MACD'] = macd
    dataframe['MACD Signal'] = macd_signal
    dataframe['MACD Hist'] = macd_hist
    dataframe = filter_data(dataframe, num_period)
    colors = ['red' if val < 0 else 'green' for val in macd_hist]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['MACD'], name='MACD', line=dict(width=2, color='orange')))
    fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['MACD Signal'], name='MACD Signal', line=dict(width=2, color='red', dash='dash')))
    fig.add_trace(go.Bar(x=dataframe['Date'], y=dataframe['MACD Hist'], name='MACD Histogram', marker_color=colors))

    fig.update_layout(
        template="plotly_dark",
        height=200,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=12),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="top", xanchor="right", x=1)
    )
    return fig


def get_data(ticker):
    stock_data = yf.download(ticker, start='2024-01-01')
    return stock_data['Close']


def stationary_check(close_price):
    adf_test = adfuller(close_price)
    p_value = round(adf_test[1], 3)
    return p_value


def get_rolling_mean(close_price):
    rolling_price = close_price.rolling(window=7).mean().dropna()
    return rolling_price


def get_differencing_order(close_price):
    p_value = stationary_check(close_price)
    d = 0
    while True:
        if p_value > 0.05:
            d += 1
            close_price = close_price.diff().dropna()
            p_value = stationary_check(close_price)
        else:
            break
    return d


def fit_model(data, differencing_order):
    model = ARIMA(data, order=(30, differencing_order, 30))
    model_fit = model.fit()
    forecast_steps = 30
    forecast = model_fit.get_forecast(steps=forecast_steps)
    predictions = forecast.predicted_mean
    return predictions


def evaluate_model(original_price, differencing_order):
    train_data, test_data = original_price[:-30], original_price[-30:]
    predictions = fit_model(train_data, differencing_order)
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    return round(rmse, 2)


def scaling(close_price):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.array(close_price).reshape(-1, 1))
    return scaled_data, scaler


def get_forecast(original_price, differencing_order):
    predictions = fit_model(original_price, differencing_order)
    start_date = datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.now() + datetime.timedelta(days=29)).strftime('%Y-%m-%d')
    forecast_index = pd.date_range(start=start_date, end=end_date, freq='D')
    forecast_df = pd.DataFrame(predictions, index=forecast_index, columns=['Close'])
    return forecast_df


def inverse_scaling(scaler, scaled_data):
    close_price = scaler.inverse_transform(np.array(scaled_data).reshape(-1,1))
    return close_price
"""

def Moving_average_forecast(forecast):
    fig = go.Figure()

    # Add trace for actual close price
    fig.add_trace(
        go.Scatter(
            x=forecast.index[:-30],
            y=forecast['Close'].iloc[:-30],
            mode='lines',
            name='Close Price',
            line=dict(width=2, color='black')
        )
    )

    # Add trace for future close price
    fig.add_trace(
        go.Scatter(
            x=forecast.index[-31:],
            y=forecast['Close'].iloc[-31:],
            mode='lines',
            name='Future Close Price',
            line=dict(width=2, color='red')
        )
    )

    # Add range slider and update layout
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        legend=dict(
            yanchor="top",
            xanchor="right"
        )
    )

    return fig

"""
