# Importing Libraries

import streamlit as st
import pandas as pd
import yfinance as yf 
import datetime
import pandas_datareader.data as web
import CAPM_function 
import plotly.graph_objects as go 


st.set_page_config(page_title="CAPM",
                   page_icon="chart_with_upwards_trend",
                   layout='wide')

st.title("Capital Asset Pricing Model")

# Getting input from user

col1,col2 = st.columns([1,1])
with col1:
    stock_list = st.multiselect("Choose 4 stocks",('TSLA','AAPL','NFLX','MSFT','MGM','AMZN','NVDA','GOOGL'),['TSLA','AAPL','GOOGL','AMZN'])
with col2:
    years = st.number_input("Number of years",1,10)


# Downloading the data for S&P 500
try:

    end = datetime.date.today()
    start = datetime.date(end.year - years, end.month, end.day)
    #datetime.date.today().year- years, datetime.date.today().month,datetime.date.today().day


    # Fetch SP500 data

    SP500 = web.DataReader(['SP500'],'fred',start,end)
    # print(SP500.head())
    SP500 = SP500.asfreq('B').ffill()  # Ensure SP500 has data for business days only
    SP500.reset_index(inplace=True)
    SP500.columns = ['Date', 'SP500']

    stock_df = pd.DataFrame()
    for stock in stock_list:
        data = yf.download(stock,start=start, end=end)  # period= f'{years}y'
        stock_df[f'{stock}'] = data['Close']

    # print(stock_df.head())

    # Reset index and align dates
    #stock_df = stock_df.set_index('Date').asfreq('B').reset_index()
    #SP500.reset_index(inplace=True)
    stock_df.reset_index(inplace=True)
    #print(stock_df.dtypes)
    #print(SP500.dtypes)
    #SP500.columns = ['Date','SP500']
    #stock_df['Date'] = stock_df['Date'].astype('datetime64[ns]')
    #stock_df['Date'] = stock_df['Date'].apply(lambda x:str(x)[:10])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df = stock_df.set_index('Date').asfreq('B').ffill().bfill().reset_index()


    # Merging the data 
    stock_df = pd.merge(stock_df,SP500,on='Date',how='inner')    # use of inner join
    #print(stock_df)

    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown("### Dataframe Head")
        st.dataframe(stock_df.head(),use_container_width=True)
    with col2:
        st.markdown("### Dataframe Tail")
        st.dataframe(stock_df.tail(),use_container_width=True)

    # If want to show the statistics description   
    #st.markdown("### Dataframe Summary")
    #st.write(stock_df.describe())


    col1,col2 = st.columns([1,1])
    with col1:
        st.markdown("### Price of all the Stocks")
        st.plotly_chart(CAPM_function.interactive_plot(stock_df))
    with col2:
        st.markdown("### Price of all the Stocks - After Normalizing")
        st.plotly_chart(CAPM_function.interactive_plot(CAPM_function.normalize(stock_df)))
        


    stock_daily_return = CAPM_function.daily_return(stock_df)
    print(stock_daily_return.head())

    beta = {}
    alpha = {}

    for i in stock_daily_return.columns:
        if i !='Date' and i !='SP500':
            b,a = CAPM_function.calculate_beta(stock_daily_return,i)


            beta[i] = b
            alpha[i] = a
    print(beta,alpha)

    beta_df = pd.DataFrame(columns= ['Stock','Beta Value'])
    beta_df['Stock'] = beta.keys()
    beta_df['Beta Value'] = [str(round(i,2)) for i in beta.values()]

    with col1:
        st.markdown(' ### Calculated Beta Value')
        st.dataframe(beta_df,use_container_width=True)


    rf = 0
    rm = stock_daily_return['SP500'].mean()*252

    return_df = pd.DataFrame()
    return_value = []
    for stock, value in beta.items():
        return_value.append(str(round( rf + (value *(rm- rf)),2 )))
    return_df['Stock']= stock_list

    return_df['Return Value'] =  return_value

    with col2:
        st.markdown(' ### Calculated Return using CAPM')
        st.dataframe(return_df,use_container_width=True)


    st.title("Stock Analysis")

    col1,col2,col3 = st.columns(3)

    today = datetime.date.today()

    with col1:
        ticker = st.text_input("Stock Ticker","TSLA")
    with col2:
        start_date = st.date_input("Choose Start Date",datetime.date(today.year - 1,today.month,today.day))
    with col3:
        end_date = st.date_input("Choose End Date",datetime.date(today.year,today.month,today.day))

    st.subheader(ticker)

    stock = yf.Ticker(ticker)

    st.write(stock.info['longBusinessSummary'])
    st.write("**Sector:**",stock.info['sector'])
    st.write("**Full Time Employees:**",stock.info['fullTimeEmployees'])
    st.write("**Website:**",stock.info['website'])
    

    col1,col2 = st.columns(2)

    # Add the column code
    with col1:
        # Creating a DataFrame with key metrics
        df = pd.DataFrame({
            "Metric": ['Market Cap', 'Beta', 'EPS', 'PE Ratio'],
            "Value": [
                stock.info.get("marketCap", "N/A"),
                stock.info.get("beta", "N/A"),
                stock.info.get("trailingEps", "N/A"),
                stock.info.get("trailingPE", "N/A")
            ]
        })
        
        # Create the Plotly table
        figure_df = CAPM_function.plotly_table(df)
        
        # Display the table in Streamlit
        st.plotly_chart(figure_df, use_container_width=True)

    # Add the column code
    with col2:
        # Creating a DataFrame with key metrics
        df = pd.DataFrame({
            "Metric": ['Quick Ratio', 'Revenue per Share', 'Profit Margins', 'Debt to Equity','Return on Equity'],
            "Value": [
                stock.info.get("quickRatio", "N/A"),
                stock.info.get("revenuePerShare", "N/A"),
                stock.info.get("profitMargins", "N/A"),
                stock.info.get("debtToEquity", "N/A"),
                stock.info.get("returnOnEquity", "N/A")
            ]
        })
        
        # Create the Plotly table
        figure_df = CAPM_function.plotly_table(df)
        
        # Display the table in Streamlit
        st.plotly_chart(figure_df, use_container_width=True)
        


    # Define your inputs (example)
   # ticker = "TSLA"  # Replace with a valid ticker
    #start_date = "2023-12-01"
    #end_date = "2023-12-31"

    
 
    # Display selected ticker as a subheader
    st.subheader(f"Selected Stock: {ticker}")

    # Fetch stock data
    try:
        stock = yf.Ticker(ticker)
        data = yf.download(ticker, start=start_date, end=end_date)

        # Validate data
        if data.empty:
            st.error("No data available for the selected stock and date range. Please adjust your inputs.")
            st.stop()

        # Display data
        st.write("Stock Data:", data)

        last_close = float(data['Close'].iloc[-1])  # Last closing price
        previous_close = float(data['Close'].iloc[-2])  # Previous day's closing price

        # Calculate daily change
        daily_change = last_close - previous_close

        # Display metrics
        col1.metric(
            "Daily Change",
            f"${last_close:,.2f}",  # Properly format the last close as a float
            f"{daily_change:+.2f}"  # Properly format daily change
        )

    except Exception as e:
        st.error(f"Error fetching stock data: {e}")


    col1,col2,col3,col4,col5,col6,col7 = st.columns([1,1,1,1,1,1,1])

    num_period = ''

    with col1:
        if st.button('5D'):
            num_period = '5d'
    with col2:
        if st.button('1M'):
            num_period = '1m'
    with col3:
        if st.button('6M'):
            num_period = '6m'
    #with col4:
     #   if st.button('YTD'):
      #      num_period = 'ytd'
    with col5:
        if st.button('1Y'):
            num_period = '1y'
    with col6:
        if st.button('5Y'):
            num_period = '5y'
    with col7:
        if st.button('MAX'):
            num_period = 'max'


    col1,col2,col3 =st.columns([1,1,4])
    with col1:
        chart_type = st.selectbox('',('Candle','Line'))
    with col2:
        if chart_type == 'Candle':

            indicators = st.selectbox('',('RSI','MACD'))
        else:
            indicators = st.selectbox('',('RSI','Moving Average','MACD'))

    ticker_ = yf.Ticker(ticker)
    new_df1 = ticker_.history(period="max")
    data1 = ticker_.history(period="max")

    if num_period == '':
        if chart_type == 'Candle' and indicators == 'RSI':
            st.plotly_chart(CAPM_function.candlestick(data1, '1y'), use_container_width=True)
            st.plotly_chart(CAPM_function.RSI(data1, '1y'), use_container_width=True)

        if chart_type == 'Candle' and indicators == 'MACD':
            st.plotly_chart(CAPM_function.candlestick(data1, '1y'), use_container_width=True)
            st.plotly_chart(CAPM_function.MACD(data1, '1y'), use_container_width=True)

        if chart_type == 'Line' and indicators == 'RSI':
            st.plotly_chart(CAPM_function.close_chart(data1, '1y'), use_container_width=True)
            st.plotly_chart(CAPM_function.RSI(data1, '1y'), use_container_width=True)

        if chart_type == 'Line' and indicators == 'Moving Average':
            st.plotly_chart(CAPM_function.Moving_average(data1, '1y'), use_container_width=True)

        if chart_type == 'Line' and indicators == 'MACD':
            st.plotly_chart(CAPM_function.close_chart(data1, '1y'), use_container_width=True)
            st.plotly_chart(CAPM_function.MACD(data1, '1y'), use_container_width=True)
    
    else:
        if chart_type == 'Candle' and indicators == 'RSI':
            st.plotly_chart(CAPM_function.candlestick(new_df1, num_period), use_container_width=True)
            st.plotly_chart(CAPM_function.RSI(new_df1, num_period), use_container_width=True)

        if chart_type == 'Candle' and indicators == 'MACD':
            st.plotly_chart(CAPM_function.candlestick(new_df1, num_period), use_container_width=True)
            st.plotly_chart(CAPM_function.MACD(new_df1, num_period), use_container_width=True)

        if chart_type == 'Line' and indicators == 'RSI':
            st.plotly_chart(CAPM_function.close_chart(new_df1, num_period), use_container_width=True)
            st.plotly_chart(CAPM_function.RSI(new_df1, num_period), use_container_width=True)

        if chart_type == 'Line' and indicators == 'Moving Average':
            st.plotly_chart(CAPM_function.Moving_average(new_df1, num_period), use_container_width=True)

        if chart_type == 'Line' and indicators == 'MACD':
            st.plotly_chart(CAPM_function.close_chart(new_df1, num_period), use_container_width=True)
            st.plotly_chart(CAPM_function.MACD(new_df1, num_period), use_container_width=True)


        """
            st.set_page_config(
            page_title="Stock Prediction",
            page_icon="chart_with_downwards_trend",
            layout="wide",
        )

        st.title("Stock Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            ticker = st.text_input("Stock Ticker", "AAPL")

        rmse = 0

        st.subheader('Predicting Next 30 Days Close Price for: ' + ticker)

        close_price = get_data(ticker)
        rolling_price = get_rolling_mean(close_price)
        differencing_order = get_differencing_order(rolling_price)
    scaled_data, scaler = scaling(rolling_price)
    rmse = evaluate_model(scaled_data, differencing_order)

    st.write("**Model RMSE Score:**", rmse)

    forecast = get_forecast(scaled_data, differencing_order)
    forecast['Close'] = inverse_scaling(scaler, forecast['Close'])

    st.write("##### Forecast Data (Next 30 days)")
    fig_tail = plotly_table(forecast.sort_index(ascending=True).round(3))
    fig_tail.update_layout(height=220)
    st.plotly_chart(fig_tail, use_container_width=True)

    forecast = pd.concat([rolling_price, forecast])

    st.plotly_chart(Moving_average_forecast(forecast.iloc[150:]), use_container_width=True)

        """







except:
    st.write("Please select valid input")

        


