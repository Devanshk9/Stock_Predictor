import pandas as pd
import streamlit as st
from datetime import date
pip install yfinance
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2013-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Predictor Using Signals")
st.text("By Devansh Khetan")
# Load the stock symbols from CSV
def load_stock_symbols():
    try:
        df = pd.read_csv('indian_stocks.csv')
        
        # Check if the necessary columns are in the DataFrame
        if 'symbol' not in df.columns or 'name' not in df.columns:
            raise ValueError("CSV file must contain 'symbol' and 'name' columns.")
        
        return df
    except FileNotFoundError:
        st.error("The file 'indian_stocks.csv' was not found.")
        return pd.DataFrame()  # Return an empty DataFrame
    except pd.errors.EmptyDataError:
        st.error("The file 'indian_stocks.csv' is empty.")
        return pd.DataFrame()
    except ValueError as e:
        st.error(f"Error in CSV file format: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()

symbols_df = load_stock_symbols()

if symbols_df.empty:
    st.warning("No stock symbols available.")
else:
    stocks = symbols_df.set_index('symbol')['name'].to_dict()
    selected_stock = st.selectbox("Select Dataset for Prediction", list(stocks.keys()))
    selected_name = stocks[selected_stock]

    n_years = st.slider("Years of prediction:", 1, 4)
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text("Load Data...")
    data = load_data(selected_stock)
    data_load_state.text("Loading data...done!")

    if not data.empty:
        st.subheader('Raw data')
        st.write(data.tail())

        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
            fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
        plot_raw_data()

        # Forecasting
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader('Forecast Data')
        st.write(forecast.tail())

        st.write('Forecast Data')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write('Forecast Components')
        fig2 = m.plot_components(forecast)
        st.write(fig2)
    else:
        st.warning("No data available to display.")
