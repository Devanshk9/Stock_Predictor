import pandas as pd
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2019-01-01"  # Reduced data size to the last 5 years
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Predictor Using Signals")
st.text("By Devansh Khetan")

# Load the stock symbols from CSV
@st.cache_data
def load_stock_symbols():
    try:
        df = pd.read_csv('indian_stocks.csv')
        
        if 'symbol' not in df.columns or 'name' not in df.columns:
            raise ValueError("CSV file must contain 'symbol' and 'name' columns.")
        
        return df
    except FileNotFoundError:
        st.error("The file 'indian_stocks.csv' was not found.")
        return pd.DataFrame()
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

    data_load_state = st.text("Loading Data...")
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

        # Initialize the progress bar and spinner
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Function for model training and forecasting
        def train_and_forecast():
            m = Prophet(
                changepoint_prior_scale=0.1,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            m.fit(df_train)
            progress_bar.progress(50)
            progress_text.text("Model training completed. Generating forecast...")

            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            progress_bar.progress(100)
            progress_text.text("Forecasting completed!")

            return m, forecast  # Return m along with forecast

        with st.spinner("Training Prophet model and generating forecast..."):
            m, forecast = train_and_forecast()

        progress_bar.empty()
        progress_text.empty()

        st.subheader('Forecast Data')
        st.write(forecast.tail())

        # Plotting Forecast
        st.write('Forecast Data')
        try:
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)
        except Exception as e:
            st.error(f"Error plotting forecast data: {e}")

        st.write('Forecast Components')
        try:
            fig2 = m.plot_components(forecast)
            st.write(fig2)
        except Exception as e:
            st.error(f"Error plotting forecast components: {e}")

    else:
        st.warning("No data available to display.")
