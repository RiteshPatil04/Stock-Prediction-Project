import streamlit as st
from datetime import date, datetime
import yfinance as yf
from prophet import Prophet
import pandas as pd
from plotly import graph_objs as go

# Set page config for full-width display
st.set_page_config(layout="wide")

# Add stock market image at the top
st.markdown(
    """
    <style>
    .header-image {
        width: 100%;
        height: 240px;  /* Approximately 6 cm */
        object-fit: cover; /* Ensures the image covers the area */
    }
    .footer {
        background-color: skyblue;  /* Change background color to sky blue */
        color: black;                /* Change text color to black */
        padding: 20px;
        text-align: center;
    }
    </style>
    <img class="header-image" src="https://g.foolcdn.com/editorial/images/472790/gettyimages-611992448.jpg" />
    """, unsafe_allow_html=True
)

# Streamlit app title
st.title("STOCK PREDICTION SYSTEM")

# User input for stock ticker
selected_stocks = st.sidebar.text_input("Enter stock ", "AAPL")

# Sidebar for number of shares input
number_of_shares = st.sidebar.number_input("Enter Number of Shares to Purchase", min_value=1, value=1, step=1)

# Function to fetch the current price
def get_live_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        return current_price
    except Exception as e:
        st.error(f"Error fetching live price: {e}")
        return 0.0  # Return 0.0 or some default value

# Fetch the live price
live_price = get_live_price(selected_stocks)

# Calculate the total price
total_price = live_price * number_of_shares

# Display the live price and total price in the sidebar
st.sidebar.markdown(f"**Live Price for {selected_stocks}:** ${live_price:.2f}")
st.sidebar.markdown(f"**Total Price for {number_of_shares} Shares:** ${total_price:.2f}")

# Highlight historical data dates with sky blue
st.sidebar.markdown('<div style="color: skyblue; font-size: 20px;">Historical Data Dates</div>', unsafe_allow_html=True)
start_date_input = st.sidebar.text_input("Enter Start Date (YYYY-MM-DD)", value=date.today().replace(month=1, day=1).strftime("%Y-%m-%d"))
end_date_input = st.sidebar.text_input("Enter End Date (YYYY-MM-DD)", value=date.today().strftime("%Y-%m-%d"))

# Validate date inputs
try:
    start_date = datetime.strptime(start_date_input, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_input, "%Y-%m-%d").date()
    if start_date > end_date:
        raise ValueError("Start date must be before end date.")
except ValueError as e:
    st.sidebar.error(f"Please enter valid dates in the format YYYY-MM-DD. {e}")
    start_date = date.today().replace(month=1, day=1)
    end_date = date.today()

# Load historical data from Yahoo Finance
@st.cache_data
def load_data(ticker, start, end):
    """Load stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

# Load data and show loading state
data_load_state = st.text("Loading data...")
data = load_data(selected_stocks, start_date, end_date)
data_load_state.text("Loading data... done!")

# Ensure Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sidebar for candlestick chart dates
st.sidebar.markdown('<div style="color: skyblue; font-size: 20px;">Candlestick Chart Dates</div>', unsafe_allow_html=True)
candlestick_start_input = st.sidebar.text_input("Enter Candlestick Start Date (YYYY-MM-DD)", value=start_date_input)
candlestick_end_input = st.sidebar.text_input("Enter Candlestick End Date (YYYY-MM-DD)", value=end_date_input)

# Validate candlestick date inputs
try:
    candlestick_start_date = datetime.strptime(candlestick_start_input, "%Y-%m-%d").date()
    candlestick_end_date = datetime.strptime(candlestick_end_input, "%Y-%m-%d").date()
    if candlestick_start_date > candlestick_end_date:
        raise ValueError("Candlestick start date must be before end date.")
except ValueError as e:
    st.sidebar.error(f"Please enter valid candlestick dates in the format YYYY-MM-DD. {e}")
    candlestick_start_date = start_date
    candlestick_end_date = end_date

# Candlestick chart
if st.checkbox("Show Closing and Opening Price Candlestick Chart"):
    candlestick_data = data[(data['Date'] >= pd.to_datetime(candlestick_start_date)) & (data['Date'] <= pd.to_datetime(candlestick_end_date))]
    if not candlestick_data.empty:
        candlestick_fig_actual = go.Figure(data=[go.Candlestick(
            x=candlestick_data['Date'],
            open=candlestick_data['Open'],
            high=candlestick_data['High'],
            low=candlestick_data['Low'],
            close=candlestick_data['Close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )])
        
        candlestick_fig_actual.update_layout(
            title='Actual Closing and Opening Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=True,
            height=800
        )
        
        # Use full width for both the chart and table
        col1, col2 = st.columns([3, 1])  # Adjust width ratio as needed
        
        with col1:
            st.plotly_chart(candlestick_fig_actual, use_container_width=True)
        
        with col2:
            st.subheader('Actual Data')
            actual_data_expander = st.expander("View Actual Data", expanded=True)
            with actual_data_expander:
                st.dataframe(data.set_index('Date'), use_container_width=True)
    else:
        st.warning("No data available for the selected candlestick date range.")

# Prepare data for forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})  # Rename columns for Prophet

# Initialize and fit the Prophet model with seasonalities
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df_train)

# Create future dataframe for predictions
future = model.make_future_dataframe(periods=30)  # Adjust the number of prediction days if needed
forecast = model.predict(future)

# Adjust forecast prices to limit daily changes to 3-4 rupees
def adjust_forecast(forecast, last_close, max_change=4):
    adjusted_prices = [last_close]  # Start with the last known closing price
    for i in range(1, len(forecast)):
        last_price = adjusted_prices[-1]
        min_price = last_price - max_change
        max_price = last_price + max_change
        new_price = min(max(forecast['yhat'].iloc[i], min_price), max_price)
        adjusted_prices.append(new_price)
        
    forecast['yhat'] = adjusted_prices
    return forecast

# Use the last closing price to adjust forecast
last_close = data['Close'].iloc[-1]  # Get the last known closing price
forecast = adjust_forecast(forecast, last_close)

# Plot forecast
st.write('Predict Plot')
forecast_fig = go.Figure()
forecast_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Price', mode='lines', line=dict(color='orange', width=2)))
forecast_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', mode='lines', line=dict(color='lightgray', dash='dash')))
forecast_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', mode='lines', line=dict(color='lightgray', dash='dash')))

forecast_fig.update_layout(
    title='Predict Prices',
    xaxis_title='Date',
    yaxis_title='Price',
    height=800
)

# Add range slider
forecast_fig.update_xaxes(rangeslider_visible=True)

# Display Predicted plot
col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(forecast_fig, use_container_width=True)

with col2:
    st.subheader('Predict data')
    forecast_expander = st.expander("View Forecast Data", expanded=True)
    with forecast_expander:
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds'), use_container_width=True)

# Volume line chart option
st.sidebar.markdown('<div style="color: skyblue; font-size: 20px;">Volume Line Chart</div>', unsafe_allow_html=True)

# Input for selecting a specific month and year
month = st.sidebar.selectbox("Select Month", list(range(1, 13)), index=date.today().month - 1)
year = st.sidebar.number_input("Select Year", value=date.today().year, min_value=2000, max_value=date.today().year)

# Filter data for the selected month
monthly_data = data[(data['Date'].dt.month == month) & (data['Date'].dt.year == year)]

# Plot volume as a line chart
if not monthly_data.empty:
    volume_fig = go.Figure(data=go.Scatter(
        x=monthly_data['Date'],
        y=monthly_data['Volume'],
        mode='lines+markers',
        name='Volume',
        line=dict(color='blue', width=2)
    ))

    volume_fig.update_layout(
        title=f'Volume for {month}/{year}',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=400
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        st.plotly_chart(volume_fig, use_container_width=True)

    with col2:
        st.subheader('Monthly Volume Data')
        st.dataframe(monthly_data[['Date', 'Volume']].set_index('Date'), use_container_width=True)
else:
    st.warning("No data available for the selected month.")

# Footer
st.markdown('<div class="footer">Created By Ritesh Patil</div>', unsafe_allow_html=True)
