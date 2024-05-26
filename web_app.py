import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
import yfinance as yf
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Nastavení počátečního a koncového data pro načítání dat
start = st.date_input('Select start date', pd.to_datetime('2018-01-01'))
end = st.date_input('Select end date', pd.to_datetime('2022-01-01'))

# Název aplikace
st.title('Stock Price Prediction')

# Uživatelský vstup pro zadání symbolu akcie
input_stock = st.text_input('Enter a stock', 'AAPL')

# Zobrazení dalších informací o společnosti
st.subheader('Company Info')


# Získání informací o společnosti na základě zadaného symbolu akcie
company_info = yf.Ticker(input_stock).info

# Název společnosti
company_name = company_info['longName']
st.write('Company Name:', company_name)

# Odvětví, ve kterém společnost působí
industry = company_info['industry']
st.write('Industry:', industry)

# Tržní kapitalizace společnosti
market_cap = company_info['marketCap']
st.write('Market Cap:', market_cap)

# P/E poměr společnosti
pe_ratio = company_info['forwardPE']
st.write('P/E Ratio:', pe_ratio)

# Načtení dat ze Stooq pomocí zadaného symbolu akcie
df = yf.download(input_stock, start=start, end=end)

# Dropping the 'Adj Close' column
df.drop('Adj Close', axis=1, inplace=True)

# Zobrazení statistik dat
st.subheader('Data from {} - {}'.format(start, end))
st.write(df.describe())




# Výpočet korelační matice
correlation_matrix = df.corr()

# Vizualizace korelační matice
st.subheader('Correlation Matrix')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
st.pyplot(fig)



# Výpočet denních výnosů
daily_returns = df['Close'].pct_change()

# Hodnota rizika (VaR) na úrovni důvěry 95%
var_95 = daily_returns.quantile(0.05)

# Podmíněná hodnota rizika (CVaR) na úrovni důvěry 95%
cvar_95 = daily_returns[daily_returns <= var_95].mean()

# Risk-free rate (předpokládaná)
risk_free_rate = 0.05

# Denní přebytečné výnosy
excess_returns = daily_returns - risk_free_rate / 252

# Roční Sharpeho poměr
sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

# Zobrazení ukazatelů rizika
st.subheader('Risk Metrics')
st.write(f'Value at Risk (VaR) at 95% confidence level: {var_95:.4f}')
st.write(f'Conditional Value at Risk (CVaR) at 95% confidence level: {cvar_95:.4f}')
st.write(f'Sharpe Ratio: {sharpe_ratio:.4f}')

# Vizualizace cen uzavření akcie v čase
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

# Funkce pro predikci cen akcií pomocí LSTM modelu
def predict_stock_price(df):
    # Příprava dat pro trénování LSTM modelu
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(df['Close'].values.reshape(-1,1))

    x_train = []
    y_train = []

    for i in range(100, len(data_scaled)):
        x_train.append(data_scaled[i-100:i, 0])
        y_train.append(data_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Načtení předtrénovaného LSTM modelu
    model = load_model('keras_model.h5')

    # Predikce cen akcií
    predicted_stock_price = model.predict(x_train)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    return predicted_stock_price


# Funkce pro predikci cen akcií pomocí LSTM modelu
predicted_prices = predict_stock_price(df)
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(df.index, df.Close, label='Original Price')
plt.plot(df.index[100:], predicted_prices, label='Predicted Price', color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Nadpis
st.subheader('Select a technical indicator')

# Selectbox
indicator = st.selectbox('', ['Moving Average (MA)', 'Relative Strength Index (RSI)', 'Moving Average Convergence Divergence (MACD)', 'Bollinger Bands', 'Stochastic Oscillator'])



# Funkce pro výpočet a vizualizaci stochastic oscillator
def plot_stochastic_oscillator(data, window, smooth_window):
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    k_percent = (data['Close'] - low_min) / (high_max - low_min) * 100
    d_percent = k_percent.rolling(window=smooth_window).mean()

    st.subheader(f'Stochastic Oscillator (K={window}, D={smooth_window})')
    fig_stoch = plt.figure(figsize=(12,6))
    plt.plot(k_percent, label='K Percent')
    plt.plot(d_percent, label='D Percent')
    plt.axhline(80, color='r', linestyle='--')
    plt.axhline(20, color='r', linestyle='--')
    plt.ylim(0, 100)
    plt.legend()
    st.pyplot(fig_stoch)


# Funkce pro výpočet a vizualizaci klouzavého průměru
def plot_moving_average(data, time_period):
    ma = data.Close.rolling(time_period).mean()
    st.subheader(f'Moving Average ({time_period} days)')
    fig_ma = plt.figure(figsize=(12,6))
    plt.plot(ma)
    plt.plot(data.Close)
    st.pyplot(fig_ma)

# Funkce pro výpočet a vizualizaci RSI
def plot_rsi(data, time_period):
    delta = data.Close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=time_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=time_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    st.subheader(f'Relative Strength Index (RSI) ({time_period} days)')
    fig_rsi = plt.figure(figsize=(12,6))
    plt.plot(rsi)
    plt.axhline(70, color='r', linestyle='--')
    plt.axhline(30, color='r', linestyle='--')
    plt.ylim(0, 100)
    st.pyplot(fig_rsi)

    # Funkce pro výpočet a vizualizaci Bollinger Bands
def plot_bollinger_bands(data, window, num_std):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)

    st.subheader(f'Bollinger Bands (Window: {window}, Std Dev: {num_std})')
    fig_bb = plt.figure(figsize=(12,6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(upper_band, label='Upper Band', color='red')
    plt.plot(lower_band, label='Lower Band', color='green')
    plt.fill_between(data.index, upper_band, lower_band, color='gray', alpha=0.1)
    plt.legend()
    st.pyplot(fig_bb)

# Funkce pro výpočet a vizualizaci MACD
def plot_macd(data, time_period):
    exp1 = data.Close.ewm(span=12, adjust=False).mean()
    exp2 = data.Close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=time_period, adjust=False).mean()  # Použití dynamického časového období pro signální linku
    st.subheader(f'Moving Average Convergence Divergence (MACD) ({time_period} days)')
    fig_macd = plt.figure(figsize=(12,6))
    plt.plot(macd, label='MACD', color='blue')
    plt.plot(signal_line, label='Signal Line', color='red')
    plt.legend()
    st.pyplot(fig_macd)



# Výběr a zobrazení technického indikátoru
if indicator == 'Moving Average (MA)':
    time_period = st.slider('Select a time period for MA', 1, 365, 30)
    plot_moving_average(df, time_period)
elif indicator == 'Relative Strength Index (RSI)':
    time_period = st.slider('Select a time period for RSI', 1, 365, 14)
    plot_rsi(df, time_period)
elif indicator == 'Moving Average Convergence Divergence (MACD)':
    time_period = st.slider('Select a time period for MACD', 1, 365, 26)
    plot_macd(df, time_period)
elif indicator == 'Bollinger Bands':
    window = st.slider('Select a window size for Bollinger Bands', 1, 365, 20)
    num_std = st.slider('Select the number of standard deviations', 1, 5, 2)
    plot_bollinger_bands(df, window, num_std)
elif indicator == 'Stochastic Oscillator':
    k_window = st.slider('Select a window size for %K', 1, 50, 14)
    d_window = st.slider('Select a window size for %D', 1, 50, 3)
    plot_stochastic_oscillator(df, k_window, d_window)



# Analýza dividend
dividends = yf.Ticker(input_stock).dividends
if not dividends.empty:
    st.subheader('Dividend Analysis')
    st.write('Dividend Payments:')
    st.write(dividends)

    # Dividendový výnos
    last_close_price = df['Close'].iloc[-1]
    dividend_yield = (dividends.sum() / last_close_price) * 100
    st.write(f'Dividend Yield: {dividend_yield:.2f}%')

    net_income = 1000000
    payout_ratio = (dividends.sum() / net_income) * 100
    st.write(f'Dividend Payout Ratio: {payout_ratio:.2f}%')
else:
    st.write('No dividend data available for this stock.')




# Monte Carlo simulace
def monte_carlo_simulation(data, num_simulations, num_days):
    returns = data.pct_change()
    last_price = data.iloc[-1]

    simulations = np.zeros((num_simulations, num_days + 1))  # Adjusted shape

    for i in range(num_simulations):
        daily_volatility = returns.std()
        daily_drift = returns.mean()
        price_series = []
        price = last_price * (1 + np.random.normal(daily_drift, daily_volatility))
        price_series.append(price)

        for d in range(num_days):
            price = price_series[d] * (1 + np.random.normal(daily_drift, daily_volatility))
            price_series.append(price)

        simulations[i, :] = price_series

    return simulations

# Monte Carlo
st.subheader('Monte Carlo Simulation')
num_simulations = 100
num_days = 365

simulated_prices = monte_carlo_simulation(df['Close'], num_simulations, num_days)

# Graf Monte Carlo
fig_mc, ax_mc = plt.subplots()
ax_mc.plot(simulated_prices.T, color='gray', alpha=0.1)
ax_mc.set_title('Monte Carlo Simulation - Future Stock Prices')
ax_mc.set_xlabel('Days')
ax_mc.set_ylabel('Price')
st.pyplot(fig_mc)




# Nastavení API klíče pro NewsAPI
newsapi_key = '7170a01af38f4602ade006e436266a09'

# Funkce pro získání novinek z NewsAPI
def get_news(ticker, country='us', page_size=10):
    url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={newsapi_key}&language=en&pageSize={page_size}'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return articles

# Zobrazení novinek a zpráv o společnosti
st.subheader('Latest News')

# Načtení novinek
articles = get_news(input_stock)

# Zobrazení novinek
for article in articles:
    st.write('###', article['title'])
    st.write(article['description'])
    st.write('[Read more](', article['url'], ')')


