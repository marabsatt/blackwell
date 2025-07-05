# Import class to create web app
import streamlit as st

# import class for financial data
import yfinance as yf 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import datetime as dt
from dateutil.relativedelta import relativedelta
import requests
from bs4 import BeautifulSoup

# from rapidfuzz import process, fuzz

sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

r = requests.get(sp500_url)
data = r.text
soup = BeautifulSoup(data, 'html.parser')
data = soup.find('table')

df_500 = pd.read_html(str(data))[0]

nas_url = 'http://en.wikipedia.org/wiki/Nasdaq-100#Components'

r = requests.get(sp500_url)
data = r.text
soup = BeautifulSoup(data, 'html.parser')
data = soup.find('table')

df_nas = pd.read_html(str(data))[0]

df = pd.concat([df_500, df_nas], ignore_index=True)

st.set_page_config(page_title="Pair's Trading Strategy Dashboard",
                   page_icon=":chart_with_upwards_trend:", 
                   layout='centered',
                   initial_sidebar_state='expanded', 
                   menu_items={
                       'Report a bug': "https://github.com/marabsatt/blackwell/issues",
                       'Connect with me': "https://www.linkedin.com/in/marabsatt/",
                       'About': "This is a stock data dashboard app to analyze and visualize cointegrated stocks. For educational purposes only."
                    }
                    )

# Title of the app
st.title("Pair's Trading Strategy Dashboard")

# Sidebar for user input
st.sidebar.header('User Input Features')
user_sector = st.sidebar.selectbox(
    'Select A Sector', 
    df[df['GICS Sub-Industry']].unique().tolist()
    )

# Page content
col1, col2 = st.columns([2, 1])

with col1:
    # Filter the DataFrame based on user input
    ticker_list = df[df['GICS Sub-Industry'] == user_sector]['Symbol'].unique().tolist()
    current_date = dt.datetime.now().strftime('%Y-%m-%d')
    start_date = current_date - relativedelta(years=5)
    start_date = start_date.strftime('%Y-%m-%d')
    df = yf.download(ticker_list, start=start_date, end=current_date)['Close']

    def cointegration_pairs(df, threshold=0.05):
        pairs = []
        for i in range(len(df.columns)):
            for j in range(i + 1, len(df.columns)):
                stock1 = df.iloc[:, i]
                stock2 = df.iloc[:, j]
                score, p_value, _ = coint(stock1, stock2)
                if p_value < threshold:
                    pairs.append((df.columns[i], df.columns[j], p_value))
        return pairs

    list_of_pairs = cointegration_pairs(df, threshold=0.05)
    sorted_pairs = sorted(list_of_pairs, key=lambda x: x[2])
    for pair in sorted_pairs:
        pvalues = [pair[2] for pair in sorted_pairs]

    highest_p_val = sorted_pairs[-1][0:2]

    # Create a matrix of p-values
    pvalues_matrix = np.zeros((len(ticker_list), len(ticker_list)))
    for pair in sorted_pairs:
        i = ticker_list.index(pair[0])
        j = ticker_list.index(pair[1])
        pvalues_matrix[i, j] = pair[2]
        pvalues_matrix[j, i] = pair[2]  # Mirror the values

    # Two stocks that have the highest p-value
    stock1 = df[f'{highest_p_val[0]}']
    stock2 = df[f'{highest_p_val[1]}']

    results = sm.OLS(stock2, stock1).fit()
    b = results.params[0]
    spread = stock2 - b * stock1

    hedge_ratio = results.params[0]

    def zscore(series):
        return (series - series.mean()) / np.std(series)

    # Spread plot with buy and sell signals
    zscore(spread).plot(figsize=(21,7))
    plt.axhline(zscore(spread).mean(), color='black', linestyle='--')
    plt.axhline(1.0, color='red', linestyle='--')
    plt.axhline(-1.0, color='green', linestyle='--')
    plt.legend(['Spread Z-Score', 'Mean', 'Upper Band (Sell Signal)', 'Lower Band (Buy Signal)'])

    fig, ax = plt.subplots(figsize=(21, 7))
    zscore(spread).plot(ax=ax)
    ax.axhline(zscore(spread).mean(), color='black', linestyle='--')
    ax.axhline(1.0, color='red', linestyle='--')
    ax.axhline(-1.0, color='green', linestyle='--')
    ax.legend([
        'Spread Z-Score',
        'Mean',
        'Upper Band (Sell Signal)',
        'Lower Band (Buy Signal)'
    ])

    # 2. Render it in Streamlit
    st.pyplot(fig, use_container_width=True)

    # Plot the closing prices of the two stocks
    plt.figure(figsize=(21, 7))
    plt.plot(stock1, lw=1.5, label=f"Close Price of {highest_p_val[0]}")
    plt.plot(stock2, lw=1.5, label=f"Close Price of {highest_p_val[1]}")
    plt.grid(True)
    plt.legend(loc=0)

    plt.axis('tight')
    plt.xlabel('Dates')
    plt.ylabel('Price')
    plt.title(f"Closing Price of {highest_p_val[0]} and {highest_p_val[1]}")

    # Backtesting performance of the strategy
    bt_df = pd.concat([zscore(spread), stock2 - b * stock1], axis=1)
    bt_df.columns = ['signal', 'position']

    bt_df['side'] = 0
    bt_df.loc[bt_df['signal'] <= -1, 'side'] = 1
    bt_df.loc[bt_df['signal'] >= 1, 'side'] = -1

    returns = bt_df.position.pct_change() * bt_df.side
    returns.cumsum().plot(figsize=(13,7), title="Cumulative Returns")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)