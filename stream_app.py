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
                   layout='wide',
                   initial_sidebar_state='expanded', 
                   menu_items={
                       'Report a bug': "https://github.com/marabsatt/blackwell/issues",
                       'Get help': "https://www.linkedin.com/in/marabsatt/",
                       'About': "This is a stock data dashboard app to analyze and visualize cointegrated stocks. For educational purposes only."
                    }
                    )

# Title of the app
st.title("Pair's Trading Strategy Dashboard")

# Sidebar for user input
st.sidebar.header('Select An Industry Sector')
user_sector = st.sidebar.selectbox(
    'Select A Sector', 
    df['GICS Sub-Industry'].unique().tolist()
    )

# Page content
col1, col2 = st.columns([2, 1])


# Filter the DataFrame based on user input
ticker_list = df[df['GICS Sub-Industry'] == user_sector]['Symbol'].unique().tolist()
current_date = dt.datetime.now().strftime('%Y-%m-%d')
start_date = (dt.datetime.now() - relativedelta(years=5)).strftime('%Y-%m-%d')

df = yf.download(ticker_list, start=start_date, end=current_date)['Close']
df.dropna(inplace=True)

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

if sorted_pairs:
    highest_p_val = sorted_pairs[-1][0:2]
else:
    st.error("No cointegrated pairs found. Please try a different sector.")
    st.stop()

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

df = df[[f"{highest_p_val[0]}", f"{highest_p_val[1]}"]]
df = df.join(
    spread.rename('spread'),
    how='inner'
)

with col1:
    # Spread plot with buy and sell signals
    fig, ax = plt.subplots(figsize=(21, 10))
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

    st.pyplot(fig, use_container_width=True)

    # Plot the closing prices of the two stocks

    fig, ax = plt.subplots(figsize=(21, 10))
    ax.plot(stock1, lw=1.5, label=f"Close Price of {highest_p_val[0]}")
    ax.plot(stock2, lw=1.5, label=f"Close Price of {highest_p_val[1]}")
    ax.grid(True)
    ax.legend(loc=0)
    ax.set(xlabel="Dates",
        ylabel="Price",
        title=f"Closing Price of {highest_p_val[0]} and {highest_p_val[1]}")
    ax.axis("tight")

    st.pyplot(fig, use_container_width=True)

    # Backtesting performance of the strategy
    bt_df = pd.concat([zscore(spread), stock2 - b * stock1], axis=1)
    bt_df.columns = ['signal', 'position']

    bt_df['side'] = 0
    bt_df.loc[bt_df['signal'] <= -1, 'side'] = 1
    bt_df.loc[bt_df['signal'] >= 1, 'side'] = -1

    returns = bt_df.position.pct_change() * bt_df.side

    cumulative_returns = returns.cumsum()
    fig, ax = plt.subplots(figsize=(21, 10))
    ax.plot(cumulative_returns, label="Cumulative Returns")
    ax.set_title("Cumulative Returns")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()

    st.pyplot(fig, use_container_width=True)

with col2:
    st.subheader('Conitegrated Pairs')
    st.dataframe(
        pd.DataFrame(sorted_pairs, columns=['Stock 1', 'Stock 2', 'P-Value']).sort_values(by='P-Value', ascending=False),
        use_container_width=True
    )
    st.subheader('Hedge Ratio')
    st.write(f"The hedge ratio for the pair {highest_p_val[0]} and {highest_p_val[1]} is: {hedge_ratio:.4f}")
    
    # Calculate spread boundaries using standard deviation
    spread_mean = df['spread'].mean()
    spread_std = df['spread'].std()

    # Define upper and lower boundaries (typically 1 or 2 standard deviations)
    upper_boundary = spread_mean + 2 * spread_std
    lower_boundary = spread_mean - 2 * spread_std

    # TODO: Need to adjust the boundaries. The current suggestions doesn't fit the visuals
    if df['spread'][-1] > upper_boundary:
        st.write(f"The current spread of {df['spread'][-1]:.2f} is above the upper boundary ({upper_boundary:.2f}), consider selling {highest_p_val[0]} and buying {highest_p_val[1]}.")
    elif df['spread'][-1] < lower_boundary:
        st.write(f"The current spread of {df['spread'][-1]:.2f} is below the lower boundary ({lower_boundary:.2f}), consider buying {highest_p_val[0]} and selling {highest_p_val[1]}.")
    else:
        st.write(f"The current spread of {df['spread'][-1]:.2f} is within the boundaries ({lower_boundary:.2f}, {upper_boundary:.2f}), no action needed.")