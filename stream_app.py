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
st.title("Pair's Trading Strategy Dashboard", 
         help="This app allows you to analyze and visualize cointegrated stocks from the S&P 500 and Nasdaq-100 indices. \n" \
         "You can select an industry sector to find pairs of stocks that are cointegrated, \n" \
         "and visualize their price movements and trading signals. \n" \
         "more information on pairs trading can be found here: \n" \
         "https://www.investopedia.com/terms/p/pairs-trading.asp"
         )

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
    lowest_p_val = sorted_pairs[-1][0:2]
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
stock1 = df[f'{lowest_p_val[0]}']
stock2 = df[f'{lowest_p_val[1]}']

results = sm.OLS(stock2, stock1).fit()
b = results.params[0]
spread = stock2 - b * stock1

hedge_ratio = results.params[0]

def zscore(series):
    return (series - series.mean()) / np.std(series)

df = df[[f"{lowest_p_val[0]}", f"{lowest_p_val[1]}"]]
df = df.join(
    spread.rename('spread'),
    how='inner'
)
ticker_1 = yf.Ticker(lowest_p_val[0])
ticker_2 = yf.Ticker(lowest_p_val[1])
ticker_1_news = ticker_1.get_news()
ticker_2_news = ticker_2.get_news()

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
    ax.plot(stock1, lw=1.5, label=f"Close Price of {lowest_p_val[0]}")
    ax.plot(stock2, lw=1.5, label=f"Close Price of {lowest_p_val[1]}")
    ax.grid(True)
    ax.legend(loc=0)
    ax.set(xlabel="Dates",
        ylabel="Price",
        title=f"Closing Price of {lowest_p_val[0]} and {lowest_p_val[1]}")
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

    st.subheader(f"Options Activity for {ticker_1.info['displayName'], lowest_p_val[0]}", divider=True,
                 help="The table below shows the options activity for the selected stock. \n" \
                 "You can see the available call and put options, their strike prices, expiration dates, and other details. \n" \
                 "This information can help you make informed decisions about trading options."
                 )
    call_opt = ticker_1.option_chain((ticker_1.options[0])).calls.sort_values('strike')
    put_opt = ticker_1.option_chain((ticker_1.options[0])).puts.sort_values('strike')
    st.write(f"{ticker_1.info['displayName']} Calls")
    st.dataframe(call_opt)
    st.markdown('***')
    st.write(f"{ticker_1.info['displayName']} Puts")
    st.dataframe(put_opt)

    st.subheader(f"Options Activity for {ticker_2.info['displayName'], lowest_p_val[1]}", divider=True,
                 help="The table below shows the options activity for the selected stock. \n" \
                 "You can see the available call and put options, their strike prices, expiration dates, and other details. \n" \
                 "This information can help you make informed decisions about trading options."
                 )
    call_opt = ticker_2.option_chain((ticker_2.options[1])).calls.sort_values('strike')
    put_opt = ticker_2.option_chain((ticker_2.options[1])).puts.sort_values('strike')
    st.write(f"{ticker_2.info['displayName']} Calls")
    st.dataframe(call_opt)
    st.markdown('***')
    st.write(f"{ticker_2.info['displayName']} Puts")
    st.dataframe(put_opt)

with col2:
    st.subheader('Conitegrated Pairs', 
                 divider=True, 
                 help= "The table below shows the pairs of stocks that are cointegrated with their p-values. \n" \
                 "A lower p-value indicates a stronger cointegration relationship. \n" \
                 "https://en.wikipedia.org/wiki/Cointegration"
                 )
    st.dataframe(
        pd.DataFrame(sorted_pairs, columns=['Stock 1', 'Stock 2', 'P-Value']).sort_values(by='P-Value', ascending=True),
        use_container_width=True
    )
    st.subheader('Hedge Ratio', 
                 divider=True, 
                 help="The hedge ratio is the ratio of the two stocks in a pair that minimizes the variance of the spread."
                 )
    st.write(f"The hedge ratio for the pair {lowest_p_val[0]} and {lowest_p_val[1]} is: {hedge_ratio:.4f}")
    
    # Calculate spread boundaries using standard deviation
    spread_mean = df['spread'].mean()
    spread_std = df['spread'].std()

    # Define upper and lower boundaries (typically 1 or 2 standard deviations), use a standard devaition of 2 for more conservative trading
    upper_boundary = spread_mean + 1 * spread_std
    lower_boundary = spread_mean - 1 * spread_std

    # TODO: Need to adjust the boundaries. The current suggestions doesn't fit the visuals
    if df['spread'][-1] > upper_boundary:
        st.write(f"The current spread of {df['spread'][-1]:.2f} is above the upper boundary ({upper_boundary:.2f}), consider selling {lowest_p_val[0]} and buying {lowest_p_val[1]}.")
    elif df['spread'][-1] < lower_boundary:
        st.write(f"The current spread of {df['spread'][-1]:.2f} is below the lower boundary ({lower_boundary:.2f}), consider buying {lowest_p_val[0]} and selling {lowest_p_val[1]}.")
    else:
        st.write(f"The current spread of {df['spread'][-1]:.2f} is within the boundaries ({lower_boundary:.2f}, {upper_boundary:.2f}), no action needed.")
    
    st.subheader(f"Recent News for {ticker_1.info['displayName']}", divider=True)
    for i in range(len(ticker_1_news[0:3])):
        st.write(f"Title: {ticker_1_news[i]['content']['title']}")
        st.write(f"Summary: {ticker_1_news[i]['content']['summary']}")
        st.write(f"URL: {ticker_1_news[i]['content']['canonicalUrl']['url']}")
        st.write(f"Published Date: {ticker_1_news[i]['content']['pubDate']}")
        st.markdown("---")

    st.subheader(f"Recent News for {ticker_2.info['displayName']}", divider=True)
    for i in range(len(ticker_2_news[0:3])):
        st.write(f"Title: {ticker_2_news[i]['content']['title']}")
        st.write(f"Summary: {ticker_2_news[i]['content']['summary']}")
        st.write(f"URL: {ticker_2_news[i]['content']['canonicalUrl']['url']}")
        st.write(f"Published Date: {ticker_2_news[i]['content']['pubDate']}")
        st.markdown("---")
