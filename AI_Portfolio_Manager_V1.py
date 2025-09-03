
import streamlit as st #web application & presentation
import yfinance as yf #Stock API configuration
import pandas as pd # Visual Tables 
import numpy as np #math calculations
import matplotlib.pyplot as plt #stock graph charter
import plotly.graph_objects as go 
from datetime import datetime, timedelta #real time adjustment
import json #further AI Chatbot improvements
import os #access & open directories for data savings future implementation
from dotenv import load_dotenv # Load AI API key, currently free tial out
import re 
import ta #technical analysis indicators
import pytz #time zone adjustment
import requests

load_dotenv()

# Set up page
st.set_page_config( 
    page_title="AI Portfolio Manager",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize portfolio data
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}  
if 'cash' not in st.session_state:
    st.session_state.cash = 10000
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []

# Process data to ensure it is timezone-aware and has the correct format by checking for current timezone converts all to Eastern time and make sure it appears on column 
def process_data(data):
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

# Add simple moving average (SMA) and exponential moving average (EMA) indicators
def add_technical_indicators(data):
    if len(data) >= 20:  # Need at least 20 data points for indicators
        data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    return data

# Fetch stock data with proper intervals
def get_stock_data(ticker, period='1d'):
    try:
        stock = yf.Ticker(ticker)

        # Create interval map and representation candlestick for each interval
        interval_map = {
            '1d': '1m',
            '1wk': '30m',
            '1mo': '1d',
            '1y': '1wk',
            'max': '1wk'
        }

        interval = interval_map.get(period, '1d')
        
        hist = stock.history(period=period, interval=interval)

        return stock, hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, pd.DataFrame()

# Get current stock price by accessing yahoo finance and getting last close price of stock
def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')
        return data['Close'].iloc[-1] if not data.empty else None
    except Exception as e:
        st.error(f"Error getting price for {ticker}: {str(e)}")
        return None

# Calculate portfolio value from all holdings
def calculate_portfolio_value():
    total = st.session_state.cash
    for ticker, holdings in st.session_state.portfolio.items():
        current_price = get_current_price(ticker)
        if current_price: 
            total += current_price * holdings['quantity']
    return total

# Buy stock function and adjust average buy price, create new holding in portfolio if new holding allocation with confirmation purchase
def buy_stock(ticker, quantity):
    current_price = get_current_price(ticker)

    if not current_price:
        return False, "Stock price unavailable to retrieve"

    if current_price * quantity > st.session_state.cash:
        return False, "Insufficient Funds"

    if ticker in st.session_state.portfolio:
        st.session_state.portfolio[ticker]['quantity'] += quantity

        #method to get avg price = sum of (old price * quant) and (new price * quant) / total quant
        st.session_state.portfolio[ticker]['avg_price'] = (
            (st.session_state.portfolio[ticker]['avg_price'] * 
            (st.session_state.portfolio[ticker]['quantity'] - quantity) + 
            current_price * quantity) / st.session_state.portfolio[ticker]['quantity']
        )
    else:
        st.session_state.portfolio[ticker] = {
            'quantity': quantity,
            'avg_price': current_price
        }

    st.session_state.cash -= current_price * quantity

    st.session_state.transaction_history.append({
        'date': datetime.now(),
        'type': 'BUY',
        'ticker': ticker,
        'quantity': quantity,
        'price': current_price,
        'total': current_price * quantity,
    })
    return True, f"Successfully purchased {quantity} shares of {ticker} at ${current_price:.2f}"

# Sell stock function by validation of enough shares
def sell_stock(ticker, quantity):
    if ticker not in st.session_state.portfolio:
        return False, f"You don't own shares of {ticker}"
    if quantity > st.session_state.portfolio[ticker]['quantity']:
        return False, f"You don't have enough shares of {ticker} to sell"

    current_price = get_current_price(ticker)
    if not current_price:
        return False, "Could not retrieve price of this stock"

    # Update portfolio
    st.session_state.portfolio[ticker]['quantity'] -= quantity
    if st.session_state.portfolio[ticker]['quantity'] == 0: 
        del st.session_state.portfolio[ticker]

    st.session_state.cash += current_price * quantity

    st.session_state.transaction_history.append({
        'date': datetime.now(),
        'type': 'SELL',
        'ticker': ticker,
        'quantity': quantity,
        'price': current_price,
        'total': current_price * quantity,
    })

    return True, f"Successfully sold {quantity} shares of {ticker} at ${current_price:.2f}"

# Get index data for sidebar to show market relevant direction and 2 days price change
def get_index_data():
    indices = {
        'S&P 500': '^GSPC',
        'Nasdaq': '^IXIC',
        'Dow Jones': '^DJI'
    }

    index_data = {}
    for name, ticker in indices.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='2d')  
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[0]
                change = current_price - prev_price
                pct_change = (change / prev_price) * 100
                index_data[name] = {
                    'price': current_price,
                    'change': change,
                    'pct_change': pct_change
                }
        except Exception as e:
            st.error(f"Error fetching {name} data: {str(e)}")

    return index_data


def get_fear_greed_index():
    
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, params={"limit": 1})
        data = response.json()

        if data and 'data' in data and len(data['data']) > 0:
            index_value = int(data['data'][0]['value'])
            classification = data['data'][0]['value_classification']
            return index_value, classification
        return None, "Data unavailable"
    except Exception as e:
        st.sidebar.error(f"Error fetching Fear & Greed Index: {str(e)}")
        return None, "Error"

# Add this function to determine the color based on the index value
def get_fear_greed_color(value):
    """Return a color based on the Fear & Greed Index value"""
    if value is None:
        return "gray"
    elif value >= 75:
        return "red"  # Extreme greed
    elif value >= 55:
        return "orange"  # Greed
    elif value >= 45:
        return "yellow"  # Neutral
    elif value >= 25:
        return "lightgreen"  # Fear
    else:
        return "green"  # Extreme fear
# Main application
def main():
    st.title("AI Portfolio Manager STONK 🐱")
    st.markdown("Manage your stock portfolio efficiently and effortlessly")

    
    # Sidebar with navigation and index prices
    st.sidebar.title("Navigation")
    # Navigation options (removed Dashboard and AI Chat)
    app_mode = st.sidebar.selectbox("Choose a section", ["Portfolio", "Stock Analysis", "News", "Transactions"])
    st.sidebar.divider()
  
    # Display index prices at the top of sidebar
    
    index_data = get_index_data()
    st.sidebar.subheader("Market Indices")
    for name, data in index_data.items():
        delta_color = "normal" if data['change'] >= 0 else "inverse"
        st.sidebar.metric(
            name, 
            f"{data['price']:.2f}", 
            f"{data['change']:.2f} ({data['pct_change']:.2f}%)",
            delta_color=delta_color
        )
   
    st.sidebar.divider()
    st.sidebar.subheader("Fear & Greed Index")

    # Fetch the index data
    fg_value, fg_classification = get_fear_greed_index()

    if fg_value is not None:
        # Create a colored box with the index value
        color = get_fear_greed_color(fg_value)
        st.sidebar.markdown(
            f"""
            <div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;">
                <h3 style="margin: 0; color: white;">{fg_value}</h3>
                <p style="margin: 0; color: white;">{fg_classification}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Add a tooltip with explanation
        with st.sidebar.expander("What does this mean?"):
            st.write("""
            The Fear & Greed Index measures market emotions:
            - 0-24: Extreme Fear (market may be oversold)
            - 25-44: Fear
            - 45-55: Neutral
            - 56-75: Greed
            - 76-100: Extreme Greed (market may be overbought)
            """)
    else:
        st.sidebar.info("Fear & Greed data unavailable")

   
    
    # Portfolio allocation chart at the top with Holdings table showing Portfolio Value, P/L, Cash allocation
    if app_mode == "Portfolio":
        st.header("Your Portfolio")
                    
        #draw pie chart from configurations using fig, plotly extension
        if st.session_state.portfolio:
            tickers = list(st.session_state.portfolio.keys())
            quantities = [holding['quantity'] for holding in st.session_state.portfolio.values()]
            current_prices = [get_current_price(ticker) for ticker in tickers]
            values = [qty * price for qty, price in zip(quantities, current_prices)]

            if st.session_state.cash > 0:
                tickers.append("CASH")
                values.append(st.session_state.cash)

            fig = go.Figure(data=[go.Pie(labels=tickers, values=values, hole=.3)])
            st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio value metrics
        portfolio_value = calculate_portfolio_value()
        init_invest = 10000
        total_gain_loss = portfolio_value - init_invest
        gain_loss_percent = (total_gain_loss / init_invest) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Value", f"${portfolio_value:.2f}")
        with col2:
            st.metric("Total Gain/Loss", f"${total_gain_loss:.2f}", f"{gain_loss_percent:.2f}%")
        with col3:
            st.metric("Available Cash", f"${st.session_state.cash:.2f}")

        # Holdings table, get ticker and holding, convert data to array for value presentation using pd, dataframe
        if not st.session_state.portfolio:
            st.info("You don't have any stocks in your portfolio yet")
        else:
            portfolio_data = []
            for ticker, holding in st.session_state.portfolio.items():
                current_price = get_current_price(ticker)
                if current_price:
                    value = current_price * holding['quantity']
                    cost_basis = holding['avg_price'] * holding['quantity']
                    gain_loss = value - cost_basis
                    gain_loss_percent = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0
                    portfolio_data.append({
                        "Stock": ticker,
                        "Quantity": holding['quantity'],
                        "Avg Price": f"${holding['avg_price']:.2f}",
                        "Current Price": f"${current_price:.2f}",
                        "Value": f"${value:.2f}",
                        "Gain/Loss": f"${gain_loss:.2f}",
                        "Gain/Loss %": f"{gain_loss_percent:.2f}%"
                    })
            st.dataframe(pd.DataFrame(portfolio_data))

    # Stock Analysis Section
    elif app_mode == "Stock Analysis":
        st.header("Stock Analysis")

        # Create main layout with chart area
        main_col = st.columns(1)[0]

        with main_col:
            # Input controls at the top
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                ticker = st.text_input("Enter stock ticker", "AAPL").upper()
            with col2:
                # Default to "1d"
                period = st.selectbox("Time period", ["1d", "1wk", "1mo", "1y", "max"], index=0)
            with col3:
                chart_type = st.selectbox('Chart Type', ['Candlestick', 'Line'])
                indicators = st.multiselect('Technical Indicators', ['SMA 20', 'EMA 20'])

            # Get stock data
            stock, hist = get_stock_data(ticker, period=period)
            if hist.empty:
                st.error("Could not retrieve stock data. Please check the ticker symbol.")
            else:
                # Process data and add technical indicators
                hist_processed = process_data(hist.copy())
                hist_processed = add_technical_indicators(hist_processed)

                # Calculate metrics
                last_close = hist_processed['Close'].iloc[-1]
                prev_close = hist_processed['Close'].iloc[0]
                change = last_close - prev_close
                pct_change = (change / prev_close) * 100
                high = hist_processed['High'].max()
                low = hist_processed['Low'].min()
                

                # Display main metrics
                if stock :
                    
                    st.subheader(f"{stock.info.get('longName', ticker)} ({stock.info.get('sector', ticker)})")
                
                st.metric(label=f"{ticker} Last Price", value=f"{last_close:.2f} USD", 
                         delta=f"{change:.2f} ({pct_change:.2f}%)")

               

                # Use fig from plotly to graph candlestick, lines with data from hist_processed after get_stock_data taht returns history of stock
                
                fig = go.Figure()

                if chart_type == 'Candlestick':
                    fig.add_trace(go.Candlestick(
                        x=hist_processed['Datetime'],
                        open=hist_processed['Open'],
                        high=hist_processed['High'],
                        low=hist_processed['Low'],
                        close=hist_processed['Close'],
                        name=f"{ticker} Price"
                    ))
                #fig chart line of all closing prices by Scatter of plotly
                else:
                    fig.add_trace(go.Scatter(
                        x=hist_processed['Datetime'], 
                        y=hist_processed['Close'],
                        mode='lines',
                        name=f"{ticker} Price",
                        line=dict(color='#FFFFFF')
                    ))

                # Add selected technical indicators to the chart
                for indicator in indicators:
                    if indicator == 'SMA 20' and 'SMA_20' in hist_processed.columns:
                        fig.add_trace(go.Scatter(x=hist_processed['Datetime'], y=hist_processed['SMA_20'], 
                                               name='SMA 20', line=dict(color='orange')))
                    elif indicator == 'EMA 20' and 'EMA_20' in hist_processed.columns:
                        fig.add_trace(go.Scatter(x=hist_processed['Datetime'], y=hist_processed['EMA_20'], 
                                               name='EMA 20', line=dict(color='red')))

                # Format graph
                fig.update_layout(
                    title=f'{ticker} {period.upper()} Chart',
                    xaxis_title='Time',
                    yaxis_title='Price (USD)',
                    xaxis_rangeslider_visible=False, 
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                
                # Display stock technical analysis info metrics (Market Cap, Time Frame Volume, Day High/Low/Open, Current Price, Open, Previous Close)
                if stock is not None:
                    info = stock.info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Volume", f"{info.get('regularMarketVolume', 'N/A'):,}")
                        market_cap = info.get('marketCap', 'N/A')
                        if market_cap != 'N/A' and market_cap is not None:
                            st.metric("Market Cap", f"${market_cap:,}")
                        else:
                            st.metric("Market Cap", "N/A")
                        
                    with col2:
                        st.metric("Current Price", f"${info.get('regularMarketPrice', 'N/A')}")
                        st.metric("Open Price", f"${info.get('regularMarketOpen', 'N/A')}")
                        st.metric("Previous Price", f"${info.get('regularMarketPreviousClose', 'N/A')}")
                    with col3:
                        # Display main metrics
                        st.metric("High", f"{high:.2f} USD")
                        st.metric("Low", f"{low:.2f} USD")
                        
                      
                    # Buy/Sell interface with validation check by success/error msg
                    st.subheader("Trade Stock")
                    trade_col1, trade_col2, trade_col3 = st.columns(3)
                    with trade_col1:
                        trade_type = st.radio("Order Type", ["Buy", "Sell"])
                    with trade_col2:
                        quantity = st.number_input("Quantity", min_value=1, value=1)
                    with trade_col3:
                        current_price = get_current_price(ticker)
                        if current_price:
                            st.write(f"Current Price: ${current_price:.2f}")
                            total = current_price * quantity
                            st.write(f"Total: ${total:.2f}")
                            if st.button("Execute Trade"):
                                if trade_type == "Buy":
                                    success, message = buy_stock(ticker, quantity)
                                else:
                                    success, message = sell_stock(ticker, quantity)
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                                  
                   
    #implement news query search by generating links with ticker name on market new broadcasters         
    elif app_mode == "News":
        st.header("📰 Stock News Search")

      
        search_query = st.text_input("Enter stock ticker symbol (e.g., AAPL, TSLA, GOOGL)", "AAPL").upper()

        if search_query:
            st.subheader(f"News for {search_query}")

            with st.spinner("Getting news..."):
                try:
                    # Get basic stock info
                    stock = yf.Ticker(search_query)
                    info = stock.info
                    company_name = info.get('longName', search_query)

                    st.write(f"**Company:** {company_name}")
                    st.divider()

                    # Create news items with actual clickable links
                    news_items = [
                        {
                            'title': f"{search_query} Latest News & Updates",
                            'summary': f"Get the most recent news and developments affecting {company_name}.",
                            'source': "Yahoo Finance",
                            'link': f"https://finance.yahoo.com/quote/{search_query}"
                        },
                        {
                            'title': f"{search_query} Price Analysis & Charts", 
                            'summary': "View detailed price charts and technical analysis for {search_query}.",
                            'source': "MarketWatch",
                            'link': f"https://www.marketwatch.com/investing/stock/{search_query}"
                        },
                        {
                            'title': f"{company_name} Earnings & Reports",
                            'summary': "Access latest earnings reports and company announcements.",
                            'source': "NASDAQ",
                            'link': f"https://www.nasdaq.com/market-activity/stocks/{search_query}"
                        },
                        {
                            'title': f"{search_query} Investor Information",
                            'summary': "Find investor resources and shareholder updates.",
                            'source': "Google Finance",
                            'link': f"https://www.google.com/finance/quote/{search_query}"
                        }
                    ]

                    # Display news with clickable links
                    for i, news in enumerate(news_items, 1):
                        st.write(f"**{i}. {news['title']}**")
                        st.write(news['summary'])
                        st.markdown(f"**[📖 Read on {news['source']}]({news['link']})**")
                        st.divider()

                    # Additional quick links section
                    st.write("### 🔍 More News Sources")
                    st.markdown(f"""
                    - **[Yahoo Finance News](https://finance.yahoo.com/quote/{search_query}/news)** - Latest news articles
                    - **[Google News](https://news.google.com/search?q={search_query}+stock)** - News search results
                    - **[Reuters](https://www.reuters.com/search/news?blob={search_query})** - Financial news
                    - **[Bloomberg](https://www.bloomberg.com/search?query={search_query})** - Market news
                    - **[CNBC](https://www.cnbc.com/quotes/{search_query})** - Business news
                    """)

                    # Current price info
                    try:
                        current_price = get_current_price(search_query)
                        if current_price:
                            st.info(f"**Current Price:** ${current_price:.2f}")
                    except:
                        pass

                except Exception as e:
                    st.error("Couldn't find data for that ticker. Please check the symbol and try again.")
                    st.info("💡 Try popular symbols like: AAPL, TSLA, MSFT, GOOGL, AMZN, NVDA")

        else:
            st.info("👆 Enter a stock ticker symbol above to see news links and resources")
            st.write("""
            **Popular tickers to try:**
            - **NVDA** - NVIDIA
            - **AAPL** - Apple
            - **TSLA** - Tesla
            - **MSFT** - Microsoft
            - **GOOGL** - Google
            - **AMZN** - Amazon
            """)
        
    # Transactions Log table with exact time format
    elif app_mode == "Transactions":
        st.header("Transaction History")
        if not st.session_state.transaction_history:
            st.info("No transactions yet.")
        else:
            transaction_data = []
            for transaction in st.session_state.transaction_history:
                transaction_data.append({
                    "Date": transaction['date'].strftime("%Y-%m-%d %H:%M"),
                    "Type": transaction['type'],
                    "Stock": transaction['ticker'],
                    "Quantity": transaction['quantity'],
                    "Price": f"${transaction['price']:.2f}",
                    "Total": f"${transaction['total']:.2f}",
                })
            st.dataframe(pd.DataFrame(transaction_data))
          
    
      

if __name__ == "__main__":
    main()