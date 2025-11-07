import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def fetch_data(symbol, start_date, end_date):
    print("\n Fetching data from Indian markets...")
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        raise ValueError("No data found. Please check the symbol or date range.")

    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    print(" Data fetched successfully for:", symbol)
    return data


def sma_strategy(data):
    print("\n Applying SMA crossover strategy...")

    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()

    data['Signal'] = 0
    data.loc[data['SMA20'] > data['SMA50'], 'Signal'] = 1
    data.loc[data['SMA20'] < data['SMA50'], 'Signal'] = -1
    data['Position'] = data['Signal'].diff()

    print("Signals generated using 20-day and 50-day SMA crossover.")
    return data



def rsi_strategy(data, period=14, overbought=70, oversold=30):
    print("\n Applying RSI strategy...")

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['Signal'] = 0
    data.loc[data['RSI'] < oversold, 'Signal'] = 1
    data.loc[data['RSI'] > overbought, 'Signal'] = -1
    data['Position'] = data['Signal'].diff()

    print("Signals generated using RSI strategy.")
    return data


def macd_strategy(data):
    print("\n⚙️ Applying MACD strategy...")

    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data['Signal'] = 0
    data.loc[data['MACD'] > data['Signal_Line'], 'Signal'] = 1
    data.loc[data['MACD'] < data['Signal_Line'], 'Signal'] = -1
    data['Position'] = data['Signal'].diff()

    print("✅ Signals generated using MACD strategy.")
    return data


def backtest(data, initial_balance):
    print("\n Running backtest...")

    data = data.dropna(subset=['Signal', 'Close'])

    balance = initial_balance
    position = 0
    trades = []

    for i in range(len(data)):
        price = data['Close'].iloc[i]
        signal = data['Signal'].iloc[i]

        
        if signal == 1 and position == 0:
            position = balance / price
            balance = 0
            trades.append((data.index[i], 'BUY', price))

        
        elif signal == -1 and position > 0:
            balance = position * price
            position = 0
            trades.append((data.index[i], 'SELL', price))

    final_value = balance + (position * data['Close'].iloc[-1])
    profit = final_value - initial_balance
    profit_percent = (profit / initial_balance) * 100

    return trades, final_value, profit, profit_percent


if __name__ == "__main__":
    print(" Algorithmic Trading Backtester for Indian Markets 🇮🇳")
    print("--------------------------------------------------------")

    
    symbol = input("Enter NSE stock symbol (e.g., INFY.NS, TCS.NS, RELIANCE.NS): ").strip()
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()

    print("\nChoose trading strategy:")
    print("1  SMA Crossover (Simple Moving Average)")
    print("2  RSI (Relative Strength Index)")
    print("3  MACD (Moving Average Convergence Divergence)")
    strategy_choice = input("Enter 1, 2, or 3: ").strip()

    initial_balance = float(input("Enter initial investment amount in ₹: "))

    
    data = fetch_data(symbol, start_date, end_date)

    
    if strategy_choice == '1':
        data = sma_strategy(data)
        strategy_name = "SMA Crossover Strategy"
    elif strategy_choice == '2':
        data = rsi_strategy(data)
        strategy_name = "RSI Strategy"
    elif strategy_choice == '3':
        data = macd_strategy(data)
        strategy_name = "MACD Strategy"
    else:
        print("Invalid choice. Defaulting to SMA Crossover.")
        data = sma_strategy(data)
        strategy_name = "SMA Crossover Strategy"

    
    trades, final_value, profit, profit_percent = backtest(data, initial_balance)


    print(f"\n=====  BACKTEST RESULTS ({strategy_name}) =====")
    for t in trades:
        print(f"{t[0].date()} - {t[1]} at ₹{t[2]:.2f}")

    print(f"\nInitial Investment: ₹{initial_balance:,.2f}")
    print(f"Final Portfolio Value: ₹{final_value:,.2f}")
    print(f"Total Profit/Loss: ₹{profit:,.2f} ({profit_percent:.2f}%)")

    plt.figure(figsize=(10,5))
    plt.plot(data['Close'], label='Stock Price (₹)', color='black')

    if strategy_choice == '1':
        plt.plot(data['SMA20'], label='20-day SMA', color='blue', linestyle='--')
        plt.plot(data['SMA50'], label='50-day SMA', color='red', linestyle='--')
    elif strategy_choice == '3':
        plt.title(f"Backtest Results for {symbol} using MACD Strategy")
        plt.plot(data['MACD'], label='MACD', color='purple', linestyle='--')
        plt.plot(data['Signal_Line'], label='Signal Line', color='orange', linestyle='--')

    plt.xlabel("Date")
    plt.ylabel("Price (₹ INR)")
    plt.title(f"{strategy_name} Backtest for {symbol} (Indian Market)")
    plt.legend()
    plt.grid(True)
    plt.show()
