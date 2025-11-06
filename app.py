import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------
# FETCH DATA
# ------------------------------------------
def fetch_data(symbol, start_date, end_date):
    print("\n📊 Fetching data from Indian markets...")
    data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        raise ValueError("No data found. Please check the symbol or date range.")

    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    print("✅ Data fetched successfully for:", symbol)
    print("Columns:", list(data.columns))
    return data


# ------------------------------------------
# GENERATE BUY/SELL SIGNALS (SMA crossover)
# ------------------------------------------
def generate_signals(data):
    print("\n⚙️ Generating trading signals...")

    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()

    data['Signal'] = 0
    data.loc[data['SMA20'] > data['SMA50'], 'Signal'] = 1   # Buy
    data.loc[data['SMA20'] < data['SMA50'], 'Signal'] = -1  # Sell

    data['Position'] = data['Signal'].diff()

    print("✅ Signals generated using 20-day and 50-day SMA crossover.")
    return data


# ------------------------------------------
# BACKTEST STRATEGY
# ------------------------------------------
def backtest(data, initial_balance=100000):
    print("\n🚀 Running backtest on Indian stock data...")

    data = data.dropna(subset=['Signal', 'Close'])

    balance = initial_balance
    position = 0
    trades = []

    for i in range(len(data)):
        price = data['Close'].iloc[i]
        signal = data['Signal'].iloc[i]

        # Buy
        if signal == 1 and position == 0:
            position = balance / price
            balance = 0
            trades.append((data.index[i], 'BUY', price))

        # Sell
        elif signal == -1 and position > 0:
            balance = position * price
            position = 0
            trades.append((data.index[i], 'SELL', price))

    # Final value
    final_value = balance + (position * data['Close'].iloc[-1])
    profit = final_value - initial_balance
    profit_percent = (profit / initial_balance) * 100

    return trades, final_value, profit, profit_percent


# ------------------------------------------
# MAIN EXECUTION
# ------------------------------------------
if __name__ == "__main__":
    print("📈 Simple Algorithmic Trading Backtester for Indian Markets 🇮🇳")
    print("--------------------------------------------------------------")

    # User Inputs
    symbol = input("Enter NSE stock symbol (e.g., INFY.NS, TCS.NS, RELIANCE.NS): ").strip()
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()

    # Fetch and process data
    data = fetch_data(symbol, start_date, end_date)
    data = generate_signals(data)

    trades, final_value, profit, profit_percent = backtest(data)

    print("\n===== 💹 BACKTEST RESULTS =====")
    for t in trades:
        print(f"{t[0].date()} - {t[1]} at ₹{t[2]:.2f}")

    print("\nInitial Portfolio Value: ₹100000.00")
    print(f"Final Portfolio Value: ₹{final_value:.2f}")
    print(f"Total Profit/Loss: ₹{profit:.2f} ({profit_percent:.2f}%)")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(data['Close'], label='Stock Price (₹)', color='green')
    plt.plot(data['SMA20'], label='20-day SMA', color='blue', linestyle='--')
    plt.plot(data['SMA50'], label='50-day SMA', color='red', linestyle='--')

    plt.title(f"Backtest Results for {symbol} (Indian Market)")
    plt.xlabel("Date")
    plt.ylabel("Price (₹ INR)")
    plt.legend()
    plt.grid(True)
    plt.show()
