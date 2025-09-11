"""
Stock Market Analysis: Tech Giants
----------------------------------
This script analyzes stock performance for Apple (AAPL),
Microsoft (MSFT), and Google (GOOG) over the past 5 years.
"""

# =====================
# 1. Setup & Data Load
# =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

plt.style.use("seaborn-v0_8")
sns.set_palette("Set2")

# Download stock data
tickers = ["AAPL", "MSFT", "GOOG"]
data = yf.download(tickers, start="2019-01-01", end="2024-12-31")["Adj Close"]

print("Data shape:", data.shape)
print(data.head())

# =====================
# 2. Closing Price Trends
# =====================
plt.figure(figsize=(10,6))
for ticker in tickers:
    plt.plot(data.index, data[ticker], label=ticker)
plt.title("Stock Price Trends (2019–2024)")
plt.xlabel("Date")
plt.ylabel("Adjusted Closing Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("images/stock_trends.png")
plt.close()

# =====================
# 3. Moving Averages
# =====================
ma = data.copy()
for ticker in tickers:
    ma[f"{ticker}_30d"] = data[ticker].rolling(30).mean()
    ma[f"{ticker}_100d"] = data[ticker].rolling(100).mean()

plt.figure(figsize=(10,6))
plt.plot(ma.index, ma["AAPL"], label="AAPL Price", color="blue")
plt.plot(ma.index, ma["AAPL_30d"], label="30-Day MA", color="orange")
plt.plot(ma.index, ma["AAPL_100d"], label="100-Day MA", color="red")
plt.title("Apple (AAPL) - Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("images/aapl_ma.png")
plt.close()

# =====================
# 4. Daily Returns & Volatility
# =====================
returns = data.pct_change().dropna()

plt.figure(figsize=(10,6))
for ticker in tickers:
    plt.plot(returns.index, returns[ticker], label=ticker, alpha=0.6)
plt.title("Daily Stock Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.legend()
plt.tight_layout()
plt.savefig("images/daily_returns.png")
plt.close()

# Volatility (30-day rolling std)
volatility = returns.rolling(30).std()

plt.figure(figsize=(10,6))
for ticker in tickers:
    plt.plot(volatility.index, volatility[ticker], label=ticker)
plt.title("Stock Volatility (30-Day Rolling Std Dev)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.tight_layout()
plt.savefig("images/volatility.png")
plt.close()

# =====================
# 5. Correlation Analysis
# =====================
corr = returns.corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation of Daily Returns")
plt.tight_layout()
plt.savefig("images/correlation.png")
plt.close()

print("\nCorrelation of daily returns:")
print(corr)

# =====================
# 6. Insights
# =====================
print("\n--- Insights ---")
print("1. All three stocks (AAPL, MSFT, GOOG) show strong upward trends over 2019–2024.")
print("2. Moving averages highlight long-term growth and short-term fluctuations.")
print("3. Volatility spikes during market stress periods (e.g., 2020, 2022).")
print("4. Correlation matrix shows tech giants move together (highly correlated).")
