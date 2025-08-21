import yfinance as yf
import os

TICKERS = ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN", "RELIANCE.NS", "INFY.NS", "TCS.NS"]
os.makedirs("cached_data", exist_ok=True)

print("Starting to cache stock data in a single, efficient batch request...")

try:
    # This is the key: Join tickers with a space and download all at once.
    # This sends only ONE request to Yahoo's servers.
    all_data = yf.download(TICKERS, period="1y", auto_adjust=True, progress=False)

    if not all_data.empty and 'Close' in all_data.columns:
        # Loop through the tickers to save each one to its own file
        for ticker in TICKERS:
            file_path = os.path.join("cached_data", f"{ticker.replace('.', '_')}.csv")
            
            # Select columns specifically for the current ticker
            # yfinance returns a multi-level column index: ('Close', 'AAPL'), ('Volume', 'MSFT'), etc.
            ticker_data = all_data.loc[:, all_data.columns.get_level_values(1) == ticker]
            
            # Check if there is actually data for this specific ticker in the result
            if not ticker_data.empty and not ticker_data.isnull().all().all():
                # Flatten the column names from ('Close', 'AAPL') to just 'Close'
                ticker_data.columns = ticker_data.columns.get_level_values(0)
                ticker_data.to_csv(file_path)
                print(f"✅ Successfully processed and saved data for {ticker}")
            else:
                 print(f"⚠️ No data was found for {ticker} within the batch download.")
    else:
        print("❌ The batch download failed to return any valid data.")

except Exception as e:
    print(f"❌ An error occurred during the batch download: {e}")

print("\nCaching process complete.")
