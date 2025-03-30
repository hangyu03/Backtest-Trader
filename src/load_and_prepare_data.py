import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

def mark_signal_oneday(df_in_one_day):
    """
    For one day of data (i.e., a subset of instruments all on the same date),
    sort by 'pred' and label the bottom/top 10% with signals -1 and +1.

    Returns
    -------
    pd.DataFrame
        The same data with an added 'signal' column where:
          - The bottom 10% (by pred) is assigned -1
          - The top 10%  (by pred) is assigned +1
          - The rest is 0
    """
    df = df_in_one_day.copy()
    df = df.sort_values("pred")
    n = len(df)
    bot_n = int(n * 0.1)
    top_n = int(n * 0.1)

    df["signal"] = 0
    df.iloc[:bot_n, df.columns.get_loc("signal")] = -1
    df.iloc[-top_n:, df.columns.get_loc("signal")] = 1

    return df

def main(result_file = "input/result0308v2.parquet", raw_file = "input/raw.parquet"):

    # 1) Read and clean 'result' data
    result = pd.read_parquet(result_file).drop(columns='ret', errors='ignore')

    # 2) Read and clean 'raw' data
    all_data = pd.read_parquet(raw_file)
    all_data = all_data.rename(columns={'ts_code': 'instrument', 'trade_date': 'Date'})
    all_data['instrument'] = all_data['instrument'].apply(
        lambda x: f"{x.split('.')[1]}{x.split('.')[0]}"
    )
    all_data['Date'] = pd.to_datetime(all_data['Date'].astype(str), format='%Y%m%d')
    all_data = all_data.sort_values(['instrument', 'Date'])

    merged = pd.merge(all_data, result, on=['instrument', 'Date'], how='inner')
    merged = merged[['instrument', 'Date', 'close', 'pred']]

    # 3) Group by date, assign signals (top 10% = +1, bottom 10% = -1)
    backtest = merged.groupby("Date", group_keys=False).apply(mark_signal_oneday)
    backtest = backtest.sort_values(["instrument", "Date"])
    backtest['signal'] = backtest.groupby('instrument')['signal'].shift(1)
    backtest.dropna(inplace=True)
    os.makedirs('output', exist_ok=True)
    backtest.to_csv('output/backtest.csv',index=False)


if __name__ == "__main__":
    main()
    
