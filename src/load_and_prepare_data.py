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

def main(
    input_dir=None, 
    output_dir=None,
    result_file="result0308v2.parquet", 
    raw_file="raw.parquet"
):

    if input_dir is None or output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if input_dir is None:
            input_dir = os.path.join(base_dir, '..', 'input')
        if output_dir is None:
            output_dir = os.path.join(base_dir, '..', 'output')

    result_path = (
        result_file 
        if os.path.isabs(result_file) else os.path.join(input_dir, result_file)
    )
    raw_path = (
        raw_file 
        if os.path.isabs(raw_file) else os.path.join(input_dir, raw_file)
    )

    result = pd.read_parquet(result_path).drop(columns='ret', errors='ignore')

    all_data = pd.read_parquet(raw_path)
    all_data = all_data.rename(columns={'ts_code': 'instrument', 'trade_date': 'Date'})
    all_data['instrument'] = all_data['instrument'].apply(
        lambda x: f"{x.split('.')[1]}{x.split('.')[0]}"
    )
    all_data['Date'] = pd.to_datetime(all_data['Date'].astype(str), format='%Y%m%d')
    all_data = all_data.sort_values(['instrument', 'Date'])

    merged = pd.merge(all_data, result, on=['instrument', 'Date'], how='inner')
    merged = merged[['instrument', 'Date', 'close', 'pred']]
    backtest = merged.groupby("Date", group_keys=False).apply(mark_signal_oneday)
    backtest = backtest.sort_values(["instrument", "Date"])
    backtest['signal'] = backtest.groupby('instrument')['signal'].shift(1)
    backtest.dropna(inplace=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'backtest.csv')
    backtest.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()

    
