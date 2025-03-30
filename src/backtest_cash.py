import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')


class CashBacktestEngine:
    def __init__(
        self,
        backtest_df,
        cost=True,
        initial_capital=1_000_000,
        trade_size=35_000,
        commission_rate=0.00025,
        stamp_duty_rate=0.001,
        annual_short_interest=0.08,
        input_dir=None,
        output_dir=None
    ):
        """
        Initialize the cash backtest engine.

        :param backtest_df: The input dataframe containing the signals and prices for backtesting.
        :param cost: Boolean indicating whether costs should be considered in the backtest.
        :param initial_capital: The starting capital.
        :param trade_size: The notional amount to trade each time a signal is triggered.
        :param commission_rate: Commission rate for transactions.
        :param stamp_duty_rate: Stamp duty rate for sell trades.
        :param annual_short_interest: Annualized interest rate for short positions.
        :param input_dir: Optional directory for inputs (if needed).
        :param output_dir: Optional directory for outputs. If not provided,
                           defaults to "<script_dir>/../output".
        """
        self.backtest_df = backtest_df
        self.initial_capital = initial_capital
        self.trade_size = trade_size
        self.cost = cost
        self.commission_rate = commission_rate if cost else 0
        self.stamp_duty_rate = stamp_duty_rate if cost else 0
        self.annual_short_interest = annual_short_interest if cost else 0

        base_dir = os.path.dirname(os.path.abspath(__file__))
        if output_dir is None:
            output_dir = os.path.join(base_dir, '..', 'output')
        self.output_dir = output_dir

        if input_dir is None:
            input_dir = os.path.join(base_dir, '..', 'input')
        self.input_dir = input_dir

    def get_trade(self, signal_df, group_num):
        """
        Extract and structure the trades (open and close) based on signal data.

        :param signal_df: Dataframe with signals for each instrument.
        :param group_num: Used to select a subset of trading dates (0, 1, 2, or 3).
        :return: A dataframe with open and close trade records.
        """
        trade_df = []
        unique_dates = signal_df["Date"].drop_duplicates().sort_values().tolist()
        date2idx = {d: i for i, d in enumerate(unique_dates) if i % 4 == group_num}

        for _, df in signal_df.groupby('instrument'):
            df = df[df.Date.isin(date2idx.keys())]
            df['close_date'] = df['Date'].shift(-1)
            df['close_close'] = df['close'].shift(-1)
            df['prev_signal'] = df['signal'].shift(fill_value=0)
            df['segment_id'] = (df['signal'] != df['prev_signal']).cumsum()
            df = df.drop(columns='prev_signal').dropna()

            for _, seg_df in df.groupby('segment_id'):
                current_signal = seg_df['signal'].iloc[0]
                if current_signal == 0:
                    continue
                else:
                    open_df = seg_df.iloc[0]
                    close_df = seg_df.iloc[-1]
                    if current_signal == 1:
                        trade_open = {
                            'ticker': open_df['instrument'],
                            'Date': open_df['Date'],
                            'close': open_df['close'],
                            'action': 'Buy to Open'
                        }
                        trade_close = {
                            'ticker': close_df['instrument'],
                            'Date': close_df['close_date'],
                            'close': close_df['close_close'],
                            'action': 'Sell to Close'
                        }
                    elif current_signal == -1:
                        trade_open = {
                            'ticker': open_df['instrument'],
                            'Date': open_df['Date'],
                            'close': open_df['close'],
                            'action': 'Short Sell'
                        }
                        trade_close = {
                            'ticker': close_df['instrument'],
                            'Date': close_df['close_date'],
                            'close': close_df['close_close'],
                            'action': 'Buy to Cover'
                        }
                    trade_df.append(trade_open)
                    trade_df.append(trade_close)

        trade_df = pd.DataFrame(trade_df)
        trade_df = trade_df.sort_values(['Date', 'ticker'])
        return trade_df

    def backtest(self, trade_df):
        """
        Perform a backtest on the provided trade dataframe.

        :param trade_df: Dataframe containing the trades (open and close).
        :return: A daily-level dataframe summarizing commission, stamp duty, interest cost, PnL changes, and cash.
        """
        # Convert to datetime
        trade_df['Date'] = pd.to_datetime(trade_df['Date'])

        # Initialize
        cash = self.initial_capital
        # positions: key is (ticker, side='long'/'short'), value is a dict with relevant trade info
        positions = {}
        records = []

        def day_diff(d1, d2):
            return (d2 - d1).days

        daily_interest_rate = self.annual_short_interest / 365.0

        for i, row in trade_df.iterrows():
            date = row['Date']
            ticker = row['ticker']
            close_price = row['close']
            action = row['action']

            before_cash = cash
            trade_shares = 0
            trade_amount = 0
            commission = 0
            stamp_duty = 0
            interest_cost = 0  # short interest
            pnl_change = 0

            # ========== Open Long ==========
            if action == 'Buy to Open':
                if cash >= self.trade_size:
                    shares = np.floor(self.trade_size / close_price)
                    cost = shares * close_price
                    commission = cost * self.commission_rate
                    stamp_duty = 0  # No stamp duty for buying
                    total_cost = cost + commission + stamp_duty

                    if cash >= total_cost:
                        cash -= total_cost
                        positions[(ticker, 'long')] = {
                            'shares': shares,
                            'open_price': close_price,
                            'open_date': date
                        }
                        trade_shares = shares
                        trade_amount = cost

            # ========== Close Long ==========
            elif action == 'Sell to Close':
                pos_key = (ticker, 'long')
                if pos_key in positions:
                    shares = positions[pos_key]['shares']
                    open_price = positions[pos_key]['open_price']
                    proceeds = shares * close_price
                    commission = proceeds * self.commission_rate
                    stamp_duty = proceeds * self.stamp_duty_rate
                    total_fee = commission + stamp_duty
                    net_proceeds = proceeds - total_fee

                    cash += net_proceeds
                    gross_pnl = (close_price - open_price) * shares
                    pnl_change = gross_pnl - total_fee
                    del positions[pos_key]
                    trade_shares = shares
                    trade_amount = proceeds

            # ========== Open Short ==========
            elif action == 'Short Sell':
                pos_key = (ticker, 'short')
                if cash >= self.trade_size:
                    shares = np.floor(self.trade_size / close_price)
                    proceeds = shares * close_price
                    commission = proceeds * self.commission_rate
                    stamp_duty = proceeds * self.stamp_duty_rate
                    total_fee = commission + stamp_duty
                    net_proceeds = proceeds - total_fee

                    # Freeze some margin
                    cash_change = net_proceeds - 0.7 * self.trade_size
                    cash += cash_change

                    positions[pos_key] = {
                        'shares': shares,
                        'open_price': close_price,
                        'open_date': date,
                        'notional': proceeds
                    }
                    trade_shares = shares
                    trade_amount = proceeds

            # ========== Close Short ==========
            elif action == 'Buy to Cover':
                pos_key = (ticker, 'short')
                if pos_key in positions:
                    shares = positions[pos_key]['shares']
                    open_price = positions[pos_key]['open_price']
                    open_date = positions[pos_key]['open_date']
                    notional = positions[pos_key]['notional']

                    cost = shares * close_price
                    commission = cost * self.commission_rate
                    stamp_duty = 0
                    total_cost = cost + commission

                    cash_change = 0.7 * self.trade_size - total_cost
                    cash += cash_change

                    days_held = day_diff(open_date, date)
                    if days_held < 0:
                        days_held = 0
                    interest_cost = notional * daily_interest_rate * days_held
                    cash -= interest_cost

                    gross_pnl = notional - cost
                    pnl_change = gross_pnl - commission - interest_cost

                    del positions[pos_key]
                    trade_shares = shares
                    trade_amount = cost

            after_cash = cash
            trade_record = {
                'Date': date,
                'ticker': ticker,
                'action': action,
                'close_price': close_price,
                'before_cash': before_cash,
                'trade_shares': trade_shares,
                'trade_amount': trade_amount,
                'commission': commission,
                'stamp_duty': stamp_duty,
                'interest_cost': interest_cost,
                'pnl_change': pnl_change,
                'after_cash': after_cash
            }
            records.append(trade_record)

        result_df = pd.DataFrame(records)
        daily_df = result_df.groupby('Date').agg({
            'commission': 'sum',
            'stamp_duty': 'sum',
            'interest_cost': 'sum',
            'pnl_change': 'sum',
            'before_cash': 'first',
            'after_cash': 'last',
            'trade_amount': 'sum',
        }).reset_index()

        return daily_df

    def cash_main(self):
        """
        Main workflow to run the backtest for each of the four groups.
        """
        desc_text = 'Computing Cash WITH Trading Costs' if self.cost else 'Computing Cash WITHOUT Trading Costs'
        cost_dir = 'with_cost' if self.cost else 'without_cost'
        out_dir = os.path.join(self.output_dir, cost_dir)

        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(13, 8))
        plt.title("Cash Curve", fontsize=22)

        for group_num in tqdm(range(4), desc=desc_text):
            # Get the trade log for this group and save
            trade_df = self.get_trade(self.backtest_df, group_num)
            if self.cost:
                trade_df.to_csv(
                    os.path.join(out_dir, f'group_{group_num}_daily_transaction_log.csv'),
                    index=False
                )
            else:
                trade_df.to_csv(
                    os.path.join(out_dir, f'group_{group_num}_daily_transaction_log_without_trading_costs.csv'),
                    index=False
                )

            # run backtest
            daily_df = self.backtest(trade_df)

            # process the daily_df to get cash plot
            start_date = daily_df['Date'].min()
            end_date = daily_df['Date'].max()
            all_dates = pd.date_range(start_date, end_date, freq='D')  

            all_days_df = pd.DataFrame({'Date': all_dates})
            merged_df = pd.merge(all_days_df, daily_df, on='Date', how='left')
            merged_df = merged_df.sort_values('Date')
            merged_df['after_cash'] = merged_df['after_cash'].ffill()
            merged_df['after_cash'] = merged_df['after_cash'].fillna(self.initial_capital)
            daily_cash = merged_df[['Date', 'after_cash']].set_index('Date')

            first_day = daily_cash.index[0]
            new_day = first_day - pd.Timedelta(days=6)
            new_row = pd.DataFrame({"after_cash": [self.initial_capital]}, index=[new_day])
            daily_cash = pd.concat([new_row, daily_cash]).after_cash

            plt.plot(daily_cash, label=f'Cash Over Trades (group {group_num})')
            plt.gca().ticklabel_format(style='plain', axis='y')

        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Cash", fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True)

        if self.cost:
            plt.savefig(os.path.join(out_dir, 'cash_plot.png'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(out_dir, 'cash_plot_cash_without_trading_costs.png'), dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = os.path.join(base_dir, '..', 'output')
    backtest_path = os.path.join(default_input_dir, 'backtest.csv')
    backtest_df = pd.read_csv(backtest_path)

    for cost_setting in [True, False]:
        engine = CashBacktestEngine(
            backtest_df=backtest_df,
            cost=cost_setting,
        )
        engine.cash_main()
