from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


class NetBacktestEngine:
    """
    Class-based backtester for demonstration of OOP style.
    """

    def __init__(self, backtest_df, cost=True, long_cost=0.0015, short_cost=0.003):
        """
        Initialize backtester with user-provided signal DataFrame.
        
        Parameters
        ----------
        backtest : pd.DataFrame
            DataFrame containing columns ['Date', 'instrument', 'pred', 'close'] 
            for signals and prices.
        cost : bool, optional
            Whether to apply transaction costs (default True).
        """
        self.backtest = backtest_df.copy()
        self.cost = cost
        self.long_cost = long_cost if cost else 0.0
        self.short_cost = short_cost if cost else 0.0
        self.index_df = pd.read_parquet('input/csi300.parquet')


    # ========== Metrics Calculation Methods ========== #
    @staticmethod
    def calc_return_metrics(data, adj = 52):
        """
        Calculate annualized return metrics (Annual Return, Annual Vol, Annual Sharpe, Annual Sortino).
        
        """
        summary = dict()
        summary["Annual Return"] = (data.mean() + 1) ** adj - 1
        summary["Annual Vol"] = data.std() * np.sqrt(adj)
        summary["Annual Sharpe"] = summary["Annual Return"] / summary["Annual Vol"]
        summary["Annual Sortino"] = summary["Annual Return"] / (
            data[data < 0].std() * np.sqrt(adj)
        )
        return pd.DataFrame(summary, index=data.columns)

    @staticmethod
    def calc_risk_metrics(data, var = 0.05):
        """
        Calculate risk metrics such as Skewness, Excess Kurtosis, VaR, CVaR, Max Drawdown.
        """
        summary = dict()
        summary["Skewness"] = data.skew()
        summary["Excess Kurtosis"] = data.kurtosis()
        summary[f"VaR ({var})"] = data.quantile(var, axis=0)
        summary[f"CVaR ({var})"] = data[data <= data.quantile(var, axis=0)].mean()

        # Calculate drawdowns
        wealth_index = 1000 * (1 + data).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        summary["Max Drawdown"] = drawdowns.min()

        return pd.DataFrame(summary, index=data.columns)

    @classmethod
    def calc_performance_metrics(cls, data, adj = 52, var = 0.05):
        """
        Convenience method to combine return & risk metrics, plus Calmar ratio.
        """
        return_summary = cls.calc_return_metrics(data, adj=adj)
        risk_summary = cls.calc_risk_metrics(data, var=var)
        summary = {**return_summary, **risk_summary}
        summary["Calmar Ratio"] = summary["Annual Return"] / abs(summary["Max Drawdown"])
        return pd.DataFrame(summary, index=data.columns).T

    # ========== Internal Utilities ========== #
    @staticmethod
    def get_trade(group):
        """
        Identify discrete trades based on signal changes.
        """
        df = group.copy()

        # Identify changes in signal
        df['prev_signal'] = df['signal'].shift(fill_value=0)
        df['segment_id'] = (df['signal'] != df['prev_signal']).cumsum()

        # Group by segments
        trades = []
        for _, seg_df in df.groupby('segment_id'):
            current_signal = seg_df['signal'].iloc[0]
            if current_signal == 0:
                continue
            total_ret = (seg_df['real_ret'] + 1).prod() - 1
            close_date = seg_df['Date'].iloc[-1]
            trades.append({
                'close_date': close_date,
                'instrument': seg_df['instrument'].iloc[0],
                'ret': total_ret,
                'signal': current_signal
            })

        return pd.DataFrame(trades)

    def process_single_day_backtest(self, group_num, signal):
        """
        Process returns for a single day-of-week (j).
        """
        backtest_df = signal.copy()
        backtest_df["Date"] = pd.to_datetime(backtest_df["Date"])
        backtest_df = backtest_df.sort_values(["Date", "instrument"]).reset_index(drop=True)

        unique_dates = backtest_df["Date"].drop_duplicates().sort_values().tolist()
        # Rebalance once a week
        date2idx = {d: i for i, d in enumerate(unique_dates) if i % 5 == group_num}
        backtest_df = backtest_df[backtest_df.Date.isin(date2idx.keys())]

        # Identify trades
        backtest_df = backtest_df.groupby('instrument').apply(self.get_trade).reset_index(drop=True)

        # PnL after costs
        backtest_df["pnl"] = (
            backtest_df["signal"] * backtest_df["ret"]
            - np.where(
                backtest_df["signal"] > 0,
                self.long_cost,
                np.where(backtest_df["signal"] < 0, self.short_cost, 0)
            ) * backtest_df["signal"].abs()
        )

        # Daily average PnL
        daily_pnl = (
            backtest_df.groupby("close_date")["pnl"]
            .mean()
            .fillna(0)
            .sort_index()
        )

        net_value = (1 + daily_pnl).cumprod()
        net_value /= net_value.iloc[0]  # normalize to start from 1
        return net_value, backtest_df['close_date']

    def plot_IC(self, signal):
        """
        Plot rolling IC (information coefficient) and save figure.
        """
        plt.figure(figsize=(13, 8))

        IC = signal.groupby('Date').apply(lambda x: x['real_ret'].corr(x['pred']))
        rolling_IC = IC.rolling(window=40).mean().dropna()
        rolling_IC.index = pd.to_datetime(rolling_IC.index)
        plt.title(f'Mean IC = {IC.mean().round(3)}', fontsize=22)
        plt.plot(rolling_IC, label='2-month Rolling IC', linewidth=2)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Rolling IC", fontsize=14)
        plt.legend(fontsize=14)
        plt.grid()
        os.makedirs('output', exist_ok=True)
        plt.savefig('output/IC.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ========== Main Driver Method ========== #
    def net_main(self):
        """
        Run the full backtest and generate plots & performance metrics.
        """
        # Shift predictions forward by 1 (avoid lookahead)
        self.backtest['pred'] = self.backtest.groupby('instrument')['pred'].shift(1)
        # Calculate future returns
        self.backtest['real_ret'] = self.backtest.groupby('instrument')['close'].transform(
            lambda x: (x.shift(-4) / x) - 1
        )
        self.backtest.dropna(inplace=True)

        desc_text = 'Computing Nets WITH Trading Costs' if self.cost else 'Computing Nets WITHOUT Trading Costs'
        output_dir = 'output/with_cost' if self.cost else 'output/without_cost'

        for group_num in tqdm(range(4), desc=desc_text):
            nv, _ = self.process_single_day_backtest(group_num, self.backtest)
            plt.figure(figsize=(13, 8))
            plt.title(f'Group {group_num} Net Curve', fontsize=22)
            plt.xlabel('Time', fontsize=14)
            plt.ylabel('Net Value', fontsize=14)
            plt.text(
                0.34, 0.955,
                f'Long cost = {self.long_cost*100:.2f}%, Short cost = {self.short_cost*100:.2f}%',
                transform=plt.gca().transAxes, fontsize=14,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor="lightgray")
            )
            plt.plot(nv, label=f'Group {group_num} Net', linewidth=2)
            perf_df = self.calc_performance_metrics(nv.pct_change().dropna().to_frame()).rename(columns={0: 'Metrics'})
            idx_slice = (self.index_df.trade_date >= nv.index.min()) & \
                        (self.index_df.trade_date <= nv.index.max())
            idx = self.index_df.copy()
            idx = idx[idx_slice].set_index('trade_date')
            index_net = (1 + idx.close.pct_change()).cumprod().fillna(1)
            plt.plot(index_net, label='CSI300 benchmark', linewidth=2)
            cell_text = perf_df.values.round(3)
            row_labels = perf_df.index.tolist()
            the_table = plt.table(
                cellText=cell_text,
                rowLabels=row_labels,
                bbox=[0.156, 0.68, 0.04, 0.3],
            )
            for (_, _), cell in the_table.get_celld().items():
                cell.set_linewidth(0)
                cell.set_edgecolor('lightgray')
                cell.set_alpha(1)
            the_table.scale(1.0, 200)
            the_table.set_zorder(2)
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(14)

            plt.tight_layout()
            plt.legend(fontsize=14, loc='lower left')
            plt.grid()
            os.makedirs(output_dir, exist_ok=True)
            if self.cost:
                plt.savefig(os.path.join(output_dir, f'group{group_num}_net_value_plot.png'), dpi=300, bbox_inches='tight')
            else:
                plt.savefig(os.path.join(output_dir, f'group{group_num}_net_value_plot_without_trading_costs.png'), dpi=300, bbox_inches='tight')
            plt.close()
        self.plot_IC(self.backtest)


if __name__ == "__main__":

    backtest_df = pd.read_csv('output/backtest.csv')
    for cost in [True, False]:
        cash_backtest_engine = NetBacktestEngine(backtest_df=backtest_df, cost=cost)
        cash_backtest_engine.net_main()