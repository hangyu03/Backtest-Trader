def task_load_and_prepare_data():
    """
    Runs load_and_prepare_data.py, which reads from `input/` and writes to `output/`.
    """
    return {
        "actions": [
            "python src/load_and_prepare_data.py"
        ],
        "file_dep": [
            "src/load_and_prepare_data.py",
            "input/result0308v2.parquet",
            "input/raw.parquet"
        ],
        "targets": [
            "output/backtest.csv"
        ],
        "verbosity": 2,
    }


def task_backtest_cash():
    """
    Runs the backtest_cash.py script, which reads 'output/backtest.csv' and
    outputs daily trade info and plots in 'output/with_cost/' or 'output/without_cost/'.
    """
    return {
        "actions": [
            "python src/backtest_cash.py"
        ],
        "file_dep": [
            "src/backtest_cash.py",
            "output/backtest.csv"
        ],
        "targets": [
            "output/with_cost/group_0_daily_trade_info.csv",
            "output/with_cost/group_1_daily_trade_info.csv",
            "output/with_cost/group_2_daily_trade_info.csv",
            "output/with_cost/group_3_daily_trade_info.csv",
            "output/with_cost/cash_plot.png",
            "output/without_cost/group_0_daily_trade_info_without_cost.csv",
            "output/without_cost/group_1_daily_trade_info_without_cost.csv",
            "output/without_cost/group_2_daily_trade_info_without_cost.csv",
            "output/without_cost/group_3_daily_trade_info_without_cost.csv",
            "output/without_cost/cash_plot_without_cash.png",
        ],
        "task_dep": ["load_and_prepare_data"],  
        "verbosity": 2,
    }


def task_backtest_net():
    """
    Runs backtest_net.py, which loads 'output/backtest.csv' and 'input/csi300.parquet'
    and produces results in 'output/with_cost/' or 'output/without_cost/'.
    """
    return {
        "actions": [
            "python src/backtest_net.py"
        ],
        "file_dep": [
            "src/backtest_net.py",
            "output/backtest.csv",
            "input/csi300.parquet",
        ],
        "targets": [
            "output/IC.png",
            'output/with_cost/group0_net_value_plot.png',
            'output/with_cost/group1_net_value_plot.png',
            'output/with_cost/group2_net_value_plot.png',
            'output/with_cost/group3_net_value_plot.png',
            'output/without_cost/group0_net_value_plot_without_trading_costs.png',
            'output/without_cost/group1_net_value_plot_without_trading_costs.png',
            'output/without_cost/group2_net_value_plot_without_trading_costs.png',
            'output/without_cost/group3_net_value_plot_without_trading_costs.png'
        ],
        "task_dep": ["load_and_prepare_data"],  
        "verbosity": 2,
    }


def task_all():
    """
    Meta-task to run everything. By depending on backtest_cash and backtest_net,
    it transitively depends on load_and_prepare_data as well.
    """
    return {
        "actions": [],
        "task_dep": [
            "backtest_cash",
            "backtest_net",
        ],
    }


 
