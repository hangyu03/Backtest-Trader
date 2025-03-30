def task_load_and_prepare_data():
    """
    Runs load_and_prepare_data.py, which reads from `input/` and writes to `output/`.
    """
    return {
        # 1) The command line action to run
        "actions": [
            "python src/load_and_prepare_data.py"
        ],
        # 2) Dependencies 
        "file_dep": [
            "src/load_and_prepare_data.py",
            "input/result0308v2.parquet",
            "input/raw.parquet"
        ],
        # 3) Targets (files produced by this script)
        "targets": [
            "output/backtest.csv"
        ],
        # (Optional) Increase verbosity so you can see what's happening
        "verbosity": 2,
    }


def task_backtest_cash():
    """
    Runs the backtest_cash.py script, which reads 'output/backtest.csv' and
    outputs daily trade info and plots in 'output/with_cost/' or 'output/without_cost/'.
    """
    return {
        # 1) The shell command to run
        "actions": [
            "python src/backtest_cash.py"
        ],
        # 2) The file dependencies
        "file_dep": [
            "src/backtest_cash.py",
            "output/backtest.csv"
        ],
        # 3) The files it produces
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
        # (Optional) verbosity for logging
        "verbosity": 2,
    }


def task_backtest_net():
    """
    Runs backtest_net.py, which loads 'output/backtest.csv' and 'input/csi300.parquet'
    and produces results in 'output/with_cost/' or 'output/without_cost/'.
    """
    return {
        # 1) The shell command to run.
        "actions": [
            "python src/backtest_net.py"
        ],
        # 2) Dependencies: the Python script, plus the data files it needs.
        "file_dep": [
            "src/backtest_net.py",
            "output/backtest.csv",
            "input/csi300.parquet",
        ],
        # 3) Targets: the files the script produces.
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
        # (Optional) verbosity for logging
        "verbosity": 2,
    }


def task_all():
    return {
        "actions": [],  
        "task_dep": [
            # "load_and_prepare_data",
            # "backtest_cash",
            "backtest_net",
        ],
    }
