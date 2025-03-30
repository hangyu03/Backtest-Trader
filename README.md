1. **Installation and Execution** (takes about **30 seconds**):

   ```bash
   cd /path/to/project
   pip install -r requirements.txt
   python -m doit

2. **Backtest Logic** (see **backtest_cash.py** and **backtest_net.py** for implementation details)

   1) Each trading day, we rank stocks by the previous day's signal value. 
      - **Top 10%**: go long  
      - **Bottom 10%**: go short  
      Positions are closed exactly 4 trading days later.

   2) Because we open new positions every day and each position is held for 4 days, there are effectively **4 overlapping branches** of positions (one for each weekday). Each branch is closed on its corresponding day in the next week.

   3) If a position's direction (long/short) for a given asset does not change from day to day, we continue to hold it rather than re-enter. 

   4) **Transaction Costs**:
      - **Long cost**: 0.15% per trade  
        (0.025% commission for both buy & sell + 0.1% stamp duty on selling)  
      - **Short cost**: 0.3% per trade  
        (includes commission, stamp duty, and annualized securities lending rate of 8% divided by 52)

   5) **Important Constraint**: Since the model forecasts 4-day returns, each branch is strictly closed after exactly 1 week (no partial closings or cross-branch adjustments). In a live trading scenario, this rule might be relaxed to accommodate real-world conditions.


