import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import mplfinance as mpf
from matplotlib.patches import Rectangle


end_date = datetime.now()
start_date = end_date - timedelta(days=180) 
interval = "1h"

print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

try:
    print("Downloading NAS100...")
    nas100 = yf.download("^NDX", start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True)
    print("Downloading US30...")
    us30 = yf.download("^DJI", start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True)
    print("Downloading S&P500...")
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True)
    

    if isinstance(nas100.columns, pd.MultiIndex):
        nas100.columns = nas100.columns.droplevel(1)
    if isinstance(us30.columns, pd.MultiIndex):
        us30.columns = us30.columns.droplevel(1)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.droplevel(1)
    

    if nas100.empty:
        print("ERROR: NAS100 download failed")
        exit()
    if us30.empty:
        print("WARNING: US30 download failed, trying alternative method...")
        
        import time
        time.sleep(2)  
        us30 = yf.download("^DJI", start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True, timeout=30)
        if isinstance(us30.columns, pd.MultiIndex):
            us30.columns = us30.columns.droplevel(1)
        if us30.empty:
            print("ERROR: US30 download failed completely. Proceeding with S&P500 comparison only.")
    if sp500.empty:
        print("ERROR: S&P500 download failed")
        exit()
    
    print("SUCCESS: Data downloaded successfully")
    print(f"NAS100 columns: {list(nas100.columns)}")
except Exception as e:
    print(f"ERROR: Error downloading data: {e}")
    exit()


print(f"Original data lengths: NAS100={len(nas100)}, US30={len(us30)}, SP500={len(sp500)}")


if not us30.empty:
    nas100, us30 = nas100.align(us30, join='inner')
    nas100, sp500 = nas100.align(sp500, join='inner')
    us30, sp500 = us30.align(sp500, join='inner')
    print(f"Aligned data length: {len(nas100)}")
else:
    nas100, sp500 = nas100.align(sp500, join='inner')
    print(f"Aligned data length (NAS100 vs SP500 only): {len(nas100)}")
    print("WARNING: US30 data not available, using S&P500 comparison only")


def detect_smt_divergence(primary, secondary, lookback=3):
    primary_low_rolling = primary['Low'].rolling(window=lookback).min()
    secondary_low_rolling = secondary['Low'].rolling(window=lookback).min()
    
    primary_new_low = primary['Low'] == primary_low_rolling
    secondary_no_new_low = secondary['Low'] > secondary_low_rolling
    
    smt_divergence = primary_new_low & secondary_no_new_low
    
    return smt_divergence.astype(int)

def detect_smt_bullish_divergence(primary, secondary, lookback=3):
    primary_high_rolling = primary['High'].rolling(window=lookback).max()
    secondary_high_rolling = secondary['High'].rolling(window=lookback).max()
    
    primary_new_high = primary['High'] == primary_high_rolling
    secondary_no_new_high = secondary['High'] < secondary_high_rolling
    
    bullish_divergence = primary_new_high & secondary_no_new_high
    
    return bullish_divergence.astype(int)

if not us30.empty:
    nas100['SMT_US30_Bearish'] = detect_smt_divergence(nas100, us30)
    nas100['SMT_US30_Bullish'] = detect_smt_bullish_divergence(nas100, us30)
else:
    nas100['SMT_US30_Bearish'] = 0
    nas100['SMT_US30_Bullish'] = 0

nas100['SMT_SP500_Bearish'] = detect_smt_divergence(nas100, sp500)
nas100['SMT_SP500_Bullish'] = detect_smt_bullish_divergence(nas100, sp500)

nas100['SMT_Bearish_Confirmed'] = (nas100['SMT_US30_Bearish'] | nas100['SMT_SP500_Bearish'])
nas100['SMT_Bullish_Confirmed'] = (nas100['SMT_US30_Bullish'] | nas100['SMT_SP500_Bullish'])

bearish_signals = nas100['SMT_Bearish_Confirmed'].sum()
bullish_signals = nas100['SMT_Bullish_Confirmed'].sum()
print(f"\nSMT Signal Summary:")
print(f"Bearish Divergences: {bearish_signals}")
print(f"Bullish Divergences: {bullish_signals}")

def create_candlestick_chart_with_signals(data, title, signals_bearish=None, signals_bullish=None):
    ohlcv_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    apds = [] 
    

    if signals_bearish is not None:
        bearish_points = data[signals_bearish == 1]
        if len(bearish_points) > 0:
            bearish_markers = pd.Series(index=data.index, dtype=float)
            bearish_markers.loc[bearish_points.index] = bearish_points['High'] * 1.002  
            apds.append(mpf.make_addplot(bearish_markers, type='scatter', markersize=100, 
                                       marker='v', color='red', alpha=0.8))
    

    if signals_bullish is not None:
        bullish_points = data[signals_bullish == 1]
        if len(bullish_points) > 0:
            bullish_markers = pd.Series(index=data.index, dtype=float)
            bullish_markers.loc[bullish_points.index] = bullish_points['Low'] * 0.998 
            apds.append(mpf.make_addplot(bullish_markers, type='scatter', markersize=100, 
                                       marker='^', color='green', alpha=0.8))
    
    mpf.plot(ohlcv_data, type='candle', style='yahoo', title=title,
             volume=True, addplot=apds if apds else None,
             figsize=(16, 10), tight_layout=True)

print("\nGenerating candlestick chart with SMT signals...")
create_candlestick_chart_with_signals(
    nas100, 
    "NAS100 Candlestick Chart with SMT Divergence Signals",
    signals_bearish=nas100['SMT_Bearish_Confirmed'],
    signals_bullish=nas100['SMT_Bullish_Confirmed']
)


def create_multi_asset_comparison():
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    
    def plot_candlestick(ax, data, title, color_up='green', color_down='red'):
        for i in range(len(data)):
            row = data.iloc[i]
            x = i
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            color = color_up if close_price >= open_price else color_down
            
            ax.plot([x, x], [low_price, high_price], color='black', linewidth=0.8)
            
        
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            if close_price >= open_price:
                ax.add_patch(Rectangle((x-0.3, body_bottom), 0.6, body_height, 
                                     facecolor=color, edgecolor='black', linewidth=0.5))
            else:
                ax.add_patch(Rectangle((x-0.3, body_bottom), 0.6, body_height, 
                                     facecolor=color, edgecolor='black', linewidth=0.5))
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-10, len(data) + 10)
    
    sample_size = min(100, len(nas100))
    nas100_sample = nas100.tail(sample_size)
    
    plot_candlestick(axes[0], nas100_sample, "NAS100 (NASDAQ 100)")
    
    bearish_signals = nas100_sample[nas100_sample['SMT_Bearish_Confirmed'] == 1]
    bullish_signals = nas100_sample[nas100_sample['SMT_Bullish_Confirmed'] == 1]
    
    if len(bearish_signals) > 0:
        bearish_indices = [nas100_sample.index.get_loc(idx) for idx in bearish_signals.index]
        axes[0].scatter(bearish_indices, bearish_signals['High'] * 1.002, 
                       marker='v', color='red', s=100, zorder=5, label='Bearish SMT')
    
    if len(bullish_signals) > 0:
        bullish_indices = [nas100_sample.index.get_loc(idx) for idx in bullish_signals.index]
        axes[0].scatter(bullish_indices, bullish_signals['Low'] * 0.998, 
                       marker='^', color='green', s=100, zorder=5, label='Bullish SMT')
    
    axes[0].legend()
    
    if not us30.empty:
        us30_sample = us30.tail(sample_size)
        plot_candlestick(axes[1], us30_sample, "US30 (Dow Jones)")
    else:
        axes[1].text(0.5, 0.5, 'US30 Data Not Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title("US30 (Dow Jones) - Data Not Available")
    
    sp500_sample = sp500.tail(sample_size)
    plot_candlestick(axes[2], sp500_sample, "S&P500")
    
    plt.tight_layout()
    plt.show()

print("Generating multi-asset comparison chart...")
create_multi_asset_comparison()

signal_summary = nas100[['Open', 'High', 'Low', 'Close', 'Volume', 
                        'SMT_Bearish_Confirmed', 'SMT_Bullish_Confirmed']].copy()
signal_summary = signal_summary[(signal_summary['SMT_Bearish_Confirmed'] == 1) | 
                               (signal_summary['SMT_Bullish_Confirmed'] == 1)]

if len(signal_summary) > 0:
    print(f"\nRecent SMT Signals:")
    print(signal_summary.tail(10).to_string())
    
    
else:
    print("\nNo SMT signals detected in the current timeframe")

def backtest_smt_strategy(data, starting_capital=10000, risk_per_trade=0.02, 
                         stop_loss_pct=0.02, take_profit_pct=0.04, hold_periods=24):
    
    capital = starting_capital
    trades = []
    equity_curve = []
    current_positions = []
    
    for i in range(len(data)):
        current_time = data.index[i]
        current_price = data.iloc[i]['Close']
        
        positions_to_close = []
        for pos in current_positions:
            if pos['type'] == 'LONG':
                unrealized_pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            else:
                unrealized_pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']
            
            should_exit = False
            exit_reason = ""
            
            if i >= pos['entry_index'] + hold_periods:
                should_exit = True
                exit_reason = "TIME_EXIT"
            
            elif unrealized_pnl_pct <= -stop_loss_pct:
                should_exit = True
                exit_reason = "STOP_LOSS"
            
            elif unrealized_pnl_pct >= take_profit_pct:
                should_exit = True
                exit_reason = "TAKE_PROFIT"
            
            if should_exit:
                realized_pnl = pos['position_size'] * unrealized_pnl_pct
                capital += realized_pnl
                
                trade_record = {
                    'entry_time': pos['entry_time'],
                    'exit_time': current_time,
                    'type': pos['type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'position_size': pos['position_size'],
                    'return_pct': unrealized_pnl_pct * 100,
                    'pnl': realized_pnl,
                    'exit_reason': exit_reason,
                    'capital_after': capital
                }
                trades.append(trade_record)
                positions_to_close.append(pos)
        
        for pos in positions_to_close:
            current_positions.remove(pos)
        
        if i < len(data) - hold_periods: 
            bearish_signal = data.iloc[i]['SMT_Bearish_Confirmed']
            bullish_signal = data.iloc[i]['SMT_Bullish_Confirmed']
            
            risk_amount = capital * risk_per_trade
            position_size = risk_amount / stop_loss_pct  
            
            if bearish_signal == 1 and len(current_positions) < 3: 
                position = {
                    'entry_time': current_time,
                    'entry_index': i,
                    'entry_price': current_price,
                    'position_size': position_size,
                    'type': 'SHORT'
                }
                current_positions.append(position)
            
            elif bullish_signal == 1 and len(current_positions) < 3:  
                position = {
                    'entry_time': current_time,
                    'entry_index': i,
                    'entry_price': current_price,
                    'position_size': position_size,
                    'type': 'LONG'
                }
                current_positions.append(position)
        

        current_equity = capital
        for pos in current_positions:
            if pos['type'] == 'LONG':
                unrealized_pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            else:
                unrealized_pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']
            
            unrealized_pnl = pos['position_size'] * unrealized_pnl_pct
            current_equity += unrealized_pnl
        
        equity_curve.append({
            'time': current_time,
            'capital': capital,
            'equity': current_equity,
            'open_positions': len(current_positions)
        })
    
    return trades, equity_curve

print("\n" + "="*50)
print("BACKTESTING SMT STRATEGY")
print("="*50)

STARTING_CAPITAL = 10000  
RISK_PER_TRADE = 0.02     
STOP_LOSS = 0.02         
TAKE_PROFIT = 0.04       
HOLD_PERIODS = 24         

print(f"Starting Capital: ${STARTING_CAPITAL:,.2f}")
print(f"Risk per Trade: {RISK_PER_TRADE*100:.1f}%")
print(f"Stop Loss: {STOP_LOSS*100:.1f}%")
print(f"Take Profit: {TAKE_PROFIT*100:.1f}%")
print(f"Hold Period: {HOLD_PERIODS} hours")

trades, equity_curve = backtest_smt_strategy(
    nas100, 
    starting_capital=STARTING_CAPITAL,
    risk_per_trade=RISK_PER_TRADE,
    stop_loss_pct=STOP_LOSS,
    take_profit_pct=TAKE_PROFIT,
    hold_periods=HOLD_PERIODS
)

if trades:
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['return_pct'] > 0])
    losing_trades = len([t for t in trades if t['return_pct'] < 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    total_pnl = sum([t['pnl'] for t in trades])
    avg_win = np.mean([t['return_pct'] for t in trades if t['return_pct'] > 0]) if winning_trades > 0 else 0
    avg_loss = np.mean([t['return_pct'] for t in trades if t['return_pct'] < 0]) if losing_trades > 0 else 0
    
    final_capital = trades[-1]['capital_after']
    total_return = ((final_capital - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
    
    returns = [t['return_pct'] for t in trades]
    max_drawdown = 0
    peak_capital = STARTING_CAPITAL
    
    for trade in trades:
        if trade['capital_after'] > peak_capital:
            peak_capital = trade['capital_after']
        drawdown = ((peak_capital - trade['capital_after']) / peak_capital) * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    print(f"\n" + "="*30)
    print("BACKTEST RESULTS")
    print("="*30)
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Average Win: {avg_win:.2f}%")
    print(f"Average Loss: {avg_loss:.2f}%")
    print(f"Profit Factor: {abs(avg_win/avg_loss) if avg_loss != 0 else 'N/A':.2f}")
    print(f"\nStarting Capital: ${STARTING_CAPITAL:,.2f}")
    print(f"Final Capital: ${final_capital:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    
    print(f"\n" + "="*30)
    print("PROFITABILITY ANALYSIS")
    print("="*30)
    if total_return > 0:
        print(f"RESULT: PROFITABLE SYSTEM")
        print(f"The strategy generated a {total_return:.2f}% return")
        print(f"Capital grew from ${STARTING_CAPITAL:,.2f} to ${final_capital:,.2f}")
    else:
        print(f"RESULT: UNPROFITABLE SYSTEM")
        print(f"The strategy lost {abs(total_return):.2f}% of capital")
        print(f"Capital decreased from ${STARTING_CAPITAL:,.2f} to ${final_capital:,.2f}")
    
    print(f"\nRecent Trades (Last 10):")
    print("-" * 80)
    recent_trades = trades_df.tail(10)
    for _, trade in recent_trades.iterrows():
        print(f"{trade['entry_time'].strftime('%Y-%m-%d %H:%M')} | {trade['type']} | "
              f"Entry: ${trade['entry_price']:.2f} | Exit: ${trade['exit_price']:.2f} | "
              f"Return: {trade['return_pct']:.2f}% | P&L: ${trade['pnl']:.2f} | "
              f"Reason: {trade['exit_reason']}")
    
    def create_equity_curve_with_trades():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        sample_size = min(200, len(nas100)) 
        nas100_sample = nas100.tail(sample_size)
        
        for i in range(len(nas100_sample)):
            row = nas100_sample.iloc[i]
            x = i
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            color = 'green' if close_price >= open_price else 'red'
            

            ax1.plot([x, x], [low_price, high_price], color='black', linewidth=0.8)
            
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            ax1.add_patch(Rectangle((x-0.3, body_bottom), 0.6, body_height, 
                                  facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7))
        
        sample_start_time = nas100_sample.index[0]
        sample_trades = [t for t in trades if t['entry_time'] >= sample_start_time]
        
        for trade in sample_trades:
            if trade['entry_time'] in nas100_sample.index:
                entry_idx = nas100_sample.index.get_loc(trade['entry_time'])
                entry_price = trade['entry_price']
                
                if trade['type'] == 'LONG':
                    ax1.scatter(entry_idx, entry_price, marker='^', color='blue', s=100, 
                               label='Long Entry' if trade == sample_trades[0] else "", zorder=5)
                else:
                    ax1.scatter(entry_idx, entry_price, marker='v', color='orange', s=100, 
                               label='Short Entry' if trade == sample_trades[0] else "", zorder=5)
        
        ax1.set_title("NAS100 Candlestick Chart with Trade Entries", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Price", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-5, len(nas100_sample) + 5)
        
        equity_df = pd.DataFrame(equity_curve)
        ax2.plot(equity_df['time'], equity_df['equity'], label='Portfolio Equity', color='blue', linewidth=2)
        ax2.axhline(y=STARTING_CAPITAL, color='red', linestyle='--', label=f'Starting Capital (${STARTING_CAPITAL:,.0f})')
        ax2.fill_between(equity_df['time'], STARTING_CAPITAL, equity_df['equity'], 
                        where=(equity_df['equity'] >= STARTING_CAPITAL), color='green', alpha=0.3, label='Profit')
        ax2.fill_between(equity_df['time'], STARTING_CAPITAL, equity_df['equity'], 
                        where=(equity_df['equity'] < STARTING_CAPITAL), color='red', alpha=0.3, label='Loss')
        
        ax2.set_title(f"Portfolio Equity Curve - Final: ${final_capital:,.2f} ({total_return:+.2f}%)", 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Equity ($)", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print("\nGenerating enhanced equity curve with trade markers...")
    create_equity_curve_with_trades()

else:
    print("\nNo trades were executed during the backtest period.")
    print("This could be due to:")
    print("- No SMT signals detected")
    print("- Insufficient data for the holding period")
    print("- Restrictive risk management parameters")

buy_hold_return = ((nas100['Close'].iloc[-1] - nas100['Close'].iloc[0]) / nas100['Close'].iloc[0]) * 100
print(f"\nBuy & Hold Comparison:")
print(f"SMT Strategy Return: {total_return:.2f}% vs Buy & Hold Return: {buy_hold_return:.2f}%")
if total_return > buy_hold_return:
    print("SMT Strategy OUTPERFORMED Buy & Hold")
else:
    print("SMT Strategy UNDERPERFORMED Buy & Hold")

print("\n" + "="*50)
print("INSTALLATION NOTE:")
print("="*50)
print("To run this script, you'll need to install mplfinance:")
print("pip install mplfinance")
print("="*50)