import os
import pandas as pd
import joblib
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb

# Ensure scripts are importable
sys.path.append(os.getcwd())

from backtester_v1.scripts.feature_engineering import build_features
from backtester_v1.scripts.backtester import load_model, run_backtest
from backtester_v1.scripts.report import compute_metrics

def generate_interactive_plot(state, ohlcv_df, features_df, model, calibration_factor, symbol, metrics):
    # Calculate Predicted vs Actual Ratios for Scatter Plot
    dmatrix = xgb.DMatrix(features_df)
    bps_raw = model.predict(dmatrix)
    
    # Sensitivity boost
    sensitivity_boost = 2.0
    bps_cal = (bps_raw * calibration_factor) * sensitivity_boost
    pred_ratios = 1.0 + (bps_cal / 10000.0)
    actual_ratios = ohlcv_df['close'].shift(-1) / ohlcv_df['close']
    
    # Create DataFrame for scatter plotting
    scatter_df = pd.DataFrame({
        'predicted': pred_ratios,
        'actual': actual_ratios,
        'action': 'none',
        'timestamp': ohlcv_df.index
    })
    
    for trade in state.trades:
        scatter_df.loc[scatter_df['timestamp'] == trade.entry_ts, 'action'] = 'entry'
        scatter_df.loc[scatter_df['timestamp'] == trade.exit_ts, 'action'] = 'exit'
    
    clean_scatter = scatter_df.dropna()

    # Create Subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f"{symbol} Cumulative PnL Over Time",
            f"{symbol} Trade Executions",
            f"{symbol} Predictive Accuracy & Trade Distribution"
        ),
        vertical_spacing=0.1,
        shared_xaxes=False
    )

    # Convert to standard Python lists for Plotly stability
    timestamps_list = state.timestamps
    pnl_list = (np.array(state.equity_curve) - state.equity_curve[0]).tolist()
    price_indices = ohlcv_df.index.tolist()
    price_values = ohlcv_df['close'].tolist()

    # Subplot 1: Equity Curve
    fig.add_trace(
        go.Scatter(x=timestamps_list, y=pnl_list, name="Net PnL ($)", line=dict(color='blue')),
        row=1, col=1
    )
    fig.update_yaxes(title_text="PnL ($)", row=1, col=1)

    # Subplot 2: Price and Trades
    fig.add_trace(
        go.Scatter(x=price_indices, y=price_values, name="Spot Price", line=dict(color='gray', width=1), opacity=0.6),
        row=2, col=1
    )
    
    # Add trades individually to ensure visibility, but with markers
    for trade in state.trades:
        color = 'green' if trade.direction == 1 else 'red'
        # Line
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_ts, trade.exit_ts],
                y=[trade.entry_price, trade.exit_price],
                mode='lines+markers',
                line=dict(color=color, dash='dash', width=2),
                marker=dict(
                    symbol=['triangle-up' if trade.direction == 1 else 'triangle-down', 'circle'],
                    size=[12, 6],
                    color=[color, 'black']
                ),
                showlegend=False
            ),
            row=2, col=1
        )
    
    fig.update_yaxes(title_text="Price (USD)", row=2, col=1)

    # Subplot 3: Scatter
    x_scat = clean_scatter['actual'].values
    y_scat = clean_scatter['predicted'].values
    slope, intercept = np.polyfit(x_scat, y_scat, 1)
    x_range = np.linspace(x_scat.min(), x_scat.max(), 100)
    y_range = slope * x_range + intercept
    
    # CI
    y_fitted = slope * x_scat + intercept
    n = len(x_scat)
    mse = np.sum((y_scat - y_fitted)**2) / (n - 2)
    x_mean = np.mean(x_scat)
    Sxx = np.sum((x_scat - x_mean)**2)
    stdev = np.sqrt(mse * (1.0/n + (x_range - x_mean)**2 / Sxx))
    ci = 1.96 * stdev

    # CI Fill
    fig.add_trace(
        go.Scatter(
            x=x_range.tolist() + x_range[::-1].tolist(),
            y=(y_range + ci).tolist() + (y_range - ci)[::-1].tolist(),
            fill='toself',
            fillcolor='rgba(255, 255, 0, 0.3)',
            line=dict(color='rgba(255, 165, 0, 0.5)'),
            name='95% CI',
            hoverinfo='skip'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=x_range.tolist(), y=y_range.tolist(), name="Regression Line", line=dict(color='blue', width=2)),
        row=3, col=1
    )
    
    # Scatter points
    colors = clean_scatter['action'].map({'none': 'black', 'entry': 'green', 'exit': 'red'}).tolist()
    fig.add_trace(
        go.Scatter(
            x=clean_scatter['actual'].tolist(),
            y=clean_scatter['predicted'].tolist(),
            mode='markers',
            marker=dict(color=colors, size=6, opacity=0.5),
            name="Predictions",
            hovertext=clean_scatter['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        ),
        row=3, col=1
    )
    
    # Parity line
    min_val = min(x_scat.min(), y_scat.min())
    max_val = max(x_scat.max(), y_scat.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], name="y=x", line=dict(color='red', dash='dash', width=1)),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Actual Ratio", row=3, col=1)
    fig.update_yaxes(title_text="Predicted Ratio", row=3, col=1)
    
    fig.update_layout(height=1500, title_text=f"Backtest Report: {symbol}", showlegend=True, template='plotly_white')
    
    return fig

def main():
    data_dir = 'backtester_v1/data/raw/multi'
    model_path = 'backtester_v1/models/xgb_regression_v1.json'
    scaler_path = 'backtester_v1/models/scaler_v1.joblib'
    calib_path = 'backtester_v1/models/calibration_v1.joblib'
    results_dir = 'backtester_v1/results'
    
    os.makedirs(results_dir, exist_ok=True)
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    calibration_factor = joblib.load(calib_path)
    
    # All 10 coins
    symbols = [
        'BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'BNB_USDT', 'XRP_USDT',
        'ADA_USDT', 'DOGE_USDT', 'DOT_USDT', 'LINK_USDT', 'LTC_USDT'
    ]
    
    html_content = """
    <html>
    <head>
        <title>Backtest Interactive Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; }
            .container { background-color: white; padding: 20px; margin-bottom: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { text-align: center; }
            h2 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f8f8f8; }
            .metric-val { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Advanced Backtest Interactive Report</h1>
    """
    
    for symbol in symbols:
        file = f"{symbol}.parquet"
        path = os.path.join(data_dir, file)
        if not os.path.exists(path):
            continue
            
        print(f"Generating interactive plot for {symbol}...")
        df = pd.read_parquet(path)
        features = build_features(df, scaler=scaler)
        ohlcv = df.loc[features.index]
        
        if features.index.tz: features.index = features.index.tz_localize(None)
        if ohlcv.index.tz: ohlcv.index = ohlcv.index.tz_localize(None)
            
        state = run_backtest(ohlcv, features, model)
        metrics = compute_metrics(state)
        
        if "Error" in metrics:
            continue
            
        fig = generate_interactive_plot(state, ohlcv, features, model, calibration_factor, symbol, metrics)
        plot_div = fig.to_html(full_html=False, include_plotlyjs=False)
        
        html_content += f"<div class='container'><h2>{symbol}</h2>"
        html_content += f"<div>{plot_div}</div>"
        
        # Stats Table
        html_content += "<table><tr>"
        items = list(metrics.items())
        # Split into two rows for better readability
        mid = len(items) // 2 + 1
        for i in range(mid):
            html_content += f"<th>{items[i][0]}</th>"
        html_content += "</tr><tr>"
        for i in range(mid):
            html_content += f"<td>{items[i][1]}</td>"
        html_content += "</tr><tr>"
        for i in range(mid, len(items)):
            html_content += f"<th>{items[i][0]}</th>"
        html_content += "</tr><tr>"
        for i in range(mid, len(items)):
            html_content += f"<td>{items[i][1]}</td>"
        html_content += "</tr></table></div>"

    html_content += "</body></html>"
    
    output_path = os.path.join(results_dir, 'interactive_report.html')
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Interactive report saved to {output_path}")

if __name__ == "__main__":
    main()
