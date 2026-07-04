import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_price_history(prices_df):
    """Line chart of normalized price history."""
    normalized = prices_df / prices_df.iloc[0] * 100
    fig = px.line(normalized, x=normalized.index, y=normalized.columns, 
                  title='Normalized Price History (Base 100)',
                  labels={'value': 'Normalized Price', 'variable': 'Ticker'})
    fig.update_layout(hovermode='x unified', template='plotly_dark')
    return fig

def plot_correlation_heatmap(returns_df):
    """Heatmap of asset return correlations."""
    corr = returns_df.corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    title='Return Correlation Matrix')
    fig.update_layout(template='plotly_dark')
    return fig

def plot_rolling_volatility(returns_df, window=21):
    """Line chart of rolling annualized volatility."""
    rolling_vol = returns_df.rolling(window=window).std() * np.sqrt(252) * 100
    fig = px.line(rolling_vol, x=rolling_vol.index, y=rolling_vol.columns,
                  title=f'{window}-Day Rolling Annualized Volatility (%)',
                  labels={'value': 'Volatility (%)', 'variable': 'Ticker'})
    fig.update_layout(hovermode='x unified', template='plotly_dark')
    return fig

def plot_return_distribution(returns_df, weights, var_dict):
    """Histogram of portfolio returns with VaR thresholds."""
    port_ret = returns_df.dot(weights) * 100  # in percentage
    
    fig = px.histogram(port_ret, nbins=100, title='Historical Portfolio Return Distribution',
                       labels={'value': 'Return (%)'}, opacity=0.7)
    
    # Add vertical lines for VaR
    colors = ['red', 'orange', 'yellow']
    for i, (method, metrics) in enumerate(var_dict.items()):
        # VaR is positive loss, so threshold is -VaR
        var_pct = -metrics['VaR'] * 100
        fig.add_vline(x=var_pct, line_dash="dash", line_color=colors[i%len(colors)],
                      annotation_text=f"{method} VaR", annotation_position="top left")
                      
    fig.update_layout(showlegend=False, template='plotly_dark')
    return fig

def plot_stress_test_results(stress_df):
    """Bar chart ranking scenario P&L."""
    fig = px.bar(stress_df, x='Portfolio Return (%)', y='Scenario', orientation='h',
                 color='Portfolio Return (%)', color_continuous_scale='RdYlGn',
                 title='Stress Test Scenario Impacts')
    fig.update_layout(template='plotly_dark')
    return fig

def plot_backtest_results(backtest_df):
    """Line chart of rolling VaR vs Actual Returns."""
    fig = go.Figure()
    
    # Actual Returns
    fig.add_trace(go.Scatter(
        x=backtest_df.index, y=backtest_df['Actual Return'] * 100,
        mode='markers', name='Actual Return (%)',
        marker=dict(color='cyan', size=4)
    ))
    
    # -VaR Threshold
    fig.add_trace(go.Scatter(
        x=backtest_df.index, y=-backtest_df['VaR Forecast'] * 100,
        mode='lines', name='-VaR Threshold (%)',
        line=dict(color='red')
    ))
    
    # Highlight Breaches
    breaches = backtest_df[backtest_df['Breach']]
    fig.add_trace(go.Scatter(
        x=breaches.index, y=breaches['Actual Return'] * 100,
        mode='markers', name='Breaches',
        marker=dict(color='red', size=8, symbol='x')
    ))
    
    fig.update_layout(title='Rolling 1-Day VaR vs Actual Returns',
                      yaxis_title='Return (%)', hovermode='x unified',
                      template='plotly_dark')
    return fig
