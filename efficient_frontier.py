import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

DOW_30_TICKER = [
    'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 
    'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 
    'MSFT', 'NKE', 'NVDA', 'PG', 'SHW', 'TRV', 'UNH', 'V', 'VZ', 'WMT'
]


data = yf.download(
    tickers=DOW_30_TICKER,
    start="2023-01-01",
    end="2025-11-01",
    progress=False,
    group_by='ticker'
)

prices = data.xs('Close', axis=1, level=1) if isinstance(data.columns, pd.MultiIndex) else data['Close']
prices = prices.ffill().bfill()
prices = prices.dropna(axis=1, thresh=int(len(prices)*0.95))

print(f"✓ {len(prices.columns)} Aktien, {len(prices)} Handelstage")

mu = expected_returns.mean_historical_return(prices, frequency=252)
S = risk_models.sample_cov(prices, frequency=252)

ef_max_sharpe = EfficientFrontier(mu, S)
ef_max_sharpe.max_sharpe()
exp_return, volatility, sharpe_ratio = ef_max_sharpe.portfolio_performance(verbose=False)


target_returns = np.linspace(mu.min(), mu.max(), 100)
frontier_vols = []
frontier_rets = []

for target in target_returns:
    try:
        ef = EfficientFrontier(mu, S)
        ef.efficient_return(target)
        perf = ef.portfolio_performance(verbose=False)
        frontier_rets.append(perf[0])
        frontier_vols.append(perf[1])
    except:
        continue

fig, ax = plt.subplots(figsize=(12, 8))

individual_rets = mu.values
individual_vols = np.sqrt(np.diag(S))
ax.scatter(individual_vols, individual_rets, 
           s=80, alpha=0.4, color='lightblue', 
           edgecolors='steelblue', linewidths=1,
           label='Einzelne Aktien', zorder=1)

ax.plot(frontier_vols, frontier_rets, 
        linewidth=3, color='darkgreen', 
           label='Efficient Frontier', zorder=2)

ax.scatter(volatility, exp_return,
           marker='s', s=300, color='red', 
           edgecolors='darkred', linewidths=2,
           label='Max Sharpe Portfolio', 
           zorder=3)

ax.set_xlabel('Volatilität', fontsize=13, fontweight='bold')
ax.set_ylabel('Erwartete Rendite', fontsize=13, fontweight='bold')

ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)

ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
plt.savefig('efficient_frontier.pdf', dpi=300, bbox_inches='tight')

print("✓ Gespeichert: efficient_frontier.png, efficient_frontier.pdf")
plt.show()
