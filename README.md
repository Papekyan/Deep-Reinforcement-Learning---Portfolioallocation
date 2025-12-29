# Deep Reinforcement Learning für die dynamische Portfolioallokation: Konzepte, Chancen und Grenzen

## Overview

Dieses Repository implementiert einen Deep Reinforcement Learning (DRL) Ansatz zur dynamischen Portfolio-Optimierung und evaluiert dessen Leistungsfähigkeit im Vergleich zu klassischen Optimierungsmethoden. 

Der DRL-Ansatz modelliert die Portfolio-Optimierung als sequentielles Entscheidungsproblem, bei dem ein Agent auf Basis historischer Marktdaten und technischer Indikatoren tägliche Allokationsentscheidungen trifft. Die Implementierung umfasst drei state-of-the-art DRL-Algorithmen (PPO, DDPG, TD3) sowie systematisches Hyperparameter-Tuning zur Maximierung der Sharpe Ratio auf einem Validierungsdatensatz.

Die Methodik wird auf dem Dow Jones Industrial Average (DJIA) Index mit 30 Aktien evaluiert und gegen drei etablierte Baselines verglichen: Buy-and-Hold Strategie, Equal-Weight Portfolio (1/N) und Mean-Variance Optimization nach Markowitz mit maximaler Sharpe Ratio.

## Key Features

### Implementierte DRL-Algorithmen

- **PPO (Proximal Policy Optimization)**
- **DDPG (Deep Deterministic Policy Gradient)**
- **TD3 (Twin Delayed DDPG)**

### Baseline-Methoden

- **DJI Buy-and-Hold**
- **Equal Weight (1/N)**
- **Markowitz Mean-Variance Optimization**


### Weitere Funktionen

- Automatisches Hyperparameter-Tuning mittels Optuna mit Tree-structured Parzen Estimator (TPE)
- Modifizierte FinRL-Umgebung mit expliziter Berücksichtigung von Transaktionskosten
- Erweiterter State Space mit Portfolio-Gewichten des Vortages zur Modellierung von Rebalancing-Kosten
- Umfassende Performance-Evaluation mit pyfolio Metriken (Sharpe Ratio, Calmar Ratio, Max Drawdown, etc.)

## Project Structure

```
Evaluierung/
├── modell_main.ipynb              # Hauptnotebook für Modelltraining und -entwicklung
├── evaluation_main.ipynb           # Evaluationsnotebook für Performance-Vergleich
├── efficient_frontier.py           # Visualisierung der Efficient Frontier für Markowitz-Baseline
├── ppo_performance.csv             # Tägliche Renditen des PPO-Modells (ohne Tuning)
├── ppo_Opt_performance.csv         # Tägliche Renditen des optimierten PPO-Modells
├── ppo_actions.csv                 # Asset-Allokationen des PPO-Modells (ohne Tuning)
├── ppo_Opt_actions.csv             # Asset-Allokationen des optimierten PPO-Modells
├── ddpg__performance.csv           # Tägliche Renditen des DDPG-Modells (ohne Tuning)
├── ddpg_Opt_performance.csv        # Tägliche Renditen des optimierten DDPG-Modells
├── ddpg_actions.csv                # Asset-Allokationen des DDPG-Modells (ohne Tuning)
├── ddpg_Opt_actions.csv            # Asset-Allokationen des optimierten DDPG-Modells
├── td3_performance.csv             # Tägliche Renditen des TD3-Modells (ohne Tuning)
├── td3_Opt_performance.csv         # Tägliche Renditen des optimierten TD3-Modells
├── td3_actions.csv                 # Asset-Allokationen des TD3-Modells (ohne Tuning)
├── td3_Opt_actions.csv             # Asset-Allokationen des optimierten TD3-Modells
├── markowitz_daily_returns.csv     # Tägliche Renditen der Markowitz-Baseline
├── markowitz_asset_allocation.csv  # Asset-Allokationen der Markowitz-Baseline
├── equal_weight_daily_returns.csv  # Tägliche Renditen der Equal-Weight-Baseline
├── equal_weight_asset_allocation.csv # Asset-Allokationen der Equal-Weight-Baseline
└── dji_baseline_daily_returns.csv  # Tägliche Renditen des DJI Buy-and-Hold Baselines
```

### Wichtige Dateien

**modell_main.ipynb**: Enthält die vollständige Pipeline für Datenverarbeitung, Umgebungserstellung, Modelltraining und Hyperparameter-Optimierung. Das Notebook implementiert eine modifizierte `StockPortfolioEnv` Klasse, die Transaktionskosten explizit im Reward-Signal berücksichtigt und die Portfolio-Gewichte des Vortages in den State Space integriert. Der Workflow umfasst: (1) Daten-Download und Feature Engineering mit technischen Indikatoren (2) Erstellung von Trainings-, Validierungs- und Test-Umgebungen, (3) Initiales Training aller drei DRL-Agenten, (4) Hyperparameter-Tuning mit Optuna auf dem Validierungsdatensatz, (5) Retraining der optimierten Modelle, (6) Implementierung der Baseline-Methoden (Markowitz und Equal Weight).

**evaluation_main.ipynb**: Führt die umfassende Performance-Evaluation aller trainierten Modelle durch. Das Notebook lädt die gespeicherten Performance-Metriken und Asset-Allokationen, berechnet statistische Kennzahlen mittels pyfolio (Sharpe Ratio, Annual Return, Volatility, Max Drawdown, Calmar Ratio), erstellt vergleichende Visualisierungen (kumulative Renditen, Drawdown-Analysen, Risk-Return Scatter Plots) und generiert Stacked Area Charts für die Portfolio-Gewichtungen über die Zeit.

**efficient_frontier.py**: Standalone-Skript zur Visualisierung der Efficient Frontier für den Testzeitraum. Berechnet die Mean-Variance Efficient Frontier und das Portfolio mit maximaler Sharpe Ratio unter Verwendung von pyportfolioopt und erstellt eine wissenschaftliche Visualisierung der Risk-Return Charakteristik.

## Requirements

Die folgenden Python-Pakete sind für die Reproduktion der Ergebnisse erforderlich:

### Core Dependencies

- `python >= 3.8`
- `numpy < 2.0` (Kompatibilität mit FinRL)
- `pandas`
- `matplotlib`
- `seaborn`

### Deep Reinforcement Learning

- `stable-baselines3` - Implementierung der DRL-Algorithmen (PPO, DDPG, TD3)
- `gymnasium` - Reinforcement Learning Umgebungsframework
- `shimmy >= 0.2.1` - Kompatibilitätsschicht zwischen Gym und Gymnasium
- `finrl` - Financial Reinforcement Learning Framework (Installation via GitHub: `git+https://github.com/AI4Finance-Foundation/FinRL.git@master`)

### Hyperparameter Optimization

- `optuna` - Automatisches Hyperparameter-Tuning mit TPE-Sampler

### Portfolio Optimization

- `pyportfolioopt` - Implementierung der Markowitz Mean-Variance Optimization
- `pyfolio` - Performance-Analyse und Metrikenberechnung für Portfolios

### Data Acquisition

- `yfinance` - Download von Finanzdaten von Yahoo Finance

### Additional Dependencies

- `swig` - Build-Tool für bestimmte Python-Pakete
- `protobuf == 3.20.3` - Spezifische Version für Kompatibilität mit FinRL
- `ta` - Technical Analysis Library (optional, für zusätzliche Indikatoren)



