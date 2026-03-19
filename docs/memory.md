# ofbot Learning Memory

`ofbot` uses DuckDB files under `data/memory`.

## Event journal

`event_journal` stores:

- raw signal decisions
- feature snapshot at decision time
- fill details (qty, price, spread, slippage)
- realized outcome rows

## Trade context

`trade_contexts` stores closed-trade records:

- symbol, direction, entry/exit timestamps
- strategy and regime
- PnL, MAE, MFE, spread and fee tags

## Policy memory

`policy_performance` stores rolling strategy stats per regime.

`policy_state` stores incumbent/challenger per regime and last promotion decision.

## Why safe

- Learning is selection-only; strategy code never mutates source files.
- Promotions are logged with gates:
  - minimum sample size
  - minimum expected improvement
  - max drawdown/bias constraints
  - confidence settings
