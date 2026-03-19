# ofbot Strategy Design

The runtime exposes three validated strategy variants for paper execution:

- `continuation`: continuation on aligned flow + queue + microprice drift
- `exhaustion`: burst/fade logic with strict weakness filters
- `hybrid`: regime-aware blend of continuation and exhaustion

All variants return `StrategyDecision` with:

- `side`: `-1`, `0`, `1`
- `target_qty`: intended absolute position target
- `target confidence`
- optional risk parameters (`stop_loss_bps`, `take_profit_bps`, `trailing_stop_bps`, `max_holding_time_s`)

Notes:

- No shorting by default (`allow_short=false` in `paper_local` config).
- All non-trade, exit and kill-switch decisions are explicit and logged.
- Regime selection is optional: a Thompson-style selector re-weights by realized PnL and can promote a challenger only if safety gates pass.
