# Soft Exit Tuning Snapshot

Date: `2026-03-22`

## Objet

Snapshot exact du profil paper utilisé après l'amélioration suivante :

- ne pas re-soumettre un `soft exit` passif tant qu'un ordre équivalent est déjà pending
- escalader `limit -> market` après un petit TTL
- escalader `limit -> market` si le book part contre nous

## Parametres exacts

Source: [config/local.paper.yaml](/home/alex/projects/Binance/config/local.paper.yaml)

```yaml
execution:
  mode: paper_local
  allow_short: true
  initial_cash: 100000.0
  taker_fee_bps: 8.0
  maker_fee_bps: 2.0
  default_target_notional_usd: 150.0
  min_target_notional_usd: 50.0
  max_target_notional_usd: 250.0
  base_spread_bps: 2.0
  fallback_half_spread_bps: 1.0
  impact_bps_per_unit_participation: 8.0
  market_sweep_depth_bps: 25.0
  latency_ms: 150
  market_participation_cap: 0.25
  limit_order_book_depth_bps: 0.0
  max_order_lifetime_s: 5
  allow_limit_orders: true
  soft_exit_limit_ttl_s: 2.0
  soft_exit_limit_adverse_bps: 0.75

risk:
  stale_data_s: 6.0
  broken_ws_s: 10.0
  abnormal_spread_bps: 220.0
  abnormal_volatility_bps: 900.0
  max_position_size: 4.0
  max_notional: 40000.0
  max_concurrent_exposure: 2
  cooldown_after_loss_s: 0
  cooldown_after_trade_s: 3
  daily_loss_limit: 0.02
  max_holding_time_s: 30
  catastrophic_stop_loss_bps: 45.0
  min_hold_before_soft_exit_s: 4
  stop_loss_bps: 8.0
  take_profit_bps: 14.0
  trailing_stop_bps: 4.0
  spread_gate_enabled: true
  volatility_gate_enabled: true
  max_consecutive_losses: 99

strategy_library:
  continuation:
    params:
      trend_alignment_threshold: 0.03
      queue_imbalance_threshold: 0.01
      microprice_drift_threshold_bps: 0.1
      min_expected_net_edge_bps: 0.75
      expected_move_discount: 0.9
      edge_fast_horizon_s: 1
      edge_slow_horizon_s: 5
      edge_fast_weight: 0.85
      edge_slow_weight: 0.15
      edge_cvd_bonus_bps: 2.0
      edge_queue_bonus_bps: 6.0
      edge_bonus_cap_bps: 20.0
      min_momentum_5s_bps: 0.4
      min_edge_cost_ratio: 1.15
      min_fee_coverage_ratio: 1.35
      require_trend_up_regime: false
      require_trend_down_regime_for_short: false
      revalidate_on_fill: true
      revalidate_grace_period_s: 2.0
      revalidate_drop_on_neutral: false
      strong_signal_expected_move_discount: 0.95
      strong_signal_min_edge_cost_ratio: 1.05
      strong_signal_min_momentum_5s_bps: 5.0
      strong_signal_min_cvd_z: 2.0
      no_trade_z: 0.35
      max_holding_time_s: 30
      min_hold_before_discretionary_exit_s: 4
      reverse_exit_cvd_z: 0.85
      reverse_exit_queue_imbalance: 0.45
      reverse_exit_microprice_drift_bps: 0.08
      reverse_exit_min_abs_pnl_bps: 3.0
      spread_bps_max: 12.0
      volatility_bps_max: 1200.0
      order_type: limit
      exit_order_type: limit
      exit_cost_order_type: limit
```

## Comportement code

Implémentation principale:

- [src/ofbot/execution/paper_broker.py](/home/alex/projects/Binance/src/ofbot/execution/paper_broker.py)
- [src/ofbot/config.py](/home/alex/projects/Binance/src/ofbot/config.py)

Logique:

- un `soft exit` de risque (`risk_max_holding`, `risk_stop_loss`, `risk_take_profit`, `risk_trailing_stop`) ne crée pas de nouvel ordre si un exit équivalent est déjà pending
- un `soft exit` pending en `limit` escalade en `market` si son âge atteint `soft_exit_limit_ttl_s`
- un `soft exit` pending en `limit` escalade aussi si le top of book bouge défavorablement d'au moins `soft_exit_limit_adverse_bps`

## Validation

Run de référence avant déduplication + escalade:

- run id: `run_20260322_113750`
- report: [report.md](/home/alex/projects/Binance/reports/run_20260322_113750/report_run_20260322_113750/report.md)
- `risk_max_holding submits`: `24`
- `Closed trades`: `2`
- `Net PnL`: `-0.021472`
- `Gross fees`: `0.147340`
- `Exit fees`: `0.073657`

Run après déduplication + escalade contrôlée:

- run id: `run_20260322_115052`
- report: [report.md](/home/alex/projects/Binance/reports/run_20260322_115052/report_run_20260322_115052/report.md)
- `risk_max_holding submits`: `5`
- `Closed trades`: `4`
- `Net PnL`: `-0.788920`
- `Gross fees`: `0.999716`
- `Exit fees`: `0.799716`

## Lecture

Ce réglage réduit fortement le spam de soumissions de `soft exit`:

- `24 -> 5` soumissions `risk_max_holding` sur les runs comparés

Il améliore la capacité de sortie, mais avec un coût clair:

- davantage de sorties effectives
- davantage de taker fees quand l'escalade `market` est déclenchée

Le prochain levier attendu est un `reprice` passif unique avant escalade, ou un TTL asymétrique selon la raison d'exit.
