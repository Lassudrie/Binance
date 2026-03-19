# Quickstart

## 1. Installer

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## 2. Baseline spot 10m

La baseline de recherche reproductible est dans `configs/spot_10m_regime_pullback_v1.toml`:

- univers spot fixe: `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `SOLUSDT`
- source: `trades`
- barres: `10m`
- stratégie: `spot_10m_regime_pullback`
- validation: `walk-forward + bootstrap`

## 3. Télécharger des trades Binance

```bash
qflow download-data --config configs/default.toml --symbol BTCUSDT --start-date 2025-01-01 --end-date 2025-01-03
```

Version multi-symboles 10m:

```bash
qflow download-data --config configs/spot_10m_regime_pullback_v1.toml
```

## 4. Construire bronze/silver

```bash
qflow build-dataset --config configs/default.toml --symbol BTCUSDT --start-date 2025-01-01 --end-date 2025-01-03
```

Version multi-symboles 10m:

```bash
qflow build-dataset --config configs/spot_10m_regime_pullback_v1.toml
```

## 5. Construire les features

```bash
qflow build-features --config configs/default.toml --symbol BTCUSDT
```

Version multi-symboles 10m:

```bash
qflow build-features --config configs/spot_10m_regime_pullback_v1.toml
```

## 6. Lancer un backtest

```bash
qflow run-backtest --config configs/default.toml --symbol BTCUSDT
```

Version multi-symboles 10m:

```bash
qflow run-backtest --config configs/spot_10m_regime_pullback_v1.toml
```

## 7. Lancer un walk-forward

```bash
qflow run-walkforward --config configs/default.toml --symbol BTCUSDT
```

Version multi-symboles 10m:

```bash
qflow run-walkforward --config configs/spot_10m_regime_pullback_v1.toml
```

## 8. Lancer un bootstrap hors échantillon

```bash
qflow run-bootstrap --config configs/spot_10m_regime_pullback_v1.toml
```

## 9. Lancer une sensibilité d’exécution

```bash
qflow run-sensitivity --config configs/default.toml --symbol BTCUSDT
```

## 10. Lancer une grid search temporelle

```bash
qflow run-grid-search --config configs/default.toml --symbol BTCUSDT --param-grid cumulative_delta_threshold=0.8,1.0,1.2 --param-grid exit_zscore_threshold=0.1,0.2,0.3
```

## 11. Scanner les features pour trouver un alpha

```bash
qflow run-alpha-scan --config configs/default.toml --symbol BTCUSDT
```

Version ciblée:

```bash
qflow run-alpha-scan \
  --config configs/default.toml \
  --symbol BTCUSDT \
  --feature cumulative_delta_rz \
  --feature trade_count_imbalance_rz \
  --feature micro_momentum_rz \
  --horizon-bars 1 \
  --horizon-bars 3 \
  --horizon-bars 6
```

## 12. Inspecter le dataset

```bash
qflow inspect-dataset --config configs/default.toml --symbol BTCUSDT --layer silver
```

Version `aggTrades`:

```bash
qflow build-dataset --config configs/default.toml --dataset-type aggTrades --symbol BTCUSDT --start-date 2025-01-01 --end-date 2025-01-03
qflow build-features --config configs/default.toml --dataset-type aggTrades --symbol BTCUSDT
```

## 13. Lancer la suite de recherche order flow only

Suite complète `3 stratégies x 2 bar sizes x 2 modes d’exécution`:

```bash
qflow run-order-flow-suite --config configs/order_flow_suite_spot_v1.toml
```

Smoke test rapide sur `BTCUSDT` / 1 jour:

```bash
qflow run-order-flow-suite --config configs/order_flow_suite_spot_smoke.toml
```

Cette config de smoke borne volontairement la grid search via `grid_search_max_points = 4`.

## 14. Approfondir le meilleur candidat `cumdelta_reversion_v1`

Étude complète `15s`, `spot`, `BTCUSDT + ETHUSDT`, `passive primary`, sur 6 mois:

```bash
qflow run-candidate-deep-dive --config configs/candidate_cumdelta_reversion_15s_spot.toml
```

Smoke test rapide sur `BTCUSDT` / 1 jour:

```bash
qflow run-candidate-deep-dive --config configs/candidate_cumdelta_reversion_15s_spot_smoke.toml
```

## Résultats

Les sorties sont écrites sous `data/reports/<run_name>_<dataset_type>/`:

- `summary.json`
- `equity_curve.csv`
- `trades.csv`
- `trade_log_detailed.csv`
- `symbol_summary.csv`
- `metrics_by_reason.csv`
- `report.md`
- `walkforward_summary.json` si applicable
- `bootstrap_summary.json` si applicable
- `sensitivity_summary.json` si applicable
- `grid_search_summary.json` si applicable
- `alpha_scan_summary.json` si applicable
- `suite_summary.json` si applicable
- `suite_summary.csv` si applicable
- `suite_ranked_candidates.csv` si applicable
- `suite_report.md` si applicable
- `study_summary.json` si applicable
- `scenario_matrix.csv` si applicable
- `phase2_grid_results.csv` si applicable
- `symbol_breakdown.csv` si applicable
- `best_candidate_trades.csv` si applicable
- `best_candidate_walkforward.json` si applicable
- `best_candidate_bootstrap.json` si applicable
- `best_candidate_passive_sensitivity.csv` si applicable
- `market_sanity_check.json` si applicable
