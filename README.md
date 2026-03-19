# Binance Order Flow Research Framework

Framework monorepo Python pour la recherche offline, le backtest événementiel et la validation robuste de stratégies de scalping order flow sur datasets historiques Binance.

Le MVP livré dans ce repo couvre un flux vertical complet:

1. téléchargement de datasets `trades` Binance spot depuis `data.binance.vision`,
2. stockage `raw -> bronze -> silver -> gold`,
3. normalisation UTC avec détection automatique `ms` vs `us`,
4. features microstructure/order flow sur barres temporelles,
5. stratégies plug-and-play,
6. backtest event-driven avec coûts, slippage et latence,
7. reporting Markdown/JSON/CSV,
8. validation walk-forward temporelle,
9. règles de risque configurables,
10. tests unitaires.

La priorité du MVP est la rigueur méthodologique:

- pas d’OHLC naïf comme source primaire,
- pas d’exécution parfaite implicite,
- coûts et slippage inclus par défaut,
- risque séparé du signal avec sorties forcées configurables,
- génération de signaux séparée de l’exécution,
- fuite temporelle évitée par design sur les features et le moteur,
- hypothèses de simulation explicitement documentées.

Voir aussi:

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Feature Catalog](docs/feature_catalog.md)
- [Methodology](docs/methodology.md)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Workflow MVP

```bash
qflow download-data --config configs/default.toml --symbol BTCUSDT --start-date 2025-01-01 --end-date 2025-01-03
qflow build-dataset --config configs/default.toml --symbol BTCUSDT --start-date 2025-01-01 --end-date 2025-01-03
qflow build-features --config configs/default.toml --symbol BTCUSDT
qflow run-backtest --config configs/default.toml --symbol BTCUSDT
qflow run-walkforward --config configs/default.toml --symbol BTCUSDT
qflow run-sensitivity --config configs/default.toml --symbol BTCUSDT
qflow run-grid-search --config configs/default.toml --symbol BTCUSDT --param-grid cumulative_delta_threshold=0.8,1.0,1.2 --param-grid exit_zscore_threshold=0.1,0.2,0.3
```

Les résultats sont écrits sous `data/reports/<run_name>_<dataset_type>/`.

## Live Paper Trading (ofbot)

Run safe paper trading using live Binance public data:

```bash
uv sync
uv run python -m ofbot.cli live --config config/local.paper.yaml
```

Optional limits for deterministic smoke checks:

```bash
uv run python -m ofbot.cli live --config config/local.paper.yaml --max-events 2000 --max-seconds 120
```

Replay from recorded raw stream:

```bash
uv run python -m ofbot.cli replay --input data/raw/raw_<run_id>.parquet --config config/local.paper.yaml
```

Build daily report:

```bash
uv run python -m ofbot.cli report --date today --config config/local.paper.yaml
```

Le pipeline supporte maintenant `trades` et `aggTrades` côté ingestion et normalisation `silver`. Exemple:

```bash
qflow build-dataset --config configs/default.toml --dataset-type aggTrades --symbol BTCUSDT --start-date 2025-01-01 --end-date 2025-01-03
qflow build-features --config configs/default.toml --dataset-type aggTrades --symbol BTCUSDT
```
