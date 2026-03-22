# Controlled Research Loop

This repo now supports a bounded and auditable replay-research loop for the live `ofbot` strategies.

## Design

- Research runs only on recorded paper/live raw parquet, never on live production flow.
- Variants are limited to explicit parameter grids, bounded random search, or one-step mutations.
- Dataset usage is time-ordered and split into `train`, `validation`, and final `test`.
- The final `test` split is not touched until one validation winner is selected.
- Robustness checks are explicit:
  - subperiod stability,
  - walk-forward stability,
  - stressed fees / impact / latency,
  - parameter-neighbor sensitivity.
- Promotion only writes a paper config. It never promotes directly to real trading.
- Promoted paper configs disable `learning.enabled`, so the paper comparison window stays controlled.

## Repo Mapping

- Live-compatible strategies: `src/ofbot/strategy/`
- Replay backtest runner: `src/ofbot/research/backtest.py`
- Dataset split logic: `src/ofbot/research/dataset.py`
- Scoring and robustness: `src/ofbot/research/evaluation.py`
- Registry: `src/ofbot/research/registry.py`
- Research orchestration: `src/ofbot/research/runner.py`
- Promotion / rollback: `src/ofbot/research/promotion.py`

## Registry

The registry is local, flat, and append-only:

- `data/research/registry/experiments.jsonl`
- `data/research/registry/candidates.jsonl`

Each line records a timestamped decision or experiment artifact path. This keeps the loop easy to audit and easy to roll back.

## Commands

Baseline replay backtest:

```bash
uv run python -m ofbot.cli research backtest-baseline \
  --config config/local.paper.yaml \
  --research-config config/research.continuation.yaml
```

Bounded research batch:

```bash
uv run python -m ofbot.cli research run \
  --config config/local.paper.yaml \
  --research-config config/research.continuation.yaml
```

Re-validate one candidate:

```bash
uv run python -m ofbot.cli research validate \
  --config config/local.paper.yaml \
  --research-config config/research.continuation.yaml \
  --candidate-id continuation_xxxxxxxx
```

Promote a validated candidate to paper-only:

```bash
uv run python -m ofbot.cli research promote-paper \
  --config config/local.paper.yaml \
  --research-config config/research.continuation.yaml \
  --candidate-id continuation_xxxxxxxx
```

Run the promoted paper candidate:

```bash
uv run python -m ofbot.cli live \
  --config data/research/deployments/active.paper.yaml \
  --max-seconds 3600
```

Rollback to the baseline paper config:

```bash
uv run python -m ofbot.cli research rollback-paper \
  --config config/local.paper.yaml \
  --research-config config/research.continuation.yaml \
  --candidate-id continuation_xxxxxxxx \
  --reason paper_underperformance
```

## Selection Rules

A candidate is only marked `validated_for_paper` if:

- it clears the direct score gates (`min_trades`, `max_drawdown`, `profit_factor`, `win_rate`),
- it beats the baseline on validation by the configured score delta,
- it does not degrade drawdown beyond the configured tolerance,
- it survives subperiod, walk-forward, stressed-cost, and neighbor-parameter checks,
- it still clears the held-out final `test`.

If any of those fail, the candidate is rejected and the reason is written to the registry and the report artifact.
