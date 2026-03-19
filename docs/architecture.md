# Architecture

## ofbot Runtime

Runtime package: `src/ofbot`.

- `gateway/`: Binance websocket and optional testnet REST adapters
- `market/`: trade/market-data events, orderbook state, feature + regime engines
- `strategy/`: continuation / exhaustion / hybrid variants and selector
- `execution/`: paper broker, fills, portfolio, risk
- `memory/`: DuckDB journal, trade context, bandit state, report export
- `replay/`: deterministic replay player
- `cli/`: live/replay/report command entrypoint

All flow is event-driven:

`live/replay event -> feature state -> regime -> strategy -> risk -> broker -> fills -> portfolio -> memory`.

The runtime default configuration is conservative and paper-only unless explicitly changed to `paper_testnet` with explicit Binance testnet keys.

## Objectif

Le repo est structuré comme un monorepo quant orienté recherche sérieuse, avec séparation explicite entre:

- ingestion des données,
- modèle de données événementiel,
- feature engineering,
- stratégies,
- exécution,
- backtest,
- validation,
- reporting,
- utilitaires et configuration.

## Arborescence

```text
/apps
  /cli
  /research
/configs
/data
  /raw
  /bronze
  /silver
  /gold
  /metadata
  /reports
/docs
/notebooks
/packages
  /core
  /data_ingestion
  /data_model
  /feature_engineering
  /strategy
  /execution
  /backtest
  /validation
  /reporting
  /utils
/tests
```

## Couche data

- `raw`: ZIP/CSV Binance téléchargés tels quels.
- `bronze`: parquet fidèle aux colonnes Binance, plus métadonnées minimales.
- `silver`: schéma homogène nettoyé, typé, trié, dédupliqué, prêt pour l’analytique.
- `gold`: datasets enrichis pour features/backtest.

Chaque partition est écrite par dataset, symbole et date:

```text
data/silver/trades/symbol=BTCUSDT/date=2025-01-01/part-0.parquet
data/silver/aggTrades/symbol=BTCUSDT/date=2025-01-01/part-0.parquet
```

## Décisions d’architecture

### Namespace unique

Le namespace Python est `quantflow.*`. Les modules sont distribués dans plusieurs dossiers `packages/*/src` pour garder une séparation monorepo sans perdre la cohérence ergonomique côté import.

### Backtest event-driven

Le moteur lit séquentiellement des `BarEvent` enrichis issus des trades. Cette couche est déjà séparée de:

- la génération du signal,
- la traduction signal -> ordre,
- la simulation d’exécution,
- la mise à jour portefeuille/PnL.

Le MVP reste bar-driven au niveau de la prise de décision, mais les données sources sont trade-driven et les features viennent des trades.

### Exécution conservative

Sans carnet complet, le moteur n’invente pas une microstructure inexistante. Les market orders sont exécutés avec:

- latence en nombre de barres,
- spread proxy,
- slippage explicite,
- impact dépendant de la participation au volume de la barre,
- partial fills si l’ordre dépasse le cap de participation,
- frais taker.

Les limit orders sont supportés en mode simplifié et documenté, avec cap de participation et expiration configurable.

### Risk overlay séparé

Le moteur applique une couche de risque distincte de la stratégie:

- stops et take profit basés sur le mark de clôture,
- trailing stop conservateur basé sur le meilleur `close` depuis l’entrée,
- `cooldown` après stop pour éviter les ré-entrées immédiates sur le même bruit de marché.

Cette séparation garde la logique alpha indépendante de la logique de protection et facilite les analyses par raison de sortie.

### Validation anti-overfitting

Le module `validation` impose:

- split strictement temporel,
- walk-forward,
- purged windows avec embargo basique.

Le framework est conçu pour décourager l’optimisation sur une seule sous-période.
