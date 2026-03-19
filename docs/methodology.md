# Methodology And Limits

## Principes

- Source primaire: trades Binance, pas OHLC exchange natif.
- Les signaux sont générés après clôture de la barre de features.
- Les ordres sont exécutés au plus tôt sur la barre suivante selon la latence configurée.
- Les coûts sont toujours appliqués.
- Les règles de risque peuvent forcer une sortie même si la stratégie voudrait rester exposée.

## Hypothèses explicites du MVP

### Données

- MVP implémente le flux vertical complet sur `spot trades`.
- `aggTrades`, `klines` et depth ont une place prévue dans l’architecture, mais le parsing de production complet n’est pas encore branché pour tous les cas.
- `aggTrades` est désormais branché en ingestion/normalisation et peut servir de source alternative de recherche.
- Les timestamps spot Binance sont détectés automatiquement en `ms` ou `us`.

### Exécution

- Sans carnet historique, le simulateur utilise un proxy conservative:
  - référence de prix = `open` de la barre d’exécution pour un market order,
  - demi-spread proxy en bps,
  - slippage explicite en bps,
  - impact additionnel dépendant du ratio `ordre / volume_de_barre`,
  - frais maker/taker en bps.
- Les market orders trop gros pour la liquidité agrégée de la barre sont découpés via un `participation cap` et peuvent donc être partiellement exécutés sur plusieurs barres.
- Les chemins publics spot depth testés sur `data.binance.vision` n’ont pas fourni d’archive exploitable dans cette itération; le framework conserve donc explicitement un modèle proxy jusqu’à disponibilité d’une source carnet historique.
- Les limit orders sont simplifiés: ils sont considérés exécutés si le prix limite est touché dans la barre, avec un modèle de partial fill dépendant du volume agrégé et une durée de vie maximale configurable.
- Le `trailing stop` du MVP suit le meilleur `close` observé depuis l’entrée, pas un plus haut intrabar, afin de rester conservateur avec des données agrégées sans chemin de prix.

### Validation

- Le walk-forward du MVP réévalue la stratégie sur fenêtres temporelles successives.
- Les splits multi-symboles sont construits sur les timestamps uniques `bar_end`, pas sur le nombre brut de lignes, pour éviter les fenêtres incohérentes quand plusieurs symboles partagent les mêmes barres.
- La commande `run-bootstrap` applique un bootstrap par blocs sur les rendements hors échantillon du portefeuille afin d’estimer un intervalle de confiance de PnL et une probabilité de rester positif.
- La commande `run-grid-search` sélectionne les params sur la partie train de chaque fenêtre puis mesure ces params sur le test suivant, afin d’éviter l’optimisation naïve sur un seul segment.
- Pour une stratégie purement rule-based sans calibration, la phase train sert surtout à mesurer la stabilité et à préparer l’extension vers l’optimisation future.
- La commande `run-alpha-scan` cherche d’abord un pouvoir prédictif feature -> forward return en walk-forward avant de transformer une intuition en stratégie tradable.

### Recherche alpha

- `run-alpha-scan` ne backteste pas une stratégie; il mesure un signal brut.
- Le scan calibre les seuils quantiles sur le train, puis applique ces seuils au test suivant.
- La métrique centrale est le `directional_test_spread`: si le train suggère `buy_high_sell_low`, le test est considéré bon seulement si cet ordre relatif reste positif.
- Un candidat n’est pas un alpha exploitable tant qu’il ne survit pas ensuite aux coûts, à la latence et au turnover via `run-backtest`.
- `run-order-flow-suite` impose un batch comparable de stratégies `order-flow only`:
  - `cumdelta_reversion_v1`
  - `delta_impulse_continuation_v1`
  - `imbalance_burst_exhaustion_v1`
- Dans ce batch, le seul filtre non-order-flow autorisé est `vol_regime >= vol_regime_min`.
- Les sorties discrétionnaires restent order-flow driven; les coûts, la latence et le risk overlay sont standardisés au niveau moteur pour toutes les familles.
- `run-candidate-deep-dive` approfondit ensuite le meilleur candidat `cumdelta_reversion_v1`:
  - barres `15s`
  - exécution `passive` en primaire
  - filtres testés: `vol_regime_min` et `session`
  - `market` utilisé seulement comme sanity check final
- La matrice de filtres du deep dive couvre:
  - baseline sans filtre explicite
  - `vol_only`
  - `session_only`
  - `combined`
- La combinaison décrite représente `12` scénarios en phase 1, pas `11`: `1 + 3 + 2 + 6`.
- Le verdict final reste conservateur:
  - `validated_edge` seulement si OOS passif positif, stabilité suffisante, sanity check market positif, bootstrap positif et symboles tous positifs
  - sinon `research_candidate`

### Risk overlay

- `max_holding_bars` force une sortie après une durée maximale de détention.
- `stop_loss_bps` et `take_profit_bps` utilisent le mark de clôture de barre.
- `trailing_stop_bps` déclenche une sortie si le PnL courant retrace d’au moins ce seuil depuis le meilleur `close` atteint depuis l’entrée.
- `cooldown_bars_after_stop` bloque les nouvelles ré-entrées non plates pendant un nombre de barres après un `risk_stop_loss` ou `risk_trailing_stop`.
- La baseline `spot_10m_regime_pullback` utilise en plus des brackets intrabar basés sur l’ATR d’entrée:
  - `stop_price = entry - stop_atr_multiple * ATR`
  - `target_price = entry + target_atr_multiple * ATR`
  - `time_stop_bars = 6` par défaut
- Si stop et target sont tous deux touchés dans la même barre, le moteur exécute le stop en premier.
- Les gaps défavorables au-delà du stop sont exécutés à l’`open` de la barre avec slippage adverse, pas au niveau idéal du stop.
- `risk_per_trade_fraction`, `daily_loss_limit_r` et `max_open_positions` sont gérés au niveau portefeuille pour éviter les séquences de sur-exposition.

## Pièges méthodologiques explicitement évités

- fuite temporelle sur les rolling highs/lows décalés,
- exécution au même bar que le signal,
- Sharpe intraday présenté sans avertissement,
- PnL sans frais,
- comparaison uniquement in-sample.
