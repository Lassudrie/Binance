# Feature Catalog

Les features du MVP sont calculées à partir du dataset spot `trades` Binance, dont les colonnes normalisées `silver` sont:

- `trade_id`
- `price`
- `qty`
- `quote_qty`
- `event_time`
- `is_buyer_maker`
- `is_best_match`

## Mapping colonnes Binance -> colonnes internes

- `price` Binance -> `price`
- `qty` Binance -> `qty`
- `quoteQty` Binance -> `quote_qty`
- `time` Binance -> `event_time`
- `isBuyerMaker` Binance -> `is_buyer_maker`

Le même schéma `silver` est désormais produit pour `aggTrades` avec:

- `aggregate_trade_id` -> `trade_id`
- `price` -> `price`
- `qty` -> `qty`
- `quote_qty` reconstruit comme `price * qty`
- `Timestamp` -> `event_time`
- `Was the buyer the maker` -> `is_buyer_maker`

## Sign convention

Sur Binance spot `trades`, `isBuyerMaker = true` signifie que le buyer était maker, donc que l’agresseur était vendeur.

Le framework définit donc:

- `side_sign = -1` si `is_buyer_maker = true`
- `side_sign = +1` sinon
- `signed_qty = qty * side_sign`
- `signed_quote_qty = quote_qty * side_sign`

## Features MVP

### Agrégats intra-barre

- `open`, `high`, `low`, `close`: calculés depuis `price`
- `volume`: somme de `qty`
- `quote_volume`: somme de `quote_qty`
- `trade_count`: nombre de trades
- `delta`: somme de `signed_qty`
- `quote_delta`: somme de `signed_quote_qty`
- `buy_count`: nombre de trades avec `side_sign > 0`
- `sell_count`: nombre de trades avec `side_sign < 0`
- `average_trade_size`: `volume / trade_count`

### Features order flow

- `cumulative_delta`: somme cumulée de `delta`
- `rolling_delta`: somme glissante de `delta`
- `trade_count_imbalance`: `(buy_count - sell_count) / trade_count`
- `burst_intensity`: `trade_count / rolling_mean_past(trade_count)`
- `tick_direction_persistence`: `abs(buy_count - sell_count) / trade_count`
- `realized_volatility`: écart-type glissant des rendements log de `close`
- `micro_momentum`: variation relative de `close` sur une fenêtre passée
- `range_high_prev`: plus haut glissant passé, décalé d’une barre
- `range_low_prev`: plus bas glissant passé, décalé d’une barre
- `micro_range_breakout`: position de `close` vis-à-vis de la micro-range passée

### Régime

- `hour_of_day`: heure UTC de la barre
- `session`: `asia`, `europe`, `us`, `off_hours`
- `vol_regime`: ratio de `realized_volatility` à sa médiane glissante passée
- `news_regime`: placeholder constant `unknown`

### Normalisation robuste

Pour certaines features, le MVP ajoute des colonnes suffixées `_rz` via:

```text
robust_z = (x_t - rolling_median_past(x)) / (rolling_mad_past(x) + epsilon)
```

La fenêtre de normalisation est décalée d’une barre pour éviter d’utiliser la valeur courante dans le dénominateur.

Les colonnes normalisées incluent désormais aussi:

- `delta_rz`
- `quote_delta_rz`

## Limites

- pas de microprice/spread/order book imbalance dans le MVP faute de carnet historique intégré,
- `spread` et `impact` sont proxy-only côté exécution,
- la granularité de décision du MVP dépend de `bar_size`.
