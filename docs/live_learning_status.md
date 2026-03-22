# Live Learning Status

Generated: `2026-03-22T06:09:30+00:00`

## Snapshot
- Alpha status: `candidate_pending_more_oos`
- Reviewed runs: `32`
- Candidates: `4` (pending=3, rejected=1)
- Best active candidate delta vs baseline: `+15.748592`

## Disk
- Free space: `24.2GB` / `74.8GB`
- Runtime raw runs: `22.9MB`
- Runtime reports: `38.5MB`
- Memory: `31.2MB`
- Temp pytest: `0B`
- Historical raw cache: `0B`
- Historical bronze: `0B`
- Historical silver: `0B`

## Retention
- Keep recent raw runs: `12`
- Keep recent report CSV sets: `8`
- Preserve pending candidate source runs: `True`

## Recent Runs
- run_20260322_060823 verdict=`negative` trades=`1` net=`-0.138429` dominant=`continuation` edge_blocks=`165`
- run_20260322_060236 verdict=`negative` trades=`2` net=`-0.530894` dominant=`continuation` edge_blocks=`626`
- run_20260322_024252 verdict=`negative` trades=`5` net=`-1.300021` dominant=`continuation` edge_blocks=`450`
- run_20260321_204136 verdict=`no_trades` trades=`0` net=`+0.000000` dominant=`continuation` edge_blocks=`2`
- run_20260321_203916 verdict=`negative` trades=`3` net=`-1.302780` dominant=`continuation` edge_blocks=`117`
- run_20260321_091349 verdict=`no_trades` trades=`0` net=`+0.000000` dominant=`continuation` edge_blocks=`147`
- run_20260321_082703 verdict=`no_trades` trades=`0` net=`+0.000000` dominant=`-` edge_blocks=`0`
- run_20260321_004004 verdict=`no_trades` trades=`0` net=`+0.000000` dominant=`continuation` edge_blocks=`256`

## Candidates
- run_20260319_181616_continuation_stricter_entry_a05acf1f status=`pending` delta=`+15.748592` baseline=`-34.636519` candidate=`-18.887927` validations=`19` trades=`9`
- run_20260319_193858_continuation_stricter_entry_824a1735 status=`pending` delta=`+0.044397` baseline=`-0.282338` candidate=`-0.237941` validations=`6` trades=`1`
- run_20260319_215950_continuation_stricter_entry_96b7dda2 status=`rejected` delta=`+0.000000` baseline=`+0.000000` candidate=`+0.000000` validations=`3` trades=`0`
- run_20260322_060823_continuation_stricter_entry_0670c1b0 status=`pending` delta=`+0.000000` baseline=`+0.000000` candidate=`+0.000000` validations=`0` trades=`0`

## Recent Blockers
- `expected_edge_blocks`: `5`
- `no_trades`: `5`
- `stricter_entry`: `5`

## Next Actions
- Collect more OOS runs for `run_20260319_181616_continuation_stricter_entry_a05acf1f` until candidate trades reach `20`.
