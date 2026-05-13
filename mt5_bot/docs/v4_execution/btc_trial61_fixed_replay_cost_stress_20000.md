# BTCUSD Fixed Replay v4

- live 적용 금지: paper-forward 검증 전용입니다.
- source_trial_id: 61
- costs: spread_points=120.0, slippage_points=20.0, commission_per_lot=0.04

## Train

- total_net_pnl: 9.71242385714131
- net_profit_factor: 1.4849331662169938
- gross_profit: 29.740799999999883
- gross_loss: 20.028376142858573
- win_count: 10
- loss_count: 27
- max_single_trade_net_loss: -0.9389791428572148
- no_trade_days_pct: 0.0
- profit_lock_saved_pnl: 16.942821142856793

## OOS

- total_trades: 18
- expectancy_net: 0.273764936507914
- net_profit_factor: 1.5357475263303018
- gross_profit: 14.12570000000014
- gross_loss: 9.197931142857689
- win_count: 5
- loss_count: 13
- max_single_trade_net_loss: -1.0451275714286428
- no_trade_days_pct: 0.0
- profit_lock_saved_pnl: 5.699560857142613

## Combined

- oos_decay_pct: 0.0
- effective_leverage_max: 11.590811428571428
- margin_used_pct_max: 2.318162285714285

## Daily Bleed Analysis

### BTCUSD

- configured_stop_after_consecutive_losses: 3
- observed_max_consecutive_losses: 16
- theoretical_consecutive_loss_halt_events: 23
- configured_daily_loss_limit_usd: 3.75
- worst_daily_pnl: -3.5391102857144086
- daily_loss_limit_breached_days: 0
- daily_bleed_halt_count_train: 161
- daily_bleed_halt_count_oos: 14
- explanation: DailyBleedGuard는 실제로 작동했다. 이전의 halt_count=0은 `daily_bleed_guard_active` block reason을 별도 halt metric으로 매핑하지 않았던 adapter 문제였고, 현재 replay에서는 해당 block reason을 `daily_bleed_halt_count`로 매핑한다.
- note: `observed_max_consecutive_losses`는 체결된 trade sequence 기준이다. 동시에 많은 신규 진입 시도는 DailyBleedGuard에 의해 차단되었으므로, paper-forward에서는 `daily_bleed_guard_active` 이벤트와 다음 진입까지의 간격을 별도 로그로 확인해야 한다.

