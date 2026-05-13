# BTCUSD Trial 61 Cost Stress Verdict

- 대상: BTCUSD-only balanced trial 61
- 판정: PASS, paper-forward config patch 후보로만 제안
- live 적용 금지
- replay rows: 20000
- baseline costs: spread_points=0, slippage_points=0, commission_per_lot=0
- stress costs: spread_points=120, slippage_points=20, commission_per_lot=0.04

## 비교 요약

| Metric | Baseline train | Baseline OOS | Cost stress train | Cost stress OOS |
|---|---:|---:|---:|---:|
| trades | 37 | 21 | 37 | 18 |
| total_net_pnl | 6.9656 | 9.7636 | 9.7124 | 4.9278 |
| expectancy_net | 0.1883 | 0.4649 | 0.2625 | 0.2738 |
| net_profit_factor | 1.3565 | 2.0260 | 1.4849 | 1.5357 |
| gross_profit | 26.5023 | 19.2795 | 29.7408 | 14.1257 |
| gross_loss | 19.5367 | 9.5159 | 20.0284 | 9.1979 |
| wins/losses | 9 / 27 | 6 / 13 | 10 / 27 | 5 / 13 |
| max_single_trade_net_loss | -0.9094 | -1.0303 | -0.9390 | -1.0451 |
| no_trade_days_pct | 0.0 | 0.0 | 0.0 | 0.0 |
| profit_lock_saved_pnl | 19.2170 | 9.0605 | 16.9428 | 5.6996 |
| daily_bleed_halt_count | 146 | 56 | 161 | 14 |

## PF 999 Sentinel 해소

기존 추천 표의 PF 999는 aggregate/report 표현상의 sentinel이었다. 고정 replay에서 실제 trade-level gross profit/gross loss로 다시 계산하면 다음과 같다.

- baseline train PF: 1.3565
- baseline OOS PF: 2.0260
- cost stress train PF: 1.4849
- cost stress OOS PF: 1.5357

따라서 trial 61은 PF=999 착시를 제거해도 비용 스트레스 OOS에서 PF 1.0 이상과 양수 expectancy를 유지한다.

## Cost Stress 해석

stress run은 단순히 기존 거래에 비용만 차감한 것이 아니라, spread/slippage/commission을 scanner와 fee-aware entry filter에 반영해 재시뮬레이션했다. 그래서 거래 선택이 달라지며 train PnL이 baseline보다 높아질 수 있다. 더 중요한 검증 항목은 OOS에서 다음 조건이 유지되는지다.

- OOS trades: 18, 1-2건짜리 후보 아님
- OOS expectancy: 0.2738, 양수
- OOS PF: 1.5357, 1.0 이상
- max loss: -1.0451, hard_max_net_loss_usd 1.25 이내
- no-trade days pct: 0.0
- profit-lock saved pnl: 5.6996

## Daily Bleed 분석

이전 `daily_bleed_halt_count=0`의 원인은 DailyBleedGuard가 작동하지 않은 것이 아니라, simulation metric adapter가 `daily_bleed_guard_active` block reason을 `daily_bleed_halt_count`로 매핑하지 않았기 때문이다.

수정 후 replay 기준:

- baseline train daily_bleed_halt_count: 146
- baseline OOS daily_bleed_halt_count: 56
- cost stress train daily_bleed_halt_count: 161
- cost stress OOS daily_bleed_halt_count: 14

또한 replay 분석상 observed max consecutive losses는 stress 기준 16으로 높다. 이는 실행된 trade sequence 기준이며, 실제 진입 시도 중 상당수가 DailyBleedGuard에 의해 block되었다. 따라서 paper-forward에서는 연속손실 후 차단 이벤트와 재진입 간격을 별도 로그로 반드시 확인해야 한다.

## Verdict

trial 61은 BTCUSD-only 20000-row fixed replay와 cost stress를 통과했다. 단, 추천은 live 적용이 아니라 paper-forward config patch 후보에 한정한다.

