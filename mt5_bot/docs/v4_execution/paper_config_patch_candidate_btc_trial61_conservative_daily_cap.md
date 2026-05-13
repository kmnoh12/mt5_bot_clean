# Conservative Paper Config Patch Candidate: BTCUSD Trial 61

- 상태: paper-forward 후보
- live 적용 금지
- 기준 후보: balanced trial 61
- 보수 변경: daily loss cap을 3.75 USD에서 3.00 USD로 낮춤

## Conservative Variant

```yaml
paper_forward_candidate:
  enabled: true
  live_apply: false
  source_trial_id: 61
  variant: conservative_daily_cap_3_00
  symbol: BTCUSD
  account_equity_reference_usd: 140
  account_leverage_reference: 500
  max_effective_leverage: 15.0
  max_margin_used_pct: 5
  target_net_loss_usd: 0.75
  hard_max_net_loss_usd: 1.25
  max_daily_net_loss_usd: 3.00

daily_bleed_guard:
  enabled: true
  max_daily_net_loss_usd: 3.00
  stop_after_consecutive_losses: 3
  cooldown_after_loss_minutes: 15
  cooldown_after_same_setup_loss_minutes: 60
  same_direction_loss_limit_per_day: 2
  same_symbol_loss_limit_per_day: 3
```

## 적용 의도

- stress replay의 worst_daily_pnl은 약 -3.5391 USD였으므로, 3.00 USD cap은 같은 상황을 더 일찍 차단하는 보수형 guard다.
- 이 variant는 live 적용이 아니라 paper-forward 로그 수집을 위한 runtime config patch다.
- paper-forward 중 daily cap hit, consecutive loss block, same-symbol/direction block을 함께 기록해야 한다.

