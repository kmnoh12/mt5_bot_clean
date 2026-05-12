# MT5 Profitability Root-Cause Report

작성일: 2026-05-12

## 1. 현재 거래 구조

- `runner.py`가 `core.config.load_config()`로 `config.yaml`과 `DEFAULT_CONFIG`를 병합한다.
- `core.runtime.TradingRuntime`이 모드에 따라 `BacktestGateway` 또는 `MT5LiveGateway`를 만든다.
- 전략은 `strategies/factory.py`를 통해 심볼별로 하나씩 매핑된다. 현재 체크인 YAML의 활성 universe는 `BTCUSD` 1개이고 전략은 `liquidity_sweep_reversal`, 타임프레임은 `TIMEFRAME_M1`이다.
- 진입 전에는 `RiskEngine`, `ExecutionChurnGuard`, `EntryQualityGuard`, `CostEdgeGuard`, `MTFConfirm`가 순서대로 손실/품질/비용 조건을 거른다.
- 보유 중에는 전략 결정, `DynamicTrailingProfitGuard`, `ExitEngine`, `ExitQualityGuard`, `ExitRetryGuard`가 청산 결정을 조정한다.
- 라이브 주문은 `MT5LiveGateway`의 하드 게이트를 통과해야 한다. 기본값은 `general.dry_run: true`, `execution.live_trading_enabled: false`이며 환경변수 `MT5_ALLOW_LIVE_TRADING=YES_I_ACCEPT_RISK`도 필요하다.
- 백테스트는 `brokers/backtest.py`의 CSV 로더와 `utils/backtest_sim.py`, `tools/staged_profit_pipeline.py`가 담당한다. 백테스트 경로는 MT5 라이브 API를 import하지 않도록 격리되어 있다.

## 2. 현재 주요 수치 기본값

`mt5_bot/config.yaml` 기준:

- 라이브 안전: `general.dry_run=true`, `execution.live_trading_enabled=false`, `execution.dry_run=false`.
- 포지션: `default_volume=0.01`, `max_positions_per_symbol=1`, `allow_opposite_position=false`, `bar_close_only=true`.
- 리스크: `risk_per_trade_pct=0.003`, `max_risk_per_trade_pct=0.005`, `daily_loss_limit_pct=0.02`, `session_loss_limit_pct=0.03`, `max_consecutive_losses=4`, `dynamic_risk_max_pct=0.005`, `dd_hard_stop_pct=0.08`.
- 진입 횟수: `max_entries_per_symbol_per_hour=1`, `max_entries_per_symbol_per_day=2`, `max_entries_global_per_day=3`, `per_symbol_daily_limits.BTCUSD=1`, `ETHUSD/SOLUSD/GOLD/Nvidia_turbo/AMD_turbo=0`.
- 재진입/청산 과다 방지: `reentry_cooldown_seconds=300`, `flip_reentry_cooldown_seconds=180`, `loss_reentry_lock_seconds=600`, `tiny_pnl_max_count_per_hour=2`, `tiny_pnl_cooldown_seconds=3600`.
- 품질 필터: `entry_quality_guard.enabled=true`, `min_score=0.62`, `min_score_risk_off=0.68`, `trend_only_symbols=[BTCUSD, ETHUSD, SOLUSD]`.
- 비용 대비 엣지: `cost_edge_guard.enabled=true`, 기본 `min_edge_to_cost_ratio=3.0`, `GOLD=2.5`.
- MTF 확인: `mtf_confirm.enabled=true`, `symbols=[BTCUSD]`, 확인 타임프레임 `TIMEFRAME_M5`.
- 청산 품질: `exit_quality_guard.enabled=true`, `tiny_profit_block_usd=2.0`, `min_hold_seconds_for_soft_exit=300`.
- LSR 전략: `TIMEFRAME_M1`, `trail_start_R=1.4`, `trail_tp_enabled=true`, `tp_R1=2.0`, `tp_R2=4.0`, `min_hold_bars=3`, `min_cooldown_bars=8`, BTC 심볼 override는 `displacement_mult=0.4`, `trend_filter_enabled=false`, `reclaim_window_sec=2400`, `zombie_bar_limit=60`, `min_cooldown_bars=3`.

## 3. 실거래/검증 데이터 인벤토리

확인 경로:

- `mt5_bot/data`: 존재하지 않음.
- `mt5_bot/reports`: 존재하지 않음.
- `mt5_bot/validation`: 존재하지 않음.
- `mt5_bot/memory`: `.gitkeep`만 있음.
- `mt5_bot/runtime/trade_event.json`: 단일 EXIT 이벤트 예시만 있으며 체결 원장이나 손익 시계열로 볼 수 없음.
- `mt5_bot/runtime/desired_state.json`: STOP 상태와 `orders_allowed=false` 메타데이터만 있음.

결론: 실제 히스토리 CSV, 체결 원장, OOS 리포트, 반복 백테스트 결과가 없어 수익률 개선을 실증할 수 없다. 따라서 이번 변경은 손실 기대값을 줄이는 구조적 안전장치, MT5-free 검증 계층, 보수적 기본값에 한정한다.

## 4. 구조적 손실 원인 순위

1. M1 노이즈 과매매
   - 기존 YAML은 `liquidity_sweep_reversal`을 `BTCUSD/TIMEFRAME_M1`에 배치하고, 일부 심볼 override가 낮은 `displacement_mult`, 짧은 cooldown, 넓은 reclaim window를 허용했다.
   - 손실 원인: 스프레드/슬리피지보다 작은 신호를 반복 진입하면 승률이 높아도 기대값이 음수가 된다.

2. 비용 대비 엣지 필터 비활성화
   - 기존 YAML은 `cost_edge_guard.enabled=false`, `min_edge_to_cost_ratio_default=1.1`이었다.
   - 손실 원인: BTC/ETH/GOLD/CFD의 스프레드, commission, slippage가 작은 R 목표를 잠식한다.

3. 진입 품질 필터 비활성화
   - 기존 YAML은 `entry_quality_guard.enabled=false`, 모든 score 임계값이 `0.1`이었다.
   - 손실 원인: 추세/ADX/EMA/M5 정렬 없는 저품질 신호가 대부분 통과한다. 특히 M1에서 no-trade day가 손실보다 낫다는 선택을 못 한다.

4. 동적 리스크 확대 가능성
   - 기존 YAML은 `risk_per_trade_pct=0.005`, `max_risk_per_trade_pct=0.01`, `dynamic_risk_max_pct=0.03`, `kelly_fraction=0.5`였다.
   - 손실 원인: 짧은 표본의 일시적 승률로 lot/risk가 커지면 연속 손실 때 계좌 변동성이 과도해진다.

5. 심볼/레짐 불일치
   - BTC/ETH/SOL/GOLD/주식 CFD는 변동성, 거래시간, 스프레드 구조가 다르다. 같은 M1 reversal 계열 파라미터를 여러 심볼에 재사용하면 레짐별 edge가 깨진다.

6. 빠른 소익절/청산 스팸
   - 기존 코드에는 `ExitQualityGuard`, `ExitRetryGuard`, `ExecutionChurnGuard`가 있으나 YAML에서 진입 품질/비용 필터가 약해 앞단 손실을 충분히 줄이지 못했다.
   - 손실 원인: 작은 이익 청산과 빠른 재진입은 비용 누적을 키우고 큰 손실 1회가 여러 작은 이익을 지우게 한다.

7. 전략 간 충돌 가능성
   - 현재 universe는 심볼당 하나의 전략이지만 `strategies/*`에는 mean-reversion, breakout, trend, LSR 계열이 공존한다.
   - 런타임 확장 시 arbitration 없이 여러 신호 소스가 같은 심볼을 밀면 상충 진입/청산이 발생할 수 있다.

8. stale state / ghost-fill / reconciliation
   - `trade_ledger_normalized`, auto-close reconcile, live gate 테스트가 존재하지만 실제 broker truth ledger가 없다.
   - 손실 원인: 이미 닫힌 포지션을 로컬이 보유 중으로 오인하거나 보호 주문 실패 후 재시도/진입이 엉키면 손실이 커질 수 있다.

9. 부분청산/trailing/profit-lock 수학
   - trailing/profit-lock은 승자를 보존할 수 있지만 활성 조건이 낮으면 정상 변동을 noise로 보고 조기 청산할 수 있다.
   - 현재는 `min_hold_seconds_for_exit`, `min_breach_count_for_exit`, `ExitQualityGuard`가 완충하지만 데이터 없이 최적값은 증명 불가다.

10. 검증 누수와 OOS 부재
    - `utils/backtest_sim.py`는 튜닝형 simulator이고 `core.validation`은 리포트 존재/패스 확인만 담당했다.
    - 손실 원인: 비용, drawdown, churn, OOS split 없는 단일 backtest 점수는 실거래 robust edge를 증명하지 못한다.

## 5. 이번 재설계와 수용 기준

구현한 방향:

- `core/performance_metrics.py`에 MT5-free 성과 지표 계층을 추가한다.
  - trades, win rate, avg win, avg loss, expectancy, profit factor, max drawdown, fees/cost estimate, exposure seconds, quick-exit/churn counts를 계산한다.
  - `walk_forward_splits()`로 train/test split scaffolding을 제공한다.
- `tools/staged_profit_pipeline.py`가 새 metrics layer를 사용하게 하여 기존 이벤트 원장 기반 리포트가 expectancy/cost/churn 지표를 포함하게 한다.
- MT5 Python 모듈이 없는 오프라인/백테스트 환경에서는 dynamic lot fallback이 요청 volume을 그대로 통과시키지 않고 risk-based sizing 경로로 내려가게 한다.
- checked-in `config.yaml`과 `DEFAULT_CONFIG`를 quality-first 보수 프로필로 맞춘다.
  - live trading default off 보존.
  - risk cap 하향.
  - global/symbol daily entry cap 강화.
  - disabled symbol cap `0`을 실제 block으로 해석.
  - entry quality, cost edge, MTF 확인 활성화.
- 테스트로 다음을 검증한다.
  - metrics 계산 정확성.
  - walk-forward split이 시간 순서를 유지한다.
  - conservative YAML 기본값이 실제로 로드된다.
  - per-symbol daily limit `0`이 진입 차단으로 동작한다.
  - 기존 live order hard gate와 backtest isolation 테스트를 계속 통과시킨다.

수용 기준:

- 실제 수익률 개선 주장은 하지 않는다.
- 백테스트/검증 경로는 MT5 라이브 API 없이 동작해야 한다.
- 라이브 주문 게이트는 기본 차단 상태이며 `order_check/order_send/modify/close`가 게이트 닫힘 상태에서 호출되면 안 된다.
- 실데이터가 들어오면 최소 3단 검증이 필요하다: in-sample 탐색, walk-forward/OOS, paper/live shadow ledger reconciliation.

## 6. 다음 데이터 요구사항

실제 최적화를 하려면 아래가 필요하다.

- 최소 3-6개월 이상의 심볼별 M1/M5 OHLCV CSV와 실제 스프레드/commission/slippage 추정치.
- MT5 history deals CSV 또는 broker truth ledger.
- 각 심볼별 거래 가능 시간, min lot/contract size/tick value/tick size.
- 전략별 signal event, skipped reason, filled price, closed price, realized PnL, fee, hold time.
- 동일 기간의 walk-forward 결과와 OOS report JSON.

라이브나 paper 전환 전 체크:

1. `python3 -m compileall -q mt5_bot`
2. focused unittest/pytest 전체 통과.
3. `utils/backtest_sim.py` 또는 staged pipeline을 실제 CSV로 실행.
4. OOS report 생성 후 `validation.require_oos_pass=true`로 live readiness 확인.
5. paper 모드에서 broker truth ledger와 local `trade_ledger_normalized` 비교.
6. 그 이후에도 live gate는 명시적 config + 환경변수 + 수동 승인 없이는 열지 않는다.

## 7. 2026-05-13 Hermes 검증 실패 대응 계획

재현 명령:

- `cd mt5_bot && ../.venv-test/bin/python -m pytest -q --ignore=tests/test_dashboard_settings.py`
- 결과: 104 passed, 9 failed.

실패 분류:

- `test_lsr_strategy.py` 5개: 보수적 기본값 도입 후 LSR unit fixture가 sweep/reclaim 진입 전에 HOLD를 반환한다. 이 영역은 production default를 완화하지 않고, 테스트 fixture가 의도한 deterministic sweep/reclaim 조건을 명시적으로 주입해야 한다.
- `test_lsr_tick_sweep_reclaim.py` 3개: tick 전략의 spread guard가 bid/ask 없는 unit tick을 `SPREAD_UNAVAILABLE_SKIP`으로 차단한다. production에서는 spread 확인이 보수적이어야 하므로 기본 guard는 유지하고, 테스트/시뮬레이션 config에서 spread guard 정책을 명시적으로 끄거나 synthetic spread를 주입해야 한다.
- `test_runtime_daily_reference_levels.py::test_run_cycle_skips_dashboard_control_channel_when_disabled`: `dashboard.enabled=false`인데 `run_cycle()`이 `control_channel.load()`를 호출한다. 런타임 동작은 dashboard disabled일 때 control channel을 읽지 않는 것이 맞으므로 code path를 복구해야 한다.

수정 원칙:

- live default OFF, live order hard gate, OOS readiness gate는 완화하지 않는다.
- `config.yaml`의 conservative defaults는 유지한다.
- 전략 테스트는 fixture config에 명시적 override를 넣어 unit 목적을 분리한다.
- runtime fix는 dashboard disabled일 때 `{}` control payload를 사용하게 하는 최소 변경으로 제한한다.
- 수정 후 같은 Hermes 명령, LSR focused tests, runtime focused test, live safety/backtest isolation focused tests, compileall만 실행한다.
