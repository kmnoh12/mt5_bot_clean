from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from utils.backtest_sim import _simulate_symbol, _coerce_ohlc_frame

def map_params(p):
    return {
        'enabled': True,
        'atr_period': int(p['atr_period']),
        'pivot_lookback_sec': int(p['lookback'])*60,
        'swing_window': int(p['lookback']),
        'sweep_buffer_atr': float(p['sweep_atr']),
        'reclaim_buffer_atr': float(p['reclaim_atr']),
        'reclaim_window_sec': 120,
        'displacement_mult': max(1.0, float(p['disp_atr']) * 4.0),
        'displacement_lookback': 20,
        'sl_atr_mult': float(p['sl_atr']),
        'stop_buffer_atr': 0.05,
        'tp_R1': max(0.8, float(p['tp_r']) * 0.5),
        'tp_R2': float(p['tp_r']),
        'be_at_R': 1.0,
        'max_hold_bars': int(p['max_hold']),
        'min_hold_bars': 1,
        'min_cooldown_bars': int(p['cooldown']),
        'fvg_enabled': False,
        'retest_enabled': False,
        'trail_tp_enabled': False,
    }

def combine(ms):
    trades=sum(int(m.get('trades',0)) for m in ms); total=sum(float(m.get('total_r',0)) for m in ms); dd=max([float(m.get('max_drawdown_r',0)) for m in ms] or [0])
    return {'trades':trades,'total_r':round(total,6),'expectancy_r':round(total/trades,6) if trades else 0,'max_drawdown_r':round(dd,6),'folds':len(ms)}

def main():
    src=json.loads(Path('mt5_bot/optimization_runs/fast_btcusd_fee004.json').read_text())
    top=src['top'][:8]
    df=_coerce_ohlc_frame(pd.read_csv('mt5_bot/data/binance/BTCUSD_TIMEFRAME_M1.csv'))
    df=df.tail(30000).reset_index(drop=True)
    folds=[]; end=len(df)
    for _ in range(5):
        folds.append((end-3000,end)); end-=6000
    folds=list(reversed(folds))
    out=[]
    for item in top:
        params=map_params(item['params'])
        ms=[]
        for s,e in folds:
            ms.append(_simulate_symbol('BTCUSD', df.iloc[s:e].copy().reset_index(drop=True), params))
        out.append({'fast_id':item['id'],'fast_rank':item['rank'],'mapped_params':params,'exact':combine(ms),'exact_folds':ms})
        print(out[-1], flush=True)
    Path('mt5_bot/optimization_runs/exact_lsr_verify_top_fee004.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
