from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.config import load_config
from core.control import (
    DESIRED_STATE_RUN,
    DESIRED_STATE_STOP,
    RuntimeControlChannel,
    write_desired_state,
)


class DashboardSettingsPayload(BaseModel):
    api_key: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    llm_enabled: Optional[bool] = None
    risk_per_trade_pct: Optional[float] = None
    daily_loss_limit_pct: Optional[float] = None
    session_loss_limit_pct: Optional[float] = None
    max_consecutive_losses: Optional[int] = None
    persist_api_key: bool = False


class DashboardContext:
    def __init__(self, config_path: Path) -> None:
        cfg = load_config(config_path)
        storage = dict(cfg.get("storage", {}))
        dashboard = dict(cfg.get("dashboard", {}))
        llm_cfg = dict(cfg.get("llm_assist", {}))
        risk_cfg = dict(cfg.get("risk_guard", {}))

        self.state_path = Path(str(storage.get("state_path"))).resolve()
        self.events_path = Path(str(storage.get("events_path"))).resolve()
        self.settings_path = Path(str(dashboard.get("settings_path", llm_cfg.get("settings_path", "./dashboard_settings.json")))).resolve()
        self.control = RuntimeControlChannel(path=str(dashboard.get("control_path", "./runtime_control.json")))

        self.runtime_settings: Dict[str, Any] = {
            "llm_assist": {
                "enabled": bool(llm_cfg.get("enabled", False)),
                "provider": str(llm_cfg.get("provider", "gemini")),
                "model": str(llm_cfg.get("model", "gemini-3-flash")),
                "api_key_env": str(llm_cfg.get("api_key_env", "GEMINI_API_KEY")),
                "base_url": str(llm_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai")),
                "api_key": "",
            },
            "risk_guard": {
                "risk_per_trade_pct": float(risk_cfg.get("risk_per_trade_pct", 0.05)),
                "daily_loss_limit_pct": float(risk_cfg.get("daily_loss_limit_pct", 0.06)),
                "session_loss_limit_pct": float(risk_cfg.get("session_loss_limit_pct", 0.12)),
                "max_consecutive_losses": int(risk_cfg.get("max_consecutive_losses", 5)),
            },
        }
        self._load_settings_file()

    def _load_settings_file(self) -> None:
        if not self.settings_path.exists():
            return
        try:
            with self.settings_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                llm = payload.get("llm_assist")
                risk = payload.get("risk_guard")
                if isinstance(llm, dict):
                    self.runtime_settings["llm_assist"].update(llm)
                if isinstance(risk, dict):
                    self.runtime_settings["risk_guard"].update(risk)
        except Exception:
            pass

    def persist_settings(self, persist_api_key: bool = False) -> None:
        payload = {
            "llm_assist": dict(self.runtime_settings.get("llm_assist", {})),
            "risk_guard": dict(self.runtime_settings.get("risk_guard", {})),
        }
        if not persist_api_key:
            payload["llm_assist"]["api_key"] = ""

        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.settings_path.with_suffix(self.settings_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp.replace(self.settings_path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _tail_events(path: Path, limit: int = 50) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    q: Deque[str] = deque(maxlen=max(1, int(limit)))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                q.append(line)

    events: List[Dict[str, Any]] = []
    for line in q:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _tail_events_by_type(path: Path, limit: int, allowed: set[str]) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    picked: List[Dict[str, Any]] = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("event") not in allowed:
            continue
        picked.append(payload)
        if len(picked) >= limit:
            break
    picked.reverse()
    return picked


def _dashboard_html() -> str:
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MT5 자동매매 대시보드</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root{--bg:#f4f8f7;--panel:#fff;--ink:#1b2a2f;--muted:#5a6d73;--primary:#0f766e;--accent:#f97316;--danger:#dc2626;--ok:#16a34a;--border:#d2e2de;}
    *{box-sizing:border-box} body{margin:0;font-family:"Pretendard","Noto Sans KR","Segoe UI",sans-serif;color:var(--ink);background:radial-gradient(1200px 400px at 10% -5%, #dff4ee 0%, transparent 60%), var(--bg);}
    .wrap{max-width:1200px;margin:0 auto;padding:20px}
    .top{display:flex;justify-content:space-between;gap:14px;margin-bottom:18px;background:linear-gradient(130deg,#0f766e,#115e59);color:#fff;border-radius:20px;padding:18px 20px;box-shadow:0 8px 24px rgba(15,118,110,.28)}
    .top h1{margin:0;font-size:30px}.top p{margin:6px 0 0;opacity:.9}.links{display:flex;gap:8px;flex-wrap:wrap}
    .pill{border:1px solid rgba(255,255,255,.35);color:#fff;text-decoration:none;padding:7px 10px;border-radius:999px;font-size:13px}
    .grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:14px}.card{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:14px;box-shadow:0 2px 8px rgba(0,0,0,.04)}
    .c-3{grid-column:span 3}.c-4{grid-column:span 4}.c-6{grid-column:span 6}.c-8{grid-column:span 8}.c-12{grid-column:span 12}
    .label{color:var(--muted);font-size:12px;margin-bottom:6px}.val{font-size:26px;font-weight:700}.good{color:var(--ok)}.bad{color:var(--danger)}.title{font-weight:700;margin:0 0 10px}
    .row{display:flex;gap:8px;flex-wrap:wrap} button{border:0;border-radius:10px;padding:10px 12px;color:#fff;cursor:pointer;font-weight:700}
    .b1{background:var(--primary)}.b2{background:#2563eb}.b3{background:var(--accent)}.b4{background:var(--danger)}
    .trade-btn{font-size:18px;padding:12px 16px;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,.14)}
    .long-btn{background:linear-gradient(135deg,#16a34a,#15803d)}
    .short-btn{background:linear-gradient(135deg,#dc2626,#b91c1c)}
    .status{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 0}.tag{padding:5px 9px;border-radius:999px;font-size:12px;border:1px solid var(--border);background:#f8fbfa}
    table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px 7px;border-bottom:1px solid #edf3f1;text-align:left}th{color:var(--muted);font-weight:700}
    .hint{font-size:12px;color:var(--muted)} input,select{width:100%;border:1px solid var(--border);border-radius:10px;padding:8px 10px;font-size:13px}
    .settings{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
    #toast{position:fixed;right:18px;bottom:18px;background:#111827;color:#fff;padding:10px 12px;border-radius:10px;opacity:0;transform:translateY(12px);transition:all .22s}
    #toast.show{opacity:.94;transform:translateY(0)}
    @media (max-width:980px){.c-3,.c-4,.c-6,.c-8{grid-column:span 12}.top{flex-direction:column}.settings{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="top">
      <div><h1>MT5 자동매매 대시보드</h1><p>일반 사용자 기준으로 핵심 상태를 한 화면에서 확인하도록 구성했습니다.</p></div>
      <div class="links"><a class="pill" href="/docs" target="_blank">개발자 API 문서</a><a class="pill" href="/openapi.json" target="_blank">OpenAPI JSON</a></div>
    </section>
    <section class="grid">
      <article class="card c-3"><div class="label">잔고</div><div id="balance" class="val">-</div></article>
      <article class="card c-3"><div class="label">평가금</div><div id="equity" class="val">-</div></article>
      <article class="card c-3"><div class="label">미실현 손익</div><div id="floating" class="val">-</div></article>
      <article class="card c-3"><div class="label">열린 포지션</div><div id="openpos" class="val">-</div></article>
      <article class="card c-6"><h3 class="title">실시간 제어</h3><div class="row"><button class="trade-btn long-btn" onclick="manualEntry('long')">▲ BTC 롱 진입</button><button class="trade-btn short-btn" onclick="manualEntry('short')">▼ BTC 숏 진입</button></div><div class="row"><button class="b1" onclick="control('pause')">일시정지</button><button class="b2" onclick="control('resume')">재개</button><button class="b3" onclick="control('flatten')">즉시 청산</button><button class="b4" onclick="control('halt')">강제 정지</button></div><div class="status" id="control-tags"></div><p class="hint">수동 진입은 기존 리스크/수량/SLTP 검증을 그대로 통과합니다.</p></article>
      <article class="card c-6"><h3 class="title">리스크 가드</h3><div class="settings"><div><div class="label">거래당 리스크(%)</div><input id="risk_per_trade_pct" type="number" step="0.001" /></div><div><div class="label">일손실 한도(%)</div><input id="daily_loss_limit_pct" type="number" step="0.001" /></div><div><div class="label">세션 손실 한도(%)</div><input id="session_loss_limit_pct" type="number" step="0.001" /></div><div><div class="label">연속 손실 최대</div><input id="max_consecutive_losses" type="number" step="1" /></div></div><div style="margin-top:10px"><button class="b1" onclick="saveRisk()">리스크 설정 저장</button></div></article>
      <article class="card c-8"><h3 class="title">최근 실현손익 추이</h3><canvas id="pnlChart" height="110"></canvas></article>
      <article class="card c-4"><h3 class="title">LLM 보조 설정</h3><div class="settings" style="grid-template-columns:1fr"><div><div class="label">제공자</div><select id="llm_provider"><option value="gemini">gemini</option><option value="openai">openai</option></select></div><div><div class="label">모델명</div><input id="llm_model" placeholder="예: gemini-3-flash"/></div><div class="row"><button class="b2" type="button" onclick="setModel('gemini-3-pro')">Gemini 3 Pro</button><button class="b2" type="button" onclick="setModel('gemini-3-flash')">Gemini 3 Flash</button></div><div><div class="label">API Key</div><input id="llm_api_key" type="password" placeholder="기본: 비영구 저장"/></div></div><div class="row" style="margin-top:10px"><button class="b1" onclick="saveLlm(true)">LLM ON</button><button class="b4" onclick="saveLlm(false)">LLM OFF</button></div><p class="hint">Gemini 선택 시 base_url/api_key_env 자동 설정됩니다.</p></article>
      <article class="card c-12"><h3 class="title">최근 거래 로그</h3><table><thead><tr><th>시간</th><th>이벤트</th><th>종목</th><th>전략/사유</th><th>손익</th><th>상태</th></tr></thead><tbody id="trades-body"></tbody></table></article>
    </section>
  </div>
  <div id="toast"></div>
  <script>
    const KRW = new Intl.NumberFormat('ko-KR', {maximumFractionDigits: 2}); let chart = null;
    const num=v=>(v===null||v===undefined||Number.isNaN(Number(v)))?'-':KRW.format(Number(v));
    const signed=v=>{if(v===null||v===undefined||Number.isNaN(Number(v)))return '-';const n=Number(v);return (n>0?'+':'')+KRW.format(n);};
    function showToast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),1600);}
    async function fetchJson(url,opt){const r=await fetch(url,opt);if(!r.ok)throw new Error('HTTP '+r.status);return await r.json();}
    async function control(kind){try{await fetchJson('/api/control/'+kind,{method:'POST'});showToast('요청 전송: '+kind);await loadAll();}catch(e){showToast('실패: '+e.message);}}
    async function manualEntry(side){try{await fetchJson('/api/control/enter/'+side,{method:'POST'});showToast('수동 진입 요청: '+side.toUpperCase());await loadAll();}catch(e){showToast('진입 실패: '+e.message);}}
    async function saveRisk(){try{const p={risk_per_trade_pct:Number(document.getElementById('risk_per_trade_pct').value),daily_loss_limit_pct:Number(document.getElementById('daily_loss_limit_pct').value),session_loss_limit_pct:Number(document.getElementById('session_loss_limit_pct').value),max_consecutive_losses:Number(document.getElementById('max_consecutive_losses').value)};await fetchJson('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});showToast('리스크 설정 저장됨');}catch(e){showToast('저장 실패: '+e.message);}}
    function setModel(name){document.getElementById('llm_model').value=name;}
    async function saveLlm(on){try{const provider=(document.getElementById('llm_provider').value||'gemini').toLowerCase();const isGemini=provider==='gemini';const p={llm_enabled:!!on,provider:provider,model:document.getElementById('llm_model').value||null,api_key:document.getElementById('llm_api_key').value||null,api_key_env:isGemini?'GEMINI_API_KEY':'OPENAI_API_KEY',base_url:isGemini?'https://generativelanguage.googleapis.com/v1beta/openai':'https://api.openai.com/v1',persist_api_key:false};await fetchJson('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});showToast(on?'LLM 보조 ON':'LLM 보조 OFF');await loadAll();}catch(e){showToast('LLM 설정 실패: '+e.message);}}
    function renderControlTags(control,risk){const el=document.getElementById('control-tags');const tags=[['일시정지',control?.paused],['수동정지',control?.manual_halt],['청산요청',control?.flatten_requested],['리스크중단',risk?.halted]];el.innerHTML=tags.map(([k,v])=>`<span class="tag">${k}: <b>${v?'ON':'OFF'}</b></span>`).join('');}
    function renderChart(items){const trades=items.filter(x=>x.event==='trade_ledger').slice(-40);const labels=trades.map((_,i)=>String(i+1));const pnls=trades.map(t=>Number(t.realized_pnl||0));const ctx=document.getElementById('pnlChart').getContext('2d');if(chart)chart.destroy();chart=new Chart(ctx,{type:'line',data:{labels,datasets:[{label:'실현손익',data:pnls,borderColor:'#0f766e',backgroundColor:'rgba(15,118,110,0.14)',tension:0.26,fill:true}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{y:{ticks:{color:'#486067'},grid:{color:'#e7f0ed'}},x:{ticks:{color:'#486067'},grid:{display:false}}}}});}
    function renderTrades(items){const tbody=document.getElementById('trades-body');const rows=items.slice(-40).reverse().map(e=>{const t=e.ts_utc||'-';const ev=e.event||'-';const symbol=e.symbol||'-';const reason=e.reason||e.strategy||'-';const pnl=(e.realized_pnl ?? e?.result?.pnl ?? null);const st=e?.result?.status||'-';const cls=Number(pnl)<0?'bad':(Number(pnl)>0?'good':'');return `<tr><td>${t}</td><td>${ev}</td><td>${symbol}</td><td>${reason}</td><td class="${cls}">${signed(pnl)}</td><td>${st}</td></tr>`;}).join('');tbody.innerHTML=rows||'<tr><td colspan="6">거래 데이터가 없습니다.</td></tr>';}
    async function loadAll(){try{const [status,pos,trades]=await Promise.all([fetchJson('/api/status'),fetchJson('/api/positions'),fetchJson('/api/trades/recent?limit=100')]);document.getElementById('balance').textContent=num(pos.balance);document.getElementById('equity').textContent=num(pos.equity);document.getElementById('floating').textContent=signed(pos.floating_pnl);document.getElementById('openpos').textContent=String(pos.open_positions??0);const f=document.getElementById('floating');f.classList.remove('good','bad');if(Number(pos.floating_pnl)>0)f.classList.add('good');if(Number(pos.floating_pnl)<0)f.classList.add('bad');const risk=status?.state?.risk_guard||{};const control=status?.control||{};renderControlTags(control,risk);const rg=status?.runtime_settings?.risk_guard||{};document.getElementById('risk_per_trade_pct').value=rg.risk_per_trade_pct??'';document.getElementById('daily_loss_limit_pct').value=rg.daily_loss_limit_pct??'';document.getElementById('session_loss_limit_pct').value=rg.session_loss_limit_pct??'';document.getElementById('max_consecutive_losses').value=rg.max_consecutive_losses??'';const llm=status?.runtime_settings?.llm_assist||{};document.getElementById('llm_provider').value=llm.provider||'gemini';document.getElementById('llm_model').value=llm.model||'';const items=trades?.items||[];renderChart(items);renderTrades(items);}catch(e){showToast('데이터 조회 실패: '+e.message);}}
    loadAll(); setInterval(loadAll,5000);
  </script>
</body>
</html>"""


def create_app(config_path: str = "./config.yaml") -> FastAPI:
    ctx = DashboardContext(config_path=Path(config_path).resolve())
    app = FastAPI(title="MT5 Bot Dashboard", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    def dashboard_home() -> HTMLResponse:
        return HTMLResponse(_dashboard_html())

    @app.get("/api/status")
    def get_status() -> Dict[str, Any]:
        state = _read_json(ctx.state_path)
        control = ctx.control.load()
        return {
            "state": state,
            "control": control,
            "runtime_settings": {
                "llm_assist": {
                    "enabled": bool(ctx.runtime_settings.get("llm_assist", {}).get("enabled", False)),
                    "provider": str(ctx.runtime_settings.get("llm_assist", {}).get("provider", "gemini")),
                    "model": str(ctx.runtime_settings.get("llm_assist", {}).get("model", "")),
                    "api_key_env": str(ctx.runtime_settings.get("llm_assist", {}).get("api_key_env", "GEMINI_API_KEY")),
                    "base_url": str(ctx.runtime_settings.get("llm_assist", {}).get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai")),
                    "api_key_loaded": bool(ctx.runtime_settings.get("llm_assist", {}).get("api_key", "")),
                },
                "risk_guard": dict(ctx.runtime_settings.get("risk_guard", {})),
            },
        }

    @app.get("/api/positions")
    def get_positions() -> Dict[str, Any]:
        state = _read_json(ctx.state_path)
        account = state.get("account", {}) if isinstance(state, dict) else {}
        return {
            "open_positions": account.get("open_positions", 0),
            "balance": account.get("balance"),
            "equity": account.get("equity"),
            "floating_pnl": account.get("floating_pnl"),
        }

    @app.get("/api/trades/recent")
    def get_recent_trades(limit: int = 50) -> Dict[str, Any]:
        limit = max(1, min(500, int(limit)))
        allowed = {
            "order_submit",
            "position_exit",
            "trade_ledger",
            "manual_flatten",
            "risk_guard_halt",
        }
        events = _tail_events_by_type(ctx.events_path, limit=limit, allowed=allowed)
        return {"items": events}

    @app.post("/api/settings")
    def post_settings(payload: DashboardSettingsPayload) -> Dict[str, Any]:
        llm = dict(ctx.runtime_settings.get("llm_assist", {}))
        risk = dict(ctx.runtime_settings.get("risk_guard", {}))

        if payload.api_key is not None:
            llm["api_key"] = str(payload.api_key)
        if payload.provider is not None:
            llm["provider"] = str(payload.provider).strip().lower()
        if payload.model is not None:
            llm["model"] = str(payload.model)
        if payload.api_key_env is not None:
            llm["api_key_env"] = str(payload.api_key_env)
        if payload.base_url is not None:
            llm["base_url"] = str(payload.base_url)
        if payload.llm_enabled is not None:
            llm["enabled"] = bool(payload.llm_enabled)

        if payload.risk_per_trade_pct is not None:
            risk["risk_per_trade_pct"] = float(payload.risk_per_trade_pct)
        if payload.daily_loss_limit_pct is not None:
            risk["daily_loss_limit_pct"] = float(payload.daily_loss_limit_pct)
        if payload.session_loss_limit_pct is not None:
            risk["session_loss_limit_pct"] = float(payload.session_loss_limit_pct)
        if payload.max_consecutive_losses is not None:
            risk["max_consecutive_losses"] = int(payload.max_consecutive_losses)

        ctx.runtime_settings["llm_assist"] = llm
        ctx.runtime_settings["risk_guard"] = risk
        ctx.persist_settings(persist_api_key=payload.persist_api_key)

        return {
            "ok": True,
            "persist_api_key": bool(payload.persist_api_key),
            "settings_path": str(ctx.settings_path),
            "llm_assist": {
                "enabled": bool(llm.get("enabled", False)),
                "provider": str(llm.get("provider", "gemini")),
                "model": str(llm.get("model", "")),
                "api_key_env": str(llm.get("api_key_env", "GEMINI_API_KEY")),
                "base_url": str(llm.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai")),
                "api_key_loaded": bool(llm.get("api_key", "")),
            },
            "risk_guard": risk,
        }

    @app.post("/api/control/pause")
    def post_pause() -> Dict[str, Any]:
        ctx.control.set_paused(True)
        return {"ok": True, "control": ctx.control.load()}

    @app.post("/api/control/resume")
    def post_resume() -> Dict[str, Any]:
        ctx.control.request_resume()
        write_desired_state(
            DESIRED_STATE_RUN,
            source="dashboard",
            reason="api_control_resume",
            metadata={"endpoint": "/api/control/resume"},
        )
        return {"ok": True, "control": ctx.control.load()}

    @app.post("/api/control/flatten")
    def post_flatten() -> Dict[str, Any]:
        ctx.control.request_flatten()
        return {"ok": True, "control": ctx.control.load()}

    @app.post("/api/control/enter/{side}")
    def post_manual_entry(side: str) -> Dict[str, Any]:
        raw = str(side or "").strip().lower()
        if raw in {"long", "buy"}:
            action = "BUY"
        elif raw in {"short", "sell"}:
            action = "SELL"
        else:
            return {"ok": False, "error": "invalid_side", "allowed": ["long", "short"]}
        ctx.control.request_manual_entry(symbol="BTCUSD", action=action, source="dashboard")
        return {"ok": True, "requested": {"symbol": "BTCUSD", "action": action}, "control": ctx.control.load()}

    @app.post("/api/control/halt")
    def post_halt() -> Dict[str, Any]:
        ctx.control.request_manual_halt(source="dashboard", reason="api_control_halt")
        write_desired_state(
            DESIRED_STATE_STOP,
            source="dashboard",
            reason="api_control_halt",
            metadata={"endpoint": "/api/control/halt"},
        )
        return {"ok": True, "control": ctx.control.load()}

    return app


app = create_app()
