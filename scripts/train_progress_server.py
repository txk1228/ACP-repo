#!/usr/bin/env python3
"""Live ACP training progress dashboard on http://127.0.0.1:8765

Spec (FFT) baselines + Conv compare runs, with live progress and physical RMSE.
"""

from __future__ import annotations

import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OUTPUTS = Path.home() / "training_outputs"
HOME = Path.home() / "ACP_fx"

# Canonical Spec baselines (archived)
SPEC_FLIP_DIR = OUTPUTS / "2026.07.17_14.42.42_flip_up_new_resnet_230"
SPEC_VASE_DIR = OUTPUTS / "2026.07.18_10.23.05_vase_wiping_resnet_230"

FLIP_SPEC_LOG = HOME / "train_flip_up.log"
VASE_SPEC_LOG = HOME / "train_vase_wiping.log"
FLIP_CONV_LOG = HOME / "train_flip_up_conv.log"
VASE_CONV_LOG = HOME / "train_vase_wiping_conv.log"

TOTAL_EPOCHS = 300
SEC_PER_STEP_FLIP = 0.33
SEC_PER_STEP_VASE = 1.0
STEPS_PER_EPOCH_FLIP = 105
STEPS_PER_EPOCH_VASE = 781
HOST, PORT = "127.0.0.1", 8765


def latest_run(pattern: str) -> Path | None:
    runs = sorted(OUTPUTS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def best_run(patterns: list[str]) -> Path | None:
    """Pick run with most train_loss epoch rows among matching dirs."""
    best, best_n = None, -1
    for pat in patterns:
        for d in OUTPUTS.glob(pat):
            if not d.is_dir():
                continue
            n = len(parse_losses(d))
            if n > best_n:
                best, best_n = d, n
    return best


def parse_losses(run_dir: Path | None) -> list[dict]:
    if not run_dir:
        return []
    path = run_dir / "logs.json.txt"
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "train_loss" in obj:
                epoch = int(obj.get("epoch", 0))
                if epoch >= TOTAL_EPOCHS:
                    continue
                rows.append(
                    {
                        "epoch": epoch,
                        "step": int(obj.get("global_step", 0)),
                        "loss": float(obj["train_loss"]),
                        "lr": float(obj.get("lr", 0)),
                    }
                )
    return rows


def parse_live_progress(log_path: Path) -> dict:
    out = {
        "epoch": None,
        "step_cur": None,
        "step_total": None,
        "epoch_pct": None,
        "live_loss": None,
    }
    if not log_path.exists():
        return out
    data = log_path.read_bytes()[-20000:].decode("utf-8", "ignore")
    matches = list(
        re.finditer(
            r"Training epoch (\d+):\s+(\d+)%\|.*?(\d+)/(\d+)\s\[[^\]]*loss=([0-9.eE+-]+)\]",
            data,
        )
    )
    if not matches:
        matches = list(
            re.finditer(
                r"Training epoch (\d+):\s+(\d+)%\|.*?(\d+)/(\d+)\s\[",
                data,
            )
        )
        if not matches:
            epochs = [int(m) for m in re.findall(r"Training epoch (\d+):", data)]
            if epochs:
                out["epoch"] = min(max(epochs), TOTAL_EPOCHS - 1)
            return out
        m = matches[-1]
        out["epoch"] = min(int(m.group(1)), TOTAL_EPOCHS - 1)
        out["epoch_pct"] = int(m.group(2))
        out["step_cur"] = int(m.group(3))
        out["step_total"] = int(m.group(4))
        return out
    m = matches[-1]
    out["epoch"] = min(int(m.group(1)), TOTAL_EPOCHS - 1)
    out["epoch_pct"] = int(m.group(2))
    out["step_cur"] = int(m.group(3))
    out["step_total"] = int(m.group(4))
    out["live_loss"] = float(m.group(5))
    return out


def scan_train_cmds() -> dict[str, bool]:
    """Detect which of the four jobs is currently training."""
    flags = {
        "flip_spec": False,
        "vase_spec": False,
        "flip_conv": False,
        "vase_conv": False,
    }
    try:
        import subprocess

        out = subprocess.check_output(["ps", "aux"], text=True)
    except Exception:
        return flags
    for line in out.splitlines():
        if "/bin/bash" in line or "bash -lc" in line or "bash -c" in line:
            continue
        if "train.py" not in line:
            continue
        if "train_conv_compare_vase_workspace" in line or (
            "train_conv" in line and "vase_wiping" in line
        ):
            flags["vase_conv"] = True
        elif "train_conv_compare_flip_workspace" in line or (
            "conv_compare" in line and "flip" in line
        ):
            flags["flip_conv"] = True
        elif "task=vase_wiping_spec" in line:
            flags["vase_spec"] = True
        elif "train_spec_workspace" in line and "logging.mode=disabled" in line:
            flags["flip_spec"] = True
    # Only one GPU job expected; prefer conv / vase if multiple match
    if flags["vase_conv"]:
        flags["flip_conv"] = flags["flip_spec"] = flags["vase_spec"] = False
    elif flags["flip_conv"]:
        flags["flip_spec"] = flags["vase_spec"] = False
    elif flags["vase_spec"]:
        flags["flip_spec"] = False
    return flags


def parse_val_metrics(run_dir: Path | None) -> dict | None:
    if not run_dir:
        return None
    candidates = [run_dir / "eval_latest_val_metrics.json"]
    epoch_files = sorted(run_dir.glob("eval_epoch_*_val_metrics.json"))
    if epoch_files:
        candidates.append(epoch_files[-1])
    for path in candidates:
        if not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        ref = obj.get("reference_pose_pos_rmse_mm")
        vt = obj.get("virtual_target_pos_rmse_mm")
        st = obj.get("stiffness_rmse_Npm")
        if ref is None and vt is None and st is None:
            continue
        return {
            "epoch": int(obj["epoch"]) if obj.get("epoch") is not None else None,
            "ref_pos_rmse_mm": float(ref) if ref is not None else None,
            "vt_pos_rmse_mm": float(vt) if vt is not None else None,
            "stiffness_rmse_Npm": float(st) if st is not None else None,
            "train_loss": float(obj["train_loss"])
            if obj.get("train_loss") is not None
            else None,
            "source": path.name,
        }
    return None


def parse_val_metrics_history(run_dir: Path | None) -> list[dict]:
    if not run_dir:
        return []
    rows = []
    for path in sorted(run_dir.glob("eval_epoch_*_val_metrics.json")):
        try:
            obj = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        epoch = obj.get("epoch")
        if epoch is None:
            continue
        epoch = int(epoch)
        if epoch >= TOTAL_EPOCHS:
            continue
        rows.append(
            {
                "epoch": epoch,
                "ref_pos_rmse_mm": float(
                    obj.get("reference_pose_pos_rmse_mm", float("nan"))
                ),
                "vt_pos_rmse_mm": float(
                    obj.get("virtual_target_pos_rmse_mm", float("nan"))
                ),
                "stiffness_rmse_Npm": float(
                    obj.get("stiffness_rmse_Npm", float("nan"))
                ),
            }
        )
    return rows


def build_job(
    name: str,
    status: str,
    run_dir: Path | None,
    losses: list[dict],
    live: dict,
    sec_per_step: float,
    steps_per_epoch: int,
    encoding: str,
) -> dict:
    epoch = live.get("epoch")
    if epoch is None and losses:
        epoch = losses[-1]["epoch"]
    step_cur = live.get("step_cur")
    step_total = live.get("step_total") or steps_per_epoch
    epoch_pct = live.get("epoch_pct")
    if epoch_pct is None and step_cur is not None and step_total:
        epoch_pct = round(100.0 * step_cur / step_total, 1)

    if epoch is None:
        overall_pct = 0.0
    else:
        frac = (step_cur / step_total) if (step_cur is not None and step_total) else 0.0
        if status == "done":
            overall_pct = 100.0
        else:
            overall_pct = round(100.0 * min((epoch + frac) / TOTAL_EPOCHS, 1.0), 2)

    loss = live.get("live_loss")
    if loss is None and losses:
        loss = losses[-1]["loss"]

    eta_hours = None
    if status == "running" and epoch is not None:
        remain_steps_this = max((step_total or steps_per_epoch) - (step_cur or 0), 0)
        remain_epochs = max(TOTAL_EPOCHS - 1 - epoch, 0)
        remain_sec = remain_steps_this * sec_per_step + remain_epochs * (
            steps_per_epoch * sec_per_step
        )
        eta_hours = round(remain_sec / 3600.0, 2)

    filled = int(round(min(max(overall_pct, 0), 100) / 2.5))
    bar = "█" * filled + "░" * (40 - filled)

    job = {
        "name": name,
        "encoding": encoding,
        "status": status,
        "run_dir": str(run_dir) if run_dir else None,
        "epoch": epoch,
        "step_cur": step_cur,
        "step_total": step_total,
        "epoch_pct": epoch_pct,
        "loss": loss,
        "pct": overall_pct,
        "eta_hours": eta_hours,
        "bar": bar,
        "losses": losses,
        "total_epochs": TOTAL_EPOCHS,
        "val_metrics": parse_val_metrics(run_dir),
        "val_metrics_history": parse_val_metrics_history(run_dir),
    }
    if status == "done":
        job["pct"] = 100.0
        job["epoch"] = TOTAL_EPOCHS - 1
        job["epoch_pct"] = 100.0
        job["bar"] = "█" * 40
        if losses:
            job["loss"] = losses[-1]["loss"]
    return job


def _status_for(running: bool, epoch, queued: bool = False) -> str:
    if running:
        return "running"
    if epoch is not None and epoch >= TOTAL_EPOCHS - 1:
        return "done"
    if queued:
        return "queued"
    return "idle"


def job_status() -> dict:
    flags = scan_train_cmds()

    flip_spec_dir = (
        SPEC_FLIP_DIR
        if SPEC_FLIP_DIR.is_dir()
        else best_run(["*flip_up*resnet*", "*flip_up*"])
    )
    vase_spec_dir = (
        SPEC_VASE_DIR
        if SPEC_VASE_DIR.is_dir()
        else best_run(["*vase_wiping*resnet*", "*vase_wiping*"])
    )
    # Prefer conv_compare dirs; avoid picking Spec dirs
    flip_conv_dir = latest_run("*flip_up*conv_compare*") or latest_run(
        "*flip_up*conv_230*"
    )
    vase_conv_dir = latest_run("*vase_wiping*conv_compare*") or latest_run(
        "*vase_wiping*conv_230*"
    )

    jobs_meta = [
        (
            "flip_spec",
            "flip_up · Spec (FFT)",
            "spec",
            flip_spec_dir,
            FLIP_SPEC_LOG,
            SEC_PER_STEP_FLIP,
            STEPS_PER_EPOCH_FLIP,
            False,
        ),
        (
            "vase_spec",
            "vase_wiping · Spec (FFT)",
            "spec",
            vase_spec_dir,
            VASE_SPEC_LOG,
            SEC_PER_STEP_VASE,
            STEPS_PER_EPOCH_VASE,
            False,
        ),
        (
            "flip_conv",
            "flip_up · Conv",
            "conv",
            flip_conv_dir,
            FLIP_CONV_LOG,
            SEC_PER_STEP_FLIP,
            STEPS_PER_EPOCH_FLIP,
            False,
        ),
        (
            "vase_conv",
            "vase_wiping · Conv",
            "conv",
            vase_conv_dir,
            VASE_CONV_LOG,
            SEC_PER_STEP_VASE,
            STEPS_PER_EPOCH_VASE,
            True,  # queued until flip_conv done
        ),
    ]

    result_jobs: dict[str, dict] = {}
    for key, name, enc, run_dir, log_path, sec, steps, can_queue in jobs_meta:
        running = flags.get(key, False)
        losses = parse_losses(run_dir)
        live = parse_live_progress(log_path) if running else {}
        if not live.get("epoch") and not running:
            live = parse_live_progress(log_path)
        epoch = losses[-1]["epoch"] if losses else live.get("epoch")
        queued = False
        if can_queue and not running:
            # Queue vase_conv while flip_conv not done
            flip_c = result_jobs.get("flip_conv")
            # flip_conv not built yet when we process vase — compute tentatively
            if key == "vase_conv":
                fc_losses = parse_losses(flip_conv_dir)
                fc_ep = fc_losses[-1]["epoch"] if fc_losses else None
                fc_done = fc_ep is not None and fc_ep >= TOTAL_EPOCHS - 1
                if not fc_done and not flags.get("flip_conv"):
                    # flip still to run or running
                    if flags.get("flip_conv") or (fc_ep is None or fc_ep < TOTAL_EPOCHS - 1):
                        if not (fc_done):
                            queued = fc_ep is None or fc_ep < TOTAL_EPOCHS - 1
                if flags.get("flip_conv"):
                    queued = True
        status = _status_for(running, epoch, queued=queued and not running)
        # Spec baselines forced done if dirs complete
        if enc == "spec" and epoch is not None and epoch >= TOTAL_EPOCHS - 1:
            status = "done"
        live_for_build = live if running else {"epoch": epoch}
        result_jobs[key] = build_job(
            name, status, run_dir, losses, live_for_build, sec, steps, enc
        )

    # Fix vase_conv queued after flip_conv known
    fc = result_jobs["flip_conv"]
    vc = result_jobs["vase_conv"]
    if vc["status"] not in ("running", "done"):
        if fc["status"] == "running" or (
            fc["status"] != "done" and fc.get("epoch") is not None
        ):
            if fc["status"] != "done":
                vc["status"] = "queued"
        elif fc["status"] == "idle" and fc.get("epoch") is None:
            vc["status"] = "queued"

    # Active = first running, else flip_conv if in progress, else vase_spec
    active_key = "vase_spec"
    for k in ("flip_conv", "vase_conv", "flip_spec", "vase_spec"):
        if result_jobs[k]["status"] == "running":
            active_key = k
            break
    else:
        if result_jobs["flip_conv"]["status"] in ("running", "idle", "queued"):
            if result_jobs["flip_conv"].get("epoch") is not None or result_jobs[
                "flip_conv"
            ]["status"] == "running":
                active_key = "flip_conv"

    # Prefer running
    for k in ("vase_conv", "flip_conv", "vase_spec", "flip_spec"):
        if result_jobs[k]["status"] == "running":
            active_key = k
            break

    return {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_epochs": TOTAL_EPOCHS,
        "active": active_key,
        "jobs": result_jobs,
        # Back-compat aliases (Spec)
        "flip": result_jobs["flip_spec"],
        "vase": result_jobs["vase_spec"],
    }


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ACP Training Progress</title>
<style>
  :root {
    --bg: #0f1216; --panel: #171b21; --text: #e7eaee; --muted: #8b939e;
    --line: #2a313a; --accent: #3d8bfd; --ok: #3ecf8e; --warn: #e6a23c;
    --conv: #c084fc;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text); min-height: 100vh; padding: 28px;
  }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
  h2 { font-size: 14px; font-weight: 600; color: var(--muted); margin: 18px 0 10px; letter-spacing: 0.04em; text-transform: uppercase; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
  .hero {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 22px 24px; margin-bottom: 8px;
  }
  .hero-title { font-size: 14px; color: var(--muted); margin-bottom: 6px; }
  .hero-pct {
    font-size: 48px; font-weight: 650; font-variant-numeric: tabular-nums;
    letter-spacing: -0.03em; line-height: 1.1;
  }
  .hero-meta { color: var(--muted); font-size: 14px; margin-top: 8px; }
  .bar-lg {
    height: 18px; background: #222831; border-radius: 99px; overflow: hidden; margin-top: 16px;
  }
  .bar-lg > i {
    display: block; height: 100%; width: 0%;
    background: var(--accent);
    border-radius: 99px; transition: width .5s ease;
  }
  .bar-sm {
    height: 8px; background: #222831; border-radius: 99px; overflow: hidden; margin-top: 6px;
  }
  .bar-sm > i {
    display: block; height: 100%; background: var(--accent);
    border-radius: 99px; transition: width .4s ease;
  }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 18px 20px;
  }
  .card.conv { border-color: color-mix(in srgb, var(--conv) 35%, var(--line)); }
  .row { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
  .name { font-size: 15px; font-weight: 600; }
  .tag {
    font-size: 11px; padding: 2px 8px; border-radius: 6px; margin-left: 8px;
    border: 1px solid var(--line); color: var(--muted); font-weight: 500;
  }
  .tag.spec { color: var(--accent); }
  .tag.conv { color: var(--conv); }
  .pill {
    font-size: 12px; padding: 3px 10px; border-radius: 999px;
    border: 1px solid var(--line); color: var(--muted);
  }
  .pill.run { color: var(--ok); border-color: color-mix(in srgb, var(--ok) 40%, var(--line)); }
  .pill.done { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 40%, var(--line)); }
  .pill.queue { color: var(--warn); }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 14px 0 10px; }
  .stats-3 { grid-template-columns: repeat(3, 1fr); }
  .stat .l { color: var(--muted); font-size: 11px; }
  .stat .v { font-size: 18px; font-weight: 600; margin-top: 2px; font-variant-numeric: tabular-nums; }
  .stat .v.sm { font-size: 15px; }
  .metrics-block {
    margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line);
  }
  .metrics-title { font-size: 12px; color: var(--muted); margin-bottom: 8px; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--muted); margin-top: 10px; }
  .chart-wrap { margin-top: 16px; }
  .chart-title { font-size: 13px; color: var(--muted); margin-bottom: 8px; }
  canvas { width: 100%; height: 180px; display: block; }
  .foot { margin-top: 16px; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
  <h1>ACP Training Progress</h1>
  <div class="sub">Spec 基线 + Conv 对比 · 每 2 秒刷新 · <span id="updated">—</span></div>
  <div class="hero" id="hero"></div>
  <h2>Spec · FFT（已完成基线）</h2>
  <div class="grid" id="cards-spec"></div>
  <h2>Conv · 时序卷积（对比实验）</h2>
  <div class="grid" id="cards-conv"></div>
  <div class="card chart-wrap">
    <div class="chart-title">Train loss · Spec 蓝/绿 · Conv 紫/橙（虚线 = vase）</div>
    <canvas id="chart" width="1100" height="180"></canvas>
  </div>
  <div class="card chart-wrap" id="rmse-card" style="display:none">
    <div class="chart-title">物理量 RMSE · 实线 Spec vase · 虚线 Conv（有数据时）· 蓝 ref / 绿 vt / 橙刚度</div>
    <canvas id="rmse-chart" width="1100" height="200"></canvas>
  </div>
  <div class="foot">目标 300 epochs（0～299）· logs.json.txt + eval_*_val_metrics.json + train_*.log</div>
<script>
function statusLabel(s) {
  if (s === 'running') return ['RUNNING', 'run'];
  if (s === 'done') return ['DONE', 'done'];
  if (s === 'queued') return ['QUEUED', 'queue'];
  return [String(s).toUpperCase(), ''];
}
function fmt(x, d=4) {
  if (x == null || Number.isNaN(x)) return '—';
  return typeof x === 'number' ? x.toFixed(d) : x;
}
function overallPct(j, total) {
  if (j.status === 'done') return 100;
  if (j.pct != null && j.pct >= 100) return 100;
  const frac = (j.step_cur != null && j.step_total) ? (j.step_cur / j.step_total) : 0;
  return Math.min(((j.epoch ?? 0) + frac) / total, 1.0) * 100;
}
function rmseBlock(m) {
  if (!m) return `<div class="metrics-block"><div class="metrics-title">物理量 RMSE</div><div style="color:var(--muted);font-size:13px">尚无 eval（checkpoint 后写入）</div></div>`;
  const ep = m.epoch != null ? ` · eval @ epoch ${m.epoch}` : '';
  return `<div class="metrics-block">
    <div class="metrics-title">物理量 RMSE（验证集）${ep}</div>
    <div class="stats stats-3">
      <div class="stat"><div class="l">参考位姿 pos</div><div class="v sm">${fmt(m.ref_pos_rmse_mm, 2)} <span style="font-size:11px;color:var(--muted);font-weight:500">mm</span></div></div>
      <div class="stat"><div class="l">虚拟目标 pos</div><div class="v sm">${fmt(m.vt_pos_rmse_mm, 2)} <span style="font-size:11px;color:var(--muted);font-weight:500">mm</span></div></div>
      <div class="stat"><div class="l">刚度</div><div class="v sm">${fmt(m.stiffness_rmse_Npm, 1)} <span style="font-size:11px;color:var(--muted);font-weight:500">N/m</span></div></div>
    </div>
  </div>`;
}
function hero(d) {
  const j = d.jobs[d.active] || d.jobs.flip_conv || d.jobs.vase_spec;
  const eta = j.eta_hours == null ? '—' : (j.eta_hours < 1 ? `${Math.round(j.eta_hours*60)} min` : `${j.eta_hours} h`);
  const step = (j.step_cur != null) ? `${j.step_cur} / ${j.step_total}` : '—';
  const total = d.total_epochs || 300;
  const pct = overallPct(j, total);
  const m = j.val_metrics;
  const rmseMeta = m
    ? ` · RMSE ref ${fmt(m.ref_pos_rmse_mm,2)} / vt ${fmt(m.vt_pos_rmse_mm,2)} mm · k ${fmt(m.stiffness_rmse_Npm,0)} N/m`
    : '';
  const enc = (j.encoding || '').toUpperCase();
  return `<div class="hero-title">当前焦点 · ${j.name} <span class="tag ${j.encoding||''}">${enc}</span></div>
    <div class="hero-pct">${fmt(pct, 2)}%</div>
    <div class="hero-meta">epoch ${j.epoch ?? '—'} / ${total - 1} · step ${step} · loss ${fmt(j.loss, 4)} · ETA ${eta}${rmseMeta}</div>
    <div class="bar-lg"><i style="width:${pct}%;background:${j.encoding==='conv'?'var(--conv)':'var(--accent)'}"></i></div>
    <div class="mono">${j.bar || ''}</div>`;
}
function card(j, totalEpochs) {
  const [lab, cls] = statusLabel(j.status);
  const total = totalEpochs || 300;
  const ep = j.epoch == null ? '—' : `${j.epoch} / ${total - 1}`;
  const step = (j.step_cur != null) ? `${j.step_cur}/${j.step_total}` : '—';
  const eta = j.eta_hours == null ? '—' : (j.eta_hours < 1 ? `${Math.round(j.eta_hours*60)}m` : `${j.eta_hours}h`);
  const pct = overallPct(j, total);
  const enc = j.encoding || 'spec';
  return `<div class="card ${enc}">
    <div class="row">
      <div class="name">${j.name} <span class="tag ${enc}">${enc.toUpperCase()}</span></div>
      <span class="pill ${cls}">${lab}</span>
    </div>
    <div class="stats">
      <div class="stat"><div class="l">Epoch</div><div class="v">${ep}</div></div>
      <div class="stat"><div class="l">本 epoch</div><div class="v" style="font-size:15px">${step}</div></div>
      <div class="stat"><div class="l">Loss</div><div class="v">${fmt(j.loss, 4)}</div></div>
      <div class="stat"><div class="l">ETA</div><div class="v">${eta}</div></div>
    </div>
    <div class="row"><span style="color:var(--muted);font-size:12px">总体</span>
      <span style="font-variant-numeric:tabular-nums">${fmt(pct,2)}%</span></div>
    <div class="bar-sm"><i style="width:${pct}%;background:${enc==='conv'?'var(--conv)':'var(--accent)'}"></i></div>
    <div class="row" style="margin-top:10px"><span style="color:var(--muted);font-size:12px">本 epoch</span>
      <span style="font-variant-numeric:tabular-nums">${fmt(j.epoch_pct,1)}%</span></div>
    <div class="bar-sm"><i style="width:${Math.min(j.epoch_pct||0,100)}%;background:var(--ok)"></i></div>
    ${rmseBlock(j.val_metrics)}
  </div>`;
}
function drawLossChart(jobs) {
  const c = document.getElementById('chart');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0,0,W,H);
  const series = [
    {pts: jobs.flip_spec?.losses, color:'#3d8bfd', dash:false},
    {pts: jobs.vase_spec?.losses, color:'#3ecf8e', dash:true},
    {pts: jobs.flip_conv?.losses, color:'#c084fc', dash:false},
    {pts: jobs.vase_conv?.losses, color:'#fb923c', dash:true},
  ].filter(s => s.pts && s.pts.length);
  if (!series.length) {
    ctx.fillStyle = '#8b939e'; ctx.font = '13px sans-serif';
    ctx.fillText('等待 epoch 结束以绘制 loss…', 20, H/2);
    return;
  }
  const all = series.flatMap(s => s.pts);
  const maxE = Math.max(299, ...all.map(p => p.epoch));
  const maxL = Math.max(...all.map(p => p.loss), 0.01);
  const minL = Math.min(...all.map(p => p.loss), 0);
  const pad = {l: 44, r: 12, t: 10, b: 28};
  const x = e => pad.l + (e / maxE) * (W - pad.l - pad.r);
  const y = l => pad.t + (1 - (l - minL) / (maxL - minL || 1)) * (H - pad.t - pad.b);
  ctx.strokeStyle = '#2a313a'; ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b); ctx.lineTo(W-pad.r, H-pad.b); ctx.stroke();
  ctx.fillStyle = '#8b939e'; ctx.font = '11px sans-serif';
  ctx.fillText(maxL.toFixed(2), 4, pad.t+8);
  ctx.fillText(minL.toFixed(2), 4, H-pad.b);
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.setLineDash(s.dash ? [6,4] : []);
    ctx.beginPath();
    s.pts.forEach((p,i) => {
      const X = x(p.epoch), Y = y(p.loss);
      if (i===0) ctx.moveTo(X,Y); else ctx.lineTo(X,Y);
    });
    ctx.stroke();
  }
  ctx.setLineDash([]);
}
function drawRmseChart(jobs) {
  const wrap = document.getElementById('rmse-card');
  const c = document.getElementById('rmse-chart');
  const series = [];
  // Spec vase (solid), Conv vase (dash), also flip if present
  const packs = [
    {hist: jobs.vase_spec?.val_metrics_history, alpha: 1, dash: false, label: 'spec-vase'},
    {hist: jobs.vase_conv?.val_metrics_history, alpha: 1, dash: true, label: 'conv-vase'},
    {hist: jobs.flip_spec?.val_metrics_history, alpha: 0.7, dash: false, label: 'spec-flip'},
    {hist: jobs.flip_conv?.val_metrics_history, alpha: 0.7, dash: true, label: 'conv-flip'},
  ];
  for (const p of packs) {
    if (p.hist && p.hist.length) series.push(p);
  }
  if (!series.length) { wrap.style.display = 'none'; return; }
  wrap.style.display = 'block';
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0,0,W,H);
  const pad = {l: 52, r: 12, t: 14, b: 28};
  const all = series.flatMap(s => s.hist);
  const maxE = Math.max(299, ...all.map(p => p.epoch));
  const posVals = all.flatMap(p => [p.ref_pos_rmse_mm, p.vt_pos_rmse_mm]).filter(x => !Number.isNaN(x));
  const maxPos = Math.max(...posVals, 0.01);
  const x = e => pad.l + (e / maxE) * (W - pad.l - pad.r);
  const yPos = l => pad.t + (1 - l / maxPos) * (H - pad.t - pad.b);
  ctx.strokeStyle = '#2a313a'; ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b); ctx.lineTo(W-pad.r, H-pad.b); ctx.stroke();
  ctx.fillStyle = '#8b939e'; ctx.font = '11px sans-serif';
  ctx.fillText(maxPos.toFixed(1)+' mm', 2, pad.t+8);
  const colors = {ref:'#3d8bfd', vt:'#3ecf8e', k:'#e6a23c'};
  for (const s of series) {
    // ref
    ctx.strokeStyle = colors.ref; ctx.lineWidth = 2; ctx.globalAlpha = s.alpha;
    ctx.setLineDash(s.dash ? [6,4] : []);
    ctx.beginPath();
    s.hist.forEach((p,i) => { const X=x(p.epoch),Y=yPos(p.ref_pos_rmse_mm); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y); });
    ctx.stroke();
    // vt
    ctx.strokeStyle = colors.vt;
    ctx.beginPath();
    s.hist.forEach((p,i) => { const X=x(p.epoch),Y=yPos(p.vt_pos_rmse_mm); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y); });
    ctx.stroke();
  }
  // stiffness — normalize per-series to chart height
  const stAll = all.map(p => p.stiffness_rmse_Npm).filter(x => !Number.isNaN(x));
  if (stAll.length) {
    const maxSt = Math.max(...stAll, 1), minSt = Math.min(...stAll, 0);
    const ySt = l => pad.t + (1 - (l - minSt) / (maxSt - minSt || 1)) * (H - pad.t - pad.b);
    for (const s of series) {
      ctx.strokeStyle = colors.k; ctx.lineWidth = 1.5; ctx.globalAlpha = s.alpha;
      ctx.setLineDash(s.dash ? [2,3] : [1,0]);
      ctx.beginPath();
      s.hist.forEach((p,i) => { const X=x(p.epoch),Y=ySt(p.stiffness_rmse_Npm); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y); });
      ctx.stroke();
    }
    ctx.fillStyle = '#e6a23c'; ctx.globalAlpha = 1;
    ctx.fillText(`k ${maxSt.toFixed(0)}→${minSt.toFixed(0)}`, W - 130, pad.t+8);
  }
  ctx.globalAlpha = 1; ctx.setLineDash([]);
  ctx.fillStyle = '#8b939e';
  ctx.fillText('实线=Spec · 虚线=Conv · 蓝ref 绿vt 橙刚度', pad.l + 8, H - 8);
}
async function tick() {
  const r = await fetch('/api/status');
  const d = await r.json();
  const jobs = d.jobs;
  document.getElementById('updated').textContent = d.updated_at;
  document.getElementById('hero').innerHTML = hero(d);
  document.getElementById('cards-spec').innerHTML =
    card(jobs.flip_spec, d.total_epochs) + card(jobs.vase_spec, d.total_epochs);
  document.getElementById('cards-conv').innerHTML =
    card(jobs.flip_conv, d.total_epochs) + card(jobs.vase_conv, d.total_epochs);
  drawLossChart(jobs);
  drawRmseChart(jobs);
  const active = jobs[d.active];
  const pct = overallPct(active, d.total_epochs || 300);
  document.title = `${pct.toFixed(1)}% · ${active.name}`;
}
tick();
setInterval(tick, 2000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path.startswith("/api/status"):
            body = json.dumps(job_status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        body = HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    s = job_status()
    a = s["jobs"][s["active"]]
    print(
        f"{a['name']} {a['status']} {a['pct']}% "
        f"epoch={a['epoch']} step={a['step_cur']}/{a['step_total']} "
        f"→ http://{HOST}:{PORT}",
        flush=True,
    )
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
