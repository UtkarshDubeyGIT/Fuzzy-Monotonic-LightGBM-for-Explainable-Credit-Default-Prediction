import json
import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def find_preprocess() -> Path:
    candidates = [
        PROJECT_ROOT / "src" / "data" / "preprocess.py",
        PROJECT_ROOT / "src" / "preprocess.py",
        Path(__file__).with_name("preprocess.py"),
        PROJECT_ROOT / "preprocess.py",
    ]
    p = first_existing(candidates)
    if not p:
        raise FileNotFoundError(f"preprocess.py not found. Checked: {', '.join(str(x) for x in candidates)}")
    return p

def find_baseline() -> Path:
    candidates = [
        PROJECT_ROOT / "src" / "models" / "baseline.py",
        PROJECT_ROOT / "src" / "models" / "train" / "baseline.py",
        PROJECT_ROOT / "src" / "baseline.py",
        PROJECT_ROOT / "baseline.py",
    ]
    p = first_existing(candidates)
    if not p:
        raise FileNotFoundError(f"baseline.py not found. Checked: {', '.join(str(x) for x in candidates)}")
    return p

def find_fuzzy_monotonic() -> Path:
    candidates = [
        PROJECT_ROOT / "src" / "models" / "fuzzy_monotonic.py",
        Path(__file__).with_name("fuzzy_monotonic.py"),
        PROJECT_ROOT / "src" / "fuzzy_monotonic.py",
        PROJECT_ROOT / "fuzzy_monotonic.py",
    ]
    p = first_existing(candidates)
    if not p:
        raise FileNotFoundError(f"fuzzy_monotonic.py not found. Checked: {', '.join(str(x) for x in candidates)}")
    return p

def parse_last_json(text: str) -> Optional[Dict]:
    # Try to parse the last JSON object present in stdout/stderr
    matches = list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))
    for m in reversed(matches):
        frag = m.group(0)
        try:
            obj = json.loads(frag)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None

def run_and_collect(cmd: List[str], env: Dict[str, str], metrics_out: Optional[Path]) -> Dict:
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    # Prefer metrics file if created
    if metrics_out and metrics_out.exists():
        try:
            with metrics_out.open("r") as f:
                return json.load(f)
        except Exception:
            pass
    # Else try to parse JSON from stdout/stderr
    for stream in (proc.stdout or "", proc.stderr or ""):
        obj = parse_last_json(stream)
        if obj:
            return obj
    # Fallback to empty dict with status info
    return {
        "status": "unknown",
        "returncode": proc.returncode,
    }

def run_preprocess(engineered: bool, variant: str, out_dir: Path) -> None:
    preprocess_py = find_preprocess()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "ENGINEER_FEATURES": "1" if engineered else "0",
        "OUTPUT_DIR": str(out_dir),
        "VARIANT": variant,
    })
    cmd = [
        sys.executable,
        str(preprocess_py),
        "--engineer-features",
        "1" if engineered else "0",
        "--output-dir",
        str(out_dir),
        "--variant",
        variant,
    ]
    subprocess.run(cmd, env=env, cwd=PROJECT_ROOT, check=True)

def run_baseline(data_dir: Path, variant: str, metrics_out: Path) -> Dict:
    baseline_py = find_baseline()
    env = os.environ.copy()  # fixed stray character
    env.update({
        "DATA_DIR": str(data_dir),
        "METRICS_OUT": str(metrics_out),
        "VARIANT": variant,
    })
    cmd = [
        sys.executable,
        str(baseline_py),
        "--data-dir",
        str(data_dir),
        "--metrics-out",
        str(metrics_out),
        "--variant",
        variant,
    ]
    return run_and_collect(cmd, env, metrics_out)

def run_fuzzy_monotonic(skip_monotonic: bool, variant: str, metrics_out: Path) -> Dict:
    fuzzy_py = find_fuzzy_monotonic()
    env = os.environ.copy()
    env.update({
        "SKIP_MONOTONIC": "1" if skip_monotonic else "0",
        "METRICS_OUT": str(metrics_out),
        "VARIANT": variant,
    })
    cmd = [
        sys.executable,
        str(fuzzy_py),
        "--metrics-out",
        str(metrics_out),
        "--variant",
        variant,
    ]
    if skip_monotonic:
        cmd += ["--skip-monotonic"]  # pass only the requested flag
    return run_and_collect(cmd, env, metrics_out)

def prioritized_keys(dicts: List[Dict]) -> List[str]:
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    keys.discard("variant")
    priority = [
        # prioritize requested metrics
        "roc_auc",
        "pr_auc",
        "brier",
        "ks",
        # sensible fallbacks
        "auc",
        "accuracy",
        "f1",
        "f1_score",
        "precision",
        "recall",
        "logloss",
        "loss",
    ]
    ordered = [k for k in priority if k in keys]
    remaining = sorted(k for k in keys if k not in priority)
    return ["variant"] + ordered + remaining

def print_markdown_table(rows: List[Dict]) -> None:
    cols = prioritized_keys(rows)
    # Header
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join("---" for _ in cols) + " |")
    # Rows
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        print("| " + " | ".join(vals) + " |")

def main() -> None:
    variants: List[Dict] = []

    # 1) baseline_raw
    raw_variant = "baseline_raw"
    raw_data_dir = PROJECT_ROOT / "data" / f"processed_{raw_variant}"
    run_preprocess(engineered=False, variant=raw_variant, out_dir=raw_data_dir)
    raw_metrics_out = RESULTS_DIR / f"{raw_variant}_metrics.json"
    raw_metrics = run_baseline(data_dir=raw_data_dir, variant=raw_variant, metrics_out=raw_metrics_out)
    raw_metrics["variant"] = raw_variant
    variants.append(raw_metrics)

    # 2) baseline_engineered
    eng_variant = "baseline_engineered"
    eng_data_dir = PROJECT_ROOT / "data" / f"processed_{eng_variant}"
    run_preprocess(engineered=True, variant=eng_variant, out_dir=eng_data_dir)
    eng_metrics_out = RESULTS_DIR / f"{eng_variant}_metrics.json"
    eng_metrics = run_baseline(data_dir=eng_data_dir, variant=eng_variant, metrics_out=eng_metrics_out)
    eng_metrics["variant"] = eng_variant
    variants.append(eng_metrics)

    # 3) fuzzy (skip monotonic constraints)
    fuzzy_variant = "fuzzy"
    fuzzy_metrics_out = RESULTS_DIR / f"{fuzzy_variant}_metrics.json"
    fuzzy_metrics = run_fuzzy_monotonic(skip_monotonic=True, variant=fuzzy_variant, metrics_out=fuzzy_metrics_out)
    fuzzy_metrics["variant"] = fuzzy_variant
    variants.append(fuzzy_metrics)

    # 4) fuzzy_monotonic (normal)
    fmon_variant = "fuzzy_monotonic"
    fmon_metrics_out = RESULTS_DIR / f"{fmon_variant}_metrics.json"
    fmon_metrics = run_fuzzy_monotonic(skip_monotonic=False, variant=fmon_variant, metrics_out=fmon_metrics_out)
    fmon_metrics["variant"] = fmon_variant
    variants.append(fmon_metrics)

    # Save combined table
    ablation_path = RESULTS_DIR / "ablation_table.json"
    with ablation_path.open("w") as f:
        json.dump(variants, f, indent=2)

    # Print markdown table
    print_markdown_table(variants)

if __name__ == "__main__":
    main()
