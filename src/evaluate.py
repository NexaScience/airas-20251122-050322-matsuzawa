import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix

# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def _json_dump(obj: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(path)


# -----------------------------------------------------------------------------
# Per-run visualisations
# -----------------------------------------------------------------------------

def _export_learning_curve(history_df: pd.DataFrame, run_id: str, out_dir: Path):
    if "train_loss" not in history_df.columns:
        return
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=np.arange(len(history_df)), y=history_df["train_loss"], label="train_loss")
    if "val_accuracy" in history_df.columns:
        sns.lineplot(x=np.arange(len(history_df)), y=history_df["val_accuracy"], label="val_accuracy")
    plt.xlabel("Update step")
    plt.ylabel("Metric value")
    plt.title(f"Learning curve – {run_id}")
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / f"{run_id}_learning_curve.pdf"
    plt.savefig(fig_path)
    plt.close()
    print(fig_path)


def _export_confusion(preds: List[str], golds: List[str], run_id: str, out_dir: Path):
    uniq = list({*preds, *golds})
    if len(uniq) == 0 or len(uniq) > 50:
        return
    cm = confusion_matrix(golds, preds, labels=uniq)
    plt.figure(figsize=(max(6, len(uniq) * 0.4), max(4, len(uniq) * 0.4)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=uniq, yticklabels=uniq)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title(f"Confusion – {run_id}")
    plt.tight_layout()
    fig_path = out_dir / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(fig_path)
    plt.close()
    print(fig_path)


def export_run_artifacts(run: "wandb.apis.public.Run", out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    history_df = run.history()
    summary = run.summary._json_dict
    config = dict(run.config)

    _json_dump({"config": config, "summary": summary, "history": history_df.to_dict("list")}, out_root / "metrics.json")

    _export_learning_curve(history_df, run.id, out_root)

    if "_preds" in summary and "_golds" in summary:
        _export_confusion(summary["_preds"], summary["_golds"], run.id, out_root)


# -----------------------------------------------------------------------------
# Aggregated analysis
# -----------------------------------------------------------------------------

def _primary_metric(metrics: Dict[str, Dict[str, float]]) -> str:
    """Pick a sensible primary metric to report."""
    for candidate in ["test_accuracy", "accuracy", "val_accuracy"]:
        if candidate in metrics:
            return candidate
    return list(metrics.keys())[0]


def _mcnemar(p1: List[str], g: List[str], p2: List[str]) -> float:
    """McNemar test p-value (two-sided, exact binomial)."""
    assert len(p1) == len(p2) == len(g)
    n01 = n10 = 0
    for a, gold, b in zip(p1, g, p2):
        correct1 = a == gold
        correct2 = b == gold
        if correct1 and not correct2:
            n10 += 1
        elif correct2 and not correct1:
            n01 += 1
    if n01 + n10 == 0:
        return 1.0
    bigger = max(n01, n10)
    return binomtest(bigger, n01 + n10, p=0.5).pvalue


def aggregate(runs: List["wandb.apis.public.Run"], results_dir: Path):
    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- collect numeric metrics ----------------
    metrics: Dict[str, Dict[str, float]] = {}
    for r in runs:
        for k, v in r.summary._json_dict.items():
            if isinstance(v, (int, float)):
                metrics.setdefault(k, {})[r.id] = float(v)

    primary = _primary_metric(metrics)

    def _best(label: str):
        rid, val = None, None
        for run_id, metric_val in metrics.get(primary, {}).items():
            if label in run_id:
                if val is None or metric_val > val:
                    rid, val = run_id, metric_val
        return rid, val

    best_prop_id, best_prop_val = _best("proposed")
    best_base_id, best_base_val = _best("comparative")
    if best_base_val is None:
        best_base_id, best_base_val = _best("baseline")

    gap = None
    if best_prop_val is not None and best_base_val is not None:
        greater_is_better = not any(word in primary.lower() for word in ["loss", "error", "perplexity"])
        raw_gap = best_prop_val - best_base_val
        if not greater_is_better:
            raw_gap *= -1
        gap = raw_gap / max(1e-8, abs(best_base_val)) * 100

    # McNemar significance for predictions if available
    p_value = None
    if best_prop_id and best_base_id:
        run_dict = {r.id: r for r in runs}
        s_prop = run_dict[best_prop_id].summary._json_dict
        s_base = run_dict[best_base_id].summary._json_dict
        if all(k in s_prop for k in ("_preds", "_golds")) and "_preds" in s_base:
            p_value = _mcnemar(s_prop["_preds"], s_prop["_golds"], s_base["_preds"])

    aggregated = {
        "primary_metric": primary,
        "metrics": metrics,
        "best_proposed": {"run_id": best_prop_id, "value": best_prop_val},
        "best_baseline": {"run_id": best_base_id, "value": best_base_val},
        "gap": gap,
        "mcnemar_p": p_value,
    }
    _json_dump(aggregated, comp_dir / "aggregated_metrics.json")

    # ---------------- comparison bar chart ----------------
    if primary in metrics:
        ids = list(metrics[primary].keys())
        vals = [metrics[primary][i] for i in ids]
        plt.figure(figsize=(max(6, len(ids) * 0.5), 4))
        sns.barplot(x=ids, y=vals, palette="viridis")
        for i, v in enumerate(vals):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        plt.ylabel(primary)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig_path = comp_dir / f"comparison_{primary}_bar_chart.pdf"
        plt.savefig(fig_path)
        plt.close()
        print(fig_path)

    # ---------------- boxplot of val_accuracy per run ----------------
    records = []
    for r in runs:
        hist = r.history(keys=["val_accuracy"])  # only fetch required column
        if "val_accuracy" in hist.columns and not hist["val_accuracy"].dropna().empty:
            for v in hist["val_accuracy"].dropna().tolist():
                records.append({"run_id": r.id, "val_accuracy": v})

    if records:
        df_box = pd.DataFrame(records)
        plt.figure(figsize=(max(6, len(runs) * 0.6), 4))
        sns.boxplot(x="run_id", y="val_accuracy", data=df_box, palette="pastel")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig_path = comp_dir / "comparison_val_accuracy_boxplot.pdf"
        plt.savefig(fig_path)
        plt.close()
        print(fig_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str, help="JSON list string of run IDs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)

    # load global wandb settings
    import yaml as _yaml

    cfg_file = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(cfg_file, "r") as f:
        root_cfg = _yaml.safe_load(f)
    entity = root_cfg["wandb"]["entity"]
    project = root_cfg["wandb"]["project"]

    api = wandb.Api()
    runs = []
    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        export_run_artifacts(run, results_dir / rid)
        runs.append(run)

    aggregate(runs, results_dir)


if __name__ == "__main__":
    main()
