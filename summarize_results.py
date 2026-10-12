"""Aggregates evaluated k-fold experiments into the paper's result tables.

Reads each experiment directory's per-fold test results (written by
evaluate.py) and training histories, and writes to --output-dir:

    per_fold_results.csv          tablature metrics per fold
    summary_mean_std.csv          mean and population std over folds
    auxiliary_head_accuracy.csv   auxiliary decoder accuracy per fold
    string_assignment_errors.csv  wrong-string rates per category (overall = 1 - TDR)
    string_pair_errors.csv        which string is confused with which, pooled over folds
    paired_statistics.csv         fold-paired comparisons against each --reference
                                  (mean difference, 95% t interval, d_z, exact Wilcoxon p)
    gcs_per_fold.csv              gradient cosine similarity per fold and task
    uncertainty_weights.csv       learned log-variances at the best and last epoch
    uncertainty_weight_stability.csv  how much those log-variances move during training
    gcs_curves.pdf / .png         GCS over epochs, mean and range over folds (--gcs-run)
    SUMMARY.md                    readable overview of all of the above

Example:
    python summarize_results.py \\
        --run proposed=outputs/cnn_mtl_<timestamp> \\
        --run single=outputs/cnn_mtl_<timestamp> \\
        --reference proposed --output-dir results
"""
import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

TAB_COLUMNS = ["tab_precision", "tab_recall", "tab_f1", "tdr", "tab_mp_precision", "tab_mp_recall",
               "tab_mp_f1", "tab_error_total_rate", "tab_error_substitution_rate", "tab_error_miss_rate",
               "tab_error_false_alarm_rate", "tab_error_duplicate_pitch_rate", "best_threshold"]
AUX_COLUMNS = ["aux_mp_f1", "aux_mp_precision", "aux_mp_recall", "hand_pos_acc_all_frames", "hand_pos_acc",
               "hand_pos_f1", "string_act_f1", "string_act_precision", "string_act_recall",
               "pitch_class_f1", "pitch_class_f1_micro_frame"]
STRING_NAMES = ["E2", "A2", "D3", "G3", "B3", "E4"]
TASK_LABELS = {"multipitch": "Multi-pitch", "hand_position": "Fret region",
               "string_activity": "String activity", "pitch_class": "Pitch class"}


def fold_dirs(run_dir: str) -> list[tuple[int, str]]:
    """(fold number, path) for every fold_<n> directory of an experiment, in order."""
    out = []
    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        if name.startswith("fold_") and os.path.isdir(path) and name[5:].isdigit():
            out.append((int(name[5:]), path))
    return sorted(out)


def load_fold_results(run_dir: str) -> list[dict]:
    """Per-fold test results: the revision campaign's test_results_revised.json if a fold has
    one, otherwise evaluate.py's test_results.json. (In the original Colab runs, a
    test_results.json next to a revised file comes from an older evaluation that picked
    the threshold on the test player, so the revised file must win.)"""
    rows = []
    for k, path in fold_dirs(run_dir):
        for name in ("test_results_revised.json", "test_results.json"):
            file = os.path.join(path, name)
            if os.path.isfile(file):
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)
                row = dict(data.get("results", data))
                row["fold"] = int(data.get("fold", row.get("fold", k)))
                rows.append(row)
                break
    return rows


def training_history_stats(run_dir: str, label: str):
    """Per-fold GCS statistics, learned log-variances, and their stability, from history.json."""
    gcs_rows, uwl_rows, stability_rows = [], [], []
    for k, path in fold_dirs(run_dir):
        file = os.path.join(path, "history.json")
        if not os.path.isfile(file):
            continue
        with open(file, encoding="utf-8") as f:
            hist = json.load(f)

        val_f1 = hist.get("val_tab_f1") or []
        best_epoch = int(np.argmax(val_f1)) if val_f1 else None

        for key in sorted(x for x in hist if x.startswith("gcs_tab_")):
            # The trainer stores 0.0 for epochs where GCS was not computed.
            vals = np.array([v for v in hist[key] if v != 0.0], dtype=float)
            if vals.size == 0:
                continue
            gcs_rows.append({
                "config": label, "fold": k, "task": key[len("gcs_tab_"):],
                "n_epochs": int(vals.size), "gcs_mean": float(vals.mean()), "gcs_std": float(vals.std()),
                "gcs_min": float(vals.min()), "gcs_max": float(vals.max()),
                "pct_epochs_negative": float(100.0 * (vals < 0).mean()),
                "gcs_first5_mean": float(vals[:5].mean()), "gcs_last5_mean": float(vals[-5:].mean()),
            })

        sigmas = hist.get("sigma_weights") or []
        for which, idx in (("best_epoch", best_epoch), ("last_epoch", len(sigmas) - 1)):
            if not sigmas or idx is None or idx >= len(sigmas):
                continue
            for task, s in sigmas[idx].items():
                s = float(s[0] if isinstance(s, list) else s)
                uwl_rows.append({"config": label, "fold": k, "epoch_type": which, "epoch": idx + 1,
                                 "task": task, "log_var": s, "precision": math.exp(-s),
                                 "loss_weight": 0.5 * math.exp(-s)})

        if sigmas:
            for task in sorted(sigmas[0]):
                traj = np.array([float(e[task][0] if isinstance(e[task], list) else e[task])
                                 for e in sigmas if task in e], dtype=float)
                if traj.size < 2:
                    continue
                steps = np.abs(np.diff(traj))
                second_half = traj[len(traj) // 2:]
                stability_rows.append({
                    "config": label, "fold": k, "task": task, "n_epochs": int(traj.size),
                    "first_epoch": float(traj[0]), "last_epoch": float(traj[-1]),
                    "total_change": float(traj[-1] - traj[0]), "max_step": float(steps.max()),
                    "mean_step": float(steps.mean()), "sd_second_half": float(second_half.std()),
                    "range_second_half": float(second_half.max() - second_half.min()),
                })
    return gcs_rows, uwl_rows, stability_rows


def plot_gcs_curves(run_dir: str, out_path: str):
    """GCS over epochs for every auxiliary task: mean over folds, shaded min-max range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = {}
    for _, path in fold_dirs(run_dir):
        file = os.path.join(path, "history.json")
        if not os.path.isfile(file):
            continue
        with open(file, encoding="utf-8") as f:
            hist = json.load(f)
        for key in (x for x in hist if x.startswith("gcs_tab_")):
            values = [v for v in hist[key] if v != 0.0]
            if values:
                curves.setdefault(key[len("gcs_tab_"):], []).append(np.asarray(values, dtype=float))
    if not curves:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for task in sorted(curves):
        folds = curves[task]
        length = min(len(f) for f in folds)
        stack = np.stack([f[:length] for f in folds])
        epochs = np.arange(1, length + 1)
        line, = ax.plot(epochs, stack.mean(0), linewidth=1.8, label=TASK_LABELS.get(task, task))
        ax.fill_between(epochs, stack.min(0), stack.max(0), alpha=0.15, color=line.get_color())
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient cosine similarity")
    ax.legend(frameon=False, ncol=2)
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png", dpi=200)
    plt.close(fig)
    return out_path


def paired_comparison(reference, other) -> dict:
    """Fold-paired comparison: mean difference, 95% t interval, Cohen's d_z, exact two-sided Wilcoxon p."""
    from scipy import stats
    a, b = np.asarray(reference, float), np.asarray(other, float)
    diff = a - b
    n = len(diff)
    out = {"n": n, "mean_reference": a.mean(), "mean_other": b.mean(), "mean_diff": diff.mean(),
           "sd_diff": diff.std(ddof=1) if n > 1 else float("nan"),
           "reference_wins": int((diff > 0).sum()), "ties": int((diff == 0).sum())}
    if n > 1 and diff.std(ddof=1) > 0:
        half_width = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / math.sqrt(n)
        out.update(ci95_low=diff.mean() - half_width, ci95_high=diff.mean() + half_width,
                   cohen_dz=diff.mean() / diff.std(ddof=1))
    if np.any(diff != 0):
        try:
            w = stats.wilcoxon(a, b, alternative="two-sided")
            out.update(wilcoxon_statistic=float(w.statistic), wilcoxon_p=float(w.pvalue))
        except Exception as e:
            out["wilcoxon_error"] = str(e)
    return out


def fmt(x, spec):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def summarize(runs: dict, references: list[str], out_dir: str, gcs_run: str | None = None) -> str:
    """Writes every table listed in the module docstring for `runs` ({label: experiment dir})."""
    os.makedirs(out_dir, exist_ok=True)

    def csv(df, name):
        df.to_csv(os.path.join(out_dir, name), index=False)

    groups = {label: load_fold_results(path) for label, path in runs.items()}
    groups = {label: rows for label, rows in groups.items() if rows}

    csv(pd.DataFrame([{"config": g, "fold": r["fold"], "test_player": r.get("test_group"),
                       **{c: r.get(c) for c in TAB_COLUMNS}}
                      for g, rows in groups.items() for r in sorted(rows, key=lambda x: x["fold"])]),
        "per_fold_results.csv")

    summary = []
    for g, rows in groups.items():
        row = {"config": g, "n_folds": len(rows)}
        for c in TAB_COLUMNS + AUX_COLUMNS:
            vals = [r[c] for r in rows if r.get(c) is not None]
            if vals:
                row[f"{c}_mean"], row[f"{c}_std"] = float(np.mean(vals)), float(np.std(vals))
        summary.append(row)
    csv(pd.DataFrame(summary), "summary_mean_std.csv")

    csv(pd.DataFrame([{"config": g, "fold": r["fold"], **{c: r.get(c) for c in AUX_COLUMNS}}
                      for g, rows in groups.items() for r in rows
                      if r.get("aux_mp_f1") is not None or r.get("hand_pos_acc") is not None]),
        "auxiliary_head_accuracy.csv")

    breakdown = []
    for g, rows in groups.items():
        pooled = {}
        for r in rows:
            sa = r.get("string_assignment", {})
            for cat, v in sa.items():
                if isinstance(v, dict) and cat != "string_pairs":
                    acc = pooled.setdefault(cat, [0, 0])
                    acc[0] += v["events"]
                    acc[1] += v["errors"]
                    breakdown.append({"config": g, "fold": r["fold"], "category": cat, **v})
            breakdown.append({"config": g, "fold": r["fold"], "category": "adjacent_string_share_of_errors",
                              "error_rate": sa.get("adjacent_string_share_of_errors")})
            breakdown.append({"config": g, "fold": r["fold"], "category": "check_one_minus_tdr",
                              "error_rate": r.get("one_minus_tdr")})
        for cat, (ev, er) in pooled.items():
            breakdown.append({"config": g, "fold": "pooled", "category": cat, "events": ev, "errors": er,
                              "error_rate": er / ev if ev else None})
        adj = sum(r.get("string_assignment", {}).get("adjacent_errors", 0) for r in rows)
        non = sum(r.get("string_assignment", {}).get("nonadjacent_errors", 0) for r in rows)
        if adj + non:
            breakdown.append({"config": g, "fold": "pooled", "category": "adjacent_string_share_of_errors",
                              "events": adj + non, "errors": adj, "error_rate": adj / (adj + non)})
    csv(pd.DataFrame(breakdown), "string_assignment_errors.csv")

    pair_rows = []
    for g, rows in groups.items():
        pooled = {}
        for r in rows:
            for pair, n in r.get("string_assignment", {}).get("string_pairs", {}).items():
                pooled[pair] = pooled.get(pair, 0) + n
        total = sum(pooled.values())
        for pair, n in sorted(pooled.items(), key=lambda kv: -kv[1]):
            ref, est = pair.split("->")
            pair_rows.append({"config": g, "reference_string": int(ref), "estimated_string": int(est),
                              "errors": n, "share_of_errors": n / total if total else None,
                              "string_distance": abs(int(ref) - int(est))})
    csv(pd.DataFrame(pair_rows), "string_pair_errors.csv")

    gcs, uwl, stability = [], [], []
    for label, path in runs.items():
        g_rows, u_rows, s_rows = training_history_stats(path, label)
        gcs += g_rows
        uwl += u_rows
        stability += s_rows
    gcs_df, uwl_df, stability_df = pd.DataFrame(gcs), pd.DataFrame(uwl), pd.DataFrame(stability)
    csv(gcs_df, "gcs_per_fold.csv")
    csv(uwl_df, "uncertainty_weights.csv")
    csv(stability_df, "uncertainty_weight_stability.csv")

    gcs_run = gcs_run or next((r for r in references if r in runs), None)
    if gcs_run:
        plot_gcs_curves(runs[gcs_run], os.path.join(out_dir, "gcs_curves"))

    stats_rows = []
    for ref_key in (r for r in references if r in groups):
        ref = {r["fold"]: r for r in groups[ref_key]}
        for other_key in (g for g in groups if g != ref_key):
            other = {r["fold"]: r for r in groups[other_key]}
            folds = sorted(set(ref) & set(other))
            if len(folds) < 2:
                continue
            for metric in ["tab_f1", "tdr", "tab_mp_f1", "tab_precision", "tab_recall"]:
                stats_rows.append({"reference": ref_key, "compared": other_key, "metric": metric,
                                   "folds": ",".join(map(str, folds)),
                                   **paired_comparison([ref[k][metric] for k in folds],
                                                       [other[k][metric] for k in folds])})
    stats_df = pd.DataFrame(stats_rows)
    csv(stats_df, "paired_statistics.csv")

    lines = [
        "# Results", "",
        f"Updated {datetime.now():%Y-%m-%d %H:%M}. Values are mean ± population std over folds. "
        "Test metrics use evaluate.py's protocol (threshold selected on the validation player, "
        "full-track inference).", "",
        "## Tablature results", "",
        "| Configuration | Folds | P | R | F1 | TDR | F_mp (tab) |",
        "|---|---|---|---|---|---|---|",
    ]
    for g, rows in groups.items():
        def ms(c):
            vals = [r[c] for r in rows if r.get(c) is not None]
            return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" if vals else "-"
        lines.append(f"| {g} | {len(rows)} | {ms('tab_precision')} | {ms('tab_recall')} | {ms('tab_f1')} | "
                     f"{ms('tdr')} | {ms('tab_mp_f1')} |")

    if not stats_df.empty:
        lines += ["", "## Paired comparisons", "",
                  "| Reference | Compared | Metric | Mean diff | 95% CI | Wins | Wilcoxon p |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in stats_df[stats_df.metric.isin(["tab_f1", "tdr"])].iterrows():
            ci = (f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
                  if not pd.isna(r.get("ci95_low", float("nan"))) else "-")
            lines.append(f"| {r['reference']} | {r['compared']} | {r['metric']} | {r['mean_diff']:+.4f} | {ci} | "
                         f"{r['reference_wins']}/{r['n']} | {fmt(r.get('wilcoxon_p'), '.3f')} |")
        lines += ["", "With six folds the smallest attainable two-sided Wilcoxon p-value is 0.031."]

    if not gcs_df.empty:
        lines += ["", "## Gradient cosine similarity", "",
                  "| Configuration | Task | Mean GCS (across folds) | Negative epochs (%) |", "|---|---|---|---|"]
        for (cfg_name, task), sub in gcs_df.groupby(["config", "task"]):
            lines.append(f"| {cfg_name} | {task} | {sub.gcs_mean.mean():+.4f} ± {sub.gcs_mean.std(ddof=0):.4f} | "
                         f"{sub.pct_epochs_negative.mean():.1f} |")

    if not uwl_df.empty:
        lines += ["", "## Uncertainty weights at the best epoch", "",
                  "| Configuration | Task | log σ² | exp(−log σ²) |", "|---|---|---|---|"]
        for (cfg_name, task), sub in uwl_df[uwl_df.epoch_type == "best_epoch"].groupby(["config", "task"]):
            lines.append(f"| {cfg_name} | {task} | {sub.log_var.mean():+.3f} ± {sub.log_var.std(ddof=0):.3f} | "
                         f"{sub.precision.mean():.3f} |")

    if not stability_df.empty:
        lines += ["", "## Stability of the learned weights over training", "",
                  "| Configuration | Task | Largest step between epochs | SD over the second half |",
                  "|---|---|---|---|"]
        for (cfg_name, task), sub in stability_df.groupby(["config", "task"]):
            lines.append(f"| {cfg_name} | {task} | {sub.max_step.max():.4f} | {sub.sd_second_half.mean():.4f} |")

    if pair_rows:
        pairs_df = pd.DataFrame(pair_rows)
        lines += ["", "## Most frequent string confusions", "",
                  "| Configuration | Reference → estimated | Share of errors |", "|---|---|---|"]
        for cfg_name, sub in pairs_df.groupby("config"):
            for _, r in sub.nlargest(3, "errors").iterrows():
                lines.append(f"| {cfg_name} | {STRING_NAMES[int(r.reference_string)]} → "
                             f"{STRING_NAMES[int(r.estimated_string)]} | {r.share_of_errors:.3f} |")

    path = os.path.join(out_dir, "SUMMARY.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                        help="An evaluated experiment directory and the label to report it under. Repeatable.")
    parser.add_argument("--reference", action="append", default=[], metavar="LABEL",
                        help="Compare every other run against this one, fold by fold. Repeatable.")
    parser.add_argument("--gcs-run", default=None, metavar="LABEL",
                        help="Run whose GCS curves are plotted; defaults to the first --reference.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        label, sep, path = item.partition("=")
        if not sep or not os.path.isdir(path):
            parser.error(f"--run expects LABEL=DIR with an existing directory, got {item!r}")
        runs[label] = path
    unknown = [r for r in args.reference + ([args.gcs_run] if args.gcs_run else []) if r not in runs]
    if unknown:
        parser.error(f"unknown run label(s): {unknown}")

    print(f"Written to {summarize(runs, args.reference, args.output_dir, args.gcs_run)}")


if __name__ == "__main__":
    main()
