"""Draws the paper's result figures, each as PDF (vector, for the paper) and PNG.

    gcs_curves           GCS between each auxiliary loss and the tablature loss over
                         epochs: mean and min-max range over folds (--gcs-run)
    string_confusion     which string a wrong-string note lands on, as a share of all
                         such errors (from summarize_results.py's string_pair_errors.csv)
    component_isolation  mean paired cost of removing each component of the multi-task
                         stage (from summarize_results.py's paired_statistics.csv)
    tab_overlay_new,     reference vs. predicted tablature / multi-pitch for one excerpt
    mp_overlay_new       of one test track (--overlay-run, needs --data-root)

All figures share the IEEE single-column width (3.5 in) and a 7 pt body font, so
they go into the paper at 1:1 scale. Any figure whose inputs are not given is skipped.

Example:
    python make_figures.py --results-dir results --gcs-run outputs/<campaign_control> \\
        --overlay-run outputs/<proposed> --data-root /path/to/enriched --out-dir figures
"""
import argparse
import csv
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

W = 3.5  # IEEE single-column width (in)
HEIGHT = {"gcs_curves": 2.4, "tab_overlay_new": 2.4, "mp_overlay_new": 2.1,
          "string_confusion": 3.0, "component_isolation": 2.6}

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5,
    "lines.linewidth": 1.0, "axes.linewidth": 0.6, "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType: IEEE rejects Type-3 fonts
})

STRINGS = ["E2", "A2", "D3", "G3", "B3", "E4"]
COLORS = {"TP": "tab:green", "SUB": "tab:orange", "FP": "tab:red", "FN": "tab:blue"}
TASK_LABEL = {"multipitch": "Multi-pitch", "hand_position": "Fret region",
              "string_activity": "String activity", "pitch_class": "Pitch class"}
# (run label in summarize_results.py, row label): rows of the component figure, top to bottom
ISOLATION_ROWS = [("E_fusion_no_auxloss", "Auxiliary losses"),
                  ("B_param_matched", "All auxiliary tasks"),
                  ("F_static_weighting", "Uncertainty weighting"),
                  ("C_no_fret_region", "Fret-region estimate"),
                  ("D_no_string_activity", "String activity"),
                  ("A_no_fusion", "Soft fusion pathway")]


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def load_gcs(run_dir):
    curves = {}
    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name, "history.json")
        if not (name.startswith("fold_") and os.path.isfile(path)):
            continue
        with open(path, encoding="utf-8") as f:
            hist = json.load(f)
        for key in (x for x in hist if x.startswith("gcs_tab_")):
            vals = [v for v in hist[key] if v != 0.0]  # 0.0 marks epochs without GCS
            if vals:
                curves.setdefault(key[len("gcs_tab_"):], []).append(np.asarray(vals, float))
    return curves


def fig_gcs(out_dir, curves):
    fig, ax = plt.subplots(figsize=(W, HEIGHT["gcs_curves"]))
    for task in sorted(curves):
        n = min(len(f) for f in curves[task])
        stack = np.stack([f[:n] for f in curves[task]])
        ep = np.arange(1, n + 1)
        line, = ax.plot(ep, stack.mean(0), linewidth=1.2, label=TASK_LABEL.get(task, task))
        ax.fill_between(ep, stack.min(0), stack.max(0), alpha=0.15, color=line.get_color())
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient cosine similarity")
    ax.legend(frameon=False, ncol=2)
    ax.margins(x=0)
    save(fig, out_dir, "gcs_curves")


def load_confusion(results_dir, label):
    """6x6 matrix of wrong-string errors in percent, reference string x estimated string."""
    path = os.path.join(results_dir, "string_pair_errors.csv")
    if not os.path.isfile(path):
        return None
    M = np.zeros((6, 6))
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["config"] == label:
                M[int(r["reference_string"]), int(r["estimated_string"])] = float(r["share_of_errors"]) * 100
    if not M.any():
        return None
    adjacent = sum(M[i, j] for i in range(6) for j in range(6) if abs(i - j) == 1)
    print(f"  (adjacent-string share of errors: {adjacent:.1f}%)")
    return M


def fig_confusion(out_dir, M):
    # The diagonal is a correct assignment, not an error, so it is masked out.
    Mm = np.ma.masked_array(M, mask=np.eye(6, dtype=bool))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.92")
    fig, ax = plt.subplots(figsize=(W, HEIGHT["string_confusion"]))
    im = ax.imshow(Mm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax).set_label("Share of string-assignment errors (%)")
    ax.set_xticks(range(6)); ax.set_xticklabels(STRINGS)
    ax.set_yticks(range(6)); ax.set_yticklabels(STRINGS)
    thr = M.max() / 1.5
    for i, j in itertools.product(range(6), range(6)):
        if i != j:
            ax.text(j, i, f"{M[i, j]:0.1f}", ha="center", va="center", fontsize=5.5,
                    color="white" if M[i, j] <= thr else "black")
    ax.set_ylabel("Reference string")
    ax.set_xlabel("Estimated string")
    save(fig, out_dir, "string_confusion")


def load_isolation(results_dir, control):
    """Mean paired cost (control minus ablation) in F-measure and TDR for each removed component."""
    path = os.path.join(results_dir, "paired_statistics.csv")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        P = {(r["reference"], r["compared"], r["metric"]): float(r["mean_diff"]) for r in csv.DictReader(f)}
    names, f1, tdr = [], [], []
    for label, row_name in ISOLATION_ROWS:
        if (control, label, "tab_f1") in P:
            names.append(row_name)
            f1.append(P[(control, label, "tab_f1")])
            tdr.append(P[(control, label, "tdr")])
    return (names, f1, tdr) if names else None


def fig_isolation(out_dir, names, f1, tdr):
    y = np.arange(len(names))[::-1]
    bh = 0.36
    fig, ax = plt.subplots(figsize=(W, HEIGHT["component_isolation"]))
    ax.barh(y + bh / 2, f1, bh, label="F-measure", color="#4C78A8")
    ax.barh(y - bh / 2, tdr, bh, label="Disambiguation rate", color="#F58518")
    ax.axvline(0.0, color="0.3", linewidth=0.8)
    span = max(max(f1), max(tdr)) - min(min(f1), min(tdr), 0.0)
    for ys, vals in ((y + bh / 2, f1), (y - bh / 2, tdr)):
        for pos, v in zip(ys, vals):
            off = 0.004 * span if v >= 0 else -0.004 * span
            ax.text(v + off, pos, f"{v:+.3f}", va="center", fontsize=5, ha="left" if v >= 0 else "right")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Cost of removing the component")
    ax.set_xlim(min(min(tdr), 0.0) - 0.004, max(f1) * 1.22)
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", handlelength=1.1, borderpad=0.3, labelspacing=0.25)
    save(fig, out_dir, "component_isolation")


def overlay_data(args):
    """Runs the overlay run's fold model on one test track and returns the requested excerpt."""
    import torch
    import yaml
    from evaluate import apply_silence_threshold, full_track_inference
    from src.data_loader import fold_split, prepare_dataset_files
    from src.models import get_model
    from src.utils.config_helpers import process_config

    with open(os.path.join(args.overlay_run, "config.yaml"), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw[raw["dataset"]]["data_root"] = args.data_root
    cfg = process_config(raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    state = torch.load(os.path.join(args.overlay_run, f"fold_{args.fold}", "model_best.pt"), map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))

    paths, groups = prepare_dataset_files(cfg)
    players = sorted(set(groups))
    by_player = {g: [p for p, gg in zip(paths, groups) if gg == g] for g in players}
    _, _, test_paths, test_player, _ = fold_split(players, by_player, args.fold - 1)
    track = [p for p in test_paths if os.path.basename(p).startswith(args.track)]
    if not track:
        raise SystemExit(f"Track {args.track} is not in fold {args.fold}'s test player ({test_player}).")

    threshold = args.threshold
    if threshold is None:  # the threshold evaluate.py selected on the validation player
        for name in ("test_results_revised.json", "test_results.json"):
            path = os.path.join(args.overlay_run, f"fold_{args.fold}", name)
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                threshold = float(data.get("results", data)["best_threshold"])
                break
    if threshold is None:
        raise SystemExit("No stored threshold for this fold; pass --threshold.")

    out = full_track_inference(track[:1], model, cfg, device)
    silence = cfg["instrument"]["silence_class"]
    preds = apply_silence_threshold(out["tab"], threshold, silence).numpy()
    targets = out["tgt"].numpy()
    a, b = args.start, min(args.end, targets.shape[0])
    P, T = preds[a:b], targets[a:b]

    tuning = np.array(cfg["instrument"]["tuning"])
    lo, hi = cfg["instrument"]["min_midi"], cfg["instrument"]["max_midi"]

    def roll(tab):
        r = np.zeros((tab.shape[0], hi - lo + 1), dtype=int)
        for t, s in zip(*np.where(tab != silence)):
            m = int(tuning[s]) + int(tab[t, s]) - lo
            if 0 <= m < r.shape[1]:
                r[t, m] = 1
        return r

    counts = {"TP": ((P == T) & (T != silence)).sum(), "FN": ((P == silence) & (T != silence)).sum(),
              "FP": ((P != silence) & (T == silence)).sum(),
              "SUB": ((P != T) & (P != silence) & (T != silence)).sum()}
    print(f"  overlay: {os.path.basename(track[0])}, frames {a}-{b}, threshold {threshold:.2f}, "
          + ", ".join(f"{k} {int(v)}" for k, v in counts.items()))
    return dict(P=P, T=T, roll_p=roll(P), roll_t=roll(T), lo=lo,
                hop=cfg["data"]["hop_length"] / cfg["data"]["sample_rate"], silence=silence)


def fig_tab_overlay(out_dir, P, T, hop, silence):
    from scipy import ndimage
    P, T = P.T, T.T  # (strings, time)
    fig, ax = plt.subplots(figsize=(W, HEIGHT["tab_overlay_new"]))
    for s in range(T.shape[0]):
        ps, ts = P[s], T[s]
        for kind, mask, src in [("TP", (ps == ts) & (ts != silence), ps),
                                ("SUB", (ps != ts) & (ps != silence) & (ts != silence), ps),
                                ("FP", (ps != silence) & (ts == silence), ps),
                                ("FN", (ps == silence) & (ts != silence), ts)]:
            lab, n_lab = ndimage.label(mask)
            for i in range(1, n_lab + 1):
                idx = np.where(lab == i)[0]
                t0, t1 = idx[0] * hop, (idx[-1] + 1) * hop
                ax.hlines(s, t0, t1, color=COLORS[kind], linewidth=1.6)
                ax.text(t0, s + 0.10, str(int(src[idx[0]])), color=COLORS[kind],
                        ha="left", va="bottom", fontsize=4.5)
    ax.set_yticks(range(T.shape[0]))
    ax.set_yticklabels(STRINGS)
    ax.set_ylim(-0.7, T.shape[0] + 1.4)  # keep the legend clear of the data
    ax.set_xlim(0, T.shape[1] * hop)
    ax.set_title("Tablature")
    ax.set_xlabel("Relative Time (s)")
    ax.set_ylabel("Strings")
    ax.grid(axis="y", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[mpatches.Patch(color=COLORS[k], label=v) for k, v in
                       [("TP", "True Positive"), ("SUB", "Substitution"), ("FP", "False Alarm"), ("FN", "Miss")]],
              loc="upper right", handlelength=1.1, borderpad=0.3, labelspacing=0.25)
    save(fig, out_dir, "tab_overlay_new")


def fig_mp_overlay(out_dir, roll_p, roll_t, hop, lo):
    from scipy import ndimage
    pr, tr = roll_p.T, roll_t.T  # (pitch, time)
    fig, ax = plt.subplots(figsize=(W, HEIGHT["mp_overlay_new"]))
    for m in range(tr.shape[0]):
        for kind, mask in [("TP", (tr[m] == 1) & (pr[m] == 1)),
                           ("FP", (tr[m] == 0) & (pr[m] == 1)),
                           ("FN", (tr[m] == 1) & (pr[m] == 0))]:
            lab, n_lab = ndimage.label(mask)
            for i in range(1, n_lab + 1):
                idx = np.where(lab == i)[0]
                ax.hlines(lo + m, idx[0] * hop, (idx[-1] + 1) * hop, color=COLORS[kind], linewidth=1.6)
    ax.set_xlim(0, tr.shape[1] * hop)
    ax.set_ylim(lo - 1, lo + tr.shape[0])
    ax.set_title("Multipitch")
    ax.set_xlabel("Relative Time (s)")
    ax.set_ylabel("MIDI Pitch")
    ax.legend(handles=[mpatches.Patch(color=COLORS[k], label=v) for k, v in
                       [("TP", "True Positive"), ("FP", "False Alarm"), ("FN", "Miss")]],
              loc="upper right", handlelength=1.1, borderpad=0.3, labelspacing=0.25)
    save(fig, out_dir, "mp_overlay_new")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--results-dir", default=None, help="summarize_results.py output directory.")
    parser.add_argument("--confusion-label", default="proposed",
                        help="Run label (as given to summarize_results.py) for the string confusion figure.")
    parser.add_argument("--control-label", default="campaign_control",
                        help="Run label of the component campaign's control. The ablations must carry "
                             "their config names as labels (A_no_fusion, ..., F_static_weighting).")
    parser.add_argument("--gcs-run", default=None, help="Experiment directory whose GCS curves are drawn.")
    parser.add_argument("--overlay-run", default=None, help="Experiment directory for the overlay figures.")
    parser.add_argument("--data-root", default=None, help="Feature directory, needed for the overlays.")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--track", default="00_SS3-98-C_solo")
    parser.add_argument("--start", type=int, default=985)
    parser.add_argument("--end", type=int, default=1035)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Silence-recovery threshold; defaults to the one stored for --fold.")
    args = parser.parse_args()

    if args.gcs_run:
        curves = load_gcs(args.gcs_run)
        if curves:
            fig_gcs(args.out_dir, curves)
    if args.results_dir:
        M = load_confusion(args.results_dir, args.confusion_label)
        if M is not None:
            fig_confusion(args.out_dir, M)
        iso = load_isolation(args.results_dir, args.control_label)
        if iso is not None:
            fig_isolation(args.out_dir, *iso)
    if args.overlay_run:
        if not args.data_root:
            parser.error("--overlay-run needs --data-root")
        d = overlay_data(args)
        fig_tab_overlay(args.out_dir, d["P"], d["T"], d["hop"], d["silence"])
        fig_mp_overlay(args.out_dir, d["roll_p"], d["roll_t"], d["hop"], d["lo"])


if __name__ == "__main__":
    main()
