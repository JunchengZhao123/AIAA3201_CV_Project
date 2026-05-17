"""
Part 1: Convergence + Quality Analysis
=======================================
Reads 3DGS (and optionally Scaffold-GS) experiment outputs and generates:
  1. Summary metrics table (PSNR / SSIM / LPIPS) for Plan A vs Plan B
  2. Convergence curves extracted from results.json (per test_iteration)
  3. Per-scene breakdown
  4. Markdown report + matplotlib plots

Usage:
    python scripts/analyze_part1.py --exp_dir ~/running_result/5_1/5_1_2_3DGS_Optimization/experiments
    python scripts/analyze_part1.py --exp_dir ~/running_result/5_1/5_1_2_3DGS_Optimization/experiments --out_dir ~/running_result/5_1/5_1_2_3DGS_Optimization/analysis
"""

import os, json, re, argparse, sys
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found; plots will be skipped.")


def parse_results_json(path):
    """Parse results.json -> {iteration: {metric: value}}"""
    with open(path) as f:
        data = json.load(f)
    parsed = {}
    for key, metrics in data.items():
        m = re.match(r"ours_(\d+)", key)
        if m:
            parsed[int(m.group(1))] = metrics
    return parsed


def parse_train_log_convergence(path):
    """Extract per-iteration test PSNR from train.log lines like:
    [ITER 7000] Evaluating test: L1 0.0576 PSNR 25.00"""
    convergence = []
    pattern = re.compile(
        r"\[ITER\s+(\d+)\]\s+Evaluating\s+test:\s+L1\s+([\d.]+)\s+PSNR\s+([\d.]+)"
    )
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    convergence.append({
                        "iteration": int(m.group(1)),
                        "l1": float(m.group(2)),
                        "psnr": float(m.group(3)),
                    })
    except FileNotFoundError:
        pass
    return convergence


def parse_train_log_loss(path):
    """Extract EMA training loss from tqdm output in train.log.
    Lines contain: Loss: 0.0123456"""
    losses = []
    pattern = re.compile(r"(\d+)%.*Loss:\s*([\d.]+)")
    iter_pattern = re.compile(r"(\d+)/(\d+)")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    losses.append(float(m.group(2)))
    except FileNotFoundError:
        pass
    return losses


def parse_cfg_args(path):
    """Read cfg_args file -> dict with source_path, model_path, iterations, etc."""
    try:
        with open(path) as f:
            text = f.read().strip()
        ns = eval(text)
        return vars(ns) if hasattr(ns, '__dict__') else {}
    except Exception:
        return {}


def discover_experiments(exp_dir, scenes, models=("3dgs", "scaffold")):
    """Find all experiment folders matching plan{A,B}_{scene}_{model}_iter*"""
    experiments = []
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        print(f"ERROR: {exp_dir} does not exist")
        sys.exit(1)

    for folder in sorted(exp_path.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        for plan in ("planA", "planB"):
            for scene in scenes:
                for model in models:
                    prefix = f"{plan}_{scene}_{model}_iter"
                    if name.startswith(prefix):
                        iters_str = name[len(prefix):]
                        try:
                            iterations = int(iters_str)
                        except ValueError:
                            continue
                        experiments.append({
                            "path": folder,
                            "name": name,
                            "plan": plan,
                            "scene": scene,
                            "model": model,
                            "iterations": iterations,
                        })
    return experiments


def build_metrics_table(experiments):
    """Build a list of dicts with final metrics for each experiment."""
    rows = []
    for exp in experiments:
        results_path = exp["path"] / "results.json"
        if not results_path.exists():
            continue
        results = parse_results_json(results_path)
        if not results:
            continue
        final_iter = max(results.keys())
        final = results[final_iter]
        rows.append({
            "plan": exp["plan"],
            "scene": exp["scene"],
            "model": exp["model"],
            "iterations": exp["iterations"],
            "final_iter": final_iter,
            "PSNR": final.get("PSNR", 0),
            "SSIM": final.get("SSIM", 0),
            "LPIPS": final.get("LPIPS", 0),
        })
    return rows


def build_convergence_data(experiments):
    """Build convergence curves from results.json (multi-checkpoint) and train.log."""
    convergence = {}
    for exp in experiments:
        key = f"{exp['plan']}_{exp['scene']}_{exp['model']}"

        results_path = exp["path"] / "results.json"
        if results_path.exists():
            results = parse_results_json(results_path)
            iters_sorted = sorted(results.keys())
            convergence[key] = {
                "iterations": iters_sorted,
                "psnr": [results[i].get("PSNR", 0) for i in iters_sorted],
                "ssim": [results[i].get("SSIM", 0) for i in iters_sorted],
                "lpips": [results[i].get("LPIPS", 0) for i in iters_sorted],
            }

        train_log = exp["path"] / "train.log"
        log_data = parse_train_log_convergence(train_log)
        if log_data:
            convergence[key + "_log"] = {
                "iterations": [d["iteration"] for d in log_data],
                "psnr": [d["psnr"] for d in log_data],
                "l1": [d["l1"] for d in log_data],
            }
    return convergence


def format_markdown_table(rows):
    """Format metrics rows into a markdown table."""
    if not rows:
        return "*No results found.*\n"

    lines = []
    lines.append("| Plan | Scene | Model | Iter | PSNR ↑ | SSIM ↑ | LPIPS ↓ |")
    lines.append("|------|-------|-------|------|--------|--------|---------|")
    for r in sorted(rows, key=lambda x: (x["scene"], x["plan"], x["model"])):
        lines.append(
            f"| {r['plan']} | {r['scene']} | {r['model']} | {r['final_iter']} "
            f"| {r['PSNR']:.2f} | {r['SSIM']:.4f} | {r['LPIPS']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def format_comparison_table(rows):
    """Create Plan A vs Plan B delta table per scene."""
    by_scene_model = defaultdict(dict)
    for r in rows:
        key = (r["scene"], r["model"])
        by_scene_model[key][r["plan"]] = r

    lines = []
    lines.append("| Scene | Model | PSNR(A) | PSNR(B) | ΔPSNR | SSIM(A) | SSIM(B) | ΔSSIM | LPIPS(A) | LPIPS(B) | ΔLPIPS |")
    lines.append("|-------|-------|---------|---------|-------|---------|---------|-------|----------|----------|--------|")

    for (scene, model) in sorted(by_scene_model.keys()):
        d = by_scene_model[(scene, model)]
        if "planA" not in d or "planB" not in d:
            continue
        a, b = d["planA"], d["planB"]
        dp = b["PSNR"] - a["PSNR"]
        ds = b["SSIM"] - a["SSIM"]
        dl = b["LPIPS"] - a["LPIPS"]
        lines.append(
            f"| {scene} | {model} "
            f"| {a['PSNR']:.2f} | {b['PSNR']:.2f} | {dp:+.2f} "
            f"| {a['SSIM']:.4f} | {b['SSIM']:.4f} | {ds:+.4f} "
            f"| {a['LPIPS']:.4f} | {b['LPIPS']:.4f} | {dl:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def get_psnr_curve(convergence, key):
    """Prefer train.log samples for convergence; fall back to results.json checkpoints."""
    log_key = key + "_log"
    if log_key in convergence and convergence[log_key].get("iterations"):
        data = convergence[log_key]
        return data["iterations"], data["psnr"], "train.log"
    if key in convergence and convergence[key].get("iterations"):
        data = convergence[key]
        return data["iterations"], data["psnr"], "results.json"
    return [], [], None


def plot_convergence(convergence, out_dir, scenes):
    """Generate PSNR convergence plots for Plan A vs Plan B per scene."""
    if not HAS_MPL:
        return []

    plots = []
    for scene in scenes:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        fig.suptitle(f"Convergence: {scene}", fontsize=14)

        plotted_sources = set()
        for model in ("3dgs", "scaffold"):
            for plan, color, ls in [("planA", "#2196F3", "-"), ("planB", "#FF5722", "--")]:
                key = f"{plan}_{scene}_{model}"
                iterations, psnr, source = get_psnr_curve(convergence, key)
                if not iterations:
                    continue

                plotted_sources.add(source)
                ax.plot(iterations, psnr, color=color, linestyle=ls,
                        marker="o", markersize=4, label=f"{plan} ({model})")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Test PSNR ↑")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if plotted_sources:
            source_note = ", ".join(sorted(plotted_sources))
            ax.text(0.01, 0.01, f"Source: {source_note}", transform=ax.transAxes,
                    fontsize=8, va="bottom", ha="left", alpha=0.7)

        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"convergence_{scene}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(plot_path)
        print(f"  Saved: {plot_path}")

    # Combined summary: all scenes side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("All Scenes: Final Metrics Comparison", fontsize=14)

    all_rows = []
    for scene in scenes:
        for model in ("3dgs", "scaffold"):
            for plan in ("planA", "planB"):
                key = f"{plan}_{scene}_{model}"
                if key in convergence and convergence[key]["iterations"]:
                    final_idx = -1
                    all_rows.append({
                        "label": f"{scene}\n{plan}\n{model}",
                        "psnr": convergence[key]["psnr"][final_idx],
                        "ssim": convergence[key]["ssim"][final_idx],
                        "lpips": convergence[key]["lpips"][final_idx],
                        "plan": plan,
                    })

    if all_rows:
        x = range(len(all_rows))
        colors = ["#2196F3" if r["plan"] == "planA" else "#FF5722" for r in all_rows]
        labels = [r["label"] for r in all_rows]

        axes[0].bar(x, [r["psnr"] for r in all_rows], color=colors)
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        axes[0].set_ylabel("PSNR ↑")

        axes[1].bar(x, [r["ssim"] for r in all_rows], color=colors)
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        axes[1].set_ylabel("SSIM ↑")

        axes[2].bar(x, [r["lpips"] for r in all_rows], color=colors)
        axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        axes[2].set_ylabel("LPIPS ↓")

        for ax in axes:
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "summary_comparison.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(plot_path)
    print(f"  Saved: {plot_path}")

    return plots


def generate_report(rows, convergence, plots, scenes, out_path):
    """Generate the full markdown analysis report."""
    lines = []
    lines.append("# Part 1: Convergence + Quality Analysis")
    lines.append("")
    lines.append("## 1. Final Metrics (Test Set)")
    lines.append("")
    lines.append(format_markdown_table(rows))
    lines.append("")
    lines.append("## 2. Plan A vs Plan B Comparison")
    lines.append("")
    lines.append(format_comparison_table(rows))
    lines.append("")
    lines.append("**Interpretation:**")
    lines.append("- ΔPSNR > 0 means Plan B (VGGT) outperforms Plan A (COLMAP)")
    lines.append("- ΔLPIPS < 0 means Plan B has better perceptual quality")
    lines.append("")

    # Per-scene convergence data
    lines.append("## 3. Convergence Data")
    lines.append("")
    for scene in scenes:
        lines.append(f"### {scene}")
        lines.append("")
        has_data = False
        for model in ("3dgs", "scaffold"):
            for plan in ("planA", "planB"):
                key = f"{plan}_{scene}_{model}"
                if key in convergence and convergence[key]["iterations"]:
                    has_data = True
                    data = convergence[key]
                    lines.append(f"**{plan} + {model}:**")
                    lines.append("")
                    lines.append("| Iteration | PSNR | SSIM | LPIPS |")
                    lines.append("|-----------|------|------|-------|")
                    for i, it in enumerate(data["iterations"]):
                        p = data["psnr"][i]
                        s = data["ssim"][i]
                        l = data["lpips"][i]
                        lines.append(f"| {it} | {p:.2f} | {s:.4f} | {l:.4f} |")
                    lines.append("")
        if not has_data:
            lines.append("*No convergence data available.*")
            lines.append("")

    # Plots
    if plots:
        lines.append("## 4. Plots")
        lines.append("")
        for p in plots:
            fname = os.path.basename(p)
            lines.append(f"![{fname}]({fname})")
            lines.append("")

    # Key observations
    lines.append("## 5. Key Observations")
    lines.append("")

    # Auto-generate observations from data
    by_scene = defaultdict(dict)
    for r in rows:
        by_scene[(r["scene"], r["model"])][r["plan"]] = r

    for (scene, model), plans in sorted(by_scene.items()):
        if "planA" in plans and "planB" in plans:
            a, b = plans["planA"], plans["planB"]
            dp = b["PSNR"] - a["PSNR"]
            winner = "Plan B (VGGT)" if dp > 0 else "Plan A (COLMAP)"
            lines.append(
                f"- **{scene} ({model}):** {winner} leads by {abs(dp):.2f} dB PSNR. "
                f"SSIM: A={a['SSIM']:.4f}, B={b['SSIM']:.4f}. "
                f"LPIPS: A={a['LPIPS']:.4f}, B={b['LPIPS']:.4f}."
            )

    lines.append("")
    lines.append("## 6. Discussion")
    lines.append("")
    lines.append("- **COLMAP (Plan A)** benefits from bundle-adjusted poses and geometrically precise "
                 "sparse triangulation, typically yielding the most accurate camera parameters.")
    lines.append("- **VGGT (Plan B)** provides zero-shot pose and depth predictions without iterative "
                 "optimization. Pose quality depends on scene complexity and image overlap.")
    lines.append("- Convergence speed (how quickly PSNR rises in early iterations) indicates "
                 "initialization quality — faster convergence suggests better initial geometry.")
    lines.append("- The gap between Plan A and Plan B at final iteration vs. early iterations "
                 "reveals whether 3DGS optimization can compensate for less accurate initialization.")
    lines.append("")

    report = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to: {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Part 1 Convergence + Quality Analysis")
    parser.add_argument("--exp_dir", required=True,
                        help="Path to experiments/ directory")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for report + plots (default: <exp_dir>/../analysis)")
    parser.add_argument("--scenes", nargs="+", default=["405841", "DL3DV-2", "Re10k-1"],
                        help="Scenes to analyze (default: mandatory 3)")
    parser.add_argument("--models", nargs="+", default=["3dgs", "scaffold"],
                        help="Models to analyze (default: 3dgs scaffold)")
    args = parser.parse_args()

    out_dir = args.out_dir or str(Path(args.exp_dir).parent / "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Experiment dir: {args.exp_dir}")
    print(f"Output dir:     {out_dir}")
    print(f"Scenes:         {args.scenes}")
    print(f"Models:         {args.models}")
    print()

    # Discover experiments
    experiments = discover_experiments(args.exp_dir, args.scenes, args.models)
    print(f"Found {len(experiments)} experiment(s):")
    for e in experiments:
        status = "OK" if (e["path"] / "results.json").exists() else "MISSING results.json"
        print(f"  {e['name']} {status}")
    print()

    if not experiments:
        print("No experiments found. Check --exp_dir and --scenes.")
        sys.exit(1)

    # Build metrics
    rows = build_metrics_table(experiments)
    print(f"Metrics collected for {len(rows)} experiment(s)\n")

    # Build convergence
    convergence = build_convergence_data(experiments)

    # Generate plots
    plots = []
    if HAS_MPL:
        print("Generating plots...")
        plots = plot_convergence(convergence, out_dir, args.scenes)
    print()

    # Save raw data
    raw_path = os.path.join(out_dir, "metrics_raw.json")
    with open(raw_path, "w") as f:
        json.dump({"final_metrics": rows, "convergence": {
            k: {kk: [float(x) for x in vv] if isinstance(vv, list) else vv
                for kk, vv in v.items()}
            for k, v in convergence.items()
        }}, f, indent=2)
    print(f"Raw data saved to: {raw_path}")

    # Generate report
    report_path = os.path.join(out_dir, "part1_analysis.md")
    generate_report(rows, convergence, plots, args.scenes, report_path)


if __name__ == "__main__":
    main()
