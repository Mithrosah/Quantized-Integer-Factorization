import csv
import argparse
from utils import run_sampler_repeated


def parse_args():
    p = argparse.ArgumentParser(description="Grid search for sampler parameters.")
    p.add_argument(
        "--metric",
        choices=["success", "ttfh"],
        default="success",
        help="Metric to optimize: 'success' (maximize run-level success rate) or 'ttfh' (minimize time-to-first-hit; failures treated as steps+1).",
    )
    p.add_argument(
        "--ttfh_stat",
        choices=["median", "mean"],
        default="median",
        help="When metric=ttfh, which statistic to minimize across runs.",
    )
    return p.parse_args()


args = parse_args()


# load problem instances
ins = []
with open('./instances.csv', 'r', encoding='utf-8') as f:
    for line in f:
        ins.append([int(x) for x in line.strip().split()])


# sampling configuration (instance-dependent steps)
burn_in = 0
sample_every = 1
seed = 42
n_runs = 1000


# parameter grids
p_good_grid = [3/4, 7/8, 15/16, 31/32, 63/64, 127/128, 255/256]
c_grid = [100, 10, 1, 1e-1, 1e-2, 1e-3]     # beta = c / 2**(x_bits+y_bits)


def total_samples_from_counts(counts) -> int:
    return sum(counts.values())


def ttfh_stat_from_hits(hit_steps, steps: int, stat: str = "median") -> float:
    """
    Compute a single Time-to-First-Hit (TTFH) statistic from per-run hit steps.
    - hit_steps: list of int or None
    - failures (None) are treated as steps+1 (a penalty just beyond the budget)
    """
    penalized = [(h if h is not None else steps + 1) for h in hit_steps]
    if len(penalized) == 0:
        return float("inf")

    if stat == "mean":
        return sum(penalized) / len(penalized)

    # median
    s = sorted(penalized)
    m = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[m])
    return 0.5 * (s[m - 1] + s[m])


rows = []

print("=" * 100)
print("Grid search")
print(f"opt_metric={args.metric}, ttfh_stat={args.ttfh_stat}")
print(f"burn_in={burn_in}, sample_every={sample_every}, n_runs={n_runs}, seed(base)={seed}")
print("steps = 5000 * (x_bits + y_bits) + burn_in")
print("=" * 100)


for F, x_bits, y_bits in ins:

    # instance-dependent steps
    steps = 1000 * (x_bits + y_bits) + burn_in

    # ---------- RULE mode ----------
    best_rule = {
        "best_param": None,
        "best_success_runs": -1,
        "best_total_samples": 0,
        "best_run_success_rate": 0.0,
        "best_ttfh": float("inf"),
    }

    for p_good in p_good_grid:
        counts, _, success_runs, success_rate, hit_steps = run_sampler_repeated(
            F=F,
            x_bits=x_bits,
            y_bits=y_bits,
            n_runs=n_runs,
            steps=steps,
            burn_in=burn_in,
            sample_every=sample_every,
            base_seed=seed,
            mode="rule",
            p_good=p_good,
            beta=0.01,
        )
        tot_samples = total_samples_from_counts(counts)
        ttfh = ttfh_stat_from_hits(hit_steps, steps=steps, stat=args.ttfh_stat)

        better = False
        if args.metric == "success":
            if success_runs > best_rule["best_success_runs"]:
                better = True
            elif success_runs == best_rule["best_success_runs"] and ttfh < best_rule.get("best_ttfh", float("inf")):
                # tie-breaker: faster first hit
                better = True
        else:  # args.metric == "ttfh"
            if ttfh < best_rule.get("best_ttfh", float("inf")):
                better = True
            elif ttfh == best_rule.get("best_ttfh", float("inf")) and success_runs > best_rule["best_success_runs"]:
                # tie-breaker: higher success
                better = True

        if better:
            best_rule.update(
                best_param=p_good,
                best_success_runs=success_runs,
                best_total_samples=tot_samples,
                best_run_success_rate=success_rate,
                best_ttfh=ttfh,
            )

    # ---------- SIGMOID mode ----------
    best_sigmoid = {
        "best_param": None,   # (c, beta)
        "best_success_runs": -1,
        "best_total_samples": 0,
        "best_run_success_rate": 0.0,
        "best_ttfh": float("inf"),
    }

    denom = 2 ** (x_bits + y_bits)
    for c in c_grid:
        beta = c / denom
        counts, _, success_runs, success_rate, hit_steps = run_sampler_repeated(
            F=F,
            x_bits=x_bits,
            y_bits=y_bits,
            n_runs=n_runs,
            steps=steps,
            burn_in=burn_in,
            sample_every=sample_every,
            base_seed=seed,
            mode="sigmoid",
            p_good=0.875,
            beta=beta,
        )
        tot_samples = total_samples_from_counts(counts)
        ttfh = ttfh_stat_from_hits(hit_steps, steps=steps, stat=args.ttfh_stat)

        better = False
        if args.metric == "success":
            if success_runs > best_sigmoid["best_success_runs"]:
                better = True
            elif success_runs == best_sigmoid["best_success_runs"] and ttfh < best_sigmoid.get("best_ttfh", float("inf")):
                # tie-breaker: faster first hit
                better = True
        else:  # args.metric == "ttfh"
            if ttfh < best_sigmoid.get("best_ttfh", float("inf")):
                better = True
            elif ttfh == best_sigmoid.get("best_ttfh", float("inf")) and success_runs > best_sigmoid["best_success_runs"]:
                # tie-breaker: higher success
                better = True

        if better:
            best_sigmoid.update(
                best_param=(c, beta),
                best_success_runs=success_runs,
                best_total_samples=tot_samples,
                best_run_success_rate=success_rate,
                best_ttfh=ttfh,
            )

    # ---------- DETERMINISTIC mode ----------
    counts, _, det_success_runs, det_success_rate, det_hit_steps = run_sampler_repeated(
        F=F,
        x_bits=x_bits,
        y_bits=y_bits,
        n_runs=n_runs,
        steps=steps,
        burn_in=burn_in,
        sample_every=sample_every,
        base_seed=seed,
        mode="deterministic",
        p_good=0.875,
        beta=0.01,
    )
    det_total_samples = total_samples_from_counts(counts)
    det_ttfh = ttfh_stat_from_hits(det_hit_steps, steps=steps, stat=args.ttfh_stat)

    # ---------- print summary ----------
    print(f"\nInstance: F={F}, x_bits={x_bits}, y_bits={y_bits}")
    print(f"steps={steps}")
    print("-" * 100)

    print(
        f"[rule]          "
        f"best p_good = {best_rule['best_param']:>8.6f} | "
        f"success_runs = {best_rule['best_success_runs']:>3d}/{n_runs:<3d} "
        f"({best_rule['best_run_success_rate']:>7.4%}) | "
        f"total_samples={best_rule['best_total_samples']} | "
        f"ttfh({args.ttfh_stat})={best_rule['best_ttfh']:.1f}"
    )

    c_best, beta_best = best_sigmoid["best_param"]
    print(
        f"[sigmoid]       "
        f"best beta = {beta_best:>10.6f} | "
        f"success_runs = {best_sigmoid['best_success_runs']:>3d}/{n_runs:<3d} "
        f"({best_sigmoid['best_run_success_rate']:>7.4%}) | "
        f"total_samples={best_sigmoid['best_total_samples']} | "
        f"ttfh({args.ttfh_stat})={best_sigmoid['best_ttfh']:.1f}"
    )

    print(
        f"[deterministic]                        | "
        f"success_runs = {det_success_runs:>3d}/{n_runs:<3d} "
        f"({det_success_rate:>7.4%}) | "
        f"total_samples={det_total_samples} | "
        f"ttfh({args.ttfh_stat})={det_ttfh:.1f}"
    )

    rows.append({
        "F": F,
        "x_bits": x_bits,
        "y_bits": y_bits,
        "steps": steps,
        "burn_in": burn_in,
        "sample_every": sample_every,
        "n_runs": n_runs,
        "seed_base": seed,

        "rule_best_p_good": best_rule["best_param"],
        "rule_success_runs": best_rule["best_success_runs"],
        "rule_run_success_rate": best_rule["best_run_success_rate"],
        "rule_total_samples": best_rule["best_total_samples"],
        "rule_best_ttfh": best_rule["best_ttfh"],

        "sigmoid_best_c": c_best,
        "sigmoid_best_beta": beta_best,
        "sigmoid_success_runs": best_sigmoid["best_success_runs"],
        "sigmoid_run_success_rate": best_sigmoid["best_run_success_rate"],
        "sigmoid_total_samples": best_sigmoid["best_total_samples"],
        "sigmoid_best_ttfh": best_sigmoid["best_ttfh"],

        "deterministic_success_runs": det_success_runs,
        "deterministic_run_success_rate": det_success_rate,
        "deterministic_total_samples": det_total_samples,
        "deterministic_ttfh": det_ttfh,
    })


# ---------- save CSV ----------
out_path = f"./gridsearch_best_results3_{args.metric}.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 100)
print(f"Saved: {out_path}")
print("=" * 100)
