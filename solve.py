import csv
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import run_sampler_repeated


def parse_args():
    p = argparse.ArgumentParser(description="Grid search for sampler parameters (parallelized).")
    p.add_argument(
        "--metric",
        choices=["success", "ttfh"],
        default="success",
        help=(
            "Metric to optimize: "
            "'success' = maximize number of successful runs; tie-breaker = smaller TTFH. "
            "'ttfh' = minimize time-to-first-hit (failures treated as steps+1); tie-breaker = more successes."
        ),
    )
    p.add_argument(
        "--ttfh_stat",
        choices=["median", "mean"],
        default="median",
        help="Statistic for time-to-first-hit across runs (median or mean).",
    )
    p.add_argument(
        "--instances",
        default="./instances.csv",
        help="Path to instances.csv (default: ./instances.csv).",
    )
    p.add_argument(
        "--max_workers",
        type=int,
        default=0,
        help="Max worker processes (default: 0 means os.cpu_count()).",
    )
    return p.parse_args()


# sampling configuration (instance-dependent steps)
burn_in = 0
sample_every = 1
seed = 42
n_runs = 1000

# parameter grids
p_good_grid = [3/4, 7/8, 15/16, 31/32, 63/64, 127/128]
c_grid = [1e-1, 1e-2, 3e-2, 1e-3, 3e-3, 1e-4, 3e-4, 1e-5]     # beta = c / 2**(x_bits+y_bits)


def total_samples_from_counts(counts) -> int:
    return sum(counts.values())


def ttfh_stat_from_hits(hit_steps, steps: int, stat: str = "median") -> float:
    """
    Compute a single Time-to-First-Hit (TTFH) statistic from per-run hit steps.
    - hit_steps: list of int or None
    - failures (None) are treated as steps+1 (a penalty just beyond the budget)
    """
    vals = [(h if h is not None else steps + 1) for h in hit_steps]
    if not vals:
        return float(steps + 1)

    if stat == "mean":
        return sum(vals) / len(vals)

    # median
    s = sorted(vals)
    m = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[m])
    return 0.5 * (s[m - 1] + s[m])


def _pick_better(a: dict, b: dict, metric: str) -> dict:
    """Return the better config result dict given metric and tie-breakers."""
    if b is None:
        return a

    if metric == "success":
        if a["success_runs"] != b["success_runs"]:
            return a if a["success_runs"] > b["success_runs"] else b
        # tie-breaker: smaller TTFH
        if a["ttfh"] != b["ttfh"]:
            return a if a["ttfh"] < b["ttfh"] else b
        return a  # identical
    else:  # metric == "ttfh"
        if a["ttfh"] != b["ttfh"]:
            return a if a["ttfh"] < b["ttfh"] else b
        # tie-breaker: more successes
        if a["success_runs"] != b["success_runs"]:
            return a if a["success_runs"] > b["success_runs"] else b
        return a


def _eval_one_config(
    *,
    F: int,
    x_bits: int,
    y_bits: int,
    steps: int,
    burn_in: int,
    sample_every: int,
    n_runs: int,
    base_seed: int,
    ttfh_stat: str,
    mode: str,
    p_good: float,
    c: float,
    beta: float,
):
    """
    Worker function (must be top-level for multiprocessing pickling).
    Returns a small dict with metrics for one parameter setting.
    """
    counts, _, success_runs, success_rate, hit_steps = run_sampler_repeated(
        F=F,
        x_bits=x_bits,
        y_bits=y_bits,
        n_runs=n_runs,
        steps=steps,
        burn_in=burn_in,
        sample_every=sample_every,
        base_seed=base_seed,
        mode=mode,
        p_good=p_good,
        beta=beta,
    )
    tot_samples = total_samples_from_counts(counts)
    ttfh = ttfh_stat_from_hits(hit_steps, steps=steps, stat=ttfh_stat)

    return {
        "mode": mode,
        "p_good": p_good,
        "c": c,
        "beta": beta,
        "success_runs": success_runs,
        "success_rate": success_rate,
        "total_samples": tot_samples,
        "ttfh": ttfh,
    }


def main():
    args = parse_args()

    # load problem instances
    ins = []
    with open(args.instances, "r", encoding="utf-8") as f:
        for line in f:
            ins.append([int(x.strip('"')) for x in line.split()])

    # keep the same slicing behavior as original script
    ins = ins[0:2]

    rows = []

    max_workers = args.max_workers if args.max_workers and args.max_workers > 0 else (os.cpu_count() or 1)

    print("=" * 100)
    print("Grid search (parallel)")
    print(f"opt_metric={args.metric}, ttfh_stat={args.ttfh_stat}")
    print(f"burn_in={burn_in}, sample_every={sample_every}, n_runs={n_runs}, seed(base)={seed}")
    print("steps = 2000 * (x_bits + y_bits) + burn_in")
    print(f"max_workers={max_workers}")
    print("=" * 100)

    # Reuse ONE process pool for all instances (less overhead).
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for F, x_bits, y_bits in ins:
            steps = 2000 * (x_bits + y_bits) + burn_in

            # ---------- RULE mode ----------
            best_rule = None
            rule_futures = []
            for p_good in p_good_grid:
                rule_futures.append(
                    ex.submit(
                        _eval_one_config,
                        F=F,
                        x_bits=x_bits,
                        y_bits=y_bits,
                        steps=steps,
                        burn_in=burn_in,
                        sample_every=sample_every,
                        n_runs=n_runs,
                        base_seed=seed,
                        ttfh_stat=args.ttfh_stat,
                        mode="rule",
                        p_good=p_good,
                        c=1,
                        beta=0.01,
                    )
                )

            for fu in as_completed(rule_futures):
                res = fu.result()
                best_rule = _pick_better(res, best_rule, args.metric)

            # ---------- SIGMOID mode ----------
            best_sigmoid = None
            denom = 2 ** (x_bits + y_bits)
            sigmoid_futures = []
            for c in c_grid:
                beta = c / denom
                sigmoid_futures.append(
                    ex.submit(
                        _eval_one_config,
                        F=F,
                        x_bits=x_bits,
                        y_bits=y_bits,
                        steps=steps,
                        burn_in=burn_in,
                        sample_every=sample_every,
                        n_runs=n_runs,
                        base_seed=seed,
                        ttfh_stat=args.ttfh_stat,
                        mode="sigmoid",
                        p_good=0.875,
                        c=c,
                        beta=beta,
                    )
                )

            for fu in as_completed(sigmoid_futures):
                res = fu.result()
                best_sigmoid = _pick_better(res, best_sigmoid, args.metric)

            # ---------- deterministic mode ----------
            det_counts, _, det_success_runs, det_success_rate, det_hit_steps = run_sampler_repeated(
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
            det_total_samples = total_samples_from_counts(det_counts)
            det_ttfh = ttfh_stat_from_hits(det_hit_steps, steps=steps, stat=args.ttfh_stat)

            # ---------- print summary ----------
            print(f"\nInstance: F={F}, x_bits={x_bits}, y_bits={y_bits}")
            print(f"steps={steps}")
            print("-" * 100)

            print(
                f"[rule]          "
                f"best p_good = {best_rule['p_good']:.6E} | "
                f"success_runs = {best_rule['success_runs']:>4d}/{n_runs:<4d} "
                f"({best_rule['success_rate']:>7.4%}) | "
                f"total_samples={best_rule['total_samples']} | "
                f"ttfh({args.ttfh_stat})={best_rule['ttfh']:.1f}"
            )

            print(
                f"[sigmoid]       "
                f"best c      = {best_sigmoid['c']:.6E} | "
                f"success_runs = {best_sigmoid['success_runs']:>4d}/{n_runs:<4d} "
                f"({best_sigmoid['success_rate']:>7.4%}) | "
                f"total_samples={best_sigmoid['total_samples']} | "
                f"ttfh({args.ttfh_stat})={best_sigmoid['ttfh']:.1f}"
            )

            print(
                f"[deterministic]                            | "
                f"success_runs = {det_success_runs:>4d}/{n_runs:<4d} "
                f"({det_success_rate:>7.4%}) | "
                f"total_samples={det_total_samples} | "
                f"ttfh({args.ttfh_stat})={det_ttfh:.1f}"
            )

            rows.append(
                {
                    "F": F,
                    "x_bits": x_bits,
                    "y_bits": y_bits,
                    "steps": steps,
                    "burn_in": burn_in,
                    "sample_every": sample_every,
                    "n_runs": n_runs,
                    "seed_base": seed,

                    "rule_best_p_good": best_rule["p_good"],
                    "rule_success_runs": best_rule["success_runs"],
                    "rule_run_success_rate": best_rule["success_rate"],
                    "rule_total_samples": best_rule["total_samples"],
                    "rule_best_ttfh": best_rule["ttfh"],

                    "sigmoid_best_c": best_sigmoid["c"],
                    "sigmoid_best_beta": best_sigmoid["beta"],
                    "sigmoid_success_runs": best_sigmoid["success_runs"],
                    "sigmoid_run_success_rate": best_sigmoid["success_rate"],
                    "sigmoid_total_samples": best_sigmoid["total_samples"],
                    "sigmoid_best_ttfh": best_sigmoid["ttfh"],

                    "deterministic_success_runs": det_success_runs,
                    "deterministic_run_success_rate": det_success_rate,
                    "deterministic_total_samples": det_total_samples,
                    "deterministic_ttfh": det_ttfh,
                }
            )

    # ---------- save CSV ----------
    out_path = f"./gridsearch_best_results3_{args.metric}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 100)
    print(f"Saved: {out_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()


# python solve.py --metric ttfh --ttfh_stat mean --max_workers 8