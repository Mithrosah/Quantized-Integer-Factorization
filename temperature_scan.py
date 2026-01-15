import csv
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import load_instances, run_sampler_repeated

# Default grids (same spirit as solve.py)
DEFAULT_C_GRID = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]

# Defaults aligned with your solve.py choices
DEFAULT_N_RUNS = 1000
DEFAULT_SEED = 42
DEFAULT_BURN_IN = 0
DEFAULT_SAMPLE_EVERY = 1
DEFAULT_STEPS_PER_BIT = 2000


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate sigmoid mode success rate for multiple c values on selected instances."
    )
    p.add_argument("--instances", default="./instances.jsonl", help="Path to instances.jsonl")
    p.add_argument(
        "--pick",
        type=str,
        default="first:5",
        help=(
            "How to pick instances. Formats:\n"
            "  first:K     -> pick first K instances\n"
            "  idx:a,b,c   -> pick specific 0-based indices\n"
            "  slice:s:e   -> pick indices [s, e) (0-based)\n"
            "Default: first:5"
        ),
    )
    p.add_argument(
        "--c_grid",
        type=str,
        default="",
        help=(
            "Comma-separated c values, e.g. '1e-2,3e-2,1e-3'. "
            "If empty, uses a built-in default grid."
        ),
    )
    p.add_argument("--n_runs", type=int, default=DEFAULT_N_RUNS, help="Runs per c value")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base seed")
    p.add_argument("--burn_in", type=int, default=DEFAULT_BURN_IN, help="Burn-in steps")
    p.add_argument("--sample_every", type=int, default=DEFAULT_SAMPLE_EVERY, help="Sampling interval")
    p.add_argument(
        "--steps_per_bit",
        type=int,
        default=DEFAULT_STEPS_PER_BIT,
        help="Steps = steps_per_bit * (x_bits + y_bits) + burn_in",
    )
    p.add_argument(
        "--max_workers",
        type=int,
        default=0,
        help="Max worker processes (0 means os.cpu_count())",
    )
    p.add_argument(
        "--out",
        type=str,
        default="./sigmoid_success_rates.csv",
        help="Output CSV path",
    )
    return p.parse_args()


def parse_c_grid(s: str):
    if not s.strip():
        return list(DEFAULT_C_GRID)
    parts = [x.strip() for x in s.split(",") if x.strip()]
    return [float(x) for x in parts]


def pick_instances(all_ins, pick_spec: str):
    spec = pick_spec.strip().lower()
    if spec.startswith("first:"):
        k = int(spec.split(":", 1)[1])
        return all_ins[:k]

    if spec.startswith("idx:"):
        idxs = spec.split(":", 1)[1]
        idx_list = [int(x.strip()) for x in idxs.split(",") if x.strip()]
        return [all_ins[i] for i in idx_list]

    if spec.startswith("slice:"):
        rng = spec.split(":", 1)[1]
        s, e = rng.split(":", 1)
        s = int(s.strip())
        e = int(e.strip())
        return all_ins[s:e]

    raise ValueError(f"Unknown --pick format: {pick_spec}")


def _eval_one_c(*, F, x_bits, y_bits, steps, burn_in, sample_every, n_runs, base_seed, c):
    denom = 2 ** (x_bits + y_bits)
    beta = c / denom
    _, _, success_runs, success_rate, _ = run_sampler_repeated(
        F=F,
        x_bits=x_bits,
        y_bits=y_bits,
        n_runs=n_runs,
        steps=steps,
        burn_in=burn_in,
        sample_every=sample_every,
        base_seed=base_seed,
        mode="sigmoid",
        p_good=0.875,  # ignored in sigmoid mode, but keep signature consistent
        beta=beta,
        collect_samples=False,
    )
    return {
        "F": F,
        "x_bits": x_bits,
        "y_bits": y_bits,
        "steps": steps,
        "burn_in": burn_in,
        "sample_every": sample_every,
        "n_runs": n_runs,
        "seed_base": base_seed,
        "c": c,
        "beta": beta,
        "success_runs": success_runs,
        "success_rate": success_rate,
    }


def main():
    args = parse_args()
    ins_all = load_instances(args.instances)
    chosen = pick_instances(ins_all, args.pick)
    c_grid = parse_c_grid(args.c_grid)

    if not chosen:
        raise ValueError("No instances selected. Check --pick.")

    max_workers = args.max_workers if args.max_workers and args.max_workers > 0 else (os.cpu_count() or 1)

    print(f"Loaded {len(ins_all)} instances, selected {len(chosen)} via --pick={args.pick}")
    print(f"c_grid={c_grid}")
    print(f"n_runs={args.n_runs}, seed={args.seed}, burn_in={args.burn_in}, sample_every={args.sample_every}")
    print(f"steps = {args.steps_per_bit} * (x_bits + y_bits) + burn_in")
    print(f"max_workers={max_workers}")
    print("=" * 100)

    rows = []

    # Parallelize across (instance, c) pairs
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for (F, x_bits, y_bits) in chosen:
            steps = args.steps_per_bit * (x_bits + y_bits) + args.burn_in
            for c in c_grid:
                futures.append(
                    ex.submit(
                        _eval_one_c,
                        F=F,
                        x_bits=x_bits,
                        y_bits=y_bits,
                        steps=steps,
                        burn_in=args.burn_in,
                        sample_every=args.sample_every,
                        n_runs=args.n_runs,
                        base_seed=args.seed,
                        c=c,
                    )
                )

        for fu in as_completed(futures):
            res = fu.result()
            rows.append(res)

    # Sort for readability
    rows.sort(key=lambda r: (r["x_bits"] + r["y_bits"], r["F"], r["c"]))

    # Print summary grouped by instance
    cur = None
    for r in rows:
        key = (r["F"], r["x_bits"], r["y_bits"], r["steps"])
        if key != cur:
            cur = key
            print(f"\nInstance: F={r['F']}, x_bits={r['x_bits']}, y_bits={r['y_bits']} | steps={r['steps']}")
            print("-" * 80)
        print(
            f"c={r['c']:.6E} | beta={r['beta']:.6E} | "
            f"success={r['success_runs']:>4d}/{r['n_runs']:<4d} ({r['success_rate']:.4%})"
        )

    # Save CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 100)
    print(f"Saved: {args.out}")
    print("=" * 100)


if __name__ == "__main__":
    main()


# python temperature_scan.py --pick idx:22,27,35,47,57
