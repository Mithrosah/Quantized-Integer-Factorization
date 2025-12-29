import csv
from utils import run_sampler_repeated


# load problem instances
ins = []
with open('./instances.csv', 'r', encoding='utf-8') as f:
    for line in f:
        ins.append([int(x) for x in line.strip().split()])


# sampling configuration (instance-dependent steps)
burn_in = 5000
sample_every = 10
seed = 42
n_runs = 1000


# parameter grids
p_good_grid = [3/4, 7/8, 15/16, 31/32, 63/64, 127/128, 255/256]
c_grid = [100, 10, 1, 1e-1, 1e-2, 1e-3]     # beta = c / 2**(x_bits+y_bits)


def total_samples_from_counts(counts) -> int:
    return sum(counts.values())


rows = []

print("=" * 100)
print("Grid search (run-level success: success_runs / n_runs)")
print(f"burn_in={burn_in}, sample_every={sample_every}, n_runs={n_runs}, seed(base)={seed}")
print("steps = 5000 * (x_bits + y_bits) + burn_in")
print("=" * 100)


for F, x_bits, y_bits in ins:

    # instance-dependent steps
    steps = 5000 * (x_bits + y_bits) + burn_in

    # ---------- RULE mode ----------
    best_rule = {
        "best_param": None,
        "best_success_runs": -1,
        "best_total_samples": 0,
        "best_run_success_rate": 0.0,
    }

    for p_good in p_good_grid:
        counts, _, success_runs, success_rate = run_sampler_repeated(
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

        if success_runs > best_rule["best_success_runs"]:
            best_rule.update(
                best_param=p_good,
                best_success_runs=success_runs,
                best_total_samples=tot_samples,
                best_run_success_rate=success_rate,
            )

    # ---------- SIGMOID mode ----------
    best_sigmoid = {
        "best_param": None,   # (c, beta)
        "best_success_runs": -1,
        "best_total_samples": 0,
        "best_run_success_rate": 0.0,
    }

    denom = 2 ** (x_bits + y_bits)
    for c in c_grid:
        beta = c / denom
        counts, _, success_runs, success_rate = run_sampler_repeated(
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

        if success_runs > best_sigmoid["best_success_runs"]:
            best_sigmoid.update(
                best_param=(c, beta),
                best_success_runs=success_runs,
                best_total_samples=tot_samples,
                best_run_success_rate=success_rate,
            )

    # ---------- DETERMINISTIC mode ----------
    counts, _, det_success_runs, det_success_rate = run_sampler_repeated(
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

    # ---------- print summary ----------
    print(f"\nInstance: F={F}, x_bits={x_bits}, y_bits={y_bits}")
    print(f"steps={steps}")
    print("-" * 100)

    print(
        f"[rule]          "
        f"best p_good = {best_rule['best_param']:>8.6f} | "
        f"success_runs = {best_rule['best_success_runs']:>3d}/{n_runs:<3d} "
        f"({best_rule['best_run_success_rate']:>7.4%}) | "
        f"total_samples={best_rule['best_total_samples']}"
    )

    c_best, beta_best = best_sigmoid["best_param"]
    print(
        f"[sigmoid]       "
        f"best beta = {beta_best:>10.6f} | "
        f"success_runs = {best_sigmoid['best_success_runs']:>3d}/{n_runs:<3d} "
        f"({best_sigmoid['best_run_success_rate']:>7.4%}) | "
        f"total_samples={best_sigmoid['best_total_samples']}"
    )

    print(
        f"[deterministic]                        | "
        f"success_runs = {det_success_runs:>3d}/{n_runs:<3d} "
        f"({det_success_rate:>7.4%}) | "
        f"total_samples={det_total_samples}"
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

        "sigmoid_best_c": c_best,
        "sigmoid_best_beta": beta_best,
        "sigmoid_success_runs": best_sigmoid["best_success_runs"],
        "sigmoid_run_success_rate": best_sigmoid["best_run_success_rate"],
        "sigmoid_total_samples": best_sigmoid["best_total_samples"],

        "deterministic_success_runs": det_success_runs,
        "deterministic_run_success_rate": det_success_rate,
        "deterministic_total_samples": det_total_samples,
    })


# ---------- save CSV ----------
out_path = "./gridsearch_best_results3.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 100)
print(f"Saved: {out_path}")
print("=" * 100)
