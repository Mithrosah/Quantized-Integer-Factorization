import random
import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =============================
# Helpers
# =============================

def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def decode_from_bits(bit_list):
    val = 0
    for b in bit_list:
        val = (val << 1) | b
    return val

def decode_xy(s: dict, x_bits: int, y_bits: int) -> tuple[int, int]:
    xb = [s[f"x{k}"] for k in range(x_bits - 1, 0, -1)] + [1]
    yb = [s[f"y{k}"] for k in range(y_bits - 1, 0, -1)] + [1]
    X = decode_from_bits(xb)
    Y = decode_from_bits(yb)
    return X, Y

def energy(s: dict, F: int, x_bits: int, y_bits: int) -> int:
    X, Y = decode_xy(s, x_bits, y_bits)
    d = X * Y - F
    return d * d

def local_energy_diff(s: dict, var: str, F: int, x_bits: int, y_bits: int) -> int:
    old = s[var]

    s[var] = 0
    e0 = energy(s, F, x_bits, y_bits)

    s[var] = 1
    e1 = energy(s, F, x_bits, y_bits)

    s[var] = old
    return e0 - e1

def update_bit(
    s: dict,
    var: str,
    F: int,
    x_bits: int,
    y_bits: int,
    mode: str = "rule",
    p_good: float = 0.875,
    beta: float = 0.01
):
    diff = local_energy_diff(s, var, F, x_bits, y_bits)
    r = random.random()

    if mode == "rule":
        if diff > 0:
            s[var] = 1 if r < p_good else 0
        elif diff < 0:
            s[var] = 0 if r < p_good else 1
        else:
            s[var] = 1 if r < 0.5 else 0

    elif mode == "sigmoid":
        p1 = sigmoid(beta * diff)
        s[var] = 1 if r < p1 else 0

    elif mode == "deterministic":
        s[var] = 1 if diff >= 0 else 0

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'rule' or 'sigmoid' or 'deterministic'.")

def run_sampler(
    F: int,
    x_bits: int,
    y_bits: int,
    steps: int = 300000,
    burn_in: int = 30000,
    sample_every: int = 10,
    seed: int = 0,
    mode: str = "rule",
    p_good: float = 0.875,
    beta: float = 0.01
):
    assert x_bits >= 2 and y_bits >= 2, "Need at least 2 bits because LSB is fixed to 1."
    random.seed(seed)

    s = {}
    for k in range(1, x_bits):
        s[f"x{k}"] = random.randint(0, 1)
    for k in range(1, y_bits):
        s[f"y{k}"] = random.randint(0, 1)

    nodes = [f"x{k}" for k in range(1, x_bits)] + [f"y{k}" for k in range(1, y_bits)]

    counts = Counter()
    samples = []

    for t in range(steps):
        v = random.choice(nodes)
        update_bit(s, v, F, x_bits, y_bits, mode=mode, p_good=p_good, beta=beta)

        if t >= burn_in and ((t - burn_in) % sample_every == 0):
            X, Y = decode_xy(s, x_bits, y_bits)
            counts[(X, Y)] += 1
            samples.append((X, Y))

    return counts, samples


# =============================
# run many chains & aggregate (with run-level success)
# =============================
def run_sampler_repeated(
    F: int,
    x_bits: int,
    y_bits: int,
    n_runs: int = 20,
    steps: int = 300000,
    burn_in: int = 30000,
    sample_every: int = 10,
    base_seed: int = 0,
    mode: str = "rule",
    p_good: float = 0.875,
    beta: float = 0.01,
    collect_samples: bool = False,
):
    """
    Repeat sampling n_runs times (independent chains), aggregate counts.

    Run-level success definition:
    - A run is SUCCESS if it ever samples any (X, Y) with X*Y == F at least once.

    Returns:
      aggregated_counts, aggregated_samples (optional), success_runs, success_rate
    """
    aggregated_counts = Counter()
    aggregated_samples = [] if collect_samples else None

    success_runs = 0

    for i in range(n_runs):
        seed_i = base_seed + i
        counts_i, samples_i = run_sampler(
            F=F, x_bits=x_bits, y_bits=y_bits,
            steps=steps, burn_in=burn_in, sample_every=sample_every,
            seed=seed_i,
            mode=mode,
            p_good=p_good,
            beta=beta
        )

        # run-level success: this run ever hits a correct factorization
        run_success = any((X * Y == F) and (c > 0) for (X, Y), c in counts_i.items())
        if run_success:
            success_runs += 1

        aggregated_counts.update(counts_i)
        if collect_samples:
            aggregated_samples.extend(samples_i)

    success_rate = (success_runs / n_runs) if n_runs > 0 else 0.0
    return aggregated_counts, aggregated_samples, success_runs, success_rate


def plot_3d_hist(counts, x_bits, y_bits, title="3D histogram of (X,Y)"):
    X_vals = list(range(1, 2**x_bits, 2))
    Y_vals = list(range(1, 2**y_bits, 2))

    total = sum(counts.values()) if sum(counts.values()) > 0 else 1

    xx, yy = np.meshgrid(np.arange(len(X_vals)), np.arange(len(Y_vals)))
    xpos = xx.ravel()
    ypos = yy.ravel()
    zpos = np.zeros_like(xpos, dtype=float)

    dz = []
    for Y in Y_vals:
        for X in X_vals:
            dz.append(counts.get((X, Y), 0) / total)
    dz = np.array(dz, dtype=float)

    dx = 0.8 * np.ones_like(dz)
    dy = 0.8 * np.ones_like(dz)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

    ax.set_xticks(np.arange(len(X_vals)) + 0.4)
    ax.set_xticklabels([str(v) for v in X_vals], rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(Y_vals)) + 0.4)
    ax.set_yticklabels([str(v) for v in Y_vals])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Probability")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


# =============================
# Main
# =============================
if __name__ == "__main__":

    F = 161
    x_bits = 3
    y_bits = 5

    steps = 50000
    burn_in = 5000
    sample_every = 10
    seed = 42

    update_mode = "rule"
    p_good = 127/128
    beta = 3 / 2**(x_bits + y_bits)

    # number of independent runs
    n_runs = 10

    counts, samples, success_runs, success_rate = run_sampler_repeated(
        F=F, x_bits=x_bits, y_bits=y_bits,
        n_runs=n_runs,
        steps=steps, burn_in=burn_in, sample_every=sample_every,
        base_seed=seed,
        mode=update_mode,
        p_good=p_good,
        beta=beta,
        collect_samples=False,
    )

    total = sum(counts.values()) if sum(counts.values()) > 0 else 1
    print(f"Aggregated over n_runs={n_runs}, total_samples={total}")
    print(f"Run success: {success_runs}/{n_runs} ({success_rate:.4%})")

    for (X, Y), c in counts.most_common(15):
        print(f"X={X}, Y={Y}, X*Y={X*Y}, freq={c/total:.4f}", end="")
        if X * Y == F:
            print(" -- CORRECT!")
        else:
            print()

    plot_3d_hist(
        counts,
        x_bits,
        y_bits,
        title=f"3D histogram (aggregated)"
    )
