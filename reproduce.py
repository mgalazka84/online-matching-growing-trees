#!/usr/bin/env python3
"""Reproduce the finite-horizon PA matching calculations and figure.

The model has attachment weights 1 + outdegree; the total at time n is
2*n - 1. Time n means n vertices are present, before arrival n+1. Edges
may only be accepted at arrival. The horizon N counts vertices.

Dependencies: Python >= 3.9, numpy, matplotlib, mpmath.
Run: python reproduce.py [--horizons 100 500 ...] [--threshold-horizon 10000]
"""

import argparse
import csv
from fractions import Fraction as F
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


def affine_dp(horizon, keep_thresholds=False):
    """O(N^2) arithmetic, O(N) memory; float64 evaluation of exact recurrence.

    beta[k] is beta_(n+1,k) before an iteration and beta_(n,k) after it.
    Unused entries need not be cleared: the next iteration reads a prefix.
    On an exact tie, accept. Floating-point decisions are not certified.
    """
    beta = np.zeros(horizon + 1, dtype=np.float64)
    workspace = np.empty(horizon, dtype=np.float64)
    weights = np.arange(1, horizon + 1, dtype=np.float64)
    thresholds = np.zeros(horizon, dtype=np.int64) if keep_thresholds else None
    constant = 0.0
    for n in range(horizon - 1, 0, -1):
        cutoff = 1.0 - beta[1]
        if thresholds is not None:
            # Sorted beta_(n+1,k+1), k=1,...,n, by the monotonicity theorem.
            thresholds[n] = np.searchsorted(beta[2:n + 2], cutoff, side="right")
        constant += beta[1]
        row = workspace[:n]
        np.maximum(cutoff, beta[2:n + 2], out=row)
        row -= beta[1:n + 1]
        row *= weights[:n]
        row /= 2 * n - 1
        row += beta[1:n + 1]
        beta[1:n + 1] = row
    return constant + beta[1] if horizon > 1 else 0.0, thresholds


def rational_affine_dp(horizon):
    """Small-horizon recurrence evaluated with exact rational arithmetic."""
    beta = [F(0)] * (horizon + 1)
    constant = F(0)
    for n in range(horizon - 1, 0, -1):
        constant += beta[1]
        beta = [F(0)] + [
            beta[k] + F(k, 2 * n - 1)
            * (max(1 - beta[1], beta[k + 1]) - beta[k])
            for k in range(1, n + 1)
        ]
    return constant + beta[1] if horizon > 1 else F(0)


def exhaustive_value(horizon, greedy=False):
    """Independent exact check using labeled vertex weights and a matched mask.

    Enumerates parent choices and both legal actions, without assuming an
    affine value function, a threshold policy, or unmatched-count sufficiency.
    Only tiny horizons are practical. The matched bitmask is updated directly.
    """
    @lru_cache(None)
    def value(weights, matched):
        n = len(weights)
        if n == horizon:
            return F(0)
        result = F(0)
        for parent, weight in enumerate(weights):
            updated = list(weights) + [1]
            updated[parent] += 1
            updated = tuple(updated)
            if matched & (1 << parent):
                reward = value(updated, matched)
            else:
                accepted = 1 + value(updated, matched | (1 << parent) | (1 << n))
                reward = accepted if greedy else max(accepted, value(updated, matched))
            result += F(weight, 2 * n - 1) * reward
        return result
    return value((1,), 0)


def greedy_expectation(horizon):
    """Closed exact formula for alpha=1, including the boundary horizons."""
    if horizon == 1:
        return F(0)
    if horizon == 2:
        return F(1)
    # The product in the general formula telescopes to 3/(2*N-3).
    return F(horizon, 4) + F(1, 8) * (1 + F(3, 2 * horizon - 3))


def deferred_value(horizon, rational=False, a=F(3, 20), b=F(9, 20)):
    """Fixed-policy backward evaluation, with the manuscript's floor cutoffs.

    Replace the Bellman maximum by the action prescribed by DG(a,b).
    The returned value is an expectation, not a sampled matching size.
    """
    first, second = int(a * horizon), int(b * horizon)
    if rational:
        beta, constant = [F(0)] * (horizon + 1), F(0)
        for n in range(horizon - 1, 0, -1):
            constant += beta[1]
            beta = [F(0)] + [
                beta[k] + F(k, 2*n-1) * (
                    (1-beta[1] if n >= second or (n >= first and k == 1)
                     else beta[k+1]) - beta[k])
                for k in range(1, n+1)]
        return constant + beta[1] if horizon > 1 else F(0)
    beta = np.zeros(horizon+1, dtype=np.float64)
    workspace = np.empty(horizon, dtype=np.float64)
    weights = np.arange(1, horizon+1, dtype=np.float64)
    constant = 0.0
    for n in range(horizon-1, 0, -1):
        cutoff = 1.0-beta[1]
        constant += beta[1]
        row = workspace[:n]
        if n >= second:
            row.fill(cutoff)
        else:
            row[:] = beta[2:n+2]
            if n >= first:
                row[0] = cutoff
        row -= beta[1:n+1]
        row *= weights[:n]
        row /= 2*n-1
        row += beta[1:n+1]
        beta[1:n+1] = row
    return constant + beta[1] if horizon > 1 else 0.0


def deferred_forward_exact(horizon, a=F(3, 20), b=F(9, 20)):
    """Independent forward expectation check, using exact weight-class counts."""
    first, second = int(a*horizon), int(b*horizon)
    counts = [F(0)]*(horizon+1)
    counts[1] = F(1)
    reward = F(0)
    for n in range(1, horizon):
        accepted = lambda k: n >= second or (n >= first and k == 1)
        accepted_weight = sum((k*counts[k] for k in range(1,n+1)
                               if accepted(k)), F(0))
        total = 2*n-1
        reward += accepted_weight/total
        updated = [F(0)]*(horizon+1)
        updated[1] = counts[1]+1-(counts[1]+accepted_weight)/total
        for k in range(2,n+2):
            inflow = (k-1)*counts[k-1] if not accepted(k-1) else F(0)
            updated[k] = counts[k]+(inflow-k*counts[k])/total
        counts = updated
    return reward


def deferred_density(digits=60):
    """Numerical quadrature of Gamma(3/20,9/20), not a certified bound."""
    with mp.workdps(digits):
        a, b = mp.mpf(3) / 20, mp.mpf(9) / 20
        q = mp.sqrt(a / b)
        leaf_mass = b / 2 + a * a / (6 * b)

        def pgf(z):
            if z == 1:
                return mp.mpf(1)
            return 3 - 2 / z - 2 * (1 - z) ** 2 / z ** 2 * mp.log1p(-z)

        def pgf_prime(z):
            if z == 1:
                return mp.mpf(2)
            return 2 * (2 - z) / z ** 2 + 4 * (1 - z) * mp.log1p(-z) / z ** 3

        def c(z):
            g = q * z / (1 - (1 - q) * z)
            return leaf_mass * z + a * (pgf(g) - mp.mpf(2) * g / 3)

        def c_prime(z):
            denominator = 1 - (1 - q) * z
            g = q * z / denominator
            return leaf_mass + a * (pgf_prime(g) - mp.mpf(2) / 3) * q / denominator ** 2

        lower = mp.sqrt(b)
        integral = mp.quad(lambda z: c_prime(z) / z ** 2, [lower, (lower + 1) / 2, 1])
        gamma = mp.mpf(1) / 4 + b * b / 4 - c(lower) / 2 + b * integral / 2
        return mp.nstr(gamma, digits - 10)


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def make_figure(directory, thresholds, threshold_horizon):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "legend.fontsize": 7.6, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "lines.linewidth": 1.5,
    })
    blue = "#1D597F"
    figure, left = plt.subplots(figsize=(4.96, 2.65))
    n = np.arange(1, threshold_horizon)
    left.step(n / threshold_horizon, thresholds[1:], where="post", color=blue)
    left.set(xlim=(0, 0.62), ylim=(-0.3, 10.4), yticks=range(0, 11),
             xlabel=r"Current time $n/N$", ylabel=r"Acceptance threshold $K_n$")
    figure.tight_layout(pad=0.65)
    # Buffer the PDF before writing, avoiding a truncated destination if a
    # filesystem layer interrupts direct savefig output.
    pdf_buffer = BytesIO()
    figure.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
    pdf_bytes = pdf_buffer.getvalue()
    if len(pdf_bytes) < 1000 or not pdf_bytes.startswith(b"%PDF-"):
        raise IOError("Matplotlib did not produce a valid nonempty PDF stream.")
    (directory / "pa_threshold.pdf").write_bytes(pdf_bytes)
    figure.savefig(directory / "pa_threshold.eps", bbox_inches="tight")
    figure.savefig(directory / "pa_threshold.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def comparison_figure(directory, records):
    """Finite-horizon policies; offline density is an asymptotic reference only."""
    data = np.asarray(records, dtype=float)
    horizons = data[:, 0]
    offline = np.exp(-2*np.pi/(3*np.sqrt(3)))
    fig, axes = plt.subplots(2, 1, figsize=(4.96, 4.35), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.25]})
    styles = [(2, "Optimal online", "#164D70", "-", "o"),
              (6, "Three-phase deferral", "#B05416", "--", "s"),
              (4, "Greedy", "#474747", ":", "^")]
    for ax in axes:
        for col, label, color, style, marker in styles:
            ax.plot(horizons, data[:, col], label=label, color=color,
                    linestyle=style, marker=marker, markersize=3.3)
        ax.set_xscale("log")
        ax.set_ylabel("Expected edges / N")
        ax.set_xlim(85, 95000)
    axes[0].axhline(offline, color="#777777", linestyle="-.", linewidth=1.1)
    axes[0].text(120, offline-0.0045, "Offline limiting density (reference)",
                 fontsize=7.6, color="#555555", va="top")
    axes[0].set_ylim(0.247, 0.304)
    axes[0].set_yticks([0.25, 0.275, 0.30])
    axes[0].text(0.02, 0.65, "(a) With offline reference", transform=axes[0].transAxes, va="top")
    axes[1].set_ylim(0.2494, 0.262)
    axes[1].set_yticks([0.250, 0.255, 0.260])
    axes[1].text(0.02, 0.95, "(b) Online policies, enlarged", transform=axes[1].transAxes,
                 va="top", fontsize=8)
    axes[1].set_xlabel(r"Horizon $N$ (vertices; logarithmic scale)")
    axes[1].legend(loc="upper right", frameon=False, fontsize=7.3)
    fig.tight_layout(pad=0.65, h_pad=0.7)
    for extension in ("pdf", "eps", "png"):
        fig.savefig(directory / f"policy_comparison.{extension}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=[100, 500, 1000, 5000, 10000, 20000, 40000, 80000])
    parser.add_argument("--threshold-horizon", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()
    if min(arguments.horizons + [arguments.threshold_horizon]) < 2:
        parser.error("Horizons must be at least 2.")
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    small = []
    for horizon in range(1, 7):
        affine = rational_affine_dp(horizon)
        exhaustive = exhaustive_value(horizon)
        greedy = exhaustive_value(horizon, greedy=True)
        if affine != exhaustive or greedy != greedy_expectation(horizon):
            raise ArithmeticError(f"Independent exact verification failed at N={horizon}")
        small.append([horizon, str(affine), str(exhaustive), str(greedy)])
    write_csv(output / "small_horizon_exact.csv",
              ["N", "affine_optimum_exact", "enumerated_optimum_exact", "enumerated_greedy_exact"], small)

    fixed_checks = []
    for a, b in [(F(3,20), F(9,20)), (F(1,3), F(2,3))]:
        for horizon in range(1,41):
            backward = deferred_value(horizon, rational=True, a=a, b=b)
            forward = deferred_forward_exact(horizon, a=a, b=b)
            floating = deferred_value(horizon, a=a, b=b)
            if backward != forward or abs(float(forward)-floating) > 1e-11:
                raise ArithmeticError(f"Deferral check failed at N={horizon}, a={a}, b={b}")
            fixed_checks.append([horizon, str(a), str(b), str(backward), str(forward)])
    write_csv(output / "deferral_exact_checks.csv",
              ["N", "a", "b", "backward_exact", "forward_exact"], fixed_checks)

    records = []
    thresholds = None
    for horizon in sorted(set(arguments.horizons)):
        start = time.perf_counter()
        optimum, stored = affine_dp(horizon, horizon == arguments.threshold_horizon)
        if stored is not None:
            thresholds = stored
        greedy = float(greedy_expectation(horizon))
        deferred = deferred_value(horizon)
        if deferred > optimum + 1e-8:
            raise ArithmeticError("A fixed policy exceeded the computed optimum.")
        records.append([horizon, optimum, optimum / horizon, greedy, greedy / horizon,
                        deferred, deferred / horizon])
        print(f"N={horizon:6d}: optimum/N={optimum/horizon:.12f}; "
              f"greedy/N={greedy/horizon:.12f}; DG/N={deferred/horizon:.12f}; "
              f"{time.perf_counter()-start:.2f}s", flush=True)
    if thresholds is None:
        _, thresholds = affine_dp(arguments.threshold_horizon, keep_thresholds=True)
    write_csv(output / "finite_horizon_density.csv",
              ["N", "optimal_expectation_float64", "optimal_density_float64",
               "greedy_expectation_float64", "greedy_density_float64",
               "deferral_expectation_float64", "deferral_density_float64"], records)
    write_csv(output / "thresholds_N10000.csv" if arguments.threshold_horizon == 10000
              else output / f"thresholds_N{arguments.threshold_horizon}.csv",
              ["current_time_n", "n_over_N", "K_n"],
              ([n, n / arguments.threshold_horizon, thresholds[n]]
               for n in range(1, arguments.threshold_horizon)))
    crossings = []
    for weight in range(1, 11):
        times = np.flatnonzero(thresholds >= weight)
        first = int(times[0]) if len(times) else None
        crossings.append([weight, first, first / arguments.threshold_horizon if first else None])
    write_csv(output / "threshold_first_admission.csv",
              ["weight_k", "first_current_time_n", "n_over_N"], crossings)
    gamma = deferred_density(60)
    gamma_check = deferred_density(80)
    # Compare at adequate precision, not mpmath's default 15 digits.
    with mp.workdps(80):
        if abs(mp.mpf(gamma) - mp.mpf(gamma_check)) > mp.mpf("1e-45"):
            raise ArithmeticError("Quadrature values failed to stabilize.")
    print(f"Numerical Gamma(3/20,9/20) = {gamma}", flush=True)
    with (output / "numerical_verification.json").open("w", encoding="utf-8") as handle:
        json.dump({"gamma_60_digit_quadrature": gamma,
                   "gamma_80_digit_quadrature": gamma_check,
                   "gamma_note": "Numerical quadrature, not an interval-certified bound.",
                   "finite_horizon_note": "Float64 evaluations of an exact Bellman recurrence; no simulations.",
                   "small_horizons": "Exact rational equality with labeled-state enumeration for N=1,...,6.",
                   "deferral_checks": "Independent forward/backward rational expectations agree for N=1,...,40, for (a,b)=(3/20,9/20) and (1/3,2/3).",
                   "deferral_convention": "Reject for n<floor(3N/20); accept only weight 1 for floor(3N/20)<=n<floor(9N/20); greedy thereafter.",
                   "threshold_convention": "n vertices present before arrival n+1; accept feasible weight k iff k<=K_n.",
                   "threshold_horizon": arguments.threshold_horizon,
                   "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
                   "mpmath_version": mp.__version__}, handle, indent=2)
    make_figure(output, thresholds, arguments.threshold_horizon)
    comparison_figure(output, records)


if __name__ == "__main__":
    main()
