# Online matching in growing random trees

Supplement to **Online matching in growing random trees: how attachment shapes optimal decisions**, by Marek Gałązka and Hanna Wdowicka.

Repository: https://github.com/mgalazka84/online-matching-growing-trees

## Requirements and use

The supplied run uses Python 3.12 with `numpy`, `matplotlib`, and `mpmath`.
Install the recorded dependency versions for reproduction:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

The default run evaluates horizons 100, 500, 1000, 5000, and 10000; generates
the threshold figure at N = 10000 and a policy-comparison figure; independently checks tiny horizons by
exact enumeration; and evaluates the three-phase policy integral at two
working precisions. To reproduce every supplied table row, run:

```sh
python reproduce.py --horizons 100 500 1000 5000 10000 20000 40000 80000
```

Choose a different figure horizon with `--threshold-horizon N`, or a different
output directory with `--output-dir PATH`. Output files in that directory are
overwritten. Running the default command replaces the supplied eight-row
density table with its five-row subset. No random sampling is used. Phase boundaries use exact integer floors, computed from rational parameters.

## Model and indices

The initial state is one unmatched root with attachment weight 1. Each new
vertex has weight 1 and chooses its parent with probability proportional to
the parent's current weight. The parent's weight then increases by 1. Thus
the weight of a vertex is 1 plus its number of children, and the total weight
with n vertices present is 2n − 1. Accepting an arriving feasible edge makes
its two endpoints matched permanently; rejecting it is irreversible.

The horizon N is the total number of vertices. The index n denotes the time
**before arrival n + 1**. In the manuscript's notation the additional-reward
value is

    V_n(x) = c_n + sum_k beta[n,k] x_k,

with c_N = 0 and beta[N,k] = 0. The code implements

    c_n = c_(n+1) + beta[n+1,1],
    beta[n,k] = beta[n+1,k]
                + k/(2n-1) * (max(1-beta[n+1,1], beta[n+1,k+1])
                              - beta[n+1,k]).

The optimal expectation from the initial root is c_1 + beta[1,1]. A feasible
edge arriving at n + 1 is accepted when

    beta[n+1,k+1] + beta[n+1,1] <= 1.

Consequently K_n is the number of weights k in {1,...,n} satisfying this
inequality. Acceptance is chosen on exact ties. `thresholds_N10000.csv`
contains the integer K_n values obtained with floating-point coefficients,
together with n and n/N. It is a complete finite-horizon table, not samples
from a limiting curve. Floating-point decisions near ties are not certified
using interval arithmetic.

The implementation uses a coefficient vector, a temporary vector, a vector
of weights, and optionally an array of thresholds. Its storage is O(N) and
its number of arithmetic operations is O(N²). NumPy evaluates each row using
vectorized operations. This is arithmetic complexity, not a bound on the
bit complexity of exact rational evaluation.

## Independent checks and numerical accuracy

For N = 1,...,6, `exhaustive_value` enumerates the labeled vertex weights,
matched-vertex mask, parent choices, and legal accept/reject actions. It
uses Python's exact `Fraction` arithmetic. It does not assume the affine
Bellman representation or a threshold policy. The computed optimum equals
the rational affine recurrence at each checked horizon. Forced greedy
enumeration also equals the separate closed formula. In particular,

    OPT_4 = 19/15,     E[M_4^Greedy] = 6/5.

For larger N, the reported optimal expectations are float64 evaluations of
the exact Bellman recurrence. The phrase “exact dynamic program” describes
the recurrence, not the rounding of the reported decimal values.

The greedy expectation is evaluated from its exact formula (N >= 3):

    E[M_N^Greedy] = N/4 + (1/8) * (1 + 3/(2N-3)).

The special cases N = 1 and N = 2 give 0 and 1 respectively. The limiting
greedy density is 1/4.

`deferred_density` evaluates the displayed one-dimensional formula for
Gamma(3/20,9/20) with `mpmath` at 60 and 80 decimal working digits. The
values must agree within 10^(-45). This stability check is numerical and
does not replace the manuscript's analytic strict-improvement proof. The
stored result is approximately 0.2553816838760049420547.

No limit theorem for the optimal finite-horizon density or threshold profile
is inferred from these computations. In particular the apparent optimum
near 0.25693 is not plotted as a proved limit.

## Three-phase policy comparison

`deferred_value` evaluates DG(3/20,9/20) at the same finite horizons as the
optimum. It replaces the maximum in the affine backward recurrence with
the prescribed action: reject all feasible edges for n < floor(3N/20),
accept only weight-1 unmatched parents until n = floor(9N/20), and then
accept every feasible edge. The root state and time convention are the
same as for the optimal-policy calculation.

`deferred_forward_exact` independently evolves expected unmatched counts
using the forward weight-class equations. Exact forward and backward
rational expectations agree for every N = 1,...,40, both for (a,b) =
(3/20,9/20) and (1/3,2/3); the latter exercises all three phases at small
horizons. Float64 backward values are also compared with these rational
values to absolute tolerance 1e-11. Larger-horizon fixed-policy expectations
are checked not to exceed the computed optimum.

`policy_comparison.pdf` plots the finite-horizon optimal, deferral and greedy
expectations per vertex on logarithmic horizon axes. The upper panel adds
the offline limiting density exp(-2*pi/(3*sqrt(3))) as a reference. This is
not the finite-horizon offline expectation and is not an error bound.
The lower panel enlarges the three online curves. The plot does not add an
extrapolated optimal limit. Connecting lines guide the eye between computed
horizons. PDF and EPS are vector output; PNG previews use 300 dpi.

## Supplied outputs

- `finite_horizon_density.csv`: optimal, greedy and three-phase deferral expectations and densities.
- `thresholds_N10000.csv`: all n = 1,...,9999 and their integer thresholds.
- `threshold_first_admission.csv`: first n at which K_n admits each weight
  k = 1,...,10, with n/N.
- `small_horizon_exact.csv`: exact optimum and greedy checks for N = 1,...,6.
- `deferral_exact_checks.csv`: independent rational forward/backward checks.
- `numerical_verification.json`: quadrature values, conventions and versions.
- `pa_threshold.pdf` and `pa_threshold.eps`: vector figure for the manuscript.
- `pa_threshold.png`: a raster preview of the same figure.
- `policy_comparison.pdf`, `.eps`, `.png`: comparison of the three online policies.
- `requirements.txt`: dependency versions used for this supplied run.

Suggested figure caption:

> The optimal weight threshold for standard preferential attachment at
> horizon N = 10000, restricted to its first ten accepted weight levels.
> At time n, before arrival n + 1, a feasible edge is accepted if and only
> if its parent's current attachment weight is at most K_n. The curve is
> obtained by floating-point evaluation of the exact backward recurrence;
> it does not assert convergence to a limiting threshold profile.
