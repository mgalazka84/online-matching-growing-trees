#!/usr/bin/env python3
"""Independent computational audit; run: python audit.py [--full].

Uses the same dependencies as reproduce.py. Exact checks use Fraction;
large-horizon checks use deterministic forward expectations, not sampling.
This verifies implementations and algebra, not mathematical proofs by a
formal proof assistant. Output: audit_results.json and age_bias_checks.csv.
"""
import argparse
import csv
from fractions import Fraction as F
from functools import lru_cache
import json
from math import comb
from pathlib import Path

import mpmath as mp
import numpy as np

import reproduce as rep


def label_value(N, model, optimal=False):
    """Labeled matched-mask enumeration, independent of count/Bellman ansatz."""
    @lru_cache(None)
    def visit(n, mask):
        if n == N:
            return F(0)
        result = F(0)
        for i in range(1, n + 1):
            p = (F(1, n) if model == 'uniform' else
                 F(2 * (i if model == 'young' else n + 1 - i), n * (n + 1)))
            if mask & (1 << (i - 1)):
                reward = visit(n + 1, mask)
            else:
                accept = 1 + visit(n + 1, mask | (1 << (i - 1)) | (1 << n))
                reward = max(accept, visit(n + 1, mask)) if optimal else accept
            result += p * reward
        return result
    return visit(1, 0)


def age_marginals(N, model, exact=False, checkpoints=()):
    """Forward unmatched probability of each label; no survival products."""
    q = [F(0)] * N if exact else np.zeros(N)
    q[0] = F(1) if exact else 1.0
    total = F(0) if exact else 0.0
    rows = []
    labels = np.arange(1, N + 1, dtype=float)
    for n in range(2, N + 1):
        if exact:
            p = [F(2 * (i if model == 'young' else n - i), n * (n - 1))
                 for i in range(1, n)]
            delta = sum((p[i] * q[i] for i in range(n - 1)), F(0))
            q[:n - 1] = [q[i] * (1 - p[i]) for i in range(n - 1)]
        else:
            p = 2 * (labels[:n - 1] if model == 'young' else n - labels[:n - 1]) / (n * (n - 1))
            delta = float(np.sum(p * q[:n - 1]))
            q[:n - 1] *= 1 - p
        q[n - 1] = 1 - delta
        total += delta
        if n in checkpoints:
            rows.append([model, n, float(total / n), float(delta)])
    assert abs(float(sum(q) - (N - 2 * total))) < 1e-7
    return total, rows


def forward_policy(N, thresholds=None, a=F(3, 20), b=F(9, 20)):
    """Forward weight-class expectation; independent of backward coefficients."""
    x = np.zeros(N + 1, dtype=np.longdouble)
    x[1] = 1
    weights = np.arange(N + 1, dtype=np.longdouble)
    total = np.longdouble(0)
    first, second = int(a * N), int(b * N)
    for n in range(1, N):
        K = int(thresholds[n]) if thresholds is not None else (n if n >= second else (1 if n >= first else 0))
        weighted = weights[1:n + 1] * x[1:n + 1]
        accepted = np.sum(weighted[:K], dtype=np.longdouble)
        total += accepted / (2 * n - 1)
        inflow = weighted.copy()
        inflow[:K] = 0
        x[2:n + 2] += (inflow - weights[2:n + 2] * x[2:n + 2]) / (2 * n - 1)
        x[1] += 1 - (x[1] + accepted) / (2 * n - 1)
        assert np.min(x[1:n + 2]) >= -1e-14
    assert abs(np.sum(x) - (N - 2 * total)) < 1e-8
    return float(total)


def rational_thresholds(N):
    b = [F(0)] * (N + 1)
    thresholds = [0] * N
    c = F(0)
    for n in range(N - 1, 0, -1):
        assert all(b[k] <= b[k + 1] for k in range(1, n + 1))
        valid = [k for k in range(1, n + 1) if b[k + 1] + b[1] <= 1]
        assert valid == list(range(1, len(valid) + 1))
        thresholds[n] = len(valid)
        c += b[1]
        b = [F(0)] + [b[k] + F(k, 2*n-1) * (max(1-b[1], b[k+1])-b[k]) for k in range(1, n+1)]
    return c + b[1] if N > 1 else F(0), thresholds


class Radical:
    """Exact ring Q[sqrt(3),sqrt(5)] in basis 1,sqrt3,sqrt5,sqrt15."""
    def __init__(self, value=0):
        self.c = tuple(map(F, value)) if isinstance(value, (tuple, list)) else (F(value), F(0), F(0), F(0))
    def __add__(self, other):
        other = other if isinstance(other, Radical) else Radical(other)
        return Radical([a+b for a,b in zip(self.c, other.c)])
    __radd__ = __add__
    def __neg__(self):
        return Radical([-a for a in self.c])
    def __sub__(self, other):
        return self + (-other if isinstance(other, Radical) else -Radical(other))
    def __rsub__(self, other):
        return -self + other
    def __mul__(self, other):
        other = other if isinstance(other, Radical) else Radical(other)
        result = [F(0)] * 4
        for i, a in enumerate(self.c):
            for j, b in enumerate(other.c):
                result[i ^ j] += a * b * (3 if i & j & 1 else 1) * (5 if i & j & 2 else 1)
        return Radical(result)
    __rmul__ = __mul__
    def __pow__(self, n):
        result = Radical(1)
        for _ in range(n):
            result = result * self
        return result


def analytic_certificate():
    """Reconstruct c_1..c_5 and S_5 independently by power-series algebra."""
    a, b = F(3, 20), F(9, 20)
    q = Radical([0, F(1, 3), 0, 0])
    root_b = Radical([0, 0, F(3, 10), 0])
    coefficients = [Radical(0), Radical(b/2 + a*a/(6*b))]
    for k in range(2, 6):
        coefficients.append(a * sum((F(4, j*(j+1)*(j+2)) * comb(k-1, j-1) * q**j * (1-q)**(k-j) for j in range(2, k+1)), Radical(0)))
    expected = [[F(7,30),0,0,0], [F(1,120),0,0,0],
                [F(1,60),-F(1,225),0,0], [F(11,360),-F(1,75),0,0],
                [F(1,18),-F(136,4725),0,0]]
    assert all(c.c == Radical(e).c for c,e in zip(coefficients[1:], expected))
    s = Radical((1-b*b)/2) + coefficients[1]*b + coefficients[2]*b
    for k in range(3, 6):
        phi = (2*(k-1)*root_b**k - k*b) * F(1, k-2)
        s += coefficients[k] * phi
    assert s.c == (F(20851,48000), F(19837,630000), F(9,500), -F(309,43750))
    log_coefficient = coefficients[2].c[0] * b
    assert log_coefficient == F(3,800)
    assert F(173206,100000)**2 > 3
    assert F(223607,100000)**2 > 5
    assert F(387298,100000)**2 < 15
    y = F(11,29)
    logarithm_lower = 2 * sum((y**(2*j+1)/F(2*j+1) for j in range(5)), F(0))
    assert logarithm_lower > F(7985,10000)
    upper_s = (s.c[0] + s.c[1]*F(173206,100000) + s.c[2]*F(223607,100000)
               + s.c[3]*F(387298,100000) - log_coefficient*F(7985,10000))
    lower_gamma = (1-upper_s)/2
    assert F(1,2) - upper_s > F(11656,10000000)
    assert lower_gamma > F(2505828,10000000)
    return {'S5_upper_exact':str(upper_s), 'Gamma_lower_exact':str(lower_gamma),
            'Gamma_lower_decimal':float(lower_gamma), 'all_rational_inequalities':True}


def affine_greedy_means():
    count = 0
    for alpha in map(F, ['0', '1/2', '1', '2', '10']):
        mean, product = F(0), F(1)
        for n in range(1, 101):
            if n >= 3:
                closed = F(n)/(alpha+3) + alpha/(2*(alpha+3))*(1+product)
                assert closed == mean
                product *= 1 - 2/((alpha+1)*n-alpha)
                count += 1
            mean += (n-2*mean)/((alpha+1)*n-alpha)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Recheck all eight stored horizons through 80000.')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = {'scope':'Computational and algebraic audit, not formal proof verification.'}
    result['analytic_certificate'] = analytic_certificate()
    print('Exact five-class analytic certificate: PASS', flush=True)
    tiny = []
    for N in range(1, 9):
        opt = label_value(N, 'uniform', True)
        greedy = label_value(N, 'uniform')
        assert opt == greedy == (F(0) if N == 1 else F(1) if N == 2 else F(N,3))
        for model in ['young', 'old']:
            enumeration = label_value(N, model)
            forward, _ = age_marginals(N, model, exact=True)
            assert enumeration == forward
            tiny.append({'model':model,'N':N,'expectation_exact':str(forward)})
    result['uniform_exact_optimality_horizons'] = list(range(1,9))
    result['age_exact_checks'] = tiny
    pa = []
    for N in range(1,8):
        optimum = rep.exhaustive_value(N)
        greedy = rep.exhaustive_value(N,greedy=True)
        assert optimum == rep.rational_affine_dp(N)
        assert greedy == rep.greedy_expectation(N)
        pa.append({'N':N,'optimal_exact':str(optimum),'greedy_exact':str(greedy)})
    result['pa_labeled_state_checks'] = pa
    print('Independent labeled-state enumeration: PASS', flush=True)
    errors, threshold_differences = [], []
    for N in range(1,51):
        exact, K = rational_thresholds(N)
        floating, floating_K = rep.affine_dp(N,True)
        errors.append(abs(float(exact)-floating))
        for n in range(1,N):
            if K[n] != floating_K[n]:
                threshold_differences.append([N,n,K[n],int(floating_K[n])])
    assert max(errors) < 1e-11
    result['rational_monotonicity_horizons'] = list(range(1,51))
    result['rational_vs_float64_max_value_error'] = max(errors)
    result['rational_vs_float64_threshold_differences'] = threshold_differences
    result['affine_greedy_exact_mean_checks'] = affine_greedy_means()
    fixed_checks = 0
    for a,b in [(F(3,20),F(9,20)), (F(1,3),F(2,3))]:
        for N in range(1,41):
            value = rep.deferred_forward_exact(N,a,b)
            assert value == rep.deferred_value(N,True,a,b)
            assert abs(float(value)-rep.deferred_value(N,False,a,b)) < 1e-11
            fixed_checks += 1
    result['exact_deferral_forward_backward_checks'] = fixed_checks
    forward = []
    for N in [100,500,1000,5000,10000]:
        value,K = rep.affine_dp(N,True)
        independent = forward_policy(N,K)
        dg = forward_policy(N)
        dg_back = rep.deferred_value(N)
        assert abs(value-independent) < 1e-8
        assert abs(dg-dg_back) < 1e-8
        forward.append({'N':N,'optimal_backward':value,'threshold_forward':independent,
                        'deferral_backward':dg_back,'deferral_forward':dg})
    result['independent_forward_policy_checks'] = forward
    mp.mp.dps = 80
    result['gamma_60_digits'] = rep.deferred_density(60)
    result['gamma_80_digits'] = rep.deferred_density(80)
    assert abs(mp.mpf(result['gamma_60_digits'])-mp.mpf(result['gamma_80_digits'])) < mp.mpf('1e-45')
    a_y = mp.quad(lambda x: 2*x*mp.exp(-2*(1-x)),[0,1])
    a_o = mp.quad(lambda x: 2*(1-x)*x*x*mp.exp(2*(1-x)),[0,1])
    assert abs(a_y-(1+mp.exp(-2))/2) < mp.mpf('1e-70')
    assert abs(a_o-(9-mp.exp(2))/4) < mp.mpf('1e-70')
    result['age_kernel_masses'] = {'young':mp.nstr(a_y,60),'old':mp.nstr(a_o,60)}
    rows=[]
    for model in ['young','old']:
        _,checks=age_marginals(20000,model,checkpoints=[100,500,1000,5000,10000,20000])
        rows.extend(checks)
    result['age_density_limits']={'young':mp.nstr(a_y/(1+a_y),60),'old':mp.nstr(a_o/(1+a_o),60)}
    print('Forward policies, age kernels, quadrature: PASS', flush=True)
    source=Path(__file__).resolve().parent
    stored=[]
    with (source/'finite_horizon_density.csv').open() as h:
        for row in csv.DictReader(h):
            N=int(row['N'])
            if N > 10000 and not args.full:
                continue
            value,_=rep.affine_dp(N)
            dg=rep.deferred_value(N)
            assert abs(value-float(row['optimal_expectation_float64'])) < 1e-8
            assert abs(dg-float(row['deferral_expectation_float64'])) < 1e-8
            assert abs(float(rep.greedy_expectation(N))-float(row['greedy_expectation_float64'])) < 1e-10
            stored.append(N)
            print(f'Stored table N={N}: PASS',flush=True)
    result['stored_table_horizons_recomputed']=stored
    _,thresholds=rep.affine_dp(10000,True)
    with (source/'thresholds_N10000.csv').open() as h:
        assert all(int(row['K_n'])==thresholds[int(row['current_time_n'])] for row in csv.DictReader(h))
    result['all_9999_stored_thresholds_recomputed']=True
    result['precision_note']='Float64 computations are not interval certificates; longdouble may equal float64 on some platforms.'
    result['versions']={'numpy':np.__version__,'mpmath':mp.__version__,
                        'longdouble_mantissa_bits':int(np.finfo(np.longdouble).nmant)}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'audit_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (args.output_dir/'age_bias_checks.csv').open('w',newline='') as h:
        writer=csv.writer(h)
        writer.writerow(['model','N','expected_matching_density','last_acceptance_probability'])
        writer.writerows(rows)
    print('ALL AUDIT CHECKS PASSED',flush=True)


if __name__=='__main__':
    main()
