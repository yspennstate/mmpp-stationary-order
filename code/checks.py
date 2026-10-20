"""Checks for "The time-stationary and Palm inter-arrival times of a Markov-modulated Poisson
process need not be ordered".

Every number quoted in the paper is printed here.  Sections:
  1. exact rational certificate for the 12-state cycle (Theorem 1.3)
  2. exact rational certificate for the 24-state block cycle (Remark 3.2)
  3. the cosine family on the n-cycle: first-order term, threshold n >= 12 (Theorem 1.2)
  4. periodic Cox process with a square-wave intensity (illustration of Theorem 1.1)
  5. reversible modulators: the order holds (Proposition 4.1), random tests
  6. convexity in the speed of modulation: the six-cycle with f''(0) = -1 (Proposition 5.1)
Run: python checks.py   (numpy, scipy, mpmath, sympy; the certificates use fractions.Fraction only)
"""
from fractions import Fraction as Fr
import math
import numpy as np
import scipy.linalg as sla
import mpmath as mp
import sympy

ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


def certify(rates, t, N):
    """MMPP on the unit-rate directed n-cycle (i -> i+1 mod n) with arrival rates `rates` (Fractions).
    Rational bounds lo <= F(t) <= hi for F(t) = (pi - alpha) e^{Ct} 1, C = Q - D, by uniformization
    with an explicit Poisson tail bound, plus separate enclosures of the two survival functions."""
    n = len(rates)
    lam = sum(rates) / n                      # pi is uniform: the cycle generator is doubly stochastic
    pi = [Fr(1, n)] * n
    alpha = [Fr(1, n) * d / lam for d in rates]
    w = [p - a for p, a in zip(pi, alpha)]
    r = max(1 + d for d in rates)             # r >= max_i (-C_ii) = 1 + d_i
    Bd = [1 - (1 + d) / r for d in rates]     # B = I + C/r: B_ii, and B_{i,i+1} = 1/r
    Bs = Fr(1) / r
    assert all(x >= 0 for x in Bd)
    a = r * t
    v = [Fr(1)] * n                           # B^k 1 (column recurrence)
    S = Fr(0); Ppi = Fr(0); Pal = Fr(0); E = Fr(0)
    coef = Fr(1)                              # a^k / k!
    for k in range(N + 1):
        assert all(0 <= vi <= 1 for vi in v)
        S += coef * sum(wi * vi for wi, vi in zip(w, v))
        Ppi += coef * sum(p * vi for p, vi in zip(pi, v))
        Pal += coef * sum(p * vi for p, vi in zip(alpha, v))
        E += coef
        v = [Bd[i] * v[i] + Bs * v[(i + 1) % n] for i in range(n)]
        coef = coef * a / (k + 1)
    assert a < N + 2
    R = coef / (1 - a / (N + 2))              # >= sum_{k > N} a^k / k!
    w1 = sum(abs(x) for x in w)
    # e^a lies in [E, E + R], and F e^a = S + tail with |tail| <= w1 R
    cands = [(S + s * w1 * R) / den for s in (-1, 1) for den in (E, E + R)]
    lo, hi = min(cands), max(cands)
    pi_hi = (Ppi + R) / E                     # P(T^pi > t) <= pi_hi
    al_lo = Pal / (E + R)                     # P(T^alpha > t) >= al_lo
    u = list(w); S2 = Fr(0); coef = Fr(1)     # row recurrence: (w B^k) 1
    for k in range(N + 1):
        S2 += coef * sum(u)
        u = [u[j] * Bd[j] + u[(j - 1) % n] * Bs for j in range(n)]
        coef = coef * a / (k + 1)
    return dict(lam=lam, alpha=alpha, lo=lo, hi=hi, pi_hi=pi_hi, al_lo=al_lo, S=S, S2=S2, R=R, r=r, a=a)


def mp_surv(rates, t, dps=40):
    mp.mp.dps = dps
    n = len(rates)
    C = mp.zeros(n, n)
    for i in range(n):
        C[i, i] = -1 - mp.mpf(rates[i].numerator) / rates[i].denominator
        C[i, (i + 1) % n] += 1
    col = mp.expm(C * (mp.mpf(t.numerator) / t.denominator)) * mp.matrix([1] * n)
    lam = sum(rates) / n
    Spi = sum(col[i] for i in range(n)) / n
    Sal = sum(col[i] * mp.mpf((rates[i] / lam / n).numerator) / (rates[i] / lam / n).denominator for i in range(n))
    return Spi, Sal


print("== 1. twelve-state cycle, rates p_i / 10^6, t = 19/2")
p12 = [1500, 1433, 1250, 1000, 750, 567, 500, 567, 750, 1000, 1250, 1433]
rates12 = [Fr(x, 10**6) for x in p12]
res = certify(rates12, Fr(19, 2), 80)
print("lambda =", res["lam"], " r =", res["r"], " a = r t =", res["a"])
print("12000 alpha_i:", [int(x * 12000) for x in res["alpha"]])
print("F(19/2) in [%.20e, %.20e]" % (float(res["lo"]), float(res["hi"])))
print("upper P(T^pi>t) - lower P(T^alpha>t) = %.12e" % float(res["pi_hi"] - res["al_lo"]))
check("12-state: row and column recurrences agree exactly", res["S"] == res["S2"])
check("12-state: F(19/2) < -1/300000 (signed enclosure)", res["hi"] < Fr(-1, 300000))
check("12-state: upper P(T^pi > t) - lower P(T^alpha > t) < -1/300000", res["pi_hi"] - res["al_lo"] < Fr(-1, 300000))
Spi, Sal = mp_surv(rates12, Fr(19, 2))
print("mpmath 40 digits: P(T^pi>t) =", mp.nstr(Spi, 15), " P(T^alpha>t) =", mp.nstr(Sal, 15), " diff =", mp.nstr(Spi - Sal, 15))
print("enclosure width = %.3e" % float(res["hi"] - res["lo"]))
check("12-state: mpmath difference within 1e-30 of the rational enclosure",
      res["lo"] - Fr(1, 10**30) <= Fr(mp.nstr(Spi - Sal, 35)) <= res["hi"] + Fr(1, 10**30))
check("12-state control: F(1/2) > 0 (the order holds at small t)", certify(rates12, Fr(1, 2), 40)["lo"] > 0)

print("== 2. 24-state block cycle, rate 2/3 on states 0..11, 0 on 12..23, t = 79/4")
rates24 = [Fr(2, 3)] * 12 + [Fr(0)] * 12
res24 = certify(rates24, Fr(79, 4), 140)
print("lambda =", res24["lam"], " F(79/4) in [%.25e, %.25e]" % (float(res24["lo"]), float(res24["hi"])))
check("24-state: F(79/4) < -0.00117 (exact enclosure)", res24["hi"] < Fr(-117, 100000))
Spi, Sal = mp_surv(rates24, Fr(79, 4))
print("mpmath: P(T^pi>t) =", mp.nstr(Spi, 12), " P(T^alpha>t) =", mp.nstr(Sal, 12))
check("24-state: survival values 0.01425361547 and 0.01542552227",
      abs(Spi - mp.mpf("0.01425361547")) < 1e-10 and abs(Sal - mp.mpf("0.01542552227")) < 1e-10)


print("== 3. cosine family on the n-cycle")
def first_order_min(n):
    th = 2 * math.pi / n
    delta, omega = 1 - math.cos(th), math.sin(th)
    t0 = 1.5 * math.pi / omega
    return (delta - omega * math.exp(-delta * t0)) / (delta**2 + omega**2), t0, delta, omega


vals = {n: first_order_min(n)[0] for n in range(3, 401)}
print("J_n(3 pi/(2 omega)) at n = 11, 12, 13:", vals[11], vals[12], vals[13])
check("first-order minimum negative exactly for n >= 12 within 3..400", [n for n in vals if vals[n] < 0] == list(range(12, 401)))
x11, x12 = math.tan(math.pi / 11), math.tan(math.pi / 12)
print("x e^{3 pi x/2} at x = tan(pi/11), tan(pi/12):", x11 * math.exp(1.5 * math.pi * x11), x12 * math.exp(1.5 * math.pi * x12))
check("criterion x e^{3 pi x/2} < 1 fails at n = 11 and holds at n = 12",
      x11 * math.exp(1.5 * math.pi * x11) > 1 > x12 * math.exp(1.5 * math.pi * x12))


def F_float(rates, t):
    n = len(rates)
    C = np.zeros((n, n))
    for i in range(n):
        C[i, i] = -1 - rates[i]
        C[i, (i + 1) % n] += 1
    col = sla.expm(C * t) @ np.ones(n)
    lam = np.mean(rates)
    return col.mean() - (rates / lam / n) @ col


for n in (12, 20):
    b = 0.5
    _, t0, dl, om = first_order_min(n)
    f = 1 + b * np.cos(2 * np.pi * np.arange(n) / n)
    pred = b * b / 2 * (dl - om * math.exp(-dl * t0)) / (dl**2 + om**2)
    for eps in (1e-3, 1e-4, 1e-5):
        fd = F_float(eps * f, t0) / eps
        print(f"n={n} eps={eps}: F/eps = {fd:.8f}, first-order value {pred:.8f}")
    check(f"n={n}: F/eps matches the first-order value (rel err < 1e-3 at eps = 1e-5)", abs(fd - pred) < 1e-3 * abs(pred))
    grid = np.linspace(0.05, 4 * t0, 400)
    closed = [b * b / 2 * (dl - math.exp(-dl * s) * (dl * math.cos(om * s) - om * math.sin(om * s))) / (dl**2 + om**2) for s in grid]
    fds = [F_float(1e-5 * f, s) / 1e-5 for s in grid]
    check(f"n={n}: closed form of (b^2/2) J_n(t) matches F/eps on a t-grid (max err < 1e-3)",
          max(abs(x - y) for x, y in zip(closed, fds)) < 1e-3)
f11 = 1 + 0.5 * np.cos(2 * np.pi * np.arange(11) / 11)
check("n = 11, eps = 1e-4: F > 0 on a grid of t in [0.1, 60]", min(F_float(1e-4 * f11, s) for s in np.linspace(0.1, 60, 600)) > 0)

print("== 4. periodic Cox process: lambda = 2 on [0,1/2), 0 on [1/2,1), period 1, lambda* = 1")
def cox_S(t, m=20000):
    th = (np.arange(m) + 0.5) / m
    def Lam(x):  # integral_0^x lambda
        k = np.floor(x)
        return k + 2 * np.minimum(x - k, 0.5)
    return np.mean(np.exp(-(Lam(th + t) - Lam(th))))


g = [math.exp(t) * cox_S(t) for t in np.linspace(0.01, 0.99, 99)]
print("g(1/4), g(1/2), g(3/4), g(1) =", [math.exp(t) * cox_S(t) for t in (0.25, 0.5, 0.75, 1.0)])
check("Cox: g > 1 on a grid in (0,1) and g(1) = 1", min(g) > 1 and abs(math.exp(1) * cox_S(1.0) - 1) < 1e-12)
h = 1e-5
haz = [-(cox_S(t + h) - cox_S(t - h)) / (2 * h) / cox_S(t) for t in (0.25, 0.75, 0.9)]
print("hazard of T^pi at t = 1/4, 3/4, 9/10:", haz)
check("Cox: hazard of T^pi exceeds lambda* = 1 at t = 3/4", haz[1] > 1)

print("== 5. reversible modulators")
rng = np.random.default_rng(1)
worst, used = 1.0, 0
for trial in range(2000):
    n = int(rng.integers(2, 8))
    W = rng.exponential(size=(n, n)) * (rng.random((n, n)) < 0.7)
    W = np.triu(W, 1); W = W + W.T
    if not np.all(np.linalg.matrix_power(W + np.eye(n), n) > 0):
        continue
    pi = rng.dirichlet(np.ones(n))
    Q = W / pi[:, None]                       # pi_i q_ij = W_ij is symmetric: reversible with respect to pi
    np.fill_diagonal(Q, 0); np.fill_diagonal(Q, -Q.sum(1))
    d = rng.exponential(size=n) * (rng.random(n) < 0.8)
    if d.sum() == 0:
        continue
    used += 1
    C = Q - np.diag(d); lam = pi @ d; al = pi * d / lam
    for t in np.linspace(0, 10, 21):
        col = sla.expm(C * t) @ np.ones(n)
        worst = min(worst, ((pi - al) @ col) / max(pi @ col, 1e-300))
print("instances:", used, " smallest (S_pi - S_alpha)/S_pi:", worst)
check("reversible: no violation beyond rounding (> -1e-9)", worst > -1e-9)

print("== 6. He 2021 Section 4.3 convexity: six-cycle, d = (1/3, 2/5, 2/3, 1, 2/3, 2/5)")
d6 = [Fr(1, 3), Fr(2, 5), Fr(2, 3), Fr(1), Fr(2, 3), Fr(2, 5)]
n = 6
Qv = lambda v: [v[(i + 1) % n] - v[i] for i in range(n)]
Dinv = lambda v: [v[i] / d6[i] for i in range(n)]
f2 = 2 * sum(Dinv(Qv(Dinv(Qv(Dinv([Fr(1)] * n)))))) / n
print("f(0) = pi D^{-1} 1 =", sum(Fr(1) / x for x in d6) / n, "  f'(0) =", sum(Dinv(Qv(Dinv([Fr(1)] * n)))) / n, "  f''(0) =", f2)
check("six-cycle: f''(0) = -1", f2 == -1)


def fx(x):  # pi (D - xQ)^{-1} 1, exact
    M = sympy.zeros(n, n)
    for i in range(n):
        M[i, i] = sympy.Rational(d6[i].numerator, d6[i].denominator) + x
        M[i, (i + 1) % n] = -x
    return sum(M.LUsolve(sympy.ones(n, 1))) / n


f1, f2_, f3 = (fx(sympy.Rational(k, 1000)) for k in (1, 2, 3))
gap = sympy.nsimplify(f1 + f3 - 2 * f2_)
print("f(1/1000) + f(3/1000) - 2 f(2/1000) =", sympy.N(gap, 12))
check("six-cycle: exact midpoint convexity gap negative", gap < 0)
check("six-cycle: f decreasing at the three points", f1 > f2_ > f3)
print("ALL PASS" if ok_all else "SOME CHECK FAILED")
