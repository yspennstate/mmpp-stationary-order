# The time-stationary and Palm inter-arrival times of a Markov-modulated Poisson process need not be ordered

Yitzchak Shmalo

For an irreducible Markov-modulated Poisson process (MMPP) with generator Q, arrival-rate matrix D and C = Q - D,
Asanjarani and Nazarathy (ANZIAM J. 64 (2022) 54-68, Conjecture 3.9; Conjecture 4.2 of arXiv:1905.01736)
conjectured that

    pi e^{Ct} 1 >= alpha e^{Ct} 1   for all t >= 0,    alpha = pi D / (pi D 1),

that is, that the time from a stationary instant to the next arrival is stochastically larger than the time between
two arrivals in the event-stationary process. He (J. Appl. Probab. 58 (2021) 880-889, Section 4.3) lists the same
property among those observed numerically for MMPP-type phase-type distributions.

The paper shows that the conjecture is false.

- The order is equivalent to the hazard rate of the time-stationary interval never exceeding the arrival rate.
- For a Cox process with a non-constant periodic intensity and uniform phase, the order fails on a set of positive measure.
- On the directed n-cycle with arrival rates eps (1 + b cos(2 pi j / n)), the order fails for all small eps > 0 whenever n >= 12.
- An explicit 12-state MMPP with positive rational rates has P(T^pi > 19/2) - P(T^alpha > 19/2) < -1/300000.
  This is proved in exact rational arithmetic.
- For reversible modulators the order holds.
- On a directed six-cycle, the mean of He's distribution X_x is strictly concave near x = 0, so it is not convex in
  the speed of modulation.

## Files

- `paper/main.pdf`, `paper/main.tex`: the paper.
- `code/checks.py`: every number in the paper. It has two parts.
  - Exact uniformization certificates for the 12-state and 24-state cycles, using `fractions.Fraction`.
  - Numerical checks: the cosine family and its threshold n >= 12, the square-wave Cox process, random reversible
    tests, and the six-cycle convexity gap.
  - Output is in `code/checks_output.txt`. Run `python checks.py` (numpy, scipy, mpmath, sympy).
- `lean/MMPP.lean`: the certificate inequality of Theorem 1.3, cleared of denominators and stated over the integers,
  and the value f''(0) = -1 of Proposition 5.1. Both are proved in Lean 4 with Mathlib by `decide +kernel`.
