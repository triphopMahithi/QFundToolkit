#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trinomial Lattice Execution / Risk Simulator
(with adjustable probabilities, closed-form Var, and reporting of std/CV/intervals)
-----------------------------------------------------------------------------------

Adds (vs. previous version):
- Adjustable non-equal probabilities p_u, p_m, p_d (constant over time; IID, no dynamics).
- Closed-form E[Proceeds] and Var[Proceeds] under IID steps.
- Enumeration now weights each path by its (p_u, p_m, p_d) product.
- Optional fractional selling via explicit b-list (e.g., 0.31, 0.25, ...).
- NEW: Report standard deviation (std), coefficient of variation (CV), and confidence intervals:
    * Normal-approx interval: mean ± z * std (z from --ci-level, default 0.95)
    * Probability-weighted quantile interval (empirical) at the same levels

Examples
--------
# Non-equal probabilities and fractional sales (3 tranches: 0.31, 0.25, 0.44)
prob_python trinomial_lattice_sim.py \
  --S0 100 --u 1.10 --p-u 0.45 --p-m 0.35 --p-d 0.20 \
  --start-t 1 --b-list 0.31,0.25,0.44 --cost-basis 200 \
  --print-first 10 --plot

# Equal probabilities, uniform selling, show 95% intervals
prob_python trinomial_lattice_sim.py \
  --S0 100 --u 1.10 \
  --start-t 1 --total-shares 500 --lot-size 100 \
  --cost-basis 200 --plot --print-first 20 --ci-level 0.95
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import itertools
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Strategy Pattern
# =========================

class TrinomialStrategy:
    """Abstract branching strategy for a trinomial step.

    Returns a list of branches, each as:
        (multiplier, probability, delta_j)
    where delta_j ∈ {+1, 0, -1} for up/mid/down transitions in a recombining tree.
    """
    def branches(self) -> List[Tuple[float, float, int]]:
        raise NotImplementedError


@dataclass
class RecombiningTrinomial(TrinomialStrategy):
    """Recombining lattice with m=1 and d=1/u; probabilities can be non-equal.

    Node index j changes by +1 (up), 0 (mid), -1 (down).
    """
    u: float
    p_u: float
    p_m: float
    p_d: float

    def __post_init__(self):
        if self.u <= 0:
            raise ValueError("u must be positive")
        if min(self.p_u, self.p_m, self.p_d) < 0:
            raise ValueError("probabilities must be non-negative")
        s = self.p_u + self.p_m + self.p_d
        if s <= 0:
            raise ValueError("sum of probabilities must be > 0")
        # normalize to sum 1
        self.p_u /= s
        self.p_m /= s
        self.p_d /= s

    def branches(self) -> List[Tuple[float, float, int]]:
        return [
            (self.u,             self.p_u, +1),  # up
            (1.0,                self.p_m,  0),  # mid
            (1.0 / self.u,       self.p_d, -1)   # down
        ]


# =========================
# Tree / Nodes
# =========================

@dataclass(frozen=True)
class Node:
    t: int   # time step
    j: int   # net up - down index


class TrinomialTree:
    """Recombining trinomial tree under m=1, d=1/u.

    Node value: S(t, j) = S0 * u^j

    With constant (p_u, p_m, p_d), the probability mass at (t, j) satisfies:
        P_{t+1}(j) = p_u * P_t(j-1) + p_m * P_t(j) + p_d * P_t(j+1),  P_0(0)=1
    """
    def __init__(self, S0: float, strategy: RecombiningTrinomial, steps: int):
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self.S0 = float(S0)
        self.strategy = strategy
        self.steps = int(steps)
        self._nodes_by_level: Dict[int, List[Node]] = {}
        self._prob_levels: Dict[int, Dict[int, float]] = {}  # P_t(j)
        self._build()
        self._build_probabilities()

    # ------- build / access -------
    def _build(self) -> None:
        self._nodes_by_level[0] = [Node(0, 0)]
        for t in range(1, self.steps + 1):
            prev = self._nodes_by_level[t - 1]
            js = set()
            for node in prev:
                for _, _, dj in self.strategy.branches():
                    js.add(node.j + dj)
            self._nodes_by_level[t] = [Node(t, j) for j in sorted(js)]

    def _build_probabilities(self) -> None:
        # DP for P_t(j)
        self._prob_levels[0] = {0: 1.0}
        p_u, p_m, p_d = self.strategy.p_u, self.strategy.p_m, self.strategy.p_d
        for t in range(1, self.steps + 1):
            prev = self._prob_levels[t - 1]
            cur: Dict[int, float] = {}
            # possible j range is [-t, +t]
            for j in range(-t, t + 1):
                prob = 0.0
                prob += p_u * prev.get(j - 1, 0.0)
                prob += p_m * prev.get(j, 0.0)
                prob += p_d * prev.get(j + 1, 0.0)
                if prob != 0.0:
                    cur[j] = prob
            self._prob_levels[t] = cur

    def nodes_at(self, t: int) -> List[Node]:
        return self._nodes_by_level.get(t, [])

    def S(self, node: Node) -> float:
        return self.S0 * (self.strategy.u ** node.j)

    def prob_mass(self, node: Node) -> float:
        return self._prob_levels.get(node.t, {}).get(node.j, 0.0)


# =========================
# Selling schedule (supports fractional)
# =========================

@dataclass
class SellingSchedule:
    start_t: int                           # first selling time (e.g., 1 to skip t=0; 0 to sell immediately)
    total_shares: Optional[float] = None   # optional if using b_list
    lot_size: Optional[float] = None       # optional if using b_list
    b_list: Optional[List[float]] = None   # explicit amounts per sale (fractional allowed)

    def sale_times(self) -> List[int]:
        if self.b_list is not None:
            N = len(self.b_list)
            if N <= 0:
                raise ValueError("b_list must contain at least one amount")
            return list(range(self.start_t, self.start_t + N))
        # uniform lots
        if self.lot_size is None or self.total_shares is None:
            raise ValueError("For uniform lots, total_shares and lot_size must be provided")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.total_shares <= 0:
            raise ValueError("total_shares must be positive")
        # determine N ~ total_shares / lot_size
        N_float = self.total_shares / self.lot_size
        N = int(round(N_float))
        if not math.isclose(N_float, N, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("total_shares must be an integer multiple of lot_size (or use b_list)")
        return list(range(self.start_t, self.start_t + N))

    def amounts(self) -> List[float]:
        if self.b_list is not None:
            return list(self.b_list)
        # uniform lots
        N = len(self.sale_times())
        return [float(self.lot_size)] * N

    def total_amount(self) -> float:
        return float(sum(self.amounts()))


# =========================
# Path enumeration (with non-equal p)
# =========================

def enumerate_paths_and_proceeds(
    S0: float,
    strategy: RecombiningTrinomial,
    steps: int,
    schedule: Optional[SellingSchedule] = None,
    cost_basis_per_share: Optional[float] = None
) -> pd.DataFrame:
    """Enumerate all 3^steps paths and compute:
       - price path S1..Ssteps
       - proceeds / P&L under a selling schedule (optional)
       Path probabilities use (p_u, p_m, p_d) per step (IID).
    """
    alphabet = [('U', strategy.u, +1, strategy.p_u),
                ('M', 1.0,        0, strategy.p_m),
                ('D', 1.0/strategy.u, -1, strategy.p_d)]
    paths = list(itertools.product(alphabet, repeat=steps))

    sale_times = schedule.sale_times() if schedule else []
    amounts = schedule.amounts() if schedule else []
    total_amount = schedule.total_amount() if schedule else 0.0

    records: List[Dict] = []
    for tup in paths:
        labels = [x[0] for x in tup]
        mults  = [x[1] for x in tup]
        probs  = [x[3] for x in tup]

        # Build S1..Ssteps
        prices = []
        running = S0
        for m in mults:
            running *= m
            prices.append(running)

        # Path probability
        path_prob = 1.0
        for p in probs:
            path_prob *= p

        rec: Dict = {"path": ''.join(labels),
                     **{f"S{k}": prices[k-1] for k in range(1, steps+1)},
                     "probability": path_prob}

        # Proceeds / P&L
        if schedule is not None:
            proceeds = 0.0
            for idx, t in enumerate(sale_times):
                amt = amounts[idx]
                if t == 0:
                    proceeds += amt * S0
                elif 1 <= t <= steps:
                    proceeds += amt * prices[t-1]
                # else (t>steps): ignore
            rec["proceeds"] = proceeds
            if cost_basis_per_share is not None:
                rec["P&L"] = proceeds - total_amount * cost_basis_per_share

        records.append(rec)

    return pd.DataFrame.from_records(records)


# =========================
# Closed-form E and Var (IID steps, constant p)
# =========================

def expected_and_variance_proceeds_closed_form(
    S0: float,
    u: float,
    schedule: SellingSchedule,
    p_u: float,
    p_m: float,
    p_d: float
) -> Tuple[float, float]:
    """Compute E[Proceeds] and Var[Proceeds] for recombining trinomial with m=1, d=1/u,
    constant probabilities (p_u, p_m, p_d), IID steps.

    Let M ∈ {u, 1, 1/u} with probs p_u, p_m, p_d.
    μ1 = E[M], μ2 = E[M^2].
    E[S_t] = S0 * μ1^t
    E[S_s S_t] = S0^2 * μ2^{min(s,t)} * μ1^{|t - s|}

    Proceeds = sum_{i} b_i S_{t_i},  t_i ∈ sale_times
    Var = sum_{i,j} b_i b_j (E[S_{t_i} S_{t_j}] - E[S_{t_i}] E[S_{t_j}])
    """
    # normalize probs
    if min(p_u, p_m, p_d) < 0:
        raise ValueError("probabilities must be non-negative")
    s = p_u + p_m + p_d
    if s <= 0:
        raise ValueError("sum of probabilities must be > 0")
    p_u, p_m, p_d = p_u/s, p_m/s, p_d/s

    μ1 = p_u * u + p_m * 1.0 + p_d * (1.0 / u)
    μ2 = p_u * (u**2) + p_m * (1.0) + p_d * ((1.0/u)**2)

    times = schedule.sale_times()
    amounts = schedule.amounts()

    # E[Proceeds]
    E = 0.0
    for t, b in zip(times, amounts):
        if t == 0:
            E += b * S0
        else:
            E += b * (S0 * (μ1 ** t))

    # Var[Proceeds]
    Var = 0.0
    for ti, bi in zip(times, amounts):
        E_sti = S0 if ti == 0 else S0 * (μ1 ** ti)
        for tj, bj in zip(times, amounts):
            E_stj = S0 if tj == 0 else S0 * (μ1 ** tj)
            m = min(ti, tj)
            k = abs(ti - tj)
            E_sti_stj = (S0**2) * (μ2 ** m) * (μ1 ** k)
            Cov = E_sti_stj - E_sti * E_stj
            Var += bi * bj * Cov

    return E, Var


# =========================
# Extremes & Median helpers (by proceeds)
# =========================

def extremes_and_median(df: pd.DataFrame, key: str = "proceeds") -> Dict[str, pd.Series]:
    if key not in df.columns:
        raise ValueError(f"Column '{key}' not found in DataFrame.")
    sorted_df = df.sort_values(by=key, kind="mergesort").reset_index(drop=True)
    n = len(sorted_df)
    worst_row = sorted_df.iloc[0]
    best_row = sorted_df.iloc[-1]
    median_row = sorted_df.iloc[n // 2]  # unweighted median by proceeds
    return {"worst": worst_row, "median": median_row, "best": best_row}


def print_extremes(rows: Dict[str, pd.Series], key: str = "proceeds") -> None:
    def fmt(label, row):
        pl_part = f", P&L={row['P&L']:.6f}" if "P&L" in row else ""
        print(f"{label:<7} path={row['path']} | {key}={row[key]:.6f}{pl_part} | prob={row['probability']:.6f}")
    print("== Best / Median / Worst (by proceeds) ==")
    fmt("Worst", rows["worst"])
    fmt("Median", rows["median"])
    fmt("Best", rows["best"])


# =========================
# Interval helpers (normal approx & weighted quantiles)
# =========================

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def normal_ppf(p: float, lo: float = -10.0, hi: float = 10.0, iters: int = 80) -> float:
    """Inverse CDF of standard normal via bisection."""
    if not (0.0 < p < 1.0):
        if p <= 0.0: return float("-inf")
        if p >= 1.0: return float("inf")
    a, b = lo, hi
    for _ in range(iters):
        m = 0.5 * (a + b)
        if normal_cdf(m) < p:
            a = m
        else:
            b = m
    return 0.5 * (a + b)

def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: List[float]) -> List[float]:
    """Probability-weighted quantiles for 1D array."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    out = []
    for q in qs:
        idx = np.searchsorted(cw, q, side="left")
        idx = min(idx, len(v) - 1)
        out.append(float(v[idx]))
    return out


# =========================
# Visualization
# =========================

def draw_recombining_trinomial_tree(tree: TrinomialTree, annotate_prob: bool = False) -> None:
    """Draw a recombining trinomial tree; node label shows S(t,j) (and probability mass if asked)."""
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []

    for t in range(0, tree.steps + 1):
        for node in tree.nodes_at(t):
            xs.append(t)
            ys.append(node.j)
            if annotate_prob and t > 0:
                lab = f"{tree.S(node):.2f}\nπ={tree.prob_mass(node):.3f}"
            else:
                lab = f"{tree.S(node):.2f}"
            labels.append(lab)

    plt.figure()
    # edges
    for t in range(0, tree.steps):
        for node in tree.nodes_at(t):
            x0, y0 = t, node.j
            for _, _, dj in tree.strategy.branches():
                x1, y1 = t + 1, node.j + dj
                plt.plot([x0, x1], [y0, y1], linewidth=1)

    # nodes
    plt.scatter(xs, ys, s=40)
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y, lab, ha='center', va='bottom', fontsize=8)

    plt.title("Recombining Trinomial Tree (node shows S(t,j))")
    plt.xlabel("Time step t")
    plt.ylabel("Index j (net up - down)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Trinomial Lattice Execution / Risk Simulator")
    ap.add_argument("--S0", type=float, required=True, help="Initial price (start of simulation)")
    ap.add_argument("--u", type=float, required=True, help="Up multiplier per step (m=1, d=1/u)")

    # Probabilities (non-equal allowed). Will be normalized internally to sum 1.
    ap.add_argument("--p-u", type=float, default=1/3, dest="p_u", help="Probability of Up")
    ap.add_argument("--p-m", type=float, default=1/3, dest="p_m", help="Probability of Mid")
    ap.add_argument("--p-d", type=float, default=1/3, dest="p_d", help="Probability of Down")

    ap.add_argument("--start-t", type=int, default=1, help="First selling time (0 to sell immediately)")

    # Two ways to specify selling amounts:
    ap.add_argument("--total-shares", type=float, default=None, help="Total amount to sell (omit if using --b-list)")
    ap.add_argument("--lot-size", type=float, default=None, help="Amount per tranche (omit if using --b-list)")
    ap.add_argument("--b-list", type=str, default=None,
                    help="Comma-separated amounts per sale (e.g., '0.31,0.25,0.44'); overrides total/lot")

    ap.add_argument("--cost-basis", type=float, default=None, help="Cost basis per share to compute P&L")
    ap.add_argument("--steps", type=int, default=None,
                    help="Total steps to simulate (auto from selling schedule if omitted)")
    ap.add_argument("--save-csv", type=str, default=None, help="Path to save paths/proceeds CSV")
    ap.add_argument("--plot", action="store_true", help="Draw the recombining trinomial tree")
    ap.add_argument("--annotate-prob", action="store_true",
                    help="Annotate node probability mass on the tree")
    ap.add_argument("--print-first", type=int, default=0,
                    help="If >0, print only the first N paths (but extremes/median are still shown)")
    ap.add_argument("--ci-level", type=float, default=0.95,
                    help="Confidence level for intervals (normal approx & weighted quantile)")

    args = ap.parse_args()

    # Parse selling schedule
    b_list = None
    if args.b_list:
        b_list = [float(x.strip()) for x in args.b_list.split(",") if x.strip() != ""]

    schedule = SellingSchedule(
        start_t=args.start_t,
        total_shares=args.total_shares if b_list is None else None,
        lot_size=args.lot_size if b_list is None else None,
        b_list=b_list
    )

    sale_times = schedule.sale_times()
    auto_steps = sale_times[-1] if sale_times else 0
    steps = args.steps if args.steps is not None else auto_steps

    # Strategy with (possibly) non-equal probabilities
    strategy = RecombiningTrinomial(u=args.u, p_u=args.p_u, p_m=args.p_m, p_d=args.p_d)

    # Tree (for plotting/prob-mass on nodes)
    tree = TrinomialTree(S0=args.S0, strategy=strategy, steps=steps)

    # Enumerate all paths & proceeds
    df = enumerate_paths_and_proceeds(
        S0=args.S0,
        strategy=strategy,
        steps=steps,
        schedule=schedule,
        cost_basis_per_share=args.cost_basis
    )

    # Best/Median/Worst (by proceeds)
    rows = extremes_and_median(df, key="proceeds")
    print_extremes(rows, key="proceeds")

    # Preview (first N)
    cols = ["path"] + [f"S{k}" for k in range(1, steps + 1)] + ["proceeds", "P&L", "probability"]
    cols = [c for c in cols if c in df.columns]
    n_print = args.print_first if args.print_first and args.print_first > 0 else len(df)
    print("\n== Preview paths ==")
    with pd.option_context("display.width", 160, "display.max_columns", None, "display.max_rows", n_print):
        print(df[cols].head(n_print).round(6))

    # Closed-form E and Var (IID steps, constant probabilities)
    E_formula, Var_formula = expected_and_variance_proceeds_closed_form(
        S0=args.S0, u=args.u, schedule=schedule,
        p_u=strategy.p_u, p_m=strategy.p_m, p_d=strategy.p_d
    )
    std_formula = math.sqrt(max(Var_formula, 0.0))
    cv_formula = (std_formula / E_formula) if E_formula != 0 else float("nan")

    # Enumeration-based E and Var (weighted by path probability)
    E_enum = float((df["proceeds"] * df["probability"]).sum())
    Var_enum = float(((df["proceeds"] - E_enum) ** 2 * df["probability"]).sum())
    std_enum = math.sqrt(max(Var_enum, 0.0))
    cv_enum = (std_enum / E_enum) if E_enum != 0 else float("nan")

    # Confidence intervals
    level = args.ci_level
    level = min(max(level, 0.0), 1.0)
    tail = (1.0 - level) / 2.0
    z = normal_ppf(1.0 - tail) if 0.0 < level < 1.0 else float("inf")

    # Normal-approx interval for proceeds (enum mean & std)
    ci_norm_lo = E_enum - z * std_enum
    ci_norm_hi = E_enum + z * std_enum

    # Weighted-quantile interval for proceeds
    qs = [tail, 0.5, 1.0 - tail] if 0.0 < level < 1.0 else [0.5]
    wq = weighted_quantiles(df["proceeds"].to_numpy(), df["probability"].to_numpy(), qs)
    if len(wq) == 3:
        q_lo, q_med, q_hi = wq
    else:
        q_lo = q_hi = float("nan")
        q_med = wq[0]

    # If cost basis provided, shift intervals to P&L
    if args.cost_basis is not None:
        total_amt = schedule.total_amount()
        const_shift = total_amt * args.cost_basis
        E_PL_formula = E_formula - const_shift
        E_PL_enum = E_enum - const_shift
        ci_norm_lo_PL = ci_norm_lo - const_shift
        ci_norm_hi_PL = ci_norm_hi - const_shift
        q_lo_PL = q_lo - const_shift if not math.isnan(q_lo) else float("nan")
        q_med_PL = q_med - const_shift
        q_hi_PL = q_hi - const_shift if not math.isnan(q_hi) else float("nan")

    # ---- REPORT ----
    print("\n== Moments & Intervals (Proceeds) ==")
    print(f"E[Proceeds] (formula) = {E_formula:.6f} | Var (formula) = {Var_formula:.6f} | std = {std_formula:.6f} | CV = {cv_formula:.6%}")
    print(f"E[Proceeds] (enum)    = {E_enum:.6f} | Var (enum)    = {Var_enum:.6f} | std = {std_enum:.6f} | CV = {cv_enum:.6%}")

    if 0.0 < level < 1.0:
        print(f"\nNormal-approx {int(level*100)}% CI (Proceeds): [{ci_norm_lo:.6f}, {ci_norm_hi:.6f}] (z ≈ {z:.6f})")
        print(f"Weighted-quantile {int(level*100)}% interval (Proceeds): [{q_lo:.6f}, {q_hi:.6f}]  | median ≈ {q_med:.6f}")

    if args.cost_basis is not None:
        print("\n== Moments & Intervals (P&L) ==")
        print(f"E[P&L] (formula) = {E_PL_formula:.6f} | E[P&L] (enum) = {E_PL_enum:.6f}")
        if 0.0 < level < 1.0:
            print(f"Normal-approx {int(level*100)}% CI (P&L): [{ci_norm_lo_PL:.6f}, {ci_norm_hi_PL:.6f}]")
            print(f"Weighted-quantile {int(level*100)}% interval (P&L): [{q_lo_PL:.6f}, {q_hi_PL:.6f}]  | median ≈ {q_med_PL:.6f}")

    # Save CSV
    if args.save_csv:
        df[cols].round(6).to_csv(args.save_csv, index=False)
        print(f"\nSaved CSV to: {args.save_csv}")

    # Plot
    if args.plot:
        draw_recombining_trinomial_tree(tree, annotate_prob=args.annotate_prob)


if __name__ == "__main__":
    main()
