#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trinomial Lattice Execution / Risk Simulator (with Extremes & Median)
---------------------------------------------------------------------

Features
- Strategy pattern for branching.
- Recombinant trinomial tree (m=1, d=1/u), equal branch probabilities.
- Exact trinomial coefficients via dynamic programming.
- Path enumeration (3^T), proceeds & P&L for a selling schedule.
- Tree plotting helper.
- NEW: Compute Best / Median / Worst paths by proceeds (and show them always).
- NEW: Option to print only the first N paths while still computing extremes/median.

Examples
--------
1) Enumerate and print only first 20 rows, but also show extremes+median
python trinomial_lattice_sim.py \
  --S0 100 --u 1.10 --start-t 1 --total-shares 500 --lot-size 100 \
  --cost-basis 200 --print-first 20 --plot

2) Save full results to CSV (may be large if steps big)
python trinomial_lattice_sim.py \
  --S0 100 --u 1.10 --start-t 1 --total-shares 500 --lot-size 100 \
  --cost-basis 200 --save-csv trinomial_paths.csv
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import itertools
import argparse

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
class UniformRecombiningTrinomial(TrinomialStrategy):
    """Recombining lattice with m=1 and d=1/u; equal branch probabilities (1/3 each).
    Node index j changes by +1 (up), 0 (mid), -1 (down).
    """
    u: float  # up multiplier per step (>0)

    def branches(self) -> List[Tuple[float, float, int]]:
        p = 1.0 / 3.0
        return [
            (self.u, p, +1),       # up
            (1.0, p, 0),           # mid
            (1.0 / self.u, p, -1)  # down
        ]


# =========================
# Tree / Nodes
# =========================

@dataclass(frozen=True)
class Node:
    t: int   # time step
    j: int   # net up - down index


class TrinomialTree:
    """Recombining trinomial tree under m=1, d=1/u, with equal probabilities.

    Node value: S(t, j) = S0 * u^j
    Probability mass at (t, j): T(t, j) * (1/3)^t, where T(t, j) are trinomial coefficients.
    """
    def __init__(self, S0: float, strategy: UniformRecombiningTrinomial, steps: int):
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if strategy.u <= 0:
            raise ValueError("u must be positive")
        self.S0 = float(S0)
        self.strategy = strategy
        self.steps = int(steps)
        self._nodes_by_level: Dict[int, List[Node]] = {}
        self._build()

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

    def nodes_at(self, t: int) -> List[Node]:
        return self._nodes_by_level.get(t, [])

    # ------- node math -------
    def S(self, node: Node) -> float:
        return self.S0 * (self.strategy.u ** node.j)

    def prob_mass(self, node: Node) -> float:
        t, j = node.t, node.j
        return self.trinomial_coeff(t, j) * (1.0 / 3.0) ** t

    @staticmethod
    def trinomial_coeff(n: int, j: int) -> int:
        """Number of distinct paths to node (n, j) in a recombining trinomial lattice.
        Uses DP for robustness.
        """
        size = 2 * n + 1
        offset = n
        T = np.zeros((n + 1, size), dtype=np.int64)
        T[0, offset] = 1
        for t in range(1, n + 1):
            for jj in range(-t, t + 1):
                total = 0
                # from (t-1, jj-1) via up
                if -(t - 1) <= jj - 1 <= (t - 1):
                    total += T[t - 1, offset + (jj - 1)]
                # from (t-1, jj) via mid
                if -(t - 1) <= jj <= (t - 1):
                    total += T[t - 1, offset + jj]
                # from (t-1, jj+1) via down
                if -(t - 1) <= jj + 1 <= (t - 1):
                    total += T[t - 1, offset + (jj + 1)]
                T[t, offset + jj] = total
        return int(T[n, offset + j])


# =========================
# Selling schedule & path proceeds
# =========================

@dataclass
class SellingSchedule:
    start_t: int            # first selling time (e.g., 1 to skip t=0; 0 to sell immediately)
    total_shares: int       # total shares to sell
    lot_size: int           # shares per tranche; must divide total_shares

    def sale_times(self) -> List[int]:
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.total_shares <= 0:
            raise ValueError("total_shares must be positive")
        if self.total_shares % self.lot_size != 0:
            raise ValueError("total_shares must be divisible by lot_size")
        N = self.total_shares // self.lot_size
        return list(range(self.start_t, self.start_t + N))


def enumerate_paths_and_proceeds(
    S0: float,
    strategy: UniformRecombiningTrinomial,
    steps: int,
    schedule: Optional[SellingSchedule] = None,
    cost_basis_per_share: Optional[float] = None
) -> pd.DataFrame:
    """Enumerate all 3^steps paths (uniform branch probabilities) and compute:
       - price path S1..Ssteps
       - proceeds / P&L under a selling schedule (optional)
    Returns a DataFrame with one row per path.
    """
    alphabet = [('U', strategy.u, +1), ('M', 1.0, 0), ('D', 1.0/strategy.u, -1)]
    paths = list(itertools.product(alphabet, repeat=steps))

    sale_times = schedule.sale_times() if schedule else []
    lot = schedule.lot_size if schedule else 0
    total_shares = schedule.total_shares if schedule else 0

    records: List[Dict] = []
    for tup in paths:
        labels = [x[0] for x in tup]
        mults  = [x[1] for x in tup]

        # Build S1..Ssteps
        prices = []
        running = S0
        for m in mults:
            running *= m
            prices.append(running)

        # Uniform path probability
        path_prob = (1.0 / 3.0) ** steps

        rec: Dict = {"path": ''.join(labels),
                     **{f"S{k}": prices[k-1] for k in range(1, steps+1)},
                     "probability": path_prob}

        # Proceeds / P&L
        if schedule is not None:
            proceeds = 0.0
            for t in sale_times:
                if t == 0:
                    proceeds += lot * S0
                elif 1 <= t <= steps:
                    proceeds += lot * prices[t-1]
            rec["proceeds"] = proceeds
            if cost_basis_per_share is not None:
                rec["P&L"] = proceeds - total_shares * cost_basis_per_share

        records.append(rec)

    return pd.DataFrame.from_records(records)


# =========================
# Analytics (Expectation)
# =========================

def expected_proceeds_formula(S0: float, u: float, schedule: SellingSchedule) -> float:
    """E[proceeds] with uniform per-branch probabilities under recombining lattice.

    For uniform p_u=p_m=p_d=1/3, one-step expected multiplier is
        mu = (u + 1 + 1/u)/3
    and E[S_t] = S0 * mu^t.

    E[Proceeds] = sum_{t in sale_times} lot * E[S_t].
    """
    mu = (u + 1.0 + 1.0/u) / 3.0
    proceeds = 0.0
    for t in schedule.sale_times():
        if t == 0:
            proceeds += schedule.lot_size * S0
        else:
            proceeds += schedule.lot_size * (S0 * (mu ** t))
    return proceeds


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
# Extremes & Median helpers
# =========================

def extremes_and_median(df: pd.DataFrame, key: str = "proceeds") -> Dict[str, pd.Series]:
    """Return dict with rows for 'worst', 'median', 'best' according to the given key."""
    if key not in df.columns:
        raise ValueError(f"Column '{key}' not found in DataFrame.")
    sorted_df = df.sort_values(by=key, kind="mergesort").reset_index(drop=True)
    n = len(sorted_df)
    worst_row = sorted_df.iloc[0]
    best_row = sorted_df.iloc[-1]
    median_row = sorted_df.iloc[n // 2]  # 50th percentile (uniform paths)
    return {"worst": worst_row, "median": median_row, "best": best_row}


def print_extremes(rows: Dict[str, pd.Series], key: str = "proceeds") -> None:
    def fmt(label, row):
        pl_part = f", P&L={row['P&L']:.4f}" if "P&L" in row else ""
        print(f"{label:<7} path={row['path']} | {key}={row[key]:.4f}{pl_part}")
    print("== Best / Median / Worst (by proceeds) ==")
    fmt("Worst", rows["worst"])
    fmt("Median", rows["median"])
    fmt("Best", rows["best"])


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Trinomial Lattice Execution / Risk Simulator")
    ap.add_argument("--S0", type=float, required=True, help="Initial price (start of simulation)")
    ap.add_argument("--u", type=float, required=True, help="Up multiplier per step (m=1, d=1/u)")
    ap.add_argument("--start-t", type=int, default=1, help="First selling time (0 to sell immediately)")
    ap.add_argument("--total-shares", type=int, required=True, help="Total shares to sell")
    ap.add_argument("--lot-size", type=int, required=True, help="Shares per tranche (must divide total)")
    ap.add_argument("--cost-basis", type=float, default=None, help="Cost basis per share to compute P&L")
    ap.add_argument("--steps", type=int, default=None,
                    help="Total steps to simulate (auto from selling schedule if omitted)")
    ap.add_argument("--save-csv", type=str, default=None, help="Path to save paths/proceeds CSV")
    ap.add_argument("--plot", action="store_true", help="Draw the recombining trinomial tree")
    ap.add_argument("--annotate-prob", action="store_true",
                    help="Annotate node probability mass on the tree")
    ap.add_argument("--print-first", type=int, default=0,
                    help="If >0, print only the first N paths (but extremes/median are still shown)")
    args = ap.parse_args()

    # Build schedule
    schedule = SellingSchedule(
        start_t=args.start_t,
        total_shares=args.total_shares,
        lot_size=args.lot_size,
    )
    sale_times = schedule.sale_times()
    auto_steps = sale_times[-1] if sale_times else 0
    steps = args.steps if args.steps is not None else auto_steps

    strategy = UniformRecombiningTrinomial(u=args.u)

    # Tree (for plotting)
    tree = TrinomialTree(S0=args.S0, strategy=strategy, steps=steps)

    # Enumerate all paths & proceeds
    df = enumerate_paths_and_proceeds(
        S0=args.S0,
        strategy=strategy,
        steps=steps,
        schedule=schedule,
        cost_basis_per_share=args.cost_basis
    )

    # Always compute and print extremes/median (by proceeds)
    rows = extremes_and_median(df, key="proceeds")
    print_extremes(rows, key="proceeds")

    # Print a preview (first N)
    cols = ["path"] + [f"S{k}" for k in range(1, steps + 1)] + ["proceeds", "P&L", "probability"]
    cols = [c for c in cols if c in df.columns]
    n_print = args.print_first if args.print_first and args.print_first > 0 else len(df)
    print("\n== Preview paths ==")
    with pd.option_context("display.width", 160, "display.max_columns", None, "display.max_rows", n_print):
        print(df[cols].head(n_print).round(6))

    # Expected proceeds (formula) for a quick check
    exp_proc = expected_proceeds_formula(args.S0, args.u, schedule)
    if args.cost_basis is not None:
        exp_pl = exp_proc - args.total_shares * args.cost_basis
        print(f"\nE[Proceeds] (formula) = {exp_proc:.6f} | E[P&L] (formula) = {exp_pl:.6f}")
    else:
        print(f"\nE[Proceeds] (formula) = {exp_proc:.6f}")

    # Save CSV (optionally limit to preview size or save full)
    if args.save_csv:
        df[cols].round(6).to_csv(args.save_csv, index=False)
        print(f"Saved CSV to: {args.save_csv}")

    # Plot (optional)
    if args.plot:
        draw_recombining_trinomial_tree(tree, annotate_prob=args.annotate_prob)


if __name__ == "__main__":
    main()
