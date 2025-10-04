#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DP-based Trinomial Lattice Execution / Risk Simulator (Dynamic Probabilities)
-----------------------------------------------------------------------------

- Recombinant trinomial lattice with m=1, d=1/u  =>  S(t,j) = S0 * u^j.
- Dynamic branch probabilities p_u(t,j), p_m(t,j), p_d(t,j) via a pluggable ProbModel:
    * ConstantProbModel: fixed (p_u, p_m, p_d).
    * SoftmaxAffineProbModel: logits = a0 + at*t + aj*j per branch, softmax -> probs.
- Pure Dynamic Programming (no 3^T enumeration):
    * Tracks P_t(j) = Prob[J_t=j], and conditional moments:
         EcondW_t(j)   = E[W_t | J_t=j]
         EcondWW_t(j)  = E[W_t^2 | J_t=j]
      where W_t = sum_{s<=t} b_s S_s (selling schedule with real amounts b_s).
    * Also tracks min/max proceeds achievable (pathwise) via DP.
- Computes:
    * E[Proceeds], Var[Proceeds], std, CV
    * Normal-approx confidence interval for Proceeds (and for P&L if cost basis is given)
    * Min/Max proceeds (pathwise extremes, no probabilities)
- Supports fractional selling:
    * Either uniform tranches: total_shares & lot_size
    * Or explicit list: --b-list "0.31,0.25,0.44"
- Optional tree plot (node labels are S(t,j); set --annotate-prob to show P_t(j) at nodes)

Example (constant probabilities):
  python dp_trinomial_dynamic.py \
    --S0 100 --u 1.10 --p-u 0.45 --p-m 0.35 --p-d 0.20 \
    --start-t 1 --total-shares 500 --lot-size 100 \
    --cost-basis 200 --ci-level 0.95 --plot --annotate-prob

Example (dynamic softmax-affine probabilities):
  python dp_trinomial_dynamic.py \
    --S0 100 --u 1.10 \
    --prob-model softmax-affine \
    --u-logit "0.1,0.00, 0.05"  --m-logit "0.0,0.00, 0.00"  --d-logit "0.1,0.00,-0.05" \
    --start-t 1 --b-list "100,100,100,100,100" \
    --cost-basis 200 --ci-level 0.95 --plot --annotate-prob
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt


# ===============================
# Selling schedule (fractional OK)
# ===============================

@dataclass
class SellingSchedule:
    start_t: int                           # first selling time (0 means sell immediately at S0)
    total_shares: Optional[float] = None   # optional if using b_list
    lot_size: Optional[float] = None       # optional if using b_list
    b_list: Optional[List[float]] = None   # explicit amounts per sale (fractional allowed)

    def sale_times(self) -> List[int]:
        if self.b_list is not None:
            N = len(self.b_list)
            if N <= 0:
                raise ValueError("b_list must contain at least one amount")
            return list(range(self.start_t, self.start_t + N))
        if self.lot_size is None or self.total_shares is None:
            raise ValueError("Provide total_shares & lot_size, or use b_list")
        if self.lot_size <= 0 or self.total_shares <= 0:
            raise ValueError("Amounts must be positive")
        N_float = self.total_shares / self.lot_size
        N = int(round(N_float))
        if not math.isclose(N_float, N, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("total_shares must be a multiple of lot_size (or use b_list)")
        return list(range(self.start_t, self.start_t + N))

    def amounts(self) -> List[float]:
        if self.b_list is not None:
            return list(self.b_list)
        N = len(self.sale_times())
        return [float(self.lot_size)] * N

    def total_amount(self) -> float:
        return float(sum(self.amounts()))


# ===============================
# Probability models p(t,j)
# ===============================

class ProbModel:
    """Interface: return (p_u, p_m, p_d) for given (t, j)."""
    def probs(self, t: int, j: int) -> Tuple[float, float, float]:
        raise NotImplementedError


@dataclass
class ConstantProbModel(ProbModel):
    p_u: float
    p_m: float
    p_d: float
    def __post_init__(self):
        if min(self.p_u, self.p_m, self.p_d) < 0:
            raise ValueError("Probabilities must be non-negative")
        s = self.p_u + self.p_m + self.p_d
        if s <= 0:
            raise ValueError("Sum of probabilities must be > 0")
        self.p_u /= s; self.p_m /= s; self.p_d /= s
    def probs(self, t: int, j: int) -> Tuple[float, float, float]:
        return (self.p_u, self.p_m, self.p_d)


@dataclass
class SoftmaxAffineProbModel(ProbModel):
    """
    Logits per branch: l_k(t,j) = a0_k + at_k * t + aj_k * j,  k ∈ {u,m,d}
    Probabilities via softmax(l_u, l_m, l_d).
    """
    a0_u: float = 0.0; at_u: float = 0.0; aj_u: float = 0.0
    a0_m: float = 0.0; at_m: float = 0.0; aj_m: float = 0.0
    a0_d: float = 0.0; at_d: float = 0.0; aj_d: float = 0.0

    @staticmethod
    def _softmax3(x: Tuple[float, float, float]) -> Tuple[float, float, float]:
        xm = max(x)
        e = [math.exp(xi - xm) for xi in x]
        s = sum(e)
        return (e[0]/s, e[1]/s, e[2]/s)

    def probs(self, t: int, j: int) -> Tuple[float, float, float]:
        lu = self.a0_u + self.at_u * t + self.aj_u * j
        lm = self.a0_m + self.at_m * t + self.aj_m * j
        ld = self.a0_d + self.at_d * t + self.aj_d * j
        return self._softmax3((lu, lm, ld))


# ===============================
# DP Engine
# ===============================

@dataclass
class DpOutputs:
    E_proceeds: float
    Var_proceeds: float
    std_proceeds: float
    CV: float
    min_proceeds: float
    max_proceeds: float
    P_last: Dict[int, float]            # P_T(j)
    nodes_by_level: Dict[int, List[int]]  # j-list per t (for plotting)


def dp_mean_var_extremes(
    S0: float,
    u: float,
    schedule: SellingSchedule,
    prob_model: ProbModel
) -> DpOutputs:
    """
    Dynamic-programming computation of:
      - E[Proceeds], Var[Proceeds], std, CV
      - min/max possible proceeds (pathwise)
    on a recombining trinomial lattice with m=1, d=1/u.

    Recursions:
      P_{t+1}(j') = Σ_{parents j} P_t(j) * p_branch(t,j; j' - j)
      EcondW_{t+1}(j') =
         (Σ [ (EcondW_t(j) + b_{t+1} S(t+1, j')) * P_t(j) * p ] ) / P_{t+1}(j')
      EcondWW_{t+1}(j') =
         (Σ [ (EcondWW_t(j) + 2 b_{t+1} S(t+1,j') EcondW_t(j) + (b_{t+1} S(t+1,j'))^2)
               * P_t(j) * p ] ) / P_{t+1}(j')
      min/max: pathwise DP via parents.
    """
    sale_times = schedule.sale_times()
    amounts    = schedule.amounts()
    T = max(sale_times) if sale_times else 0

    # Accessor for sale amount at time t
    b_at = {t: amt for t, amt in zip(sale_times, amounts)}

    def S_price(t: int, j: int) -> float:
        return S0 * (u ** j)

    # Initialize level t=0
    P_prev: Dict[int, float] = {0: 1.0}
    EW_prev: Dict[int, float] = {0: (b_at.get(0, 0.0) * S0)}
    EWW_prev: Dict[int, float] = {0: (b_at.get(0, 0.0) * S0) ** 2}
    minW_prev: Dict[int, float] = {0: (b_at.get(0, 0.0) * S0)}
    maxW_prev: Dict[int, float] = {0: (b_at.get(0, 0.0) * S0)}
    nodes_by_level: Dict[int, List[int]] = {0: [0]}

    # For reporting E[W_t], E[W_t^2] if needed
    # E_W_prev = sum_j P_prev(j) * EW_prev(j)
    # E_WW_prev = sum_j P_prev(j) * EWW_prev(j)

    for t in range(1, T + 1):
        P_cur: Dict[int, float] = {}
        EW_cur: Dict[int, float] = {}
        EWW_cur: Dict[int, float] = {}
        minW_cur: Dict[int, float] = {}
        maxW_cur: Dict[int, float] = {}

        # current j range
        j_list = list(range(-t, t + 1))
        nodes_by_level[t] = j_list

        # For each child node j', gather contributions from feasible parents
        for jprime in j_list:
            add_amount = b_at.get(t, 0.0) * S_price(t, jprime)

            # parents: jprime from (jprime-1 via Up), (jprime via Mid), (jprime+1 via Down)
            parents = [
                (jprime - 1, +1),  # came via Up from jprime-1
                (jprime,     0),   # came via Mid from jprime
                (jprime + 1, -1),  # came via Down from jprime+1
            ]

            # Accumulators for probability-weighted sums
            prob_sum = 0.0
            num_EW = 0.0
            num_EWW = 0.0

            # For min/max DP
            min_candidates = []
            max_candidates = []

            for j_parent, dj in parents:
                Pj = P_prev.get(j_parent, 0.0)
                if Pj == 0.0:
                    # Even if probability is 0, still relevant for min/max; but min/max are pathwise regardless of prob.
                    # For pathwise min/max we assume branches exist from any node, so include them.
                    pass

                # Branch probability from parent -> this child
                pu, pm, pd = prob_model.probs(t - 1, j_parent)
                if dj == +1:
                    p_trans = pu
                elif dj == 0:
                    p_trans = pm
                else:
                    p_trans = pd

                # --- Probabilities / moments ---
                if Pj > 0.0 and p_trans > 0.0:
                    w_parent = EW_prev[j_parent]
                    ww_parent = EWW_prev[j_parent]
                    flow = Pj * p_trans
                    prob_sum += flow
                    num_EW  += (w_parent + add_amount) * flow
                    num_EWW += (ww_parent + 2.0 * add_amount * w_parent + add_amount * add_amount) * flow

                # --- Min/Max pathwise DP ---
                # If branch is allowed (p_trans > 0), consider parent candidate.
                # If p_trans can be zero for forbidden branch, we skip it in min/max too (edge absent).
                if p_trans > 0.0 and (j_parent in minW_prev):
                    min_candidates.append(minW_prev[j_parent] + add_amount)
                    max_candidates.append(maxW_prev[j_parent] + add_amount)

            # Set probability & conditional moments (guard zero prob)
            if prob_sum > 0.0:
                P_cur[jprime] = prob_sum
                EW_cur[jprime] = num_EW / prob_sum
                EWW_cur[jprime] = num_EWW / prob_sum

            # Set min/max if any valid parent/branch existed
            if min_candidates:
                minW_cur[jprime] = min(min_candidates)
                maxW_cur[jprime] = max(max_candidates)

        # Normalize P_cur to sum ~1 (optional numerical hygiene)
        totalP = sum(P_cur.values())
        if totalP > 0:
            for k in list(P_cur.keys()):
                P_cur[k] /= totalP

        # Step forward
        P_prev, EW_prev, EWW_prev = P_cur, EW_cur, EWW_cur
        minW_prev, maxW_prev = minW_cur, maxW_cur

    # Final time = last sale time
    E_proceeds = sum(P_prev[j] * EW_prev[j] for j in P_prev)
    E2_proceeds = sum(P_prev[j] * EWW_prev[j] for j in P_prev)
    Var_proceeds = max(E2_proceeds - E_proceeds * E_proceeds, 0.0)
    std_proceeds = math.sqrt(Var_proceeds)
    CV = (std_proceeds / E_proceeds) if E_proceeds != 0 else float("nan")
    min_proceeds = min(minW_prev.values()) if minW_prev else 0.0
    max_proceeds = max(maxW_prev.values()) if maxW_prev else 0.0

    return DpOutputs(
        E_proceeds=E_proceeds,
        Var_proceeds=Var_proceeds,
        std_proceeds=std_proceeds,
        CV=CV,
        min_proceeds=min_proceeds,
        max_proceeds=max_proceeds,
        P_last=P_prev,
        nodes_by_level=nodes_by_level
    )


# ===============================
# Plot
# ===============================

def draw_tree(S0: float, u: float, nodes_by_level: Dict[int, List[int]], P_levels: Optional[Dict[int, Dict[int, float]]] = None):
    """Draw the recombining tree; annotate S(t,j). If P_levels is provided, annotate probabilities too."""
    xs, ys, labels = [], [], []

    max_t = max(nodes_by_level.keys())
    # Build helper to get P_t(j) if provided
    def P_at(t: int, j: int) -> float:
        if P_levels is None:
            return float("nan")
        return P_levels.get(t, {}).get(j, 0.0)

    # Nodes
    for t, j_list in nodes_by_level.items():
        for j in j_list:
            xs.append(t)
            ys.append(j)
            S_tj = S0 * (u ** j)
            if P_levels is None or t == 0:
                lab = f"{S_tj:.2f}"
            else:
                lab = f"{S_tj:.2f}\nπ={P_at(t,j):.3f}"
            labels.append(lab)

    # Edges
    plt.figure()
    for t in range(0, max_t):
        for j in nodes_by_level[t]:
            for dj in (+1, 0, -1):
                j2 = j + dj
                if j2 in nodes_by_level.get(t+1, []):
                    plt.plot([t, t+1], [j, j2], linewidth=1)

    # Scatter nodes & labels
    plt.scatter(xs, ys, s=40)
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y, lab, ha='center', va='bottom', fontsize=8)

    plt.title("Recombining Trinomial Tree (S(t,j) shown; π shown if provided)")
    plt.xlabel("Time step t")
    plt.ylabel("Index j (net up - down)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ===============================
# Normal quantiles (for CI)
# ===============================

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def normal_ppf(p: float, lo: float = -10.0, hi: float = 10.0, iters: int = 80) -> float:
    if not (0.0 < p < 1.0):
        return float("-inf") if p <= 0 else float("inf")
    a, b = lo, hi
    for _ in range(iters):
        m = 0.5 * (a + b)
        if normal_cdf(m) < p:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


# ===============================
# CLI
# ===============================

def parse_triplet(arg: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) != 3:
        raise ValueError("Logit triplet must have 3 comma-separated numbers: a0,at,aj")
    return (float(parts[0]), float(parts[1]), float(parts[2]))

def main():
    ap = argparse.ArgumentParser(description="DP-based Trinomial Lattice (Dynamic Probabilities)")
    ap.add_argument("--S0", type=float, required=True, help="Initial price")
    ap.add_argument("--u", type=float, required=True, help="Up multiplier per step (m=1, d=1/u)")

    # Selling schedule
    ap.add_argument("--start-t", type=int, default=1, help="First selling time (0 to sell immediately)")
    ap.add_argument("--total-shares", type=float, default=None, help="Total amount to sell (omit if using --b-list)")
    ap.add_argument("--lot-size", type=float, default=None, help="Amount per tranche (omit if using --b-list)")
    ap.add_argument("--b-list", type=str, default=None,
                    help="Comma-separated amounts per sale (e.g., '100,100,100'); overrides total/lot")

    # Probabilities
    ap.add_argument("--prob-model", type=str, choices=["constant","softmax-affine"], default="constant")
    ap.add_argument("--p-u", type=float, default=1/3, dest="p_u", help="(constant) p_up")
    ap.add_argument("--p-m", type=float, default=1/3, dest="p_m", help="(constant) p_mid")
    ap.add_argument("--p-d", type=float, default=1/3, dest="p_d", help="(constant) p_down")
    ap.add_argument("--u-logit", type=str, default="0,0,0", help="(softmax-affine) 'a0,at,aj' for UP")
    ap.add_argument("--m-logit", type=str, default="0,0,0", help="(softmax-affine) 'a0,at,aj' for MID")
    ap.add_argument("--d-logit", type=str, default="0,0,0", help="(softmax-affine) 'a0,at,aj' for DOWN")

    # Reporting
    ap.add_argument("--cost-basis", type=float, default=None, help="Cost basis per share to report P&L stats")
    ap.add_argument("--ci-level", type=float, default=0.95, help="Normal-approx CI level for Proceeds/P&L")
    ap.add_argument("--plot", action="store_true", help="Draw the lattice with S(t,j) (and π if desired)")
    ap.add_argument("--annotate-prob", action="store_true", help="If plotting, annotate node probabilities")

    args = ap.parse_args()

    # Build schedule
    if args.b_list:
        b_list = [float(x.strip()) for x in args.b_list.split(",") if x.strip() != ""]
        schedule = SellingSchedule(start_t=args.start_t, b_list=b_list)
    else:
        schedule = SellingSchedule(start_t=args.start_t, total_shares=args.total_shares, lot_size=args.lot_size)

    # Prob model
    if args.prob_model == "constant":
        prob_model = ConstantProbModel(p_u=args.p_u, p_m=args.p_m, p_d=args.p_d)
    else:
        au0, aut, auj = parse_triplet(args.u_logit)
        am0, amt, amj = parse_triplet(args.m_logit)
        ad0, adt, adj = parse_triplet(args.d_logit)
        prob_model = SoftmaxAffineProbModel(
            a0_u=au0, at_u=aut, aj_u=auj,
            a0_m=am0, at_m=amt, aj_m=amj,
            a0_d=ad0, at_d=adt, aj_d=adj
        )

    # DP compute
    out = dp_mean_var_extremes(S0=args.S0, u=args.u, schedule=schedule, prob_model=prob_model)

    # Report
    print("== DP Results (Proceeds) ==")
    print(f"E[Proceeds] = {out.E_proceeds:.6f}")
    print(f"Var         = {out.Var_proceeds:.6f}")
    print(f"std         = {out.std_proceeds:.6f}")
    print(f"CV          = {out.CV:.6%}")
    print(f"Min path proceeds = {out.min_proceeds:.6f}")
    print(f"Max path proceeds = {out.max_proceeds:.6f}")

    # Normal-approx CI
    level = max(0.0, min(args.ci_level, 1.0))
    if 0.0 < level < 1.0:
        z = normal_ppf(0.5 + level/2.0)
        lo = out.E_proceeds - z * out.std_proceeds
        hi = out.E_proceeds + z * out.std_proceeds
        print(f"Normal-approx {int(level*100)}% CI (Proceeds): [{lo:.6f}, {hi:.6f}] (z≈{z:.6f})")

    # P&L shift, if cost basis is provided
    if args.cost_basis is not None:
        total_amt = schedule.total_amount()
        E_PL = out.E_proceeds - total_amt * args.cost_basis
        print("\n== DP Results (P&L) ==")
        print(f"E[P&L] = {E_PL:.6f}")
        if 0.0 < level < 1.0:
            lo_pl = lo - total_amt * args.cost_basis
            hi_pl = hi - total_amt * args.cost_basis
            print(f"Normal-approx {int(level*100)}% CI (P&L): [{lo_pl:.6f}, {hi_pl:.6f}]")

    # Plot
    if args.plot:
        # Build P_levels for all t for annotation
        P_levels: Dict[int, Dict[int, float]] = {0: {0: 1.0}}
        sale_ts = schedule.sale_times()
        T = max(sale_ts) if sale_ts else 0

        for t in range(1, T + 1):
            P_levels[t] = {}
            for jprime in range(-t, t + 1):
                prob_sum = 0.0
                for j_parent, dj in ((jprime - 1, +1), (jprime, 0), (jprime + 1, -1)):
                    Pj = P_levels[t - 1].get(j_parent, 0.0)
                    if Pj == 0.0:
                        continue
                    pu, pm, pd = prob_model.probs(t - 1, j_parent)
                    p_trans = pu if dj == +1 else (pm if dj == 0 else pd)
                    prob_sum += Pj * p_trans
                if prob_sum > 0.0:
                    P_levels[t][jprime] = prob_sum
            # normalize
            sP = sum(P_levels[t].values())
            if sP > 0:
                for k in list(P_levels[t].keys()):
                    P_levels[t][k] /= sP

        # Nodes_by_level for plotting
        nodes_by_level = {0: [0]}
        for t in range(1, T + 1):
            nodes_by_level[t] = list(range(-t, t + 1))

        # FIX: use args.annotate_prob (underscored), not args.annotate-prob
        draw_tree(args.S0, args.u, nodes_by_level, P_levels if args.annotate_prob else None)

if __name__ == "__main__":
    main()
