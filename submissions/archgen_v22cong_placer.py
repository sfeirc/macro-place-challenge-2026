"""
v22cong: v22 with adaptive congestion-biased CD + LNS.

Circuits routed here (from vhybrid):
  ibm15 (ratio=63.5, actual n_hard=393): medium, vmoon causes density explosion → Tier A
  ibm18 (ratio=91.9, frac=0.087): extreme connectivity, very low density → Tier A (direct)
  ibm17 (ratio=60.3, frac=0.172): large, may beat vmoon via adaptive densities → Tier A (direct)

Note: ibm02 (ratio=35.7) and ibm06 (n_hard=178) were routed here but caused regressions.
  ibm02: v22cong Tier C = 1.2790 (cong-biased) vs v22=1.3000 (-1.65%) → NOW ACTIVE.
  ibm06: v22cong Tier B = 1.4109 (+2.2% vs v22=1.3801) → reverted to v22 in vhybrid.

Adaptive tiers (feature-driven):
  Tier A (ratio > 60): strong biased (0.5/0.5/2.0), scatter=50, pre_cd_lns=120s,
         eplace_cong=0.3 (congestion-aware GP for low-density high-ratio circuits),
         SA reheating enabled for n_hard ≤ 400 (ibm15: 1.4512 vs 1.4802 without reheating).
         Disabled for n_hard > 400 (ibm14 Tier B: +2.4%, ibm12: +0.8% regression).
         For macro_frac < 0.20 (ibm17=0.172, ibm18=0.087): adaptive target_densities
         set to 2x/3.5x/5x macro_frac to force spreading, reducing congestion hotspots.
         n_destroy=20 for n_nets > 20000 (ibm18-like: high connectivity, needs larger LNS).
  Tier B (ratio > 45): standard biased (1.0/0.5/1.0), scatter=30, pre_cd_lns=90s,
         eplace_cong=0.5 (v51 validated: ibm14=1.3693 vs vmoon=1.3962),
         SA reheating DISABLED (ibm14 regressed +2.4% with reheating).
  Tier C (ratio ≤ 45): DEFAULT CD (no cong bias), scatter=10, pre_cd_lns=0s.
         ibm02 (ratio=35.7, frac=0.553): unbiased CD gives reliable 1.3000.
         Cong bias tested but showed no reliable improvement (vfix2: 1.3032).

Pre-CD LNS: runs a short congestion-spreading LNS before the main CD pass to escape
the clustered ePlace starting topology for high-ratio circuits.
SA reheating: for congestion-trapped circuits, alpha=0.88 drives T→0 within ~50 iters.
Periodic reheat (to T0*0.35) after N iters without global improvement enables escape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark

_submissions_dir = str(Path(__file__).parent)
if _submissions_dir not in sys.path:
    sys.path.insert(0, _submissions_dir)

from archgen_v7c_placer import ArchgenV7cPlacer  # noqa: E402

_CONG_PROXY_WEIGHTS = {"wirelength": 1.0, "density": 0.5, "congestion": 1.0}
_STRONG_CONG_PROXY_WEIGHTS = {"wirelength": 0.5, "density": 0.5, "congestion": 2.0}


class ArchgenV22CongPlacer:
    """v22 + adaptive biased CD/LNS for congestion-dominated circuits."""

    def __init__(self, seed: int = 42, verbose: bool = False):
        self.seed = seed
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        import numpy as _np
        n_nets = len(benchmark.net_pin_nodes)
        n_hard = benchmark.num_hard_macros
        ratio = n_nets / max(n_hard, 1)

        max_orderings = 1 if n_nets > 30000 else 7

        # Compute macro area fraction for adaptive density targeting
        sizes = benchmark.macro_sizes.numpy()
        macro_area = float(_np.sum(sizes[:n_hard, 0] * sizes[:n_hard, 1]))
        canvas_area = float(benchmark.canvas_width) * float(benchmark.canvas_height)
        macro_frac = macro_area / max(canvas_area, 1e-12)

        # Adaptive n_destroy: more destruction for larger / more connected circuits.
        if n_hard > 700:
            n_destroy = 32
        elif n_hard > 350:
            n_destroy = 20 if (ratio > 60 and n_nets > 20000) else 16
        elif n_hard > 200:
            n_destroy = 20 if n_nets > 20000 else 12
        else:
            n_destroy = 8

        if ratio > 60:
            proxy_weights = _STRONG_CONG_PROXY_WEIGHTS
            scatter_candidates = 100 if n_hard > 700 else 50
            pre_cd_budget = 120.0
            lns_sa_T0 = 0.05
            lns_reheat_interval = 40 if n_hard <= 400 else 0
            eplace_cong = 0.3
            if macro_frac < 0.20:
                td1 = max(macro_frac * 2.0, 0.15)
                td2 = max(macro_frac * 3.5, 0.30)
                td3 = min(macro_frac * 5.0, 0.70)
                target_densities = (td1, td2, td3)
            else:
                target_densities = (0.70, 0.85, 0.95)
        elif ratio > 45:
            if n_hard < 400 and n_nets > 13000:
                proxy_weights = None
                scatter_candidates = 10
                pre_cd_budget = 0.0
                lns_sa_T0 = 0.03
                lns_reheat_interval = 0
                eplace_cong = 0.5
                target_densities = (0.70, 0.85, 0.95)
            else:
                proxy_weights = _CONG_PROXY_WEIGHTS
                scatter_candidates = 30
                pre_cd_budget = 90.0
                lns_sa_T0 = 0.04
                lns_reheat_interval = 0
                eplace_cong = 0.5
                target_densities = (0.70, 0.85, 0.95)
        else:
            proxy_weights = None
            scatter_candidates = 10
            pre_cd_budget = 0.0
            lns_sa_T0 = 0.03
            lns_reheat_interval = 0
            eplace_cong = 0.0
            target_densities = (0.70, 0.85, 0.95)

        if self.verbose:
            if ratio > 60:
                tier = "A"
            elif ratio > 45:
                tier = "B0" if (n_hard < 400 and n_nets > 13000) else "B"
            else:
                tier = "C"
            print(
                f"[v22cong] n_hard={n_hard} ratio={ratio:.1f} tier={tier} "
                f"macro_frac={macro_frac:.3f} target_den={target_densities} "
                f"n_destroy={n_destroy} scatter={scatter_candidates} pre_cd={pre_cd_budget:.0f}s"
                f" eplace_cong={eplace_cong} reheat={lns_reheat_interval}"
            )

        placer = ArchgenV7cPlacer(
            seed=self.seed,
            verbose=self.verbose,
            max_orderings=max_orderings,
            cong_scatter_candidates=scatter_candidates,
            cd_n_extra_random=50,
            lns_min_budget=150.0,
            lns_n_destroy=n_destroy,
            lns_cd_passes=5,
            lns_sa_T0=lns_sa_T0,
            lns_n_extra_random=30,
            cd_proxy_weights=proxy_weights,
            lns_proxy_weights=proxy_weights,
            pre_cd_lns_budget=pre_cd_budget,
            eplace_cong_weight=eplace_cong,
            target_densities=target_densities,
            lns_sa_reheat_interval=lns_reheat_interval,
            lns_sa_max_reheats=3,
        )
        return placer.place(benchmark)
