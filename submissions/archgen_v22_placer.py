"""
v22 (Archgen): conservative v21 + cong_scatter_candidates=10.

Builds directly on the revised v21 design (data-driven threshold):
  - n_nets ≤ 30000: max_ord=7 + v7c exact defaults (NO regressions)
  - n_nets > 30000: max_ord=1 + v7c exact defaults (ibm14/16/17 only)

Adds: cong_scatter_candidates=10 for ALL benchmarks.
During each LNS scatter, destroyed macros try 10 candidate positions and pick
the one in the lowest-congestion routing grid cell. This biases scattered macros
toward routing channels with spare capacity — most beneficial for high-congestion
benchmarks (ibm06 cong=1.949, ibm16 cong=1.979, ibm17 cong=2.385, ibm18 cong=2.415).

Overhead: FastIncrementalProxy creation per LNS iteration for congestion grid.
Small benchmarks (ibm01): ~90s overhead. Large (ibm17): ~400s but fewer iters anyway.
"""

from __future__ import annotations
import sys
from pathlib import Path

_submissions_dir = str(Path(__file__).parent)
if _submissions_dir not in sys.path:
    sys.path.insert(0, _submissions_dir)

from macro_place.benchmark import Benchmark  # noqa: E402
import torch  # noqa: E402

from archgen_v7c_placer import ArchgenV7cPlacer  # noqa: E402

# v7c's exact default LNS parameters
_V7C_DEFAULTS = dict(
    cd_n_extra_random=50,
    lns_min_budget=150.0,
    lns_n_destroy=8,
    lns_cd_passes=5,
    lns_n_extra_random=30,
)


class ArchgenV22Placer:
    """
    Archgen V22: conservative v21 + congestion-guided LNS scatter.

    All within-budget benchmarks (n_nets ≤ 23000) use v7c exact parameters —
    zero risk of ibm02-type regressions. Over-budget benchmarks get max_ord=1
    to free LNS time. cong_scatter_candidates=10 biases LNS toward low-congestion
    regions across all benchmarks.
    """

    def __init__(self, seed: int = 42, verbose: bool = False):
        self.seed = seed
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        n_nets = len(benchmark.net_pin_nodes)
        max_orderings = 1 if n_nets > 30000 else 7

        placer = ArchgenV7cPlacer(
            seed=self.seed,
            verbose=self.verbose,
            max_orderings=max_orderings,
            cong_scatter_candidates=10,
            **_V7C_DEFAULTS,
        )
        return placer.place(benchmark)
