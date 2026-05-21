"""
FastIncrementalProxy: drop-in replacement for IncrementalRealProxy with
vectorized density, routing, and congestion smoothing.

Only the hot-path methods are overridden; all other logic is inherited
unchanged from IncrementalRealProxy.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_submissions_dir = str(Path(__file__).resolve().parent)
if _submissions_dir not in sys.path:
    sys.path.insert(0, _submissions_dir)

from incremental_real_proxy import IncrementalRealProxy  # noqa: E402


class FastIncrementalProxy(IncrementalRealProxy):
    """Vectorized drop-in for IncrementalRealProxy — same public API."""

    # ── Density ──────────────────────────────────────────────────────────────

    def _add_density(self, macro_idx: int, sign: float, sx: float = None, sy: float = None):
        if sx is None:
            sx = self.sizes[macro_idx, 0]
        if sy is None:
            sy = self.sizes[macro_idx, 1]
        x = self.pos[macro_idx, 0]
        y = self.pos[macro_idx, 1]
        x_lo = max(x - sx / 2, 0.0); x_hi = min(x + sx / 2, self.cw)
        y_lo = max(y - sy / 2, 0.0); y_hi = min(y + sy / 2, self.ch)
        if x_hi <= x_lo or y_hi <= y_lo:
            return

        col_lo = max(0, int(x_lo / self.grid_width))
        col_hi = max(0, min(int((x_hi - 1e-12) / self.grid_width), self.gc - 1))
        row_lo = max(0, int(y_lo / self.grid_height))
        row_hi = max(0, min(int((y_hi - 1e-12) / self.grid_height), self.gr - 1))

        # Row overlaps: vectorised over columns for each row slice
        rows = np.arange(row_lo, row_hi + 1)
        ry_lo_arr = rows * self.grid_height
        ry_hi_arr = ry_lo_arr + self.grid_height
        oy_arr = np.clip(np.minimum(y_hi, ry_hi_arr) - np.maximum(y_lo, ry_lo_arr), 0.0, None)

        cols = np.arange(col_lo, col_hi + 1)
        rx_lo_arr = cols * self.grid_width
        rx_hi_arr = rx_lo_arr + self.grid_width
        ox_arr = np.clip(np.minimum(x_hi, rx_hi_arr) - np.maximum(x_lo, rx_lo_arr), 0.0, None)

        contrib = sign * np.outer(oy_arr, ox_arr)
        self.density[row_lo:row_hi + 1, col_lo:col_hi + 1] += contrib

    # ── Two-pin net routing ───────────────────────────────────────────────────

    def _add_two_pin(self, source_gc: Tuple[int, int], sink_gc: Tuple[int, int],
                     weight: float, sign: float):
        col_min = min(sink_gc[1], source_gc[1]); col_max = max(sink_gc[1], source_gc[1])
        row_min = min(sink_gc[0], source_gc[0]); row_max = max(sink_gc[0], source_gc[0])
        if col_max > col_min:
            self.H_net[source_gc[0], col_min:col_max] += sign * weight
        if row_max > row_min:
            self.V_net[row_min:row_max, sink_gc[1]] += sign * weight

    # ── L / T / three-pin routing ────────────────────────────────────────────

    def _add_l_routing(self, sorted_gcells: List[Tuple[int, int]], weight: float,
                       sign: float, y1: int = None):
        y1_v, x1 = sorted_gcells[0]
        y2, x2   = sorted_gcells[1]
        y3, x3   = sorted_gcells[2]
        if x2 > x1:
            self.H_net[y1_v, x1:x2] += sign * weight
        if x3 > x2:
            self.H_net[y2, x2:x3] += sign * weight
        r_lo, r_hi = min(y1_v, y2), max(y1_v, y2)
        if r_hi > r_lo:
            self.V_net[r_lo:r_hi, x2] += sign * weight
        r_lo, r_hi = min(y2, y3), max(y2, y3)
        if r_hi > r_lo:
            self.V_net[r_lo:r_hi, x3] += sign * weight

    def _add_t_routing(self, sorted_gcells: List[Tuple[int, int]], weight: float,
                       sign: float, y1: int = None):
        y1_v, x1 = sorted_gcells[0]
        y2, x2   = sorted_gcells[1]
        y3, x3   = sorted_gcells[2]
        xmin = min(x1, x2, x3); xmax = max(x1, x2, x3)
        if xmax > xmin:
            self.H_net[y2, xmin:xmax] += sign * weight
        r_lo, r_hi = min(y1_v, y2), max(y1_v, y2)
        if r_hi > r_lo:
            self.V_net[r_lo:r_hi, x1] += sign * weight
        r_lo, r_hi = min(y2, y3), max(y2, y3)
        if r_hi > r_lo:
            self.V_net[r_lo:r_hi, x3] += sign * weight

    def _add_three_pin(self, gcells_set: frozenset, weight: float, sign: float,
                       temp: list = None):
        if temp is None:
            temp = sorted(gcells_set, key=lambda g: (g[1], g[0]))
        y1, x1 = temp[0]; y2, x2 = temp[1]; y3, x3 = temp[2]
        if x1 < x2 and x2 < x3 and min(y1, y3) < y2 < max(y1, y3):
            self._add_l_routing(temp, weight, sign)
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            if x2 > x1:
                self.H_net[y1, x1:x2] += sign * weight
            r_lo = y1; r_hi = max(y2, y3)
            if r_hi > r_lo:
                self.V_net[r_lo:r_hi, x2] += sign * weight
        elif y2 == y3:
            if x2 > x1:
                self.H_net[y1, x1:x2] += sign * weight
            if x3 > x2:
                self.H_net[y2, x2:x3] += sign * weight
            r_lo, r_hi = min(y2, y1), max(y2, y1)
            if r_hi > r_lo:
                self.V_net[r_lo:r_hi, x2] += sign * weight
        else:
            temp2 = sorted(gcells_set)
            self._add_t_routing(temp2, weight, sign)

    # ── Congestion cost ───────────────────────────────────────────────────────

    def congestion_cost(self, v_net_norm=None, h_net_norm=None,
                        v_macro_norm=None, h_macro_norm=None) -> float:
        if v_net_norm is None:
            return super().congestion_cost()
        # Vectorized congestion computation with pre-normalised grids
        v_cong = np.maximum(0.0, v_net_norm + v_macro_norm - 1.0)
        h_cong = np.maximum(0.0, h_net_norm + h_macro_norm - 1.0)
        smooth_r = getattr(self, 'smooth_range', 2)
        if smooth_r > 0:
            from scipy.ndimage import uniform_filter
            v_cong = uniform_filter(v_cong, size=smooth_r * 2 + 1, mode='constant')
            h_cong = uniform_filter(h_cong, size=smooth_r * 2 + 1, mode='constant')
        return float(np.mean(v_cong) + np.mean(h_cong))
