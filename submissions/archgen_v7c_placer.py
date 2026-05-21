"""
v7c (Archgen): v4 + checkpoint reset in _real_proxy_cd.

Builds on v11's ePlace global placement + multi-start legalization
with an outer reweighting loop inspired by Lagrangian / game-theoretic
"raise the weight of the worst component, re-solve":

  GLOBAL ──► REWEIGHT ──► QA
  (DREAMPlace) (per-net)   (legalize+check)
       ↑                    │
       └──────loop──────────┘

Per-iteration reweighting:
  1. After each ePlace + legalize, compute per-net HPWL.
  2. Identify "long" nets — those whose HPWL is in the top K% of
     (HPWL / sqrt(net_pin_count)).
  3. Boost their net weights by a multiplicative factor (e.g. 1.5).
  4. Re-run ePlace global placement starting from current pos with
     new net_weights.
  5. Legalize, compute actual proxy. Keep best.

QA step (legalization) is unchanged from v11 — multi-start spiral
search with 7 orderings + soft-Jacobi update.

The intuition: when we identify nets that are "stretched" (high HPWL),
boosting their weight tells the next ePlace solve to prioritize
shrinking them, which routes around the local optimum we're stuck in.

Otherwise identical to v11 (Nesterov+BB, FFT density, filler nodes,
γ + density-weight schedules).

OLD v11 NOTES below.

v11: DREAMPlace-faithful pure-PyTorch reimplementation.

Closely follows the algorithms in external/DREAMPlace/dreamplace/:
  - PlaceObj.obj_fn:    wirelength + density_weight × electrostatic_density
  - WeightedAverageWirelength: γ-smoothed weighted-average HPWL per net
  - electric_potential.ElectricPotentialFunction: ePlace electrostatic
    density via FFT Poisson solve (here using rfft2; DREAMPlace uses
    DCT2/IDCT2/IDXST_IDCT for Neumann BCs; the energy is still
    monotonic in real density, so the gradient direction matches).
  - NesterovAcceleratedGradientOptimizer: Nesterov + Barzilai-Borwein
    adaptive step size.
  - PlaceObj.update_gamma:  γ = base_γ × 10^((overflow - 0.1)·20/9 - 1)
  - PlaceObj.update_density_weight (HPWL mode):
        if Δ_hpwl < 0: μ = 1.05 × max(0.9999^iter, 0.98)
        else:           μ = 1.05 × clamp(1.05^(-Δhpwl/ref_hpwl), 0.95, 1.05)
        density_weight *= μ
  - Filler nodes: fictitious cells filling empty area so density
    spreads uniformly to target_density. Critical — without filler,
    real cells over-cluster.

Pipeline:
  1. ePlace global placement (Nesterov + electrostatic, with filler).
  2. Greedy minimum-displacement legalization of hard macros.
  3. Soft-macro Jacobi update (HPWL-optimal centroid, from v3).
  4. Compare ePlace-warmed vs initial.plc-warmed pipelines; keep best.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark

# Ensure sibling files (incremental_real_proxy.py) are importable
_submissions_dir = str(Path(__file__).parent)
if _submissions_dir not in sys.path:
    sys.path.insert(0, _submissions_dir)


# ───────────────────────── PLC loader ───────────────────────── #
def _load_plc(name: str):
    from macro_place.loader import load_benchmark, load_benchmark_from_dir
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    # Map canonical NG45 name → directory name (handles both "ariane133" and "ariane133_ng45")
    _ng45_dir = {"ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
                 "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile",
                 "ariane133": "ariane133", "ariane136": "ariane136",
                 "nvdla": "nvdla", "mempool_tile": "mempool_tile"}
    d = _ng45_dir.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


# ───────────────────── pin/net data builder ──────────────────── #
def _build_pin_data(benchmark: Benchmark, device: torch.device):
    n_macros = benchmark.num_macros
    n_ports = benchmark.port_positions.shape[0]
    n_owners = n_macros + n_ports
    max_pins = max((p.shape[0] for p in benchmark.macro_pin_offsets), default=1)
    max_pins = max(max_pins, 1)
    owner_pin_offsets = torch.zeros(n_owners, max_pins, 2, device=device)
    for i, offsets in enumerate(benchmark.macro_pin_offsets):
        if offsets.shape[0] > 0:
            owner_pin_offsets[i, : offsets.shape[0]] = offsets.to(device)

    net_owner_list, net_pinidx_list, net_id_list = [], [], []
    for nid, pins in enumerate(benchmark.net_pin_nodes):
        for row in pins.tolist():
            net_owner_list.append(int(row[0]))
            net_pinidx_list.append(int(row[1]))
            net_id_list.append(nid)
    if not net_owner_list:
        return None
    net_owners = torch.tensor(net_owner_list, dtype=torch.long, device=device)
    net_pinidx = torch.tensor(net_pinidx_list, dtype=torch.long, device=device)
    net_ids = torch.tensor(net_id_list, dtype=torch.long, device=device)
    n_nets = len(benchmark.net_pin_nodes)
    pin_offset_xy = owner_pin_offsets[net_owners, net_pinidx]
    return dict(
        n_macros=n_macros, n_ports=n_ports, n_owners=n_owners, n_nets=n_nets,
        net_owners=net_owners, net_ids=net_ids, pin_offset_xy=pin_offset_xy,
    )


# ─────────────────── smooth WA wirelength (same as DP) ─────────── #
def _wa_hpwl_dim(x: torch.Tensor, net_ids: torch.Tensor, n_nets: int, gamma: float) -> torch.Tensor:
    """Per-net weighted-average wirelength along one axis, smoothed by γ.
    Returns [n_nets] tensor of per-net (WA_max - WA_min) values (NOT yet
    multiplied by net weights — caller does that)."""
    device = x.device
    big = 1e18
    true_max = torch.full((n_nets,), -big, device=device)
    true_max.scatter_reduce_(0, net_ids, x / gamma, reduce="amax", include_self=True)
    true_min = torch.full((n_nets,), big, device=device)
    true_min.scatter_reduce_(0, net_ids, x / gamma, reduce="amin", include_self=True)
    tmax = true_max.detach()
    tmin = true_min.detach()
    ep = torch.exp(x / gamma - tmax[net_ids])
    sum_ep = torch.zeros(n_nets, device=device).scatter_add(0, net_ids, ep)
    sum_xep = torch.zeros(n_nets, device=device).scatter_add(0, net_ids, x * ep)
    wa_max = sum_xep / (sum_ep + 1e-12)
    en = torch.exp(-x / gamma + tmin[net_ids])
    sum_en = torch.zeros(n_nets, device=device).scatter_add(0, net_ids, en)
    sum_xen = torch.zeros(n_nets, device=device).scatter_add(0, net_ids, x * en)
    wa_min = sum_xen / (sum_en + 1e-12)
    return wa_max - wa_min


# ─────────────────── density map (rectangle/bin overlap) ────────── #
def _density_map(pos: torch.Tensor, sizes: torch.Tensor, cw: float, ch: float, n_bins: int) -> torch.Tensor:
    """
    Differentiable density map = exact macro/bin overlap area.
    Returns [n_bins, n_bins] tensor (y-row, x-col).
    """
    device = pos.device
    bin_w = cw / n_bins
    bin_h = ch / n_bins
    bin_x_lo = torch.arange(n_bins, device=device).float() * bin_w
    bin_x_hi = bin_x_lo + bin_w
    bin_y_lo = torch.arange(n_bins, device=device).float() * bin_h
    bin_y_hi = bin_y_lo + bin_h
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    mx_lo = (pos[:, 0] - half_w).unsqueeze(1)
    mx_hi = (pos[:, 0] + half_w).unsqueeze(1)
    my_lo = (pos[:, 1] - half_h).unsqueeze(1)
    my_hi = (pos[:, 1] + half_h).unsqueeze(1)
    ox = (torch.minimum(mx_hi, bin_x_hi.unsqueeze(0))
          - torch.maximum(mx_lo, bin_x_lo.unsqueeze(0))).clamp(min=0)
    oy = (torch.minimum(my_hi, bin_y_hi.unsqueeze(0))
          - torch.maximum(my_lo, bin_y_lo.unsqueeze(0))).clamp(min=0)
    return (oy.unsqueeze(2) * ox.unsqueeze(1)).sum(dim=0)


# ─────────────── ePlace electrostatic energy via FFT ──────────── #
def _eplace_energy(rho: torch.Tensor, cw: float, ch: float) -> torch.Tensor:
    """
    Solve −∇²φ = ρ on a periodic grid via 2D FFT (Poisson in frequency
    domain), then compute electrostatic energy E = (1/2) ∫ ρ φ dA.
    """
    n_y, n_x = rho.shape
    bin_w = cw / n_x
    bin_h = ch / n_y
    rho_centered = rho - rho.mean()  # charge neutrality
    rho_hat = torch.fft.rfft2(rho_centered)
    fy = torch.fft.fftfreq(n_y, d=bin_h, device=rho.device)
    fx = torch.fft.rfftfreq(n_x, d=bin_w, device=rho.device)
    ky = (2 * math.pi * fy).unsqueeze(1)
    kx = (2 * math.pi * fx).unsqueeze(0)
    k2 = kx * kx + ky * ky
    k2_safe = k2.clone()
    k2_safe[0, 0] = 1.0
    phi_hat = rho_hat / k2_safe
    phi_hat[0, 0] = 0.0
    phi = torch.fft.irfft2(phi_hat, s=(n_y, n_x))
    energy = 0.5 * (rho_centered * phi).sum() * bin_w * bin_h
    return energy


# ───────────── Nesterov accelerated gradient + BB step ────────── #
class NesterovBB:
    """
    Nesterov + Barzilai-Borwein step size, ported from
    external/DREAMPlace/dreamplace/NesterovAcceleratedGradientOptimizer.py
    """
    def __init__(self, init_pos: torch.Tensor, obj_grad_fn, constraint_fn, lr_init: float):
        self.obj_grad_fn = obj_grad_fn
        self.constraint_fn = constraint_fn
        # v_k carries gradient
        self.v_k = init_pos.detach().clone().requires_grad_(True)
        self.u_k = self.v_k.detach().clone()
        self.a_k = torch.tensor(1.0, device=init_pos.device)

        obj_k, g_k = obj_grad_fn(self.v_k)
        self.g_k = g_k.detach().clone()
        self.obj_k = obj_k.detach().clone()

        v_k_1 = (self.v_k.detach() - lr_init * self.g_k).requires_grad_(True)
        obj_k_1, g_k_1 = obj_grad_fn(v_k_1)
        self.v_k_1 = v_k_1.detach()
        self.g_k_1 = g_k_1.detach().clone()
        denom = (self.g_k - self.g_k_1).norm(p=2).clamp(min=1e-12)
        self.alpha_k = ((self.v_k.detach() - self.v_k_1).norm(p=2) / denom).abs()

    def step(self):
        s_k = self.v_k.detach() - self.v_k_1
        y_k = self.g_k - self.g_k_1
        sk_dot_yk = (s_k * y_k).sum().clamp(min=1e-20)
        bb_short = (sk_dot_yk / (y_k * y_k).sum().clamp(min=1e-20)).abs()
        lip_step = (s_k.norm(p=2) / y_k.norm(p=2).clamp(min=1e-12)).abs()
        if bb_short.item() > 0:
            step_size = bb_short
        else:
            step_size = torch.minimum(lip_step, self.alpha_k)

        a_kp1 = (1 + (4 * self.a_k.pow(2) + 1).sqrt()) / 2
        coef = (self.a_k - 1) / a_kp1

        u_kp1 = self.v_k.detach() - step_size * self.g_k
        v_kp1 = u_kp1 + coef * (u_kp1 - self.u_k)
        v_kp1 = self.constraint_fn(v_kp1)
        v_kp1 = v_kp1.detach().clone().requires_grad_(True)
        obj_kp1, g_kp1 = self.obj_grad_fn(v_kp1)

        self.v_k_1 = self.v_k.detach().clone()
        self.g_k_1 = self.g_k.clone()
        self.alpha_k = step_size
        self.u_k = u_kp1.detach().clone()
        self.v_k = v_kp1
        self.g_k = g_kp1.detach().clone()
        self.obj_k = obj_kp1.detach().clone()
        self.a_k = a_kp1
        return obj_kp1.item()


# ───────────────────── Routing congestion energy ──────────────── #
def _routing_cong_energy_1d(
    pin_xy: "torch.Tensor",
    net_ids: "torch.Tensor",
    n_nets: int,
    cw: float,
    ch: float,
    n_rows: int,
    n_cols: int,
    gamma_frac: float = 0.02,
    hroutes_per_micron: float = 0.0,  # 0 = auto-normalise to mean demand
    vroutes_per_micron: float = 0.0,
) -> "torch.Tensor":
    """Differentiable 1D routing congestion energy for ePlace.

    Matches evaluator's 10×10 bin ABU-5% metric. Uses actual benchmark
    routing capacity (hroutes_per_micron, vroutes_per_micron) when provided.
    Horizontal demand HD[r] = Σ_nets hpwl_x_n × span_y_n_r (sigmoid span).
    Capacity = bin_height × hroutes × bin_width when provided, else mean demand.
    Energy = Σ max(0, HD[r]/cap_h - 1)² + Σ max(0, VD[c]/cap_v - 1)².
    """
    device = pin_xy.device
    dtype = pin_xy.dtype
    px, py = pin_xy[:, 0], pin_xy[:, 1]

    # Per-net bounding boxes via scatter_reduce (grad flows to argmax pin)
    _neg_inf = torch.full((n_nets,), -1e9, device=device, dtype=dtype)
    xmax_n = torch.scatter_reduce(_neg_inf, 0, net_ids, px, reduce="amax", include_self=True)
    xmin_n = -torch.scatter_reduce(_neg_inf, 0, net_ids, -px, reduce="amax", include_self=True)
    ymax_n = torch.scatter_reduce(_neg_inf, 0, net_ids, py, reduce="amax", include_self=True)
    ymin_n = -torch.scatter_reduce(_neg_inf, 0, net_ids, -py, reduce="amax", include_self=True)

    hpwl_x = (xmax_n - xmin_n).clamp(min=0.0)  # horizontal span per net
    hpwl_y = (ymax_n - ymin_n).clamp(min=0.0)  # vertical span per net

    # Horizontal routing demand through each row r
    bh = ch / n_rows
    row_c = (torch.arange(n_rows, device=device, dtype=dtype) + 0.5) * bh  # [n_rows]
    g_y = max(gamma_frac * ch, bh)
    span_y = (torch.sigmoid((ymax_n.unsqueeze(1) - row_c) / g_y) *
              torch.sigmoid((row_c - ymin_n.unsqueeze(1)) / g_y))  # [n_nets, n_rows]
    HD = (hpwl_x.unsqueeze(1) * span_y).sum(0)  # [n_rows]

    # Vertical routing demand through each col c
    bw = cw / n_cols
    col_c = (torch.arange(n_cols, device=device, dtype=dtype) + 0.5) * bw  # [n_cols]
    g_x = max(gamma_frac * cw, bw)
    span_x = (torch.sigmoid((xmax_n.unsqueeze(1) - col_c) / g_x) *
              torch.sigmoid((col_c - xmin_n.unsqueeze(1)) / g_x))  # [n_nets, n_cols]
    VD = (hpwl_y.unsqueeze(1) * span_x).sum(0)  # [n_cols]

    # Capacity: use benchmark routing capacity if provided, else auto-normalise
    if hroutes_per_micron > 0.0 and vroutes_per_micron > 0.0:
        h_cap = float(hroutes_per_micron * bh * cw)  # available H tracks in row bin
        v_cap = float(vroutes_per_micron * bw * ch)  # available V tracks in col bin
    else:
        h_cap = float(HD.detach().mean().clamp(min=1e-8).item())
        v_cap = float(VD.detach().mean().clamp(min=1e-8).item())
    h_viol = torch.clamp(HD / h_cap - 1.0, min=0.0)
    v_viol = torch.clamp(VD / v_cap - 1.0, min=0.0)
    return h_viol.pow(2).sum() + v_viol.pow(2).sum()


# ───────────────────── ePlace global placement ──────────────── #
def _eplace_global(
    benchmark: Benchmark,
    device: torch.device,
    n_iters: int = 600,
    n_bins: int = 64,
    max_seconds: float = 0.0,  # 0 = no limit
    target_density: float = 0.85,
    base_gamma_factor: float = 4.0,
    init_lr_frac: float = 1e-3,
    add_filler: bool = True,
    verbose: bool = False,
    net_weights: "torch.Tensor | None" = None,
    init_pos_real: "torch.Tensor | None" = None,
    macro_halo_frac: float = 0.0,  # NEW: halo as fraction of canvas (each side)
    eplace_cong_weight: float = 0.0,  # congestion penalty weight (0 = disabled)
):
    """DREAMPlace-style global placement. Returns (final_real_pos, hist_dict).

    With macro_halo_frac > 0, hard macros are inflated by 2 * halo on each
    dimension during density computation only (AutoDMP trick). Pin offsets
    and final macro positions are unchanged. The halo reserves space around
    macros so legalization barely needs to move them.
    """
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
    n_real = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    sizes_real = benchmark.macro_sizes.to(device).float()
    movable_real = benchmark.get_movable_mask().to(device)
    fixed_pos = benchmark.macro_positions.to(device).float().clone()

    # Macro halos: inflate hard macros only (soft macros are already small)
    halo_size = macro_halo_frac * max(cw, ch)
    if halo_size > 0:
        sizes_inflated = sizes_real.clone()
        sizes_inflated[:n_hard, 0] += 2 * halo_size
        sizes_inflated[:n_hard, 1] += 2 * halo_size
    else:
        sizes_inflated = sizes_real

    # Filler cells (use REAL areas to compute filler need; halo doesn't add real area)
    canvas_area = cw * ch
    real_area = float((sizes_real[:, 0] * sizes_real[:, 1]).sum().item())
    target_total_area = canvas_area * target_density
    filler_area = max(target_total_area - real_area, 0.0)
    filler_size = math.sqrt(filler_area / max(n_real, 1)) if filler_area > 0 else 0.0
    n_filler = int(filler_area / max(filler_size * filler_size, 1e-12)) if add_filler and filler_size > 0 else 0
    if verbose:
        print(f"  [eplace] cw×ch={cw:.1f}×{ch:.1f}  real_area={real_area:.1f} "
              f"target={target_total_area:.1f}  fillers={n_filler}")

    if n_filler > 0:
        filler_sizes = torch.full((n_filler, 2), filler_size, device=device)
        all_sizes = torch.cat([sizes_inflated, filler_sizes], dim=0)
    else:
        all_sizes = sizes_inflated
    n_total = n_real + n_filler

    # Pin data
    pin_data = _build_pin_data(benchmark, device)
    if pin_data is None:
        return fixed_pos[:n_real].cpu(), {}
    n_owners = pin_data["n_owners"]; n_nets = pin_data["n_nets"]; n_ports = pin_data["n_ports"]
    net_owners = pin_data["net_owners"]; net_ids = pin_data["net_ids"]
    pin_offset_xy = pin_data["pin_offset_xy"]
    port_pos = benchmark.port_positions.to(device).float()

    # Initial positions: warm-start if provided
    pos = torch.zeros(n_total, 2, device=device)
    if init_pos_real is not None:
        pos[:n_real] = init_pos_real.to(device).float()
    else:
        pos[:n_real] = fixed_pos
    if n_filler > 0:
        ng = max(1, int(math.ceil(math.sqrt(n_filler))))
        for k in range(n_filler):
            ix = k % ng; iy = k // ng
            pos[n_real + k, 0] = (ix + 0.5) * cw / ng
            pos[n_real + k, 1] = (iy + 0.5) * ch / ng

    half_w_all = all_sizes[:, 0] / 2
    half_h_all = all_sizes[:, 1] / 2

    movable_all = torch.zeros(n_total, dtype=torch.bool, device=device)
    movable_all[:n_real] = movable_real
    if n_filler > 0:
        movable_all[n_real:] = True
    fixed_pos_full = pos.clone()  # for restoring fixed entries

    def constraint_fn(p: torch.Tensor) -> torch.Tensor:
        p_clamped = p.clone()
        p_clamped[:, 0] = p_clamped[:, 0].clamp(min=half_w_all, max=cw - half_w_all)
        p_clamped[:, 1] = p_clamped[:, 1].clamp(min=half_h_all, max=ch - half_h_all)
        if (~movable_all).any():
            p_clamped[~movable_all] = fixed_pos_full[~movable_all]
        return p_clamped

    bin_size = (cw / n_bins) + (ch / n_bins)
    base_gamma = base_gamma_factor * bin_size
    bin_w = cw / n_bins
    bin_h = ch / n_bins
    target_cap = target_density * bin_w * bin_h

    # Per-net weights tensor (default 1.0 each)
    if net_weights is None:
        nw_tensor = torch.ones(n_nets, device=device)
    else:
        nw_tensor = net_weights.to(device).float()
        assert nw_tensor.shape[0] == n_nets, f"net_weights len {nw_tensor.shape[0]} != n_nets {n_nets}"

    state = dict(density_weight=1.0e-5, gamma=10 * base_gamma, prev_hpwl=None,
                 hpwl_history=[], overflow_history=[], dw_history=[])

    def compute_overflow_val(rho: torch.Tensor) -> float:
        excess = (rho - target_cap).clamp(min=0).sum().item()
        denom = real_area + (n_filler * filler_size * filler_size if n_filler > 0 else 0.0)
        return excess / max(denom, 1e-12)

    def update_gamma(overflow: float):
        coef = 10.0 ** ((overflow - 0.1) * 20.0 / 9.0 - 1.0)
        return base_gamma * coef

    def update_density_weight(cur_hpwl: float, prev_hpwl: float, iteration: int):
        if prev_hpwl is None:
            return state["density_weight"]
        ref_hpwl = max(abs(prev_hpwl) * 0.1, 1e-3)
        delta = cur_hpwl - prev_hpwl
        if delta < 0:
            mu = 1.05 * max(0.9999 ** float(iteration), 0.98)
        else:
            inner = 1.05 ** (-delta / ref_hpwl)
            mu = 1.05 * float(np.clip(inner, 0.95, 1.05))
        return state["density_weight"] * mu

    def obj_grad_fn(p: torch.Tensor):
        if p.grad is not None:
            p.grad.zero_()
        owner_pos = torch.cat([p[:n_real], port_pos], dim=0) if n_ports > 0 else p[:n_real]
        pin_xy = owner_pos[net_owners] + pin_offset_xy
        wl_x = _wa_hpwl_dim(pin_xy[:, 0], net_ids, n_nets, state["gamma"])
        wl_y = _wa_hpwl_dim(pin_xy[:, 1], net_ids, n_nets, state["gamma"])
        # Per-net WL multiplied by net weight
        wl_total = (nw_tensor * wl_x).sum() + (nw_tensor * wl_y).sum()
        rho = _density_map(p, all_sizes, cw, ch, n_bins)
        density_energy = _eplace_energy(rho, cw, ch)
        loss = wl_total + state["density_weight"] * density_energy
        if eplace_cong_weight > 0.0:
            gf = float(state["gamma"]) / max(cw, ch)  # track ePlace gamma schedule
            cong_e = _routing_cong_energy_1d(
                pin_xy, net_ids, n_nets, cw, ch, 10, 10, gamma_frac=gf,
            )
            loss = loss + eplace_cong_weight * state["density_weight"] * cong_e
        if loss.requires_grad:
            loss.backward()
        with torch.no_grad():
            if (~movable_all).any() and p.grad is not None:
                p.grad[~movable_all] = 0.0
        return loss, p.grad if p.grad is not None else torch.zeros_like(p)

    pos.requires_grad_(True)
    canvas_size = max(cw, ch)
    init_lr = init_lr_frac * canvas_size
    optimizer = NesterovBB(pos, obj_grad_fn, constraint_fn, init_lr)

    # Estimate initial density_weight: balance gradient magnitudes
    with torch.no_grad():
        owner_pos0 = torch.cat([optimizer.v_k.detach()[:n_real], port_pos], dim=0) if n_ports > 0 else optimizer.v_k.detach()[:n_real]
        pin_xy0 = owner_pos0[net_owners] + pin_offset_xy
        wl0 = ((nw_tensor * _wa_hpwl_dim(pin_xy0[:, 0], net_ids, n_nets, state["gamma"])).sum().item()
               + (nw_tensor * _wa_hpwl_dim(pin_xy0[:, 1], net_ids, n_nets, state["gamma"])).sum().item())
        rho0 = _density_map(optimizer.v_k.detach(), all_sizes, cw, ch, n_bins)
        de0 = abs(_eplace_energy(rho0, cw, ch).item())
        if de0 > 1e-12:
            state["density_weight"] = 8e-5 * abs(wl0) / de0
        if verbose:
            print(f"  [eplace] init wl={wl0:.2f}  density_energy={de0:.2f}  λ0={state['density_weight']:.3e}")

    best_overflow = float("inf")
    best_pos = optimizer.v_k.detach().clone()
    diverged_count = 0
    _eplace_t0 = time.time()

    for it in range(n_iters):
        if max_seconds > 0 and time.time() - _eplace_t0 > max_seconds:
            break
        with torch.no_grad():
            rho_cur = _density_map(optimizer.v_k.detach(), all_sizes, cw, ch, n_bins)
            overflow = compute_overflow_val(rho_cur)
            state["gamma"] = update_gamma(overflow)
            owner_pos_cur = torch.cat([optimizer.v_k.detach()[:n_real], port_pos], dim=0) if n_ports > 0 else optimizer.v_k.detach()[:n_real]
            pin_xy_cur = owner_pos_cur[net_owners] + pin_offset_xy
            cur_hpwl = ((nw_tensor * _wa_hpwl_dim(pin_xy_cur[:, 0], net_ids, n_nets, state["gamma"])).sum().item()
                        + (nw_tensor * _wa_hpwl_dim(pin_xy_cur[:, 1], net_ids, n_nets, state["gamma"])).sum().item())

        new_dw = update_density_weight(cur_hpwl, state["prev_hpwl"], it)
        state["density_weight"] = float(np.clip(new_dw, 1e-10, 1e3))

        try:
            optimizer.step()
        except Exception as e:
            if verbose:
                print(f"  [eplace] iter {it}: step failed: {e}")
            break

        state["prev_hpwl"] = cur_hpwl
        state["hpwl_history"].append(cur_hpwl)
        state["overflow_history"].append(overflow)
        state["dw_history"].append(state["density_weight"])

        if overflow < best_overflow:
            best_overflow = overflow
            best_pos = optimizer.v_k.detach().clone()

        if verbose and it % 50 == 0:
            print(f"  [eplace] it={it} γ={state['gamma']:.3f} λ={state['density_weight']:.3e} "
                  f"hpwl={cur_hpwl:.2f} ovfl={overflow:.3f}")

        # Convergence
        if overflow < 0.10 and it > 100 and cur_hpwl > state["hpwl_history"][-10]:
            if verbose:
                print(f"  [eplace] converged at iter {it}: overflow={overflow:.3f}")
            break

        # Divergence
        if it > 50 and len(state["hpwl_history"]) > 20:
            if cur_hpwl > 5 * state["hpwl_history"][20]:
                diverged_count += 1
                if diverged_count > 5:
                    if verbose:
                        print(f"  [eplace] iter {it}: HPWL diverged, stopping")
                    break

    final_pos = optimizer.v_k.detach()[:n_real].cpu().clone()
    return final_pos, state


# ─────────────────────── Legalization (v3) ───────────────────── #
def _legalize_with_order(
    pos: np.ndarray, movable: np.ndarray, sizes: np.ndarray,
    cw: float, ch: float, n_hard: int, fixed_pos: np.ndarray, order: List[int],
) -> np.ndarray:
    pos = pos.copy()
    for i in range(n_hard):
        if not movable[i]:
            pos[i] = fixed_pos[i]
    gap = 0.001
    sx = sizes[:n_hard, 0]; sy = sizes[:n_hard, 1]
    sep_x_mat = (sx[:, None] + sx[None, :]) / 2 + gap
    sep_y_mat = (sy[:, None] + sy[None, :]) / 2 + gap
    half_w = sx / 2; half_h = sy / 2
    placed = np.zeros(n_hard, dtype=bool)
    for i in range(n_hard):
        if not movable[i]:
            placed[i] = True

    def has_overlap(idx, x, y):
        if not placed.any():
            return False
        dx = np.abs(x - pos[:n_hard, 0]); dy = np.abs(y - pos[:n_hard, 1])
        o = (dx < sep_x_mat[idx]) & (dy < sep_y_mat[idx]) & placed
        o[idx] = False
        return bool(o.any())

    for idx in order:
        if placed[idx]:
            continue
        x0 = float(np.clip(pos[idx, 0], half_w[idx], cw - half_w[idx]))
        y0 = float(np.clip(pos[idx, 1], half_h[idx], ch - half_h[idx]))
        if not has_overlap(idx, x0, y0):
            pos[idx, 0], pos[idx, 1] = x0, y0
            placed[idx] = True
            continue
        step = max(sx[idx], sy[idx]) * 0.20
        best_x, best_y, best_d = x0, y0, float("inf")
        found_any = False
        for r in range(1, 250):
            ring_found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = float(np.clip(x0 + dxm * step, half_w[idx], cw - half_w[idx]))
                    cy = float(np.clip(y0 + dym * step, half_h[idx], ch - half_h[idx]))
                    if has_overlap(idx, cx, cy):
                        continue
                    d = (cx - x0) ** 2 + (cy - y0) ** 2
                    if d < best_d:
                        best_d = d; best_x, best_y = cx, cy
                        ring_found = True; found_any = True
            if found_any and ring_found:
                break
        pos[idx, 0], pos[idx, 1] = best_x, best_y
        placed[idx] = True
    return pos


def _build_orderings(sizes_np, fixed_pos, n_hard, cw, ch):
    orderings = []
    area = sizes_np[:n_hard, 0] * sizes_np[:n_hard, 1]
    orderings.append(("area_desc", list(np.argsort(-area))))
    orderings.append(("area_asc", list(np.argsort(area))))
    cx, cy = cw / 2, ch / 2
    dc = (fixed_pos[:n_hard, 0] - cx) ** 2 + (fixed_pos[:n_hard, 1] - cy) ** 2
    orderings.append(("center_first", list(np.argsort(dc))))
    orderings.append(("edges_first", list(np.argsort(-dc))))
    orderings.append(("width_desc", list(np.argsort(-sizes_np[:n_hard, 0]))))
    for s in (1, 7):
        rng = np.random.RandomState(s)
        order = list(range(n_hard))
        rng.shuffle(order)
        orderings.append((f"random_{s}", order))
    return orderings


# ─────────────────────── Soft Jacobi (v3) ────────────────────── #
def _soft_jacobi_update(pos_np, benchmark, n_iters=3, damping=0.5):
    n_macros = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    n_soft = n_macros - n_hard
    if n_soft == 0:
        return pos_np
    n_ports = benchmark.port_positions.shape[0]
    n_owners = n_macros + n_ports
    max_pins = max((p.shape[0] for p in benchmark.macro_pin_offsets), default=1)
    max_pins = max(max_pins, 1)
    owner_pin_offsets = np.zeros((n_owners, max_pins, 2), dtype=np.float64)
    for i, offsets in enumerate(benchmark.macro_pin_offsets):
        if offsets.shape[0] > 0:
            owner_pin_offsets[i, : offsets.shape[0]] = offsets.numpy()

    net_owner_list, net_pinidx_list, net_id_list = [], [], []
    for nid, pins in enumerate(benchmark.net_pin_nodes):
        for row in pins.tolist():
            net_owner_list.append(int(row[0]))
            net_pinidx_list.append(int(row[1]))
            net_id_list.append(nid)
    if not net_owner_list:
        return pos_np
    net_owners = np.array(net_owner_list, dtype=np.int64)
    net_pinidx = np.array(net_pinidx_list, dtype=np.int64)
    net_ids = np.array(net_id_list, dtype=np.int64)
    n_nets = len(benchmark.net_pin_nodes)
    pin_offset_xy = owner_pin_offsets[net_owners, net_pinidx]
    port_pos_np = benchmark.port_positions.numpy().astype(np.float64)

    soft_owner_mask = (net_owners >= n_hard) & (net_owners < n_macros)
    soft_pin_indices = np.nonzero(soft_owner_mask)[0]
    soft_macro_idx = net_owners[soft_pin_indices]
    soft_pin_net_ids = net_ids[soft_pin_indices]

    pos_np = pos_np.copy()
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
    soft_sizes = benchmark.macro_sizes[n_hard:].numpy().astype(np.float64)
    soft_half_w = soft_sizes[:, 0] / 2; soft_half_h = soft_sizes[:, 1] / 2

    for _ in range(n_iters):
        owner_pos = np.zeros((n_owners, 2), dtype=np.float64)
        owner_pos[:n_macros] = pos_np
        if n_ports > 0:
            owner_pos[n_macros:] = port_pos_np
        pin_xy = owner_pos[net_owners] + pin_offset_xy
        net_sum_x = np.zeros(n_nets); net_sum_y = np.zeros(n_nets); net_count = np.zeros(n_nets, dtype=np.int64)
        np.add.at(net_sum_x, net_ids, pin_xy[:, 0])
        np.add.at(net_sum_y, net_ids, pin_xy[:, 1])
        np.add.at(net_count, net_ids, 1)

        soft_pin_x = pin_xy[soft_pin_indices, 0]
        soft_pin_y = pin_xy[soft_pin_indices, 1]
        soft_net_sum_x = net_sum_x[soft_pin_net_ids]
        soft_net_sum_y = net_sum_y[soft_pin_net_ids]
        soft_net_count = net_count[soft_pin_net_ids]
        contrib_x = soft_net_sum_x - soft_pin_x
        contrib_y = soft_net_sum_y - soft_pin_y
        contrib_n = soft_net_count - 1

        soft_local_idx = soft_macro_idx - n_hard
        sum_x = np.zeros(n_soft); sum_y = np.zeros(n_soft); sum_n = np.zeros(n_soft, dtype=np.int64)
        np.add.at(sum_x, soft_local_idx, contrib_x)
        np.add.at(sum_y, soft_local_idx, contrib_y)
        np.add.at(sum_n, soft_local_idx, contrib_n)

        valid = sum_n > 0
        new_x = pos_np[n_hard:, 0].copy(); new_y = pos_np[n_hard:, 1].copy()
        new_x[valid] = sum_x[valid] / sum_n[valid]
        new_y[valid] = sum_y[valid] / sum_n[valid]
        pos_np[n_hard:, 0] = (1 - damping) * pos_np[n_hard:, 0] + damping * new_x
        pos_np[n_hard:, 1] = (1 - damping) * pos_np[n_hard:, 1] + damping * new_y
        pos_np[n_hard:, 0] = np.clip(pos_np[n_hard:, 0], soft_half_w, cw - soft_half_w)
        pos_np[n_hard:, 1] = np.clip(pos_np[n_hard:, 1], soft_half_h, ch - soft_half_h)
    return pos_np


# ─────────────────────── Fast incremental CD (from v7) ────── #
class IncrementalProxy:
    """Incremental WL+density surrogate (HPWL per net + density grid)."""

    def __init__(self, benchmark: Benchmark, full_pos: np.ndarray,
                 wl_weight: float = 1.0, den_weight: float = 0.5):
        self.benchmark = benchmark
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.n_macros = benchmark.num_macros
        self.n_hard = benchmark.num_hard_macros
        self.sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        self.wl_weight = wl_weight
        self.den_weight = den_weight

        n_ports = benchmark.port_positions.shape[0]
        self.n_ports = n_ports
        self.n_owners = self.n_macros + n_ports
        self.port_pos = (benchmark.port_positions.numpy().astype(np.float64)
                         if n_ports > 0 else np.zeros((0, 2)))

        max_pins = max((p.shape[0] for p in benchmark.macro_pin_offsets), default=1)
        max_pins = max(max_pins, 1)
        self.owner_pin_offsets = np.zeros((self.n_owners, max_pins, 2), dtype=np.float64)
        for i, offsets in enumerate(benchmark.macro_pin_offsets):
            if offsets.shape[0] > 0:
                self.owner_pin_offsets[i, : offsets.shape[0]] = offsets.numpy()

        n_nets = len(benchmark.net_pin_nodes)
        self.n_nets = n_nets

        net_owner_list, net_pinidx_list, net_id_list = [], [], []
        for nid, pins in enumerate(benchmark.net_pin_nodes):
            for row in pins.tolist():
                net_owner_list.append(int(row[0]))
                net_pinidx_list.append(int(row[1]))
                net_id_list.append(nid)
        self.net_owners = np.array(net_owner_list, dtype=np.int64)
        self.net_pinidx = np.array(net_pinidx_list, dtype=np.int64)
        self.net_ids = np.array(net_id_list, dtype=np.int64)

        self.owner_to_pin_entries = [None] * self.n_owners
        for owner in range(self.n_owners):
            self.owner_to_pin_entries[owner] = np.nonzero(self.net_owners == owner)[0]
        self.pin_offset_xy = self.owner_pin_offsets[self.net_owners, self.net_pinidx]
        self.net_to_pin_entries = [None] * n_nets
        for nid in range(n_nets):
            self.net_to_pin_entries[nid] = np.nonzero(self.net_ids == nid)[0]

        self.pos = full_pos.copy()
        self.pin_xy = self._pin_xy_full()
        self.net_hpwl = np.zeros(n_nets, dtype=np.float64)
        self._recompute_all_hpwl()

        self.gr = benchmark.grid_rows
        self.gc = benchmark.grid_cols
        self.bin_w = self.cw / self.gc
        self.bin_h = self.ch / self.gr
        self.density = np.zeros((self.gr, self.gc), dtype=np.float64)
        self._recompute_density_full()

    def _pin_xy_full(self) -> np.ndarray:
        owner_pos = np.zeros((self.n_owners, 2), dtype=np.float64)
        owner_pos[:self.n_macros] = self.pos
        if self.n_ports > 0:
            owner_pos[self.n_macros:] = self.port_pos
        return owner_pos[self.net_owners] + self.pin_offset_xy

    def _recompute_all_hpwl(self):
        for nid in range(self.n_nets):
            entries = self.net_to_pin_entries[nid]
            if len(entries) <= 1:
                self.net_hpwl[nid] = 0.0
                continue
            xs = self.pin_xy[entries, 0]; ys = self.pin_xy[entries, 1]
            self.net_hpwl[nid] = (xs.max() - xs.min()) + (ys.max() - ys.min())

    def _recompute_density_full(self):
        self.density.fill(0.0)
        for i in range(self.n_macros):
            self._add_density_contribution(i, +1.0)

    def _add_density_contribution(self, macro_idx: int, sign: float):
        sx = self.sizes[macro_idx, 0]; sy = self.sizes[macro_idx, 1]
        x = self.pos[macro_idx, 0]; y = self.pos[macro_idx, 1]
        x_lo = max(x - sx / 2, 0.0); x_hi = min(x + sx / 2, self.cw)
        y_lo = max(y - sy / 2, 0.0); y_hi = min(y + sy / 2, self.ch)
        if x_hi <= x_lo or y_hi <= y_lo:
            return
        col_lo = int(x_lo / self.bin_w); col_hi = int(min((x_hi - 1e-12) / self.bin_w, self.gc - 1))
        row_lo = int(y_lo / self.bin_h); row_hi = int(min((y_hi - 1e-12) / self.bin_h, self.gr - 1))
        col_lo = max(0, min(col_lo, self.gc - 1)); col_hi = max(0, min(col_hi, self.gc - 1))
        row_lo = max(0, min(row_lo, self.gr - 1)); row_hi = max(0, min(row_hi, self.gr - 1))

        for r in range(row_lo, row_hi + 1):
            ry_lo = r * self.bin_h; ry_hi = ry_lo + self.bin_h
            oy = max(0.0, min(y_hi, ry_hi) - max(y_lo, ry_lo))
            if oy <= 0: continue
            for c in range(col_lo, col_hi + 1):
                rx_lo = c * self.bin_w; rx_hi = rx_lo + self.bin_w
                ox = max(0.0, min(x_hi, rx_hi) - max(x_lo, rx_lo))
                if ox <= 0: continue
                self.density[r, c] += sign * ox * oy

    def total_hpwl(self) -> float:
        return float(self.net_hpwl.sum())

    def density_cost(self) -> float:
        flat = self.density.flatten()
        n_top = max(1, int(np.ceil(len(flat) * 0.1)))
        top_vals = np.partition(flat, -n_top)[-n_top:]
        bin_area = self.bin_w * self.bin_h
        return float(top_vals.mean() / bin_area)

    def surrogate_cost(self) -> float:
        wl = self.total_hpwl() / max(self.n_nets * (self.cw + self.ch), 1e-12)
        return self.wl_weight * wl + self.den_weight * self.density_cost()

    def proposed_move_cost(self, macro_idx: int, new_x: float, new_y: float) -> float:
        old_x = self.pos[macro_idx, 0]; old_y = self.pos[macro_idx, 1]
        affected_nets = np.unique(self.net_ids[self.owner_to_pin_entries[macro_idx]])
        old_hpwl_for_nets = {nid: self.net_hpwl[nid] for nid in affected_nets}
        new_hpwl_for_nets = {}
        for entry in self.owner_to_pin_entries[macro_idx]:
            self.pin_xy[entry, 0] = new_x + self.pin_offset_xy[entry, 0]
            self.pin_xy[entry, 1] = new_y + self.pin_offset_xy[entry, 1]
        for nid in affected_nets:
            entries = self.net_to_pin_entries[nid]
            if len(entries) <= 1:
                new_hpwl_for_nets[nid] = 0.0
                continue
            xs = self.pin_xy[entries, 0]; ys = self.pin_xy[entries, 1]
            new_hpwl_for_nets[nid] = (xs.max() - xs.min()) + (ys.max() - ys.min())
        for entry in self.owner_to_pin_entries[macro_idx]:
            self.pin_xy[entry, 0] = old_x + self.pin_offset_xy[entry, 0]
            self.pin_xy[entry, 1] = old_y + self.pin_offset_xy[entry, 1]
        delta_hpwl = sum(new_hpwl_for_nets[nid] - old_hpwl_for_nets[nid]
                         for nid in affected_nets)
        new_total_hpwl = self.total_hpwl() + delta_hpwl

        self._add_density_contribution(macro_idx, -1.0)
        self.pos[macro_idx, 0] = new_x; self.pos[macro_idx, 1] = new_y
        self._add_density_contribution(macro_idx, +1.0)
        new_density_cost = self.density_cost()
        self._add_density_contribution(macro_idx, -1.0)
        self.pos[macro_idx, 0] = old_x; self.pos[macro_idx, 1] = old_y
        self._add_density_contribution(macro_idx, +1.0)

        wl_norm = new_total_hpwl / max(self.n_nets * (self.cw + self.ch), 1e-12)
        return self.wl_weight * wl_norm + self.den_weight * new_density_cost

    def commit_move(self, macro_idx: int, new_x: float, new_y: float):
        old_x = self.pos[macro_idx, 0]; old_y = self.pos[macro_idx, 1]
        self._add_density_contribution(macro_idx, -1.0)
        self.pos[macro_idx, 0] = new_x; self.pos[macro_idx, 1] = new_y
        self._add_density_contribution(macro_idx, +1.0)
        for entry in self.owner_to_pin_entries[macro_idx]:
            self.pin_xy[entry, 0] = new_x + self.pin_offset_xy[entry, 0]
            self.pin_xy[entry, 1] = new_y + self.pin_offset_xy[entry, 1]
        affected_nets = np.unique(self.net_ids[self.owner_to_pin_entries[macro_idx]])
        for nid in affected_nets:
            entries = self.net_to_pin_entries[nid]
            if len(entries) <= 1:
                self.net_hpwl[nid] = 0.0
                continue
            xs = self.pin_xy[entries, 0]; ys = self.pin_xy[entries, 1]
            self.net_hpwl[nid] = (xs.max() - xs.min()) + (ys.max() - ys.min())


def _try_swap(
    inc: "IncrementalProxy", i: int, j: int,
    sep_x: np.ndarray, sep_y: np.ndarray,
    half_w: np.ndarray, half_h: np.ndarray, cw: float, ch: float,
) -> "float | None":
    pos_i_old = (inc.pos[i, 0], inc.pos[i, 1])
    pos_j_old = (inc.pos[j, 0], inc.pos[j, 1])
    n_hard = len(half_w)

    nx_i = float(np.clip(pos_j_old[0], half_w[i], cw - half_w[i]))
    ny_i = float(np.clip(pos_j_old[1], half_h[i], ch - half_h[i]))
    nx_j = float(np.clip(pos_i_old[0], half_w[j], cw - half_w[j]))
    ny_j = float(np.clip(pos_i_old[1], half_h[j], ch - half_h[j]))

    ddx = np.abs(nx_i - inc.pos[:n_hard, 0]); ddy = np.abs(ny_i - inc.pos[:n_hard, 1])
    mask_i = (ddx < sep_x[i]) & (ddy < sep_y[i]); mask_i[i] = False; mask_i[j] = False
    if mask_i.any():
        return None
    ddx = np.abs(nx_j - inc.pos[:n_hard, 0]); ddy = np.abs(ny_j - inc.pos[:n_hard, 1])
    mask_j = (ddx < sep_x[j]) & (ddy < sep_y[j]); mask_j[i] = False; mask_j[j] = False
    if mask_j.any():
        return None
    if abs(nx_i - nx_j) < sep_x[i, j] and abs(ny_i - ny_j) < sep_y[i, j]:
        return None

    inc.commit_move(i, nx_i, ny_i)
    inc.commit_move(j, nx_j, ny_j)
    cost = inc.surrogate_cost()
    inc.commit_move(i, pos_i_old[0], pos_i_old[1])
    inc.commit_move(j, pos_j_old[0], pos_j_old[1])
    return cost


def _real_proxy_cd(
    full_pos: torch.Tensor, benchmark: Benchmark, plc,
    n_passes: int, step_fracs: tuple, rng_seed: int,
    do_swaps: bool = True, n_swap_neighbors: int = 10,
    n_extra_random: int = 0,
    swap_every: int = 5,
    checkpoint_every: int = 10,
    max_seconds: float = 1800.0,
    proxy_weights: "dict | None" = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, float]:
    """
    Real-proxy CD using IncrementalRealProxy (~1240x faster than
    compute_proxy_cost). Optimizes the EXACT TILOS proxy.

    Enhanced vs v14: swap pass every `swap_every` passes (not just last),
    checkpointing every `checkpoint_every` passes, more neighbors,
    time-based budget via `max_seconds`.
    """
    import sys, os
    _v14_dir = os.path.dirname(os.path.abspath(__file__))
    if _v14_dir not in sys.path:
        sys.path.insert(0, _v14_dir)
    from fast_incremental_proxy import FastIncrementalProxy as IncrementalRealProxy  # type: ignore
    from macro_place.objective import compute_proxy_cost

    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
    sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
    movable = benchmark.get_movable_mask().numpy()
    movable_hard = movable[:n_hard]
    movable_idx = np.where(movable_hard)[0]
    sx = sizes_np[:n_hard, 0]; sy = sizes_np[:n_hard, 1]
    sep_x = (sx[:, None] + sx[None, :]) / 2 + 0.001
    sep_y = (sy[:, None] + sy[None, :]) / 2 + 0.001
    half_w = sx / 2; half_h = sy / 2

    pos_np = full_pos.numpy().astype(np.float64).copy()
    inc = IncrementalRealProxy(benchmark, pos_np, plc=plc)

    rng = random.Random(rng_seed)
    initial_real = compute_proxy_cost(torch.from_numpy(pos_np).float(), benchmark, plc)["proxy_cost"]
    best_real = initial_real
    best_pos = full_pos.clone()
    cd_start = time.time()
    deadline = cd_start + max_seconds
    if verbose:
        print(f"    [cd] start plc={initial_real:.4f} inc={inc.proxy_cost():.4f}")

    for p_idx, step_frac in enumerate(step_fracs[:n_passes]):
        if time.time() > deadline:
            if verbose:
                print(f"    [cd] stopping at pass {p_idx} — time limit {max_seconds:.0f}s")
            break
        step = max(cw, ch) * step_frac
        offsets = [(step, 0), (-step, 0), (0, step), (0, -step),
                   (step, step), (-step, step), (step, -step), (-step, -step)]
        order = list(movable_idx); rng.shuffle(order)
        improved = 0
        baseline_inc = inc.proxy_cost(proxy_weights)
        _timed_out = False

        for i in order:
            if time.time() > deadline:  # enforce budget mid-pass for large benchmarks
                _timed_out = True
                break
            ox = inc.pos[i, 0]; oy = inc.pos[i, 1]
            best_local = (ox, oy, baseline_inc)
            for dx, dy in offsets:
                nx = float(np.clip(ox + dx, half_w[i], cw - half_w[i]))
                ny = float(np.clip(oy + dy, half_h[i], ch - half_h[i]))
                ddx = np.abs(nx - inc.pos[:n_hard, 0]); ddy = np.abs(ny - inc.pos[:n_hard, 1])
                mask = (ddx < sep_x[i]) & (ddy < sep_y[i]); mask[i] = False
                if mask.any():
                    continue
                c = inc.proposed_move_cost(i, nx, ny, weights=proxy_weights)
                if c < best_local[2]:
                    best_local = (nx, ny, c)
            for _ in range(n_extra_random):
                nx = rng.uniform(half_w[i], cw - half_w[i])
                ny = rng.uniform(half_h[i], ch - half_h[i])
                ddx = np.abs(nx - inc.pos[:n_hard, 0]); ddy = np.abs(ny - inc.pos[:n_hard, 1])
                mask = (ddx < sep_x[i]) & (ddy < sep_y[i]); mask[i] = False
                if mask.any():
                    continue
                c = inc.proposed_move_cost(i, nx, ny, weights=proxy_weights)
                if c < best_local[2]:
                    best_local = (nx, ny, c)
            if best_local[2] < baseline_inc:
                inc.commit_move(i, best_local[0], best_local[1])
                baseline_inc = best_local[2]
                improved += 1

        if _timed_out:
            # Checkpoint partial-pass progress before returning
            new_pos_t = torch.from_numpy(inc.pos).float()
            new_real = compute_proxy_cost(new_pos_t, benchmark, plc)["proxy_cost"]
            if verbose:
                print(f"    [cd] timeout mid-pass {p_idx+1} shifts={improved} plc={new_real:.4f}")
            if new_real < best_real:
                best_real = new_real
                best_pos = new_pos_t.clone()
            break

        # Swap pass every `swap_every` passes and on the final pass
        if do_swaps and (p_idx % swap_every == (swap_every - 1) or p_idx == n_passes - 1):
            swap_baseline = inc.proxy_cost()  # always use standard proxy for swaps
            for i in order:
                ix, iy = inc.pos[i, 0], inc.pos[i, 1]
                d = np.hypot(inc.pos[movable_idx, 0] - ix, inc.pos[movable_idx, 1] - iy)
                neighbor_local = np.argsort(d)[1 : n_swap_neighbors + 1]
                neighbors = movable_idx[neighbor_local]
                for j in neighbors:
                    if j == i: continue
                    new_cost = _try_swap_real(inc, int(i), int(j), sep_x, sep_y, half_w, half_h, cw, ch)
                    if new_cost is not None and new_cost < swap_baseline:
                        pi = (inc.pos[i, 0], inc.pos[i, 1])
                        pj = (inc.pos[j, 0], inc.pos[j, 1])
                        nx_i = float(np.clip(pj[0], half_w[i], cw - half_w[i]))
                        ny_i = float(np.clip(pj[1], half_h[i], ch - half_h[i]))
                        nx_j = float(np.clip(pi[0], half_w[j], cw - half_w[j]))
                        ny_j = float(np.clip(pi[1], half_h[j], ch - half_h[j]))
                        inc.commit_move(int(i), nx_i, ny_i)
                        inc.commit_move(int(j), nx_j, ny_j)
                        swap_baseline = new_cost
            baseline_inc = inc.proxy_cost(proxy_weights)  # update biased baseline after swaps

        # Checkpoint with real plc every `checkpoint_every` passes and on the final pass
        if p_idx % checkpoint_every == (checkpoint_every - 1) or p_idx == n_passes - 1:
            new_pos_t = torch.from_numpy(inc.pos).float()
            new_real = compute_proxy_cost(new_pos_t, benchmark, plc)["proxy_cost"]
            if verbose:
                print(f"    [cd] pass {p_idx+1} step={step_frac:.2%} "
                      f"shifts={improved} inc={baseline_inc:.4f} plc={new_real:.4f}")
            if new_real < best_real:
                best_real = new_real
                best_pos = new_pos_t.clone()
            elif proxy_weights is None:
                # Standard mode: reset to best known position to prevent drift
                inc = IncrementalRealProxy(
                    benchmark, best_pos.numpy().astype(np.float64).copy(), plc=plc
                )
            # Biased mode (proxy_weights != None): allow exploration without reset

    return best_pos, best_real


def _try_swap_real(
    inc, i: int, j: int, sep_x, sep_y, half_w, half_h, cw, ch,
):
    """Try swap with IncrementalRealProxy."""
    pos_i_old = (inc.pos[i, 0], inc.pos[i, 1])
    pos_j_old = (inc.pos[j, 0], inc.pos[j, 1])
    n_hard = len(half_w)

    nx_i = float(np.clip(pos_j_old[0], half_w[i], cw - half_w[i]))
    ny_i = float(np.clip(pos_j_old[1], half_h[i], ch - half_h[i]))
    nx_j = float(np.clip(pos_i_old[0], half_w[j], cw - half_w[j]))
    ny_j = float(np.clip(pos_i_old[1], half_h[j], ch - half_h[j]))

    ddx = np.abs(nx_i - inc.pos[:n_hard, 0]); ddy = np.abs(ny_i - inc.pos[:n_hard, 1])
    mask_i = (ddx < sep_x[i]) & (ddy < sep_y[i]); mask_i[i] = False; mask_i[j] = False
    if mask_i.any(): return None
    ddx = np.abs(nx_j - inc.pos[:n_hard, 0]); ddy = np.abs(ny_j - inc.pos[:n_hard, 1])
    mask_j = (ddx < sep_x[j]) & (ddy < sep_y[j]); mask_j[i] = False; mask_j[j] = False
    if mask_j.any(): return None
    if abs(nx_i - nx_j) < sep_x[i, j] and abs(ny_i - ny_j) < sep_y[i, j]:
        return None

    inc.commit_move(i, nx_i, ny_i)
    inc.commit_move(j, nx_j, ny_j)
    cost = inc.proxy_cost()
    inc.commit_move(i, pos_i_old[0], pos_i_old[1])
    inc.commit_move(j, pos_j_old[0], pos_j_old[1])
    return cost


def _fast_cd(
    full_pos: torch.Tensor, benchmark: Benchmark, plc,
    n_passes: int, step_fracs: tuple, rng_seed: int,
    do_swaps: bool = True, n_swap_neighbors: int = 5,
    verbose: bool = False,
) -> Tuple[torch.Tensor, float]:
    """Fast surrogate-cost CD with shifts + swaps + real-proxy verification."""
    from macro_place.objective import compute_proxy_cost
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
    sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
    movable = benchmark.get_movable_mask().numpy()
    movable_hard = movable[:n_hard]
    movable_idx = np.where(movable_hard)[0]
    sx = sizes_np[:n_hard, 0]; sy = sizes_np[:n_hard, 1]
    sep_x = (sx[:, None] + sx[None, :]) / 2 + 0.001
    sep_y = (sy[:, None] + sy[None, :]) / 2 + 0.001
    half_w = sx / 2; half_h = sy / 2

    pos_np = full_pos.numpy().astype(np.float64).copy()
    inc = IncrementalProxy(benchmark, pos_np)

    rng = random.Random(rng_seed)
    cur_real = compute_proxy_cost(torch.from_numpy(pos_np).float(), benchmark, plc)["proxy_cost"]
    best_real = cur_real
    best_pos = full_pos.clone()
    if verbose:
        print(f"    [fast-cd] start real proxy={cur_real:.4f}")

    for p_idx, step_frac in enumerate(step_fracs[:n_passes]):
        step = max(cw, ch) * step_frac
        offsets = [(step, 0), (-step, 0), (0, step), (0, -step),
                   (step, step), (-step, step), (step, -step), (-step, -step)]
        order = list(movable_idx); rng.shuffle(order)
        improved = 0
        baseline_surr = inc.surrogate_cost()

        for i in order:
            ox = inc.pos[i, 0]; oy = inc.pos[i, 1]
            best_local = (ox, oy, baseline_surr)
            for dx, dy in offsets:
                nx = float(np.clip(ox + dx, half_w[i], cw - half_w[i]))
                ny = float(np.clip(oy + dy, half_h[i], ch - half_h[i]))
                ddx = np.abs(nx - inc.pos[:n_hard, 0]); ddy = np.abs(ny - inc.pos[:n_hard, 1])
                mask = (ddx < sep_x[i]) & (ddy < sep_y[i]); mask[i] = False
                if mask.any():
                    continue
                c = inc.proposed_move_cost(i, nx, ny)
                if c < best_local[2]:
                    best_local = (nx, ny, c)
            if best_local[2] < baseline_surr:
                inc.commit_move(i, best_local[0], best_local[1])
                baseline_surr = best_local[2]
                improved += 1

        if do_swaps and p_idx == n_passes - 1:
            for i in order:
                ix, iy = inc.pos[i, 0], inc.pos[i, 1]
                d = np.hypot(inc.pos[movable_idx, 0] - ix, inc.pos[movable_idx, 1] - iy)
                neighbor_local = np.argsort(d)[1 : n_swap_neighbors + 1]
                neighbors = movable_idx[neighbor_local]
                for j in neighbors:
                    if j == i: continue
                    new_cost = _try_swap(inc, int(i), int(j), sep_x, sep_y, half_w, half_h, cw, ch)
                    if new_cost is not None and new_cost < baseline_surr:
                        pi = (inc.pos[i, 0], inc.pos[i, 1])
                        pj = (inc.pos[j, 0], inc.pos[j, 1])
                        nx_i = float(np.clip(pj[0], half_w[i], cw - half_w[i]))
                        ny_i = float(np.clip(pj[1], half_h[i], ch - half_h[i]))
                        nx_j = float(np.clip(pi[0], half_w[j], cw - half_w[j]))
                        ny_j = float(np.clip(pi[1], half_h[j], ch - half_h[j]))
                        inc.commit_move(int(i), nx_i, ny_i)
                        inc.commit_move(int(j), nx_j, ny_j)
                        baseline_surr = new_cost

        new_pos_t = torch.from_numpy(inc.pos).float()
        new_real = compute_proxy_cost(new_pos_t, benchmark, plc)["proxy_cost"]
        if verbose:
            print(f"    [fast-cd] pass {p_idx+1} step={step_frac:.2%}: "
                  f"shifts={improved}/{len(order)} surrogate={baseline_surr:.4f} real={new_real:.4f}")
        if new_real < best_real:
            best_real = new_real
            best_pos = new_pos_t.clone()
        else:
            inc.pos = best_pos.numpy().astype(np.float64).copy()
            inc.pin_xy = inc._pin_xy_full()
            inc._recompute_all_hpwl()
            inc._recompute_density_full()

    return best_pos, best_real


# ─────────────────────── Helpers for v12 ───────────────────── #
def _compute_per_net_hpwl(
    pos_t: torch.Tensor,
    benchmark: Benchmark,
    device: torch.device,
) -> np.ndarray:
    """Compute true per-net HPWL (max−min of pin x + max−min of pin y) at given pos."""
    pin_data = _build_pin_data(benchmark, device)
    if pin_data is None:
        return np.zeros(0)
    n_nets = pin_data["n_nets"]
    n_real = pin_data["n_macros"]
    n_ports = pin_data["n_ports"]
    net_owners = pin_data["net_owners"]
    net_ids = pin_data["net_ids"]
    pin_offset_xy = pin_data["pin_offset_xy"]
    port_pos = benchmark.port_positions.to(device).float()

    p = pos_t.to(device).float()
    owner_pos = torch.cat([p[:n_real], port_pos], dim=0) if n_ports > 0 else p[:n_real]
    pin_xy = owner_pos[net_owners] + pin_offset_xy

    big = 1e18
    max_x = torch.full((n_nets,), -big, device=device).scatter_reduce_(
        0, net_ids, pin_xy[:, 0], reduce="amax", include_self=True)
    min_x = torch.full((n_nets,), big, device=device).scatter_reduce_(
        0, net_ids, pin_xy[:, 0], reduce="amin", include_self=True)
    max_y = torch.full((n_nets,), -big, device=device).scatter_reduce_(
        0, net_ids, pin_xy[:, 1], reduce="amax", include_self=True)
    min_y = torch.full((n_nets,), big, device=device).scatter_reduce_(
        0, net_ids, pin_xy[:, 1], reduce="amin", include_self=True)
    hpwl = (max_x - min_x) + (max_y - min_y)
    # Replace -inf for empty nets
    hpwl = torch.where(hpwl < 0, torch.zeros_like(hpwl), hpwl)
    return hpwl.detach().cpu().numpy()


def _multi_start_legalize_and_score(
    spos: np.ndarray,
    fixed_pos: np.ndarray,
    sizes_np: np.ndarray,
    movable_hard: np.ndarray,
    n_hard: int,
    cw: float,
    ch: float,
    benchmark: Benchmark,
    plc,
    soft_iters: int,
    soft_damping: float,
    orderings: List[Tuple[str, List[int]]],
) -> Tuple[float, torch.Tensor, str]:
    """Run multi-start legalize + soft jacobi from spos. Returns (best_cost, best_pos_t, best_label)."""
    from macro_place.objective import compute_proxy_cost
    best_cost = float("inf")
    best_pos = None
    best_label = "none"
    for oname, order in orderings:
        pos_np = spos.copy()
        legal = _legalize_with_order(
            pos_np[:n_hard].copy(), movable_hard, sizes_np, cw, ch, n_hard,
            fixed_pos[:n_hard], order,
        )
        pos_np[:n_hard] = legal

        full_legal = torch.from_numpy(pos_np).float()
        if plc is not None:
            cd = compute_proxy_cost(full_legal, benchmark, plc)
            if cd["overlap_count"] == 0 and cd["proxy_cost"] < best_cost:
                best_cost = cd["proxy_cost"]
                best_pos = full_legal.clone()
                best_label = f"{oname}/legal"

        pos_refined = _soft_jacobi_update(
            pos_np, benchmark, n_iters=soft_iters, damping=soft_damping,
        )
        full_ref = torch.from_numpy(pos_refined).float()
        if plc is not None:
            cd = compute_proxy_cost(full_ref, benchmark, plc)
            if cd["overlap_count"] == 0 and cd["proxy_cost"] < best_cost:
                best_cost = cd["proxy_cost"]
                best_pos = full_ref.clone()
                best_label = f"{oname}/refined"
    return best_cost, best_pos, best_label


def _wire_mask_sweep(
    pos: torch.Tensor,
    benchmark: Benchmark,
    plc,
    n_passes: int = 3,
    grid_n: int = 12,
    max_seconds: float = 120.0,
    proxy_weights: "dict | None" = None,
    rng_seed: int = 0,
    verbose: bool = False,
) -> "Tuple[torch.Tensor, float]":
    """
    Wire-mask greedy refinement (WireMask-BBO, NeurIPS 2023): for each macro try
    all positions on a grid_n×grid_n grid and accept the globally best improvement.
    Systematically covers the canvas unlike step-based CD, enabling large positional
    jumps that escape CD local minima. Safe: validates legality post-sweep.
    """
    from macro_place.objective import compute_proxy_cost
    import os as _os, sys as _sys
    _dir = _os.path.dirname(_os.path.abspath(__file__))
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)
    from fast_incremental_proxy import FastIncrementalProxy as _IRP

    n_hard = benchmark.num_hard_macros
    movable = benchmark.get_movable_mask().numpy()
    movable_hard = movable[:n_hard]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
    half_w = sizes_np[:n_hard, 0] / 2
    half_h = sizes_np[:n_hard, 1] / 2

    pos_np = pos.numpy().astype(np.float64).copy()
    try:
        cd = compute_proxy_cost(torch.from_numpy(pos_np).float(), benchmark, plc)
        if cd["overlap_count"] > 0:
            return pos, float("inf")
        cur_cost = cd["proxy_cost"]
    except Exception:
        return pos, float("inf")

    inc = _IRP(benchmark, pos_np.copy(), plc=plc)

    # Grid: grid_n internal positions, avoiding canvas edges
    x_grid = np.linspace(0, cw, grid_n + 2)[1:-1]
    y_grid = np.linspace(0, ch, grid_n + 2)[1:-1]

    rng = np.random.default_rng(rng_seed)
    order = np.arange(n_hard)

    t0 = time.time()
    total_improved = 0

    for pass_idx in range(n_passes):
        if time.time() - t0 > max_seconds:
            break
        rng.shuffle(order)
        pass_improved = 0

        for i in order:
            if not movable_hard[i]:
                continue
            if time.time() - t0 > max_seconds:
                break

            best_c = cur_cost
            best_nx = pos_np[i, 0]
            best_ny = pos_np[i, 1]

            for gx in x_grid:
                nx = float(np.clip(gx, half_w[i], cw - half_w[i]))
                for gy in y_grid:
                    ny = float(np.clip(gy, half_h[i], ch - half_h[i]))
                    c = inc.proposed_move_cost(i, nx, ny, weights=proxy_weights)
                    if c < best_c:
                        best_c = c
                        best_nx = nx
                        best_ny = ny

            if best_nx != pos_np[i, 0] or best_ny != pos_np[i, 1]:
                pos_np[i, 0] = best_nx
                pos_np[i, 1] = best_ny
                inc.update(pos_np)
                cur_cost = best_c
                pass_improved += 1

        total_improved += pass_improved
        if verbose:
            print(f"  [wire-mask] pass {pass_idx + 1}: {pass_improved} moves → proxy={cur_cost:.4f}")
        if pass_improved == 0:
            break

    result_pos = torch.from_numpy(pos_np).float()
    return result_pos, cur_cost


def _orientation_search(
    best_pos: torch.Tensor, benchmark: Benchmark, plc,
    rng_seed: int = 0, max_seconds: float = 120.0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, float]:
    """
    Multi-pass greedy orientation search: for each hard macro, try all orientations
    in its group (NS: N/FN/S/FS, EW: E/FE/W/FW). Respects initial orientation group
    (same constraint as competition's coordinate_descent_placer).
    Repeats passes until no improvement found (convergence) or time runs out.
    """
    from macro_place.objective import compute_proxy_cost
    _NS = ["N", "FN", "S", "FS"]
    _EW = ["E", "FE", "W", "FW"]
    _NS_SET = frozenset(_NS)
    n_hard = benchmark.num_hard_macros
    movable = benchmark.get_movable_mask().numpy()
    movable_hard = movable[:n_hard]

    cur_pos = best_pos.clone()
    cur_cost_dict = compute_proxy_cost(cur_pos, benchmark, plc)
    cur_cost = cur_cost_dict["proxy_cost"]
    if cur_cost_dict["overlap_count"] > 0:
        return best_pos, float("inf")

    # Precompute orientation group per macro (NS or EW, based on initial orientation)
    _orient_group = {}
    for i in range(n_hard):
        plc_idx = benchmark.hard_macro_indices[i]
        init_o = plc.modules_w_pins[plc_idx].get_orientation()
        _orient_group[i] = _NS if (init_o in _NS_SET) else _EW

    rng = random.Random(rng_seed)
    order = list(range(n_hard))
    total_improved = 0
    n_passes = 0
    t0 = time.time()

    while time.time() - t0 < max_seconds:
        rng.shuffle(order)
        pass_improved = 0

        for i in order:
            if not movable_hard[i]:
                continue
            if time.time() - t0 > max_seconds:
                break
            plc_idx = benchmark.hard_macro_indices[i]
            cur_orient = plc.modules_w_pins[plc_idx].get_orientation()
            best_orient = cur_orient
            best_orient_cost = cur_cost

            for orient in _orient_group[i]:
                if orient == cur_orient:
                    continue
                plc.update_macro_orientation(plc_idx, orient)
                cd = compute_proxy_cost(cur_pos, benchmark, plc)
                if cd["overlap_count"] == 0 and cd["proxy_cost"] < best_orient_cost:
                    best_orient_cost = cd["proxy_cost"]
                    best_orient = orient
                # Restore to current best for this macro before trying next
                plc.update_macro_orientation(plc_idx, cur_orient)

            if best_orient != cur_orient:
                plc.update_macro_orientation(plc_idx, best_orient)
                cur_cost = best_orient_cost
                pass_improved += 1

        n_passes += 1
        total_improved += pass_improved
        if pass_improved == 0:
            break  # converged

    if verbose:
        print(f"  [orient] {n_passes} pass(es), {total_improved} improvements → proxy={cur_cost:.4f}")

    return cur_pos, cur_cost


# ─────────────────────── Archgen V3 Placer ───────────────────── #
# v3 over v2: LNS phase (large neighborhood search) added after main CD.
# v2 over v1: FastIncrementalProxy (1.57× CD speedup), 60 CD passes (up from 40).
_ARCHGEN_STEP_FRACS = (
    0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.055, 0.05,  # coarse: 8 passes
    0.045, 0.04, 0.035, 0.03, 0.025, 0.022, 0.020, 0.018,  # medium: 8 passes
    0.016, 0.014, 0.012, 0.010, 0.009, 0.008, 0.007, 0.006,  # medium-fine: 8 passes
    0.005, 0.004, 0.003, 0.0025, 0.002, 0.0018, 0.0015, 0.001,  # fine: 8 passes
    0.0008, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.00005,  # very fine: 8 passes
)  # 40 total
def _score_macros_congestion(
    pos_np: np.ndarray,
    benchmark,
    plc,
) -> np.ndarray:
    """Score each hard macro by max (V_net_norm + H_net_norm) in its grid-cell bbox."""
    import sys as _sys, os as _os
    _dir = _os.path.dirname(_os.path.abspath(__file__))
    if _dir not in _sys.path:
        _sys.path.insert(0, _dir)
    from fast_incremental_proxy import FastIncrementalProxy as _IRP
    n_hard = benchmark.num_hard_macros
    proxy = _IRP(benchmark, pos_np.copy(), plc=plc)
    v_norm = proxy.V_net / max(float(proxy.grid_v_routes), 1e-12)
    h_norm = proxy.H_net / max(float(proxy.grid_h_routes), 1e-12)
    cong = v_norm + h_norm
    sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
    scores = np.zeros(n_hard, dtype=np.float64)
    for m in range(n_hard):
        x, y = pos_np[m, 0], pos_np[m, 1]
        sx, sy = sizes_np[m, 0] / 2, sizes_np[m, 1] / 2
        col_lo = max(0, min(int(max(x - sx, 0.0) / proxy.grid_width), proxy.gc - 1))
        col_hi = max(0, min(int((min(x + sx, proxy.cw) - 1e-12) / proxy.grid_width), proxy.gc - 1))
        row_lo = max(0, min(int(max(y - sy, 0.0) / proxy.grid_height), proxy.gr - 1))
        row_hi = max(0, min(int((min(y + sy, proxy.ch) - 1e-12) / proxy.grid_height), proxy.gr - 1))
        scores[m] = cong[row_lo:row_hi + 1, col_lo:col_hi + 1].max()
    return scores


def _lns_phase(
    best_pos: "torch.Tensor",
    best_cost: float,
    benchmark,
    plc,
    rng_seed: int = 0,
    max_seconds: float = 300.0,
    n_destroy: int = 8,
    n_cd_passes: int = 5,
    cd_step_fracs: tuple = _LNS_STEP_FRACS,
    sa_T0: float = 0.0,
    sa_alpha: float = 0.88,
    lns_n_extra_random: int = 30,
    cong_scatter_candidates: int = 0,
    lns_proxy_weights: "dict | None" = None,
    sa_reheat_interval: int = 0,
    sa_max_reheats: int = 3,
    verbose: bool = False,
) -> "Tuple[torch.Tensor, float]":
    """
    Large Neighborhood Search: each iteration destroys the K most-congested macros,
    scatters them randomly, runs a short CD pass, accepts if global proxy improves.

    sa_reheat_interval > 0: reheat SA temperature after this many iterations without
    global improvement (to T0*0.35). Helps escape congestion traps for large circuits
    where alpha=0.88 causes T→0 within ~50 iterations, leaving 200+ pure greedy iters.
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
    movable = benchmark.get_movable_mask().numpy()
    movable_idx = np.where(movable[:n_hard])[0]
    if len(movable_idx) == 0 or max_seconds < 30.0:
        return best_pos, best_cost
    half_w = sizes_np[:n_hard, 0] / 2
    half_h = sizes_np[:n_hard, 1] / 2
    # Pre-compute widths/heights for overlap check
    w_hard = sizes_np[:n_hard, 0]
    h_hard = sizes_np[:n_hard, 1]
    eligible = movable_idx.tolist()
    k = min(n_destroy, len(eligible))
    rng = np.random.default_rng(rng_seed)

    # SA temperature: T0=0 means greedy (same as v3); T0>0 enables SA acceptance
    temperature = sa_T0 if sa_T0 > 0 else 0.0
    _n_reheats = 0
    _iters_since_global_improvement = 0

    t0 = time.time()
    cur_pos = best_pos.numpy().astype(np.float64).copy()
    cur_cost = best_cost
    best_pos_lns = best_pos.clone()
    best_cost_lns = best_cost
    n_iters = n_accepted = 0

    while time.time() - t0 < max_seconds - 5.0:
        # Score macros by local congestion; pick top-2K, randomly choose K
        scores = _score_macros_congestion(cur_pos, benchmark, plc)
        eligible_arr = np.array(eligible)
        top_2k = min(k * 2, len(eligible_arr))
        top_local = np.argsort(-scores[eligible_arr])[:top_2k]
        k_actual = min(k, len(top_local))
        chosen = rng.choice(top_local, size=k_actual, replace=False)
        destroy_set = [int(eligible_arr[c]) for c in chosen]

        old_pos = cur_pos.copy()
        if cong_scatter_candidates > 1:
            # Congestion-aware scatter: try N candidates, pick lowest-cong region
            import sys as _sys, os as _os
            _dir = _os.path.dirname(_os.path.abspath(__file__))
            if _dir not in _sys.path:
                _sys.path.insert(0, _dir)
            from fast_incremental_proxy import FastIncrementalProxy as _IRP
            _proxy_tmp = _IRP(benchmark, cur_pos.copy(), plc=plc)
            _v_norm = _proxy_tmp.V_net / max(float(_proxy_tmp.grid_v_routes), 1e-12)
            _h_norm = _proxy_tmp.H_net / max(float(_proxy_tmp.grid_h_routes), 1e-12)
            _cong_grid = _v_norm + _h_norm
            _gw = _proxy_tmp.grid_width; _gh = _proxy_tmp.grid_height
            _gc = _proxy_tmp.gc; _gr = _proxy_tmp.gr
            del _proxy_tmp
            for m in destroy_set:
                best_x = float(rng.uniform(half_w[m], cw - half_w[m]))
                best_y = float(rng.uniform(half_h[m], ch - half_h[m]))
                best_c = _cong_grid[min(int(best_y / _gh), _gr - 1), min(int(best_x / _gw), _gc - 1)]
                for _ in range(cong_scatter_candidates - 1):
                    nx = float(rng.uniform(half_w[m], cw - half_w[m]))
                    ny = float(rng.uniform(half_h[m], ch - half_h[m]))
                    cc = _cong_grid[min(int(ny / _gh), _gr - 1), min(int(nx / _gw), _gc - 1)]
                    if cc < best_c:
                        best_c = cc; best_x = nx; best_y = ny
                cur_pos[m, 0] = best_x
                cur_pos[m, 1] = best_y
        else:
            for m in destroy_set:
                cur_pos[m, 0] = float(rng.uniform(half_w[m], cw - half_w[m]))
                cur_pos[m, 1] = float(rng.uniform(half_h[m], ch - half_h[m]))

        time_left = max_seconds - (time.time() - t0) - 5.0
        if time_left < 30.0:
            cur_pos = old_pos
            break

        new_pos_t, new_cost = _real_proxy_cd(
            torch.from_numpy(cur_pos).float(),
            benchmark, plc,
            n_passes=n_cd_passes,
            step_fracs=cd_step_fracs[:n_cd_passes],
            rng_seed=int(rng.integers(0, 2 ** 31)),
            do_swaps=True, n_swap_neighbors=10,
            n_extra_random=lns_n_extra_random, swap_every=3,
            checkpoint_every=n_cd_passes,
            max_seconds=time_left,
            proxy_weights=lns_proxy_weights,
            verbose=False,
        )
        # Reject physically invalid placements (overlap check)
        pos_np = new_pos_t.numpy().astype(np.float64)
        px, py = pos_np[:n_hard, 0], pos_np[:n_hard, 1]
        dx = np.abs(px[:, None] - px[None, :])
        dy = np.abs(py[:, None] - py[None, :])
        min_sep_x = (w_hard[:, None] + w_hard[None, :]) / 2.0
        min_sep_y = (h_hard[:, None] + h_hard[None, :]) / 2.0
        ox = np.maximum(0.0, min_sep_x - dx)
        oy = np.maximum(0.0, min_sep_y - dy)
        np.fill_diagonal(ox, 0.0)
        np.fill_diagonal(oy, 0.0)
        has_overlap = bool(((ox > 0) & (oy > 0)).any())

        n_iters += 1
        delta = new_cost - cur_cost  # positive = worse
        if not has_overlap and (delta < 0 or (temperature > 1e-6 and rng.random() < math.exp(-delta / temperature))):
            accepted = True
            n_accepted += 1
            cur_pos = new_pos_t.numpy().astype(np.float64).copy()
            cur_cost = new_cost
            if new_cost < best_cost_lns:
                best_cost_lns = new_cost
                best_pos_lns = new_pos_t.clone()
                _iters_since_global_improvement = 0
            else:
                _iters_since_global_improvement += 1
        else:
            accepted = False
            cur_pos = old_pos
            _iters_since_global_improvement += 1
        if temperature > 0:
            temperature *= sa_alpha
        # Periodic SA reheat: after sa_reheat_interval iters without global improvement,
        # reheat to T0*0.35 to escape congestion traps. Safe: best_pos_lns tracks global
        # best independently, so reheating can never worsen the final returned result.
        if (sa_reheat_interval > 0 and sa_T0 > 0 and temperature < 1e-4
                and _iters_since_global_improvement >= sa_reheat_interval
                and _n_reheats < sa_max_reheats):
            temperature = sa_T0 * 0.35
            _n_reheats += 1
            _iters_since_global_improvement = 0
            if verbose:
                print(f"  [LNS] reheat #{_n_reheats}: T={temperature:.5f}")

        if verbose:
            reason = "ACCEPT" if accepted else ("OVERLAP" if has_overlap else "reject")
            print(f"  [LNS {n_iters}] {reason} "
                  f"proxy={new_cost:.4f}  best={best_cost_lns:.4f}  T={temperature:.5f}")

    if verbose:
        print(f"  [LNS] done: {n_iters} iters, {n_accepted} accepted, "
              f"best={best_cost_lns:.4f}")
    return best_pos_lns, best_cost_lns


class ArchgenV7cPlacer:
    """
    Archgen V7c: V4 + checkpoint reset in _real_proxy_cd.

    Key fix: when a CD checkpoint shows the real proxy is worse than the best
    seen so far, reset the FastIncrementalProxy state to the best known position.
    This prevents accumulated bad moves from leading to worse local optima
    (the ibm13 regression in V4 where CD over-optimizes WL at the cost of
    congestion due to FastIncrementalProxy drift between checkpoints).
    """

    def __init__(
        self, seed: int = 42, device: "torch.device | None" = None,
        gp_iters: int = 500, gp_iters_warm: int = 250,
        n_bins: int = 64,
        target_densities: tuple = (0.70, 0.85, 0.95),
        verbose: bool = False, soft_iters: int = 3, soft_damping: float = 0.5,
        try_init_plc: bool = True,
        n_outer_iters: int = 1,
        reweight_top_frac: float = 0.10,
        reweight_boost: float = 1.5, reweight_decay: float = 0.95,
        do_cd_refine: bool = True,
        cd_passes: int = 60,
        cd_step_fracs: tuple = _ARCHGEN_STEP_FRACS,
        macro_halo_frac: float = 0.0,
        lns_min_budget: float = 150.0,
        lns_n_destroy: int = 8,
        lns_cd_passes: int = 5,
        lns_sa_T0: float = 0.03,
        lns_sa_alpha: float = 0.88,
        lns_sa_reheat_interval: int = 0,
        lns_sa_max_reheats: int = 3,
        cd_n_extra_random: int = 50,
        eplace_per_run_seconds: float = 0.0,
        lns_n_extra_random: int = 30,
        cd_proxy_weights: "dict | None" = None,
        lns_proxy_weights: "dict | None" = None,
        cong_scatter_candidates: int = 0,
        pre_cd_lns_budget: float = 0.0,
        pre_cd_n_destroy: int = 20,
        max_orderings: int = 7,
        eplace_cong_weight: float = 0.0,
        total_budget: float = 3300.0,
    ):
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gp_iters = gp_iters
        self.gp_iters_warm = gp_iters_warm
        self.n_bins = n_bins
        self.target_densities = target_densities
        self.verbose = verbose
        self.soft_iters = soft_iters
        self.soft_damping = soft_damping
        self.try_init_plc = try_init_plc
        self.n_outer_iters = n_outer_iters
        self.reweight_top_frac = reweight_top_frac
        self.reweight_boost = reweight_boost
        self.reweight_decay = reweight_decay
        self.do_cd_refine = do_cd_refine
        self.cd_passes = cd_passes
        self.cd_step_fracs = cd_step_fracs
        self.macro_halo_frac = macro_halo_frac
        self.lns_min_budget = lns_min_budget
        self.lns_n_destroy = lns_n_destroy
        self.lns_cd_passes = lns_cd_passes
        self.lns_sa_T0 = lns_sa_T0
        self.lns_sa_alpha = lns_sa_alpha
        self.lns_sa_reheat_interval = lns_sa_reheat_interval
        self.lns_sa_max_reheats = lns_sa_max_reheats
        self.cd_n_extra_random = cd_n_extra_random
        self.eplace_per_run_seconds = eplace_per_run_seconds
        self.lns_n_extra_random = lns_n_extra_random
        self.cd_proxy_weights = cd_proxy_weights
        self.lns_proxy_weights = lns_proxy_weights
        self.cong_scatter_candidates = cong_scatter_candidates
        self.pre_cd_lns_budget = pre_cd_lns_budget
        self.pre_cd_n_destroy = pre_cd_n_destroy
        self.max_orderings = max_orderings
        self.eplace_cong_weight = eplace_cong_weight
        self.total_budget = total_budget

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        _place_start = time.time()
        _total_budget = self.total_budget  # default 3300s; override per-variant for slow benchmarks

        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
        sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
        movable = benchmark.get_movable_mask().numpy()
        movable_hard = movable[:n_hard]
        fixed_pos = benchmark.macro_positions.numpy().astype(np.float64)

        plc = _load_plc(benchmark.name)
        n_nets = len(benchmark.net_pin_nodes)
        if n_nets == 0:
            return torch.from_numpy(fixed_pos).float()

        # Net pin counts (for HPWL normalization)
        net_pin_counts = np.array(
            [pins.shape[0] for pins in benchmark.net_pin_nodes], dtype=np.float64
        )

        orderings = _build_orderings(sizes_np, fixed_pos, n_hard, cw, ch)
        if self.max_orderings < len(orderings):
            orderings = orderings[:self.max_orderings]

        best_cost = float("inf"); best_pos = None; best_label = None

        # ─── Baseline path: init.plc warm + multi-start legalize + CD ───
        if self.try_init_plc:
            cost_init, pos_init, label_init = _multi_start_legalize_and_score(
                fixed_pos, fixed_pos, sizes_np, movable_hard, n_hard, cw, ch,
                benchmark, plc, self.soft_iters, self.soft_damping, orderings,
            )
            if pos_init is not None and cost_init < best_cost:
                best_cost = cost_init
                best_pos = pos_init
                best_label = f"init/{label_init}"
            if self.verbose:
                print(f"  [init.plc legal] proxy={cost_init:.4f}")
            # CD refinement on init.plc legalized
            if pos_init is not None and self.do_cd_refine and plc is not None:
                _cd_budget = max(60.0, _total_budget - (time.time() - _place_start))
                refined, refined_cost = _real_proxy_cd(
                    pos_init, benchmark, plc,
                    n_passes=self.cd_passes, step_fracs=self.cd_step_fracs,
                    rng_seed=self.seed, do_swaps=True, n_swap_neighbors=10,
                    n_extra_random=self.cd_n_extra_random, swap_every=5, checkpoint_every=10,
                    max_seconds=_cd_budget,
                    proxy_weights=self.cd_proxy_weights,
                    verbose=self.verbose,
                )
                if refined_cost < best_cost:
                    best_cost = refined_cost
                    best_pos = refined
                    best_label = f"init/CD"
                if self.verbose:
                    print(f"  [init.plc + CD] proxy={refined_cost:.4f}")

        # ─── Multi-target-density DREAMPlace runs ────────────────
        for td in self.target_densities:
            net_weights = torch.ones(n_nets, device=self.device)
            cur_pos_real = None

            for it in range(self.n_outer_iters):
                iters_this = self.gp_iters if it == 0 else self.gp_iters_warm
                gp_pos_t, _ = _eplace_global(
                    benchmark, self.device,
                    n_iters=iters_this, n_bins=self.n_bins,
                    target_density=td, verbose=False,
                    net_weights=net_weights,
                    init_pos_real=cur_pos_real,
                    macro_halo_frac=self.macro_halo_frac,
                    eplace_cong_weight=self.eplace_cong_weight,
                    max_seconds=self.eplace_per_run_seconds,
                )
                cur_pos_real = gp_pos_t.clone()
                gp_full = np.zeros_like(fixed_pos)
                gp_full[:benchmark.num_macros] = gp_pos_t.numpy().astype(np.float64)

                cost_iter, pos_iter, label_iter = _multi_start_legalize_and_score(
                    gp_full, fixed_pos, sizes_np, movable_hard, n_hard, cw, ch,
                    benchmark, plc, self.soft_iters, self.soft_damping, orderings,
                )

                if pos_iter is not None and cost_iter < best_cost:
                    best_cost = cost_iter
                    best_pos = pos_iter
                    best_label = f"td={td}/iter{it}/{label_iter}"
                if self.verbose:
                    print(f"  [td={td:.2f} iter {it}] gp_iters={iters_this} legal → proxy={cost_iter:.4f}  best={best_cost:.4f}")

                # CD refinement on this iteration's legalized result
                if pos_iter is not None and self.do_cd_refine and plc is not None:
                    _cd_budget = max(60.0, _total_budget - (time.time() - _place_start))
                    refined, refined_cost = _real_proxy_cd(
                        pos_iter, benchmark, plc,
                        n_passes=self.cd_passes, step_fracs=self.cd_step_fracs,
                        rng_seed=self.seed + it + 1, do_swaps=True, n_swap_neighbors=10,
                        n_extra_random=self.cd_n_extra_random, swap_every=5, checkpoint_every=10,
                        max_seconds=_cd_budget,
                        proxy_weights=self.cd_proxy_weights,
                        verbose=self.verbose,
                    )
                    if refined_cost < best_cost:
                        best_cost = refined_cost
                        best_pos = refined
                        best_label = f"td={td}/iter{it}/CD"
                    if self.verbose:
                        print(f"  [td={td:.2f} iter {it}] + CD → proxy={refined_cost:.4f}  best={best_cost:.4f}")

                # REWEIGHT (only if multiple outer iters)
                if it < self.n_outer_iters - 1 and pos_iter is not None:
                    hpwl_per_net = _compute_per_net_hpwl(pos_iter, benchmark, self.device)
                    normalized = hpwl_per_net / np.sqrt(np.maximum(net_pin_counts, 1.0))
                    k = max(1, int(self.reweight_top_frac * n_nets))
                    top_idx = np.argsort(-normalized)[:k]
                    nw_np = net_weights.cpu().numpy()
                    nw_np = 1.0 + (nw_np - 1.0) * self.reweight_decay
                    nw_np[top_idx] *= self.reweight_boost
                    nw_np = np.clip(nw_np, 0.5, 10.0)
                    net_weights = torch.from_numpy(nw_np).float().to(self.device)

        # ─── Pre-CD congestion-spreading LNS (v16) ───────────────────────
        if self.pre_cd_lns_budget > 0 and best_pos is not None and plc is not None:
            _elapsed = time.time() - _place_start
            _remaining = _total_budget - _elapsed
            if _remaining > self.pre_cd_lns_budget + self.lns_min_budget + 60:
                if self.verbose:
                    print(f"  [pre-CD LNS] spreading congestion for {self.pre_cd_lns_budget:.0f}s ...")
                spread_pos, spread_cost = _lns_phase(
                    best_pos, best_cost, benchmark, plc,
                    rng_seed=self.seed + 1234,
                    max_seconds=self.pre_cd_lns_budget,
                    n_destroy=self.pre_cd_n_destroy,
                    n_cd_passes=1,
                    cd_step_fracs=(_LNS_STEP_FRACS[0],),
                    cong_scatter_candidates=max(self.cong_scatter_candidates, 10),
                    lns_proxy_weights=self.lns_proxy_weights,
                    lns_n_extra_random=5,
                    verbose=self.verbose,
                )
                if spread_cost < best_cost:
                    best_cost = spread_cost
                    best_pos = spread_pos
                    if self.verbose:
                        print(f"  [pre-CD LNS] improved → proxy={spread_cost:.4f}")

        # ─── Intermediate orientation search (before LNS) ─────────────────
        # Fixing orientations before LNS improves LNS initial state; the final
        # orientation search after LNS further refines from the LNS result.
        _t_mid = time.time() - _place_start
        _mid_o_budget = min(60.0, _total_budget - _t_mid - self.lns_min_budget - 60.0)
        if _mid_o_budget > 20.0 and best_pos is not None and plc is not None:
            try:
                _mid_op, _mid_oc = _orientation_search(
                    best_pos, benchmark, plc,
                    rng_seed=self.seed + 222,
                    max_seconds=_mid_o_budget,
                    verbose=False,
                )
                if _mid_oc < best_cost:
                    best_cost = _mid_oc
                    best_pos = _mid_op
            except Exception:
                pass

        # ─── LNS phase ────────────────────────────────────────────────────
        lns_budget = _total_budget - (time.time() - _place_start)
        if lns_budget > self.lns_min_budget and best_pos is not None and plc is not None:
            if self.verbose:
                print(f"  [LNS] starting with {lns_budget:.0f}s budget ...")
            lns_pos, lns_cost = _lns_phase(
                best_pos, best_cost, benchmark, plc,
                rng_seed=self.seed + 777,
                max_seconds=lns_budget - 30.0,
                n_destroy=self.lns_n_destroy,
                n_cd_passes=self.lns_cd_passes,
                cd_step_fracs=_LNS_STEP_FRACS[:self.lns_cd_passes],
                sa_T0=self.lns_sa_T0,
                sa_alpha=self.lns_sa_alpha,
                lns_n_extra_random=self.lns_n_extra_random,
                cong_scatter_candidates=self.cong_scatter_candidates,
                lns_proxy_weights=self.lns_proxy_weights,
                sa_reheat_interval=self.lns_sa_reheat_interval,
                sa_max_reheats=self.lns_sa_max_reheats,
                verbose=self.verbose,
            )
            if lns_cost < best_cost:
                best_cost = lns_cost
                best_pos = lns_pos
                if self.verbose:
                    print(f"  [LNS] improved → proxy={lns_cost:.4f}")

        if best_pos is None:
            best_pos = torch.from_numpy(fixed_pos.copy()).float()

        # Wire-mask sweep: global positional refinement after LNS.
        _t_wm = time.time() - _place_start
        _wm_budget = min(120.0, _total_budget + 60.0 - _t_wm - 60.0)
        if _wm_budget > 30.0 and best_pos is not None and plc is not None:
            try:
                _wm_p, _wm_c = _wire_mask_sweep(
                    best_pos, benchmark, plc,
                    n_passes=3, grid_n=16,
                    max_seconds=_wm_budget,
                    proxy_weights=self.cd_proxy_weights,
                    rng_seed=self.seed + 555,
                    verbose=False,
                )
                if _wm_c < best_cost:
                    best_cost = _wm_c
                    best_pos = _wm_p
                    if self.verbose:
                        print(f"  [wire-mask] improved → proxy={_wm_c:.4f}")
            except Exception:
                pass

        # Orientation search: try N/FN/S/FS for each macro after CD+LNS.
        # Uses up to 120s beyond the standard budget (wall clock ~3420s, within 3600s limit).
        t_elapsed = time.time() - _place_start
        orient_budget = min(120.0, _total_budget + 120.0 - t_elapsed - 15.0)
        if orient_budget > 20.0 and plc is not None:
            try:
                orient_pos, orient_cost = _orientation_search(
                    best_pos, benchmark, plc,
                    rng_seed=self.seed + 9999,
                    max_seconds=orient_budget,
                    verbose=self.verbose,
                )
                if orient_cost < best_cost:
                    if self.verbose:
                        print(f"  [orient] improved: {best_cost:.4f} → {orient_cost:.4f}")
                    best_cost = orient_cost
                    best_pos = orient_pos
            except Exception as e:
                if self.verbose:
                    print(f"  [orient] error: {e}")

        if self.verbose:
            print(f"  ▶ ArchgenV4 best: proxy={best_cost:.4f}")

        return best_pos
