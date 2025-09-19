"""Sediment transport and deposition helpers."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .rivers import RiverNetwork


def _node_slopes(points: np.ndarray,
                 neighbors: Sequence[np.ndarray],
                 heights: np.ndarray) -> np.ndarray:
    """Compute maximum downslope gradient for each node."""
    slopes = np.zeros(len(points), dtype=np.float64)

    for idx, nbrs in enumerate(neighbors):
        if len(nbrs) == 0:
            continue

        delta = heights[idx] - heights[nbrs]
        if delta.size == 0:
            continue

        offsets = points[nbrs] - points[idx]
        distances = np.linalg.norm(offsets, axis=1)
        valid = distances > 1e-8
        if not np.any(valid):
            continue

        local = np.zeros_like(delta)
        local[valid] = delta[valid] / distances[valid]
        local = np.maximum(local, 0.0)
        if local.size:
            slopes[idx] = np.max(local)

    return slopes


def _transport_capacity(discharge: np.ndarray,
                        slopes: np.ndarray,
                        params) -> np.ndarray:
    """Compute sediment transport capacity per node."""
    s_eff = np.maximum(slopes - params.tau_crit_slope, 0.0)
    capacity = np.power(discharge, params.m_capacity, dtype=np.float64)
    capacity *= np.power(s_eff, params.n_capacity, dtype=np.float64)
    return params.k_capacity * np.nan_to_num(capacity)


def _topological_order(upstream: Sequence[Iterable[int]],
                       downstream: Sequence[Optional[int]]) -> List[int]:
    """Return nodes in upstream-to-downstream order for routing."""
    indegree = np.fromiter((len(u) for u in upstream), dtype=np.int64,
                           count=len(upstream))
    stack = [int(idx) for idx in np.where(indegree == 0)[0]]
    order: List[int] = []

    while stack:
        node = stack.pop()
        order.append(node)
        dn = downstream[node]
        if dn is not None:
            indegree[dn] -= 1
            if indegree[dn] == 0:
                stack.append(dn)

    return order


def _graph_laplacian(neighbors: Sequence[np.ndarray],
                     values: np.ndarray) -> np.ndarray:
    """Simple unweighted graph Laplacian."""
    lap = np.zeros_like(values)

    for idx, nbrs in enumerate(neighbors):
        if len(nbrs) == 0:
            continue
        lap[idx] = values[nbrs].mean() - values[idx]

    return lap


def morphodynamic_step(points: np.ndarray,
                       neighbors: Sequence[np.ndarray],
                       heights: np.ndarray,
                       land_mask: np.ndarray,
                       river_network: RiverNetwork,
                       params,
                       baseline: Optional[np.ndarray]) -> np.ndarray:
    """Perform a single Exner-style sediment routing step."""
    discharge = river_network.volume
    slopes = _node_slopes(points, neighbors, heights)
    slopes[~land_mask] = 0.0
    capacity = _transport_capacity(discharge, slopes, params)

    load = np.zeros_like(heights)
    deposition = np.zeros_like(heights)
    erosion = np.zeros_like(heights)

    order = _topological_order(river_network.upstream, river_network.downstream)
    downstream = river_network.downstream
    solid_fraction = max(1e-6, 1.0 - params.porosity)

    for node in order:
        incoming = load[node]
        cap = capacity[node]

        if not land_mask[node]:
            deposition[node] += incoming
            outgoing = 0.0
        else:
            if incoming > cap:
                excess = incoming - cap
                deposit_amount = min(params.k_deposition * excess, incoming)
                deposition[node] += deposit_amount
                outgoing = incoming - deposit_amount
            else:
                deficit = cap - incoming
                erosion_amount = params.k_erosion * deficit
                erosion[node] += erosion_amount
                outgoing = incoming + erosion_amount

            if slopes[node] < params.tau_crit_slope and outgoing > 0.0:
                lake_fill = min(params.lake_fill_factor * outgoing, outgoing)
                deposition[node] += lake_fill
                outgoing -= lake_fill

        if downstream[node] is None and outgoing > 0.0 and params.delta_enhance > 1.0:
            extra = outgoing * (params.delta_enhance - 1.0) / params.delta_enhance
            extra = min(extra, outgoing)
            deposition[node] += extra
            outgoing -= extra

        dn = downstream[node]
        if dn is not None and outgoing > 0.0:
            load[dn] += outgoing

    net_change = (deposition - erosion) * (params.dt_morpho / solid_fraction)
    if baseline is not None:
        net_change = np.clip(net_change, 0.0, params.sediment_max_step)
    new_heights = heights + net_change
    if baseline is not None:
        new_heights = np.maximum(new_heights, baseline)
    new_heights = np.maximum(new_heights, 0.0)
    new_heights[~land_mask] = 0.0

    if params.floodplain_diffusion > 0.0:
        lap = _graph_laplacian(neighbors, new_heights)
        flat_mask = slopes < params.tau_crit_slope * 5.0
        smoothing = params.floodplain_diffusion * lap * flat_mask.astype(np.float64)
        new_heights = new_heights + smoothing
        new_heights = np.maximum(new_heights, 0.0)
        new_heights[~land_mask] = 0.0
        if baseline is not None:
            new_heights = np.maximum(new_heights, baseline)

    return new_heights


def morphodynamic_update(points: np.ndarray,
                         neighbors: Sequence[np.ndarray],
                         heights: np.ndarray,
                         land_mask: np.ndarray,
                         river_network: RiverNetwork,
                         params,
                         baseline: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply several morphodynamic iterations and derive net deposition."""
    if params.morpho_iterations <= 0 or not params.enable_sediment:
        return heights, np.zeros_like(heights)

    initial_heights = heights.copy()
    updated = heights.copy()
    for _ in range(params.morpho_iterations):
        updated = morphodynamic_step(points, neighbors, updated, land_mask,
                                     river_network, params, baseline)

    reference = baseline if baseline is not None else initial_heights
    deposition_map = np.maximum(updated - reference, 0.0)
    deposition_map[~land_mask] = 0.0
    return updated, deposition_map
