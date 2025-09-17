"""River network generation and computation."""

import numpy as np
import heapq
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass

from .utils import lerp

@dataclass
class RiverNetwork:
    """Container for river network data."""
    upstream: List[Set[int]]
    downstream: List[Optional[int]]
    volume: np.ndarray
    watershed: np.ndarray

class RiverGenerator:
    """Generates river networks on terrain."""
    
    def __init__(self, directional_inertia: float = 0.2,
                 default_water_level: float = 1.0,
                 evaporation_rate: float = 0.2):
        self.directional_inertia = directional_inertia
        self.default_water_level = default_water_level
        self.evaporation_rate = evaporation_rate
    
    def compute_network(self, points: np.ndarray, 
                       neighbors: List[List[int]],
                       heights: np.ndarray,
                       land_mask: np.ndarray) -> RiverNetwork:
        """Compute complete river network."""
        num_points = len(points)
        
        # Find flow directions
        downstream = self._compute_flow_directions(
            points, neighbors, heights, land_mask
        )
        
        # Build upstream connections
        upstream = self._build_upstream_connections(downstream)
        
        # Compute water volume
        volume = self._compute_water_volume(upstream, num_points)

        # Compute watershed ownership for each sample point
        watershed = self._compute_watersheds(downstream, land_mask)

        return RiverNetwork(upstream, downstream, volume, watershed)
    
    def _compute_flow_directions(self, points: np.ndarray,
                                neighbors: List[List[int]], 
                                heights: np.ndarray,
                                land_mask: np.ndarray) -> List[Optional[int]]:
        """Compute downstream flow direction for each point."""
        num_points = len(points)
        
        def unit_delta(i, j):
            delta = points[j] - points[i]
            norm = np.linalg.norm(delta)
            return delta / norm if norm > 0 else delta
        
        # Initialize priority queue with coastal points
        q = []
        roots = set()
        
        for i in range(num_points):
            if land_mask[i]:
                continue
            
            is_root = True
            for j in neighbors[i]:
                if not land_mask[j]:
                    continue
                is_root = True
                heapq.heappush(q, (-1.0, (i, j, unit_delta(i, j))))
            
            if is_root:
                roots.add(i)
        
        # Compute flow directions
        downstream = [None] * num_points
        
        while len(q) > 0:
            (_, (i, j, direction)) = heapq.heappop(q)
            
            if downstream[j] is not None:
                continue
            
            downstream[j] = i
            
            # Process neighbors
            for k in neighbors[j]:
                if (heights[k] < heights[j] or 
                    downstream[k] is not None or 
                    not land_mask[k]):
                    continue
                
                neighbor_direction = unit_delta(j, k)
                priority = -np.dot(direction, neighbor_direction)
                
                weighted_direction = lerp(
                    neighbor_direction, direction,
                    self.directional_inertia
                )
                
                heapq.heappush(q, (priority, (j, k, weighted_direction)))
        
        return downstream
    
    def _build_upstream_connections(self, 
                                   downstream: List[Optional[int]]) -> List[Set[int]]:
        """Build upstream connections from downstream data."""
        num_points = len(downstream)
        upstream = [set() for _ in range(num_points)]
        
        for i, j in enumerate(downstream):
            if j is not None:
                upstream[j].add(i)
        
        return upstream
    
    def _compute_water_volume(self, upstream: List[Set[int]], 
                             num_points: int) -> np.ndarray:
        """Compute water volume at each point."""
        volume = [None] * num_points

        def compute_volume(i):
            if volume[i] is not None:
                return

            v = self.default_water_level
            for j in upstream[i]:
                compute_volume(j)
                v += volume[j]

            volume[i] = v * (1 - self.evaporation_rate)

        for i in range(num_points):
            compute_volume(i)

        return np.array(volume)

    def _compute_watersheds(self, downstream: List[Optional[int]],
                            land_mask: np.ndarray) -> np.ndarray:
        """Assign a watershed identifier to each sample point."""
        num_points = len(downstream)
        # -1 indicates unassigned (or off-map water cell)
        watershed = np.full(num_points, -1, dtype=np.int32)

        next_id = 1  # Start at 1 so 0 can remain background if desired

        for i in range(num_points):
            if not land_mask[i]:
                # Ocean / water cells act as sinks; ensure they have stable ids
                if watershed[i] == -1:
                    watershed[i] = next_id
                    next_id += 1
                continue

            path = []
            current = i

            # Walk downstream until we reach an assigned node or exit the network
            while current is not None and watershed[current] == -1:
                path.append(current)
                current = downstream[current]

            if current is None:
                basin_id = next_id
                next_id += 1
            else:
                basin_id = watershed[current]
                if basin_id == -1:
                    basin_id = next_id
                    next_id += 1
                    watershed[current] = basin_id

            for node in path:
                watershed[node] = basin_id

        return watershed
