"""Utility functions for terrain generation."""

import numpy as np
import collections
import scipy.spatial
from typing import Tuple, List, Optional

def normalize(x: np.ndarray, bounds: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """Renormalizes the values of x to bounds."""
    if x.max() == x.min():
        return np.full_like(x, bounds[0])
    return np.interp(x, (x.min(), x.max()), bounds)

def gaussian_blur(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Performs a gaussian blur of a."""
    freqs = tuple(np.fft.fftfreq(n, d=1.0 / n) for n in a.shape)
    freq_radial = np.hypot(*np.meshgrid(*freqs))
    sigma2 = sigma**2
    g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
    kernel = g(freq_radial)
    kernel /= kernel.sum()
    return np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(kernel)).real

def gaussian_gradient(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Returns the gradient of the gaussian blur of a encoded as a complex number."""
    [fy, fx] = np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in a.shape))
    sigma2 = sigma**2
    g = lambda x: ((2 * np.pi * sigma2) ** -0.5) * np.exp(-0.5 * (x / sigma)**2)
    dg = lambda x: g(x) * (x / sigma2)
    
    fa = np.fft.fft2(a)
    dy = np.fft.ifft2(np.fft.fft2(dg(fy) * g(fx)) * fa).real
    dx = np.fft.ifft2(np.fft.fft2(g(fy) * dg(fx)) * fa).real
    return 1j * dx + dy

def lerp(x: np.ndarray, y: np.ndarray, a: float) -> np.ndarray:
    """Linear interpolation of x to y with respect to a."""
    return (1.0 - a) * x + a * y

def make_grid_points(shape: Tuple[int, int]) -> np.ndarray:
    """Returns a list of grid coordinates for every (x, y) position."""
    [Y, X] = np.meshgrid(np.arange(shape[0]), np.arange(shape[1])) 
    grid_points = np.column_stack([X.flatten(), Y.flatten()])
    return grid_points

def bump(shape: Tuple[int, int], sigma: float) -> np.ndarray:
    """Returns an array with a bump centered in the middle."""
    [y, x] = np.meshgrid(*map(np.arange, shape))
    r = np.hypot(x - shape[0] / 2, y - shape[1] / 2)
    c = min(shape) / 2
    return np.tanh(np.maximum(c - r, 0.0) / sigma)

def dist_to_mask(mask: np.ndarray) -> np.ndarray:
    """Returns distance to nearest False value for all True values in mask."""
    border_mask = (np.maximum.reduce([
        np.roll(mask, 1, axis=0), np.roll(mask, -1, axis=0),
        np.roll(mask, -1, axis=1), np.roll(mask, 1, axis=1)]) * (1 - mask))
    border_points = np.column_stack(np.where(border_mask > 0))
    
    if len(border_points) == 0:
        return np.zeros_like(mask, dtype=float)
    
    kdtree = scipy.spatial.cKDTree(border_points)
    grid_points = make_grid_points(mask.shape)
    
    return kdtree.query(grid_points)[0].reshape(mask.shape)

def poisson_disc_sampling(shape: Tuple[int, int], radius: float,
                         retries: int = 16) -> np.ndarray:
    """Fast Poisson-disc sampling (Bridson) with numpy grid acceleration.

    - Preserves API and output format of the previous implementation.
    - Uses a dense integer grid for O(1) neighbor lookups instead of a dict.
    - Inlines simple math to reduce numpy call overhead in tight loops.
    """
    # Convert shape for arithmetic and keep integer dims for bounds
    H, W = int(shape[0]), int(shape[1])
    shape_arr = np.array([H, W], dtype=float)

    if radius <= 0:
        # Degenerate case: return empty set
        return np.empty((0, 2), dtype=float)

    cell_size = float(radius) / np.sqrt(2.0)
    if cell_size <= 0:
        return np.empty((0, 2), dtype=float)

    # Grid dimensions (rows=Y, cols=X)
    grid_rows = int(np.ceil(H / cell_size))
    grid_cols = int(np.ceil(W / cell_size))
    grid = np.full((grid_rows, grid_cols), -1, dtype=np.int32)

    # Neighbor cell offsets to search (covering a 5x5 cross + diagonals sufficient for r)
    neighbor_offsets = (
        (0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2)
    )

    # Active list and point storage
    active = collections.deque()
    pts_x: list = []
    pts_y: list = []

    r2 = float(radius) * float(radius)
    max_r2 = 4.0 * r2  # (2r)^2

    def to_cell_ix(x: float, y: float) -> Tuple[int, int]:
        return int(x / cell_size), int(y / cell_size)

    def occupied_within_radius(x: float, y: float) -> bool:
        cx, cy = to_cell_ix(x, y)
        for off_x, off_y in neighbor_offsets:
            nx = cx + off_x
            ny = cy + off_y
            if 0 <= ny < grid_rows and 0 <= nx < grid_cols:
                idx = grid[ny, nx]
                if idx != -1:
                    dx = pts_x[idx] - x
                    dy = pts_y[idx] - y
                    if (dx * dx + dy * dy) <= r2:
                        return True
        return False

    def add_point_xy(x: float, y: float):
        cx, cy = to_cell_ix(x, y)
        grid[cy, cx] = len(pts_x)
        active.append((x, y))
        pts_x.append(x)
        pts_y.append(y)

    # First point uniformly in domain
    first = shape_arr * np.random.rand(2)
    add_point_xy(float(first[0]), float(first[1]))

    # Main loop: pop active point and try to place new points around it
    while active:
        px, py = active.pop()  # LIFO works well and keeps cache locality
        for _ in range(retries):
            # Random candidate in annulus [r, 2r)
            rx, ry = 2.0 * radius * (2.0 * np.random.rand(2) - 1.0)
            d2 = rx * rx + ry * ry
            if not (r2 < d2 < max_r2):
                continue
            nx = px + rx
            ny = py + ry
            # Fast bounds check first
            if nx < 0.0 or nx >= W or ny < 0.0 or ny >= H:
                continue
            # Neighbor radius check
            if not occupied_within_radius(nx, ny):
                add_point_xy(nx, ny)

    if not pts_x:
        return np.empty((0, 2), dtype=float)

    points = np.column_stack((np.asarray(pts_x, dtype=float),
                              np.asarray(pts_y, dtype=float)))
    return points

def remove_lakes(mask: np.ndarray) -> np.ndarray:
    """Removes bodies of water enclosed by land."""
    import skimage.measure
    labels = skimage.measure.label(~mask, connectivity=1)
    new_mask = np.zeros_like(mask, dtype=bool)
    new_mask[labels != labels[0, 0]] = True
    return new_mask

def render_triangulation(shape: Tuple[int, int], tri, values: np.ndarray) -> np.ndarray:
    """Renders values for each triangle on an array."""
    import matplotlib.tri
    points = make_grid_points(shape)
    triangulation = matplotlib.tri.Triangulation(
        tri.points[:,0], tri.points[:,1], tri.simplices)
    interp = matplotlib.tri.LinearTriInterpolator(triangulation, values)
    return interp(points[:,0], points[:,1]).reshape(shape).filled(0.0)

def identify_inland_seas(land_mask: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Identify inland seas (water bodies not connected to ocean).
    
    Args:
        land_mask: Boolean array where True = land, False = water
        
    Returns:
        ocean_mask: Boolean mask of the main ocean
        inland_seas: List of boolean masks for each inland sea
    """
    from skimage import measure
    
    # Label water bodies (inverse of land mask)
    water_mask = ~land_mask
    labeled_water = measure.label(water_mask, connectivity=1)
    
    # Find the ocean (largest water body touching the edge)
    edge_labels = set()
    h, w = land_mask.shape
    
    # Collect labels that touch the edges
    edge_labels.update(labeled_water[0, :])    # top
    edge_labels.update(labeled_water[-1, :])   # bottom  
    edge_labels.update(labeled_water[:, 0])    # left
    edge_labels.update(labeled_water[:, -1])   # right
    edge_labels.discard(0)  # Remove background
    
    # Ocean is the largest edge-touching water body
    ocean_label = 0
    ocean_size = 0
    for label in edge_labels:
        size = np.sum(labeled_water == label)
        if size > ocean_size:
            ocean_size = size
            ocean_label = label
    
    # If no edge water, pick the largest water body
    if ocean_label == 0 and labeled_water.max() > 0:
        labels, counts = np.unique(labeled_water[labeled_water > 0], return_counts=True)
        ocean_label = labels[np.argmax(counts)]
    
    ocean_mask = (labeled_water == ocean_label)
    
    # Find all inland seas
    inland_seas = []
    for label in np.unique(labeled_water):
        if label > 0 and label != ocean_label:
            inland_seas.append(labeled_water == label)
    
    return ocean_mask, inland_seas

def carve_channel_to_ocean(heightmap: np.ndarray, land_mask: np.ndarray,
                          inland_sea_mask: np.ndarray, ocean_mask: np.ndarray,
                          carve_depth: float = 0.05) -> np.ndarray:
    """
    Carve a channel from inland sea to ocean.
    
    Args:
        heightmap: Current terrain heightmap
        land_mask: Boolean mask of land areas
        inland_sea_mask: Boolean mask of the specific inland sea
        ocean_mask: Boolean mask of the ocean
        carve_depth: How deep to carve the channel
        
    Returns:
        Modified heightmap with carved channel
    """
    from scipy.ndimage import distance_transform_edt, binary_dilation
    from skimage.graph import route_through_array
    
    h, w = heightmap.shape
    
    # Find closest points between inland sea and ocean
    inland_dist = distance_transform_edt(~inland_sea_mask)
    ocean_dist = distance_transform_edt(~ocean_mask)
    
    # Find a point on the inland sea edge
    dilated_inland = binary_dilation(inland_sea_mask)
    inland_edge = dilated_inland & ~inland_sea_mask
    if not np.any(inland_edge):
        return heightmap
    
    inland_edge_points = np.argwhere(inland_edge)
    # Pick the point closest to ocean
    min_dist_idx = np.argmin([ocean_dist[p[0], p[1]] for p in inland_edge_points])
    start_point = tuple(inland_edge_points[min_dist_idx])
    
    # Find the closest ocean point
    dilated_ocean = binary_dilation(ocean_mask)
    ocean_edge = dilated_ocean & ~ocean_mask
    if not np.any(ocean_edge):
        return heightmap
    
    ocean_edge_points = np.argwhere(ocean_edge)
    distances = [np.sqrt((p[0]-start_point[0])**2 + (p[1]-start_point[1])**2) 
                 for p in ocean_edge_points]
    end_point = tuple(ocean_edge_points[np.argmin(distances)])
    
    # Create cost array for pathfinding
    # Lower cost for lower terrain, high cost for going uphill
    cost = np.ones_like(heightmap)
    # Normalize heightmap for cost calculation
    norm_height = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)
    cost = 1.0 + norm_height * 10.0  # Prefer lower areas
    
    # Find path using least-cost pathfinding
    try:
        path_indices, _ = route_through_array(
            cost, start_point, end_point, fully_connected=True
        )
    except:
        # Fallback to straight line if pathfinding fails
        num_points = int(np.sqrt((end_point[0]-start_point[0])**2 + 
                                 (end_point[1]-start_point[1])**2))
        if num_points > 0:
            t = np.linspace(0, 1, num_points)
            path_indices = np.array([
                (int(start_point[0] * (1-ti) + end_point[0] * ti),
                 int(start_point[1] * (1-ti) + end_point[1] * ti))
                for ti in t
            ])
        else:
            return heightmap
    
    # Carve the channel
    modified = heightmap.copy()
    
    # Create a smooth channel profile
    channel_width = 3  # pixels
    for y, x in path_indices:
        # Carve the channel point and its neighbors
        for dy in range(-channel_width, channel_width+1):
            for dx in range(-channel_width, channel_width+1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    dist = np.sqrt(dy**2 + dx**2)
                    if dist <= channel_width:
                        # Gaussian falloff for smooth edges
                        falloff = np.exp(-dist**2 / (channel_width**2))
                        carve_amount = carve_depth * falloff
                        # Carve down but don't go below 0
                        modified[ny, nx] = max(0, modified[ny, nx] - carve_amount)
    
    # Smooth the carved area
    from scipy.ndimage import gaussian_filter
    # Create mask of carved area
    carved_mask = np.zeros_like(land_mask, dtype=bool)
    for y, x in path_indices:
        for dy in range(-channel_width-1, channel_width+2):
            for dx in range(-channel_width-1, channel_width+2):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    carved_mask[ny, nx] = True
    
    # Apply smoothing only to carved area
    if np.any(carved_mask):
        smoothed = gaussian_filter(modified, sigma=1.0)
        modified[carved_mask] = smoothed[carved_mask]
    
    return modified


def connect_inland_seas(heightmap: np.ndarray, land_mask: np.ndarray,
                       min_sea_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Connect all inland seas to the ocean or fill them if too small.
    
    Args:
        heightmap: Terrain heightmap  
        land_mask: Boolean mask where True = land
        min_sea_size: Minimum size for inland seas (smaller ones are filled)
        
    Returns:
        Modified heightmap and land mask
    """
    ocean_mask, inland_seas = identify_inland_seas(land_mask)
    
    if not inland_seas:
        # No inland seas found
        return heightmap, land_mask
    
    modified_heightmap = heightmap.copy()
    
    print(f"Found {len(inland_seas)} inland water bodies")
    
    for i, sea_mask in enumerate(inland_seas):
        sea_size = np.sum(sea_mask)
        
        if sea_size < min_sea_size:
            # Fill small lakes
            print(f"  Filling small lake {i+1} (size: {sea_size})")
            # Set to slightly above water level
            modified_heightmap[sea_mask] = 0.01  
        else:
            # Carve channel for larger seas
            print(f"  Carving channel for inland sea {i+1} (size: {sea_size})")
            modified_heightmap = carve_channel_to_ocean(
                modified_heightmap, land_mask, sea_mask, ocean_mask,
                carve_depth=0.1  # Make carving more aggressive
            )
    
    # Recompute land mask after modifications
    # Water is anything at or below 0
    new_land_mask = modified_heightmap > 0.001
    
    # Ensure carved channels are properly marked as water
    modified_heightmap = np.where(new_land_mask, modified_heightmap, 0)
    
    return modified_heightmap, new_land_mask