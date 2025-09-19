"""Export functionality for terrain data."""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional

from ..core.utils import normalize

class TerrainExporter:
    """Handles terrain data export."""
    
    @staticmethod
    def export_heightmap(heightmap: np.ndarray, filepath: str, 
                        format: str = "PNG_8"):
        """Export heightmap to image file."""
        exporters = {
            "PNG_8": TerrainExporter._export_png_8bit,
            "PNG_16": TerrainExporter._export_png_16bit,
            "TIFF_32": TerrainExporter._export_tiff_32bit
        }
        
        if format in exporters:
            exporters[format](heightmap, Path(filepath))
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    @staticmethod
    def export_flow_mask(river_volume: np.ndarray, land_mask: np.ndarray,
                        filepath: str, format: str = "PNG_8"):
        """Export flow mask to image file."""
        # Prepare flow data
        flow_data = river_volume.copy()

        # Set non-land areas to 0
        if land_mask is not None:
            flow_data[~land_mask] = 0

        # Normalize to 0-1 range
        if flow_data.max() > 0:
            flow_data = flow_data / flow_data.max()

        # Export using same methods as heightmap
        TerrainExporter.export_heightmap(flow_data, filepath, format)

    @staticmethod
    def export_watershed_mask(watershed_mask: np.ndarray, land_mask: np.ndarray,
                              filepath: str, format: str = "PNG_8"):
        """Export watershed mask to image file."""
        mask_data = watershed_mask.astype(np.float32)

        if land_mask is not None:
            mask_data = mask_data.copy()
            mask_data[~land_mask] = 0

        if mask_data.max() > 0:
            mask_data = mask_data / mask_data.max()

        TerrainExporter.export_heightmap(mask_data, filepath, format)

    @staticmethod
    def export_sediment_mask(sediment_deposition: Optional[np.ndarray], land_mask: np.ndarray,
                             filepath: str, format: str = "PNG_8"):
        """Export sediment deposition mask to image file."""
        if sediment_deposition is None:
            raise ValueError("No sediment deposition data available for export.")

        mask_data = sediment_deposition.astype(np.float32)

        if land_mask is not None:
            mask_data = mask_data.copy()
            mask_data[~land_mask] = 0.0

        max_value = float(mask_data.max())
        if max_value > 0.0:
            mask_data = mask_data / max_value

        TerrainExporter.export_heightmap(mask_data, filepath, format)

    @staticmethod
    def _export_png_8bit(data: np.ndarray, filepath: Path):
        """Export as 8-bit PNG."""
        normalized = normalize(data.astype(np.float32), (0, 255))
        # Round to nearest to minimize banding from truncation
        img_data = np.clip(np.rint(normalized), 0, 255).astype(np.uint8)
        img = Image.fromarray(img_data, mode='L')
        img.save(filepath)
    
    @staticmethod
    def _export_png_16bit(data: np.ndarray, filepath: Path):
        """Export as 16-bit PNG."""
        # Normalize to full 16-bit range using float math, then round
        normalized = normalize(data.astype(np.float32), (0, 65535))
        img_data = np.clip(np.rint(normalized), 0, 65535).astype(np.uint16)
        img = Image.fromarray(img_data, mode='I;16')
        img.save(filepath)
    
    @staticmethod
    def _export_tiff_32bit(data: np.ndarray, filepath: Path):
        """Export as 32-bit float TIFF."""
        img = Image.fromarray(data.astype(np.float32), mode='F')
        img.save(filepath)
