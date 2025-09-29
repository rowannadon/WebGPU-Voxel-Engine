#!/usr/bin/env python3
"""
Terrain amplification (super-resolution) — Python CLI reimplementation
with optional patch jitter and final [0,1] normalization.

Reference:
Sparse representation of terrains for procedural modeling
Guérin, Digne, Galin, Peytavie (Eurographics 2016)

Usage example:
python terrain_sr.py \
  --factor 4 \
  --exemplarhr grandcanyonhr.png \
  --inputterrain sketchlr.png \
  --output out.png \
  --masksize 16 \
  --offset-analysis 8 \
  --offset-synthesis 8 \
  --jitter-px 2 \
  --scale-jitter 0.05 \
  --seed 0
"""
import argparse
import time
from math import floor
from typing import Optional, Tuple

import numpy as np
from PIL import Image


# ------------------------------
# Utilities
# ------------------------------

def imread_grayscale_like_matlab(path: str) -> np.ndarray:
    """
    Read an image and reproduce the MATLAB conversions used in the exemplar code.

    For Iexemplarhr:
      - If 3 channels: I = (R + 255*G + 255*255*B) / 65535.0
      - Else if uint16: I = double(I) / 255.0
      - Else: double(I)
    For input terrain:
      - If uint16: double(I) / 255.0
      - Else: double(I)

    Returns a numpy array (float64 or will be cast later), channel-handling
    is applied by the caller to match the original MATLAB behavior.
    """
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 3 and arr.shape[2] >= 3:
        # Keep RGB (ignore alpha if present)
        arr = arr[:, :, :3]
    elif arr.ndim == 2:
        pass
    else:
        # Convert everything else to L mode
        arr = np.array(img.convert("L"))

    return arr


def build_mask(masksize: int) -> np.ndarray:
    """
    MATLAB build_mask: polynomial mask with clipping at 0, then squared.
    """
    assert masksize % 2 == 0, "masksize should be an even integer."
    offset = 1.0 - 1.0 / masksize
    radius = (masksize - 1) * 0.5

    i = np.arange(1, masksize + 1)[:, None]
    j = np.arange(1, masksize + 1)[None, :]
    x = (i - 1 - radius) / radius
    y = (j - 1 - radius) / radius

    val = 1.0 - offset * (x ** 2 + y ** 2)
    val[val < 0] = 0.0
    return (val ** 2).astype(np.float64)


def terrain_dilate(terrain: np.ndarray, s: int) -> np.ndarray:
    """
    MATLAB terrain_dilate: expands the terrain to size + s, with edge clamping.
    """
    H, W = terrain.shape
    radius = s // 2
    out = np.empty((H + s, W + s), dtype=terrain.dtype)

    i_idx = np.arange(H + s) - radius
    i_idx = np.clip(i_idx, 0, H - 1)

    j_idx = np.arange(W + s) - radius
    j_idx = np.clip(j_idx, 0, W - 1)

    out[:, :] = terrain[i_idx[:, None], j_idx[None, :]]
    return out


def imresize_float(arr: np.ndarray, scale: float) -> np.ndarray:
    """
    Approximate MATLAB imresize with bicubic, keeping values in float domain.
    """
    H, W = arr.shape
    newH = max(1, int(round(H * scale)))
    newW = max(1, int(round(W * scale)))
    im = Image.fromarray(arr)
    im = im.resize((newW, newH), resample=Image.Resampling.BICUBIC)
    return np.array(im, dtype=np.float64)


def extract_patch_means(img: np.ndarray, masksize: int, offset: int, d1: int, d2: int) -> np.ndarray:
    means = np.zeros((d1, d2), dtype=np.float64)
    for i in range(d1):
        for j in range(d2):
            patch = img[i * offset:i * offset + masksize, j * offset:j * offset + masksize]
            means[i, j] = patch.mean()
    return means


def normalize_columns(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize columns (2D matrix) to unit L2 norm, skip zeros.
    Returns:
      out: normalized columns
      norms: original norms for kept columns
      keep_mask: boolean mask of kept columns
    """
    norms = np.linalg.norm(mat, axis=0)
    keep_mask = norms > 0
    out = mat[:, keep_mask] / norms[keep_mask]
    return out, norms[keep_mask], keep_mask


def make_jittered_coords(
    H: int, W: int, mask: int, offset: int, d1: int, d2: int,
    jitter: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns two (d1, d2) integer arrays of top-left (row, col) for each patch,
    starting from the regular grid (i*offset, j*offset), then adding clamped jitter
    so the mask stays inside the image.
    """
    base_i = np.arange(d1)[:, None] * offset
    base_j = np.arange(d2)[None, :] * offset

    if jitter <= 0:
        i0 = np.broadcast_to(base_i, (d1, d2)).copy()
        j0 = np.broadcast_to(base_j, (d1, d2)).copy()
        return i0.astype(int), j0.astype(int)

    ji = rng.integers(-jitter, jitter + 1, size=(d1, d2))
    jj = rng.integers(-jitter, jitter + 1, size=(d1, d2))

    i0 = base_i + ji
    j0 = base_j + jj

    i0 = np.clip(i0, 0, max(0, H - mask))
    j0 = np.clip(j0, 0, max(0, W - mask))
    return i0.astype(int), j0.astype(int)


# ------------------------------
# Core pipeline
# ------------------------------

def terrain_super_resolution(
    factor: int,
    exemplarhr_path: str,
    inputterrain_path: str,
    output_path: str,
    masksize: int,
    offset_analysis: int,
    offset_synthesis: int,
    jitter_px: int = 0,
    scale_jitter: float = 0.0,
    seed: Optional[int] = None,
) -> None:
    start = time.time()
    rng = np.random.default_rng(seed)

    # --- Read high-res exemplar, reproduce MATLAB channel handling ---
    Iexemplarhr_raw = imread_grayscale_like_matlab(exemplarhr_path)

    if Iexemplarhr_raw.ndim == 2:
        # Single channel
        if Iexemplarhr_raw.dtype == np.uint16:
            Iexemplarhr = Iexemplarhr_raw.astype(np.float64) / 255.0
        else:
            Iexemplarhr = Iexemplarhr_raw.astype(np.float64)
    else:
        # 3 channels exactly
        R = Iexemplarhr_raw[:, :, 0].astype(np.float64)
        G = Iexemplarhr_raw[:, :, 1].astype(np.float64)
        B = Iexemplarhr_raw[:, :, 2].astype(np.float64)
        Iexemplarhr = (R + 255.0 * G + 255.0 * 255.0 * B) / 65535.0

    # --- Low-res exemplar is resized version of HR exemplar ---
    Iexemplarlr = imresize_float(Iexemplarhr, 1.0 / float(factor))

    # --- Read input terrain (single-channel behavior like MATLAB) ---
    Iinputterrain_raw = imread_grayscale_like_matlab(inputterrain_path)
    if Iinputterrain_raw.dtype == np.uint16:
        Iinputterrain = Iinputterrain_raw.astype(np.float64) / 255.0
    else:
        Iinputterrain = Iinputterrain_raw.astype(np.float64)

    # -------------------------------
    # Optimize dictionary for exemplar (LR)
    # -------------------------------
    mask = build_mask(masksize)
    offset = offset_analysis

    d1 = (Iexemplarlr.shape[0] - masksize) // offset
    d2 = (Iexemplarlr.shape[1] - masksize) // offset
    if d1 <= 0 or d2 <= 0:
        raise ValueError("Mask/offset too large for exemplar LR dimensions.")

    # Means on LR exemplar patches
    exemplarmeans = extract_patch_means(Iexemplarlr, masksize, offset, d1, d2)

    # Build X (terrain - mean) * mask for LR exemplar
    num_patches = d1 * d2
    X_lr = np.zeros((num_patches, masksize * masksize), dtype=np.float64)
    nl = 0
    for i in range(d1):
        for j in range(d2):
            patch = Iexemplarlr[i * offset:i * offset + masksize, j * offset:j * offset + masksize]
            v = patch - exemplarmeans[i, j]
            v = v * mask
            X_lr[nl, :] = v.reshape(-1)
            nl += 1

    # Dictionary init and normalization (LR)
    Dinit_lr = X_lr.T.copy()  # (mask^2, num_patches_raw)
    D_lr, Dinit_lr_norms, keep_cols = normalize_columns(Dinit_lr)

    # -------------------------------
    # Optimize terrain with the dictionary (LR OMP with sparsity=1)
    # -------------------------------
    offset = offset_synthesis
    Isyn = terrain_dilate(Iinputterrain, masksize * 2)

    d1_syn = (Isyn.shape[0] - masksize) // offset
    d2_syn = (Isyn.shape[1] - masksize) // offset
    if d1_syn <= 0 or d2_syn <= 0:
        raise ValueError("Mask/offset too large for synthesized LR size.")

    # Build jittered LR patch positions for synthesis
    i0_syn, j0_syn = make_jittered_coords(
        Isyn.shape[0], Isyn.shape[1], masksize, offset, d1_syn, d2_syn, jitter_px, rng
    )

    # Means on synthesized LR (jittered positions)
    means_syn = np.zeros((d1_syn, d2_syn), dtype=np.float64)
    for i in range(d1_syn):
        for j in range(d2_syn):
            ii, jj = int(i0_syn[i, j]), int(j0_syn[i, j])
            patch = Isyn[ii:ii + masksize, jj:jj + masksize]
            means_syn[i, j] = patch.mean()

    # Build X for synthesized LR (jittered positions)
    num_patches_syn = d1_syn * d2_syn
    X_syn = np.zeros((num_patches_syn, masksize * masksize), dtype=np.float64)
    nl = 0
    for i in range(d1_syn):
        for j in range(d2_syn):
            ii, jj = int(i0_syn[i, j]), int(j0_syn[i, j])
            patch = Isyn[ii:ii + masksize, jj:jj + masksize]
            v = patch - means_syn[i, j]
            v = v * mask
            X_syn[nl, :] = v.reshape(-1)
            nl += 1

    # OMP with sparsity 1:
    # Choose atom k maximizing |d_k^T x|, coeff = d_k^T x (atoms are normalized)
    G = D_lr.T @ X_syn.T  # (n_atoms_kept, n_patches_syn)
    idx_max = np.argmax(np.abs(G), axis=0)  # length n_patches_syn
    coeffs = G[idx_max, np.arange(G.shape[1])]  # length n_patches_syn

    newcoeffs = np.zeros_like(G)
    newcoeffs[idx_max, np.arange(G.shape[1])] = coeffs

    # Optional per-patch random gain in [1-S, 1+S]
    if scale_jitter > 0.0:
        gains = 1.0 + rng.uniform(-scale_jitter, scale_jitter, size=newcoeffs.shape[1])
        newcoeffs *= gains[np.newaxis, :]

    print(f"Number of patches : {newcoeffs.shape[1]}")

    # -------------------------------
    # Build the high-resolution dictionary
    # -------------------------------
    masksizehr = masksize * factor
    maskhr = build_mask(masksizehr)
    offsethra = offset_analysis * factor
    offsethrs = offset_synthesis * factor

    d1_lr = (Iexemplarlr.shape[0] - masksize) // offset_analysis
    d2_lr = (Iexemplarlr.shape[1] - masksize) // offset_analysis
    if d1_lr <= 0 or d2_lr <= 0:
        raise ValueError("Mask/offset too large for LR exemplar in HR step.")

    # HR patch means from HR exemplar (using offsets scaled by factor)
    noisemeanshr = np.zeros((d1_lr, d2_lr), dtype=np.float64)
    for i in range(d1_lr):
        for j in range(d2_lr):
            patch_hr = Iexemplarhr[
                i * offsethra:i * offsethra + masksizehr,
                j * offsethra:j * offsethra + masksizehr
            ]
            noisemeanshr[i, j] = patch_hr.mean()

    # HR centered & masked patches stacked
    X_hr = np.zeros((d1_lr * d2_lr, masksizehr * masksizehr), dtype=np.float64)
    nl = 0
    for i in range(d1_lr):
        for j in range(d2_lr):
            patch_hr = Iexemplarhr[
                i * offsethra:i * offsethra + masksizehr,
                j * offsethra:j * offsethra + masksizehr
            ]
            v = patch_hr - noisemeanshr[i, j]
            v = v * maskhr
            X_hr[nl, :] = v.reshape(-1)
            nl += 1

    # High-res dictionary init
    Dinithr = X_hr.T.copy()  # (maskhr^2, num_atoms_raw)

    # Match LR/HR atom normalization like MATLAB:
    # D_hr[:,k] = Dinithr[:,k] / norm(Dinit_lr[:,k]) for columns kept in LR.
    Dinithr_kept = Dinithr[:, keep_cols]
    D_hr = Dinithr_kept / Dinit_lr_norms  # (maskhr^2, n_atoms_kept)

    # -------------------------------
    # Reconstruct HR patches and stitch with jittered positions
    # -------------------------------
    # Y = newcoeffs' * D_hr'   (n_patches_syn x maskhr^2)
    Y = (newcoeffs.T @ D_hr.T)  # shape (n_patches_syn, maskhr^2)

    H_in, W_in = Iinputterrain.shape
    Isynhr = np.zeros((H_in * factor + 2 * masksizehr, W_in * factor + 2 * masksizehr), dtype=np.float64)
    Crechr = np.zeros_like(Isynhr)

    # Scale LR jitter to HR
    i0_syn_hr = (i0_syn * factor).astype(int)
    j0_syn_hr = (j0_syn * factor).astype(int)

    # Stitch patches with jittered positions; weight overlap by maskhr
    idx = 0
    for i in range(d1_syn):
        for j in range(d2_syn):
            ii_hr = int(i0_syn_hr[i, j])
            jj_hr = int(j0_syn_hr[i, j])
            block = Y[idx, :].reshape(masksizehr, masksizehr)

            Isynhr[ii_hr:ii_hr + masksizehr, jj_hr:jj_hr + masksizehr] += block + means_syn[i, j] * maskhr
            Crechr[ii_hr:ii_hr + masksizehr, jj_hr:jj_hr + masksizehr] += maskhr
            idx += 1

    nz = Crechr != 0
    Isynhr[nz] = Isynhr[nz] / Crechr[nz]

    # Crop central area
    Isynhr = Isynhr[
        masksizehr:masksizehr + H_in * factor,
        masksizehr:masksizehr + W_in * factor
    ]

    # -------------------------------
    # Normalize output to [0,1] range (simple contrast stretch)
    # -------------------------------
    Isynhr = Isynhr - Isynhr.min()
    maxv = Isynhr.max()
    if maxv > 0:
        Isynhr = Isynhr / maxv

    # Save as 16-bit PNG spanning full dynamic range
    Image.fromarray((Isynhr * 65535.0).astype(np.uint16)).save(output_path)

    print(f"Done in {time.time() - start:.2f} s. Wrote: {output_path}")


# ------------------------------
# CLI
# ------------------------------

def main():
    p = argparse.ArgumentParser(description="Terrain amplification (super-resolution) — Python CLI")
    p.add_argument("--factor", type=int, required=False, default=4, help="Amplification factor (e.g., 4)")
    p.add_argument("--exemplarhr", type=str, required=True, help="High-resolution exemplar filename")
    p.add_argument("--inputterrain", type=str, required=True, help="Input terrain filename")
    p.add_argument("--output", type=str, required=True, help="Output filename (e.g., out.png)")
    p.add_argument("--masksize", type=int, required=True, help="Mask (patch) diameter on LR images (even integer)")
    p.add_argument("--offset-analysis", type=int, required=True, help="Patch offset for LR exemplar analysis")
    p.add_argument("--offset-synthesis", type=int, required=True, help="Patch offset used for synthesis")

    # New jitter-related controls
    p.add_argument("--jitter-px", type=int, default=0,
                   help="Max random +/- pixel jitter applied to each LR patch (default: 0 = off)")
    p.add_argument("--scale-jitter", type=float, default=0.0,
                   help="Per-patch random gain in [1-S, 1+S] (default: 0.0 = off)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for reproducibility (default: None)")

    args = p.parse_args()

    terrain_super_resolution(
        factor=args.factor,
        exemplarhr_path=args.exemplarhr,
        inputterrain_path=args.inputterrain,
        output_path=args.output,
        masksize=args.masksize,
        offset_analysis=args.offset_analysis,
        offset_synthesis=args.offset_synthesis,
        jitter_px=args.jitter_px,
        scale_jitter=args.scale_jitter,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
