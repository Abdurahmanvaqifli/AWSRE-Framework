"""
AWSRE Framework
Content-adaptive (HVS-masking-based) embedding-strength module.

THIS IS THE DISSERTATION'S CORE SCIENTIFIC CONTRIBUTION: every AWSRE method
so far (DCT, DWT, DCT-SVD, DWT-SVD, BLOCK-SVD) embeds with a single, global
alpha applied identically to every block regardless of what is actually in
that block. This module replaces that with a PER-BLOCK embedding-strength
MULTIPLIER derived from the block's own local content -- grounded in the
well-established Human Visual System (HVS) masking principle used
throughout the perceptual/JND (Just-Noticeable-Distortion) watermarking
literature: the human eye is measurably less sensitive to small intensity
perturbations in busy, high-texture, high-edge regions than in smooth,
flat regions (sky, skin, a medical scan's dark background, etc.).

Concretely, for each block:
    score = log1p(local_variance) + log1p(local_edge_energy)
      -> local_variance:    how "flat vs. busy" the block's own pixel
                             intensities are
      -> local_edge_energy: mean Sobel gradient magnitude, how much real
                             structure (edges) the block contains
    (log1p keeps a handful of extreme blocks -- e.g. one hard silhouette
    edge -- from dominating the normalization below; both quantities are
    heavy-tailed across natural/medical images.)

The raw scores are then min-max normalized AGAINST THIS IMAGE'S OWN block
score distribution (not a fixed global threshold), so the method adapts to
each host image individually, and mapped into [MIN_MULTIPLIER,
MAX_MULTIPLIER] (default 0.5x - 1.6x of the base alpha):
    - smooth / flat blocks       -> multiplier near 0.5  (weaker embedding,
                                     where distortion would be most visible)
    - highly textured/edgy blocks -> multiplier near 1.6 (stronger
                                     embedding, where the eye already masks
                                     more distortion, AND which tends to
                                     survive compression/geometric attacks
                                     better since it's exactly where the
                                     signal has room to hide)

Why this needed NO changes to any extract() function anywhere in the
codebase: every AWSRE method decodes a watermark bit from the SIGN of
(processed_coefficient - original_coefficient) -- see
DCTWatermarker.extract() / DCTSVDWatermarker.extract() -- never from its
magnitude relative to a fixed alpha. Varying the embedding strength
per-block only changes how large that difference is, never its sign, so
the existing non-blind, sign-based extraction logic decodes an
adaptively-embedded watermark exactly as reliably as it decodes a
classic, fixed-alpha one. This is what makes DCTAdaptiveWatermarker /
DCTSVDWatermarker's adaptive counterpart (see dct.py / dct_svd.py) a
strictly additive, backward-compatible change: registered under new
method names ("DCT-Adaptive", "DCT-SVD-Adaptive"), so the classic
DCT/DWT/DCT-SVD methods and every scientific measurement already collected
for them (experiments/scientific_results*) are completely untouched.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MIN_MULTIPLIER = 0.5
DEFAULT_MAX_MULTIPLIER = 1.6
DEFAULT_CONTEXT_MARGIN = 4


def compute_block_strength_map(
    host_gray: np.ndarray,
    block_size: int,
    n_rows: int,
    n_cols: int,
    min_multiplier: float = DEFAULT_MIN_MULTIPLIER,
    max_multiplier: float = DEFAULT_MAX_MULTIPLIER,
    context_margin: int = DEFAULT_CONTEXT_MARGIN,
) -> np.ndarray:
    """
    Return an (n_rows, n_cols) float64 array of per-block alpha
    multipliers in [min_multiplier, max_multiplier].

    Parameters
    ----------
    host_gray:
        The ORIGINAL grayscale host image (not the watermarked one).
        Because AWSRE's methods are non-blind, extraction always has
        access to this same original image, so callers on the extraction
        side could in principle recompute the identical map -- though in
        practice they don't even need to, since decoding is sign-based
        (see module docstring).
    block_size:
        Must match the watermarker's own block_size, so multiplier[row,
        col] lines up with the exact same pixel region embed()/extract()
        address via block_coordinates(row, col, block_size).
    n_rows, n_cols:
        Watermark shape (bit_row count, bit_column count) -- one
        multiplier per embedded bit/block, in the same row-major order
        embed() iterates over.
    min_multiplier, max_multiplier:
        Range the per-block strength is scaled into. 1.0 would reproduce
        the classic fixed-alpha behaviour for every block; the default
        [0.5, 1.6] keeps the same *mean* embedding strength in roughly the
        same ballpark as a fixed alpha while giving meaningful per-block
        contrast between smooth and textured content.
    context_margin:
        Extra pixels read around each block (clipped at image borders) so
        the texture/edge score reflects a little of the block's immediate
        surroundings too, not just its own raw 8x8 (or similar) contents,
        which is a very small and sometimes noisy sample on its own.

    Returns
    -------
    np.ndarray
        Shape (n_rows, n_cols). A perfectly flat / degenerate image (no
        variation anywhere to discriminate blocks by) returns an all-ones
        map, i.e. every block falls back to exactly the classic, uniform
        alpha behaviour.
    """
    host = host_gray.astype(np.float32)
    height, width = host.shape[:2]

    # Sobel-style gradient magnitude for the whole image, computed once.
    # Central differences (rather than importing OpenCV's Sobel here) --
    # this module intentionally has no dependency beyond numpy so it can
    # be unit-tested and reused independently of OpenCV/watermarking
    # internals.
    gradient_x = np.zeros_like(host)
    gradient_y = np.zeros_like(host)
    gradient_x[:, 1:-1] = host[:, 2:] - host[:, :-2]
    gradient_y[1:-1, :] = host[2:, :] - host[:-2, :]
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    raw_scores = np.zeros((n_rows, n_cols), dtype=np.float64)

    for row in range(n_rows):
        for col in range(n_cols):
            y0 = max(0, row * block_size - context_margin)
            y1 = min(height, (row + 1) * block_size + context_margin)
            x0 = max(0, col * block_size - context_margin)
            x1 = min(width, (col + 1) * block_size + context_margin)

            window = host[y0:y1, x0:x1]
            edge_window = gradient_magnitude[y0:y1, x0:x1]

            local_variance = float(np.var(window))
            local_edge_energy = float(np.mean(edge_window))

            raw_scores[row, col] = (
                np.log1p(local_variance) + np.log1p(local_edge_energy)
            )

    score_min = float(raw_scores.min())
    score_max = float(raw_scores.max())
    spread = score_max - score_min

    if spread < 1e-9:
        # No basis to discriminate blocks (e.g. a perfectly flat test
        # image, or a 1x1 watermark) -- fall back to the classic, uniform
        # behaviour rather than dividing by ~zero.
        return np.ones((n_rows, n_cols), dtype=np.float64)

    normalized = (raw_scores - score_min) / spread  # in [0, 1]
    return min_multiplier + normalized * (max_multiplier - min_multiplier)
