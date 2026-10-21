
import cv2
import heapq
import numpy as np
from numba import njit, prange
from a import get_gaussian_weights


@njit(parallel=True, cache=True)
def calculate_distances(image, cand_y, cand_x, target, known, weights):

    n = len(cand_y)
    win = target.shape[0]
    channels = target.shape[2]
    pad = win // 2
    distances = np.zeros(n, dtype=np.float32)

    for i in prange(n):
        cy, cx = cand_y[i], cand_x[i]
        total = 0.0

        for dy in range(win):
            for dx in range(win):
                if known[dy, dx]:
                    for c in range(channels):
                        diff = (image[cy + dy - pad, cx + dx - pad, c] - target[dy, dx, c])
                        total += weights[dy, dx] * diff * diff
        distances[i] = total
    return distances

def inpainting(image, mask, window_size=9, tol_factor=1.1, candidate_stride=3, max_candidates=2500, random_seed=42):

    if image is None or mask is None:
        raise ValueError("Image and mask must not be None")

    if mask.ndim != 2 or mask.shape != image.shape[:2]:
        raise ValueError("Mask must be 2D and match the image dimensions")

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    if candidate_stride < 1 or max_candidates < 1:
        raise ValueError("Candidate settings must be positive")

    rng = np.random.default_rng(random_seed)

    original_gray = (image.ndim == 2)
    img = image.astype(np.float32)

    if img.max() > 1.0:
        img /= 255.0

    if original_gray:
        img = img[:, :, None]

    h, w, channels = img.shape
    pad = window_size // 2

    if h < window_size or w < window_size:
        raise ValueError("Image is smaller than the neighborhood window")

    hole = (mask > 0)
    known = ~hole
    output = img.copy()

    if not np.any(hole):
        return image.copy()

    if not np.any(known):
        raise ValueError("Cannot inpaint an image with no known pixels")

    integral = cv2.integral(hole.astype(np.uint8))

    def get_source_centers(stride):
        ys = np.arange(pad, h - pad, stride)
        xs = np.arange(pad, w - pad, stride)

        if len(ys) == 0 or len(xs) == 0:
            return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        sums = (
            integral[yy + pad + 1, xx + pad + 1]
            - integral[yy - pad, xx + pad + 1]
            - integral[yy + pad + 1, xx - pad]
            + integral[yy - pad, xx - pad]
        )

        valid = (sums == 0)
        return yy[valid].astype(np.int32), xx[valid].astype(np.int32)

    cand_y, cand_x = get_source_centers(candidate_stride)

    if len(cand_y) == 0:
        cand_y, cand_x = get_source_centers(1)

    if len(cand_y) == 0:
        raise ValueError(
            "No fully known source patches. Reduce window_size "
            "or provide a larger known region."
        )

    if len(cand_y) > max_candidates:
        chosen = rng.choice(len(cand_y), max_candidates, replace=False)
        cand_y = cand_y[chosen]
        cand_x = cand_x[chosen]

    gaussian = get_gaussian_weights(window_size, window_size / 6.4)

    heap = []
    versions = np.zeros((h, w), dtype=np.int32)
    counter = 0

    def neighbor_count(y, x):
        return int(np.sum(known[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)]))

    def add_to_heap(y, x):
        nonlocal counter

        if known[y, x]:
            return

        count = neighbor_count(y, x)
        if count == 0:
            return

        versions[y, x] += 1

        heapq.heappush(heap, (-count, counter, y, x, versions[y, x]))
        counter += 1

    counts = cv2.filter2D(
        known.astype(np.float32),
        -1,
        np.ones((3, 3), dtype=np.float32),
        borderType=cv2.BORDER_CONSTANT
    )

    ys, xs = np.nonzero(hole & (counts > 0))

    for y, x in zip(ys, xs):
        add_to_heap(int(y), int(x))

    while heap:
        neg_count, _, y, x, version = heapq.heappop(heap)

        if known[y, x] or version != versions[y, x]:
            continue

        count = neighbor_count(y, x)

        if count == 0:
            continue

        if count != -neg_count:
            add_to_heap(y, x)
            continue

        y1, y2 = max(0, y-pad), min(h, y+pad+1)
        x1, x2 = max(0, x-pad), min(w, x+pad+1)

        wy1, wy2 = pad-(y-y1), pad+(y2-y)
        wx1, wx2 = pad-(x-x1), pad+(x2-x)

        target = np.zeros((window_size, window_size, channels), dtype=np.float32)
        target_known = np.zeros((window_size, window_size), dtype=bool)

        target[wy1:wy2, wx1:wx2] = output[y1:y2, x1:x2]
        target_known[wy1:wy2, wx1:wx2] = known[y1:y2, x1:x2]

        weights = gaussian * target_known
        total_weight = weights.sum()

        if total_weight == 0:
            continue

        weights /= total_weight

        distances = calculate_distances(img, cand_y, cand_x, target, target_known, weights)

        best = float(distances.min())
        matches = np.flatnonzero(distances <= tol_factor * max(best, 1e-5))

        idx = int(rng.choice(matches))
        sy, sx = int(cand_y[idx]), int(cand_x[idx])

        new_pixels = []

        for ty in range(y1, y2):
            for tx in range(x1, x2):
                if not known[ty, tx]:
                    py = sy + (ty - y)
                    px = sx + (tx - x)

                    output[ty, tx] = img[py, px]
                    known[ty, tx] = True
                    new_pixels.append((ty, tx))

        for ty, tx in new_pixels:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = ty + dy, tx + dx

                    if (0 <= ny < h and 0 <= nx < w and not known[ny, nx]):
                        add_to_heap(ny, nx)

    if not known.all():
        raise RuntimeError("Some pixels remain unfilled. Try a smaller window_size.")

    result = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    if original_gray:
        return result[:, :, 0]

    return result
