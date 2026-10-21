# a
import cv2
import heapq
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def get_gaussian_weights(window_size, sigma):
    ax = np.arange(-(window_size // 2), window_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    weights = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return weights / np.sum(weights)

def calculate_distances(sample_patches, target, known, weights):
    n = sample_patches.shape[0]
    win = target.shape[0]
    channels = target.shape[2]
    distances = np.zeros(n, dtype=np.float32)

    for i in range(n):
        total = 0.0
        for dy in range(win):
            for dx in range(win):
                if known[dy, dx]:
                    for c in range(channels):
                        diff = (
                            sample_patches[i, dy, dx, c]
                            - target[dy, dx, c]
                        )
                        total += weights[dy, dx] * diff * diff
        distances[i] = total

    return distances


def synthesize_texture(sample_img, output_size=(256, 256), window_size=11, tol_factor=1.1, random_seed=42):

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    is_gray = sample_img.ndim == 2

    if sample_img.ndim not in (2, 3):
        raise ValueError("Input must be a grayscale or color image")

    sample_img = sample_img.astype(np.float32)

    if sample_img.max() > 1.0:
        sample_img /= 255.0

    if is_gray:
        sample_img = sample_img[:, :, np.newaxis]

    h_s, w_s, channels = sample_img.shape
    h_out, w_out = output_size
    pad = window_size // 2

    if h_s > h_out or w_s > w_out:
        raise ValueError("Sample texture must fit inside output")

    if h_s < window_size or w_s < window_size:
        raise ValueError("Sample is too small for the chosen window")

    rng = np.random.default_rng(random_seed)

    output = np.zeros((h_out, w_out, channels), dtype=np.float32)
    mask = np.zeros((h_out, w_out), dtype=bool)

    start_y = (h_out - h_s) // 2
    start_x = (w_out - w_s) // 2

    output[start_y:start_y+h_s, start_x:start_x+w_s] = sample_img
    mask[start_y:start_y+h_s, start_x:start_x+w_s] = True

    sigma = window_size / 6.4
    gaussian = get_gaussian_weights(window_size, sigma)

    patches = sliding_window_view(sample_img, (window_size, window_size), axis=(0, 1))

    patches = patches.transpose(0, 1, 3, 4, 2)

    sample_patches = np.ascontiguousarray(patches.reshape(-1, window_size, window_size, channels))

    heap = []
    versions = np.zeros((h_out, w_out), dtype=np.int32)
    counter = 0

    def neighbor_count(y, x):
        y1 = max(0, y - 1)
        y2 = min(h_out, y + 2)
        x1 = max(0, x - 1)
        x2 = min(w_out, x + 2)
        return int(np.sum(mask[y1:y2, x1:x2]))

    def add_to_heap(y, x):
        nonlocal counter

        if mask[y, x]:
            return

        count = neighbor_count(y, x)
        if count == 0:
            return

        versions[y, x] += 1

        heapq.heappush(heap, (-count, counter, y, x, versions[y, x]))
        counter += 1

    boundary = cv2.filter2D(
        mask.astype(np.uint8),
        cv2.CV_16U,
        np.ones((3, 3), dtype=np.uint8)
    )
    ys, xs = np.nonzero((~mask) & (boundary > 0))

    for y, x in zip(ys, xs):
        add_to_heap(int(y), int(x))

    while heap:
        neg_count, _, y, x, version = heapq.heappop(heap)

        if mask[y, x] or version != versions[y, x]:
            continue

        current_count = neighbor_count(y, x)

        if current_count == 0:
            continue

        if current_count != -neg_count:
            add_to_heap(y, x)
            continue

        y1 = max(0, y - pad)
        y2 = min(h_out, y + pad + 1)
        x1 = max(0, x - pad)
        x2 = min(w_out, x + pad + 1)

        target = np.zeros((window_size, window_size, channels), dtype=np.float32)
        known = np.zeros((window_size, window_size), dtype=bool)

        wy1 = pad - (y - y1)
        wy2 = pad + (y2 - y)
        wx1 = pad - (x - x1)
        wx2 = pad + (x2 - x)

        target[wy1:wy2, wx1:wx2] = output[y1:y2, x1:x2]
        known[wy1:wy2, wx1:wx2] = mask[y1:y2, x1:x2]

        weights = gaussian * known
        weights /= weights.sum()

        distances = calculate_distances(sample_patches, target, known, weights)

        min_dist = float(distances.min())

        valid = np.flatnonzero(distances <= tol_factor * max(min_dist, 1e-5))

        chosen = rng.choice(valid)

        center_y = chosen // (w_s - window_size + 1) + pad
        center_x = chosen % (w_s - window_size + 1) + pad

        output[y, x] = sample_img[center_y, center_x]
        mask[y, x] = True

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx

                if (0 <= ny < h_out and 0 <= nx < w_out and not mask[ny, nx]):
                    add_to_heap(ny, nx)

    if not mask.all():
        raise RuntimeError("Texture synthesis stopped before filling output")

    result = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    if is_gray:
        return result[:, :, 0]

    return result
