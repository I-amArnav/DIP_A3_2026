
import numpy as np

def _mean_shift_update(oldz, density, converged, hr, hs):
    n = oldz.shape[0]
    m = density.shape[0]

    newz = oldz.copy()

    hr2 = 2.0 * hr * hr
    hs2 = 2.0 * hs * hs

    for i in range(n):
        if converged[i]:
            continue

        zr = oldz[i, 0]
        zg = oldz[i, 1]
        zb = oldz[i, 2]
        zx = oldz[i, 3]
        zy = oldz[i, 4]

        denominator = 0.0
        sums = np.zeros(5, dtype=np.float64)

        for j in range(m):
            dr = zr - density[j, 0]
            dg = zg - density[j, 1]
            db = zb - density[j, 2]

            dx = zx - density[j, 3]
            dy = zy - density[j, 4]

            color_dist = dr * dr + dg * dg + db * db
            spatial_dist = dx * dx + dy * dy

            weight = np.exp(-color_dist / hr2 - spatial_dist / hs2)

            denominator += weight

            for k in range(5):
                sums[k] += weight * density[j, k]

        if denominator > 1e-12:
            for k in range(5):
                newz[i, k] = sums[k] / denominator

    return newz


def my_mean_shift_segmentation(image, hr, hs, fixed_density_approx=None, max_iter=30, eps=0.5, sample_count=100, seed=0):
    H, W = image.shape[:2]

    z0 = np.empty((H, W, 5), dtype=np.float32)
    z0[:, :, :3] = image[:, :, :3].astype(np.float32)
    z0[:, :, 3] = np.arange(H, dtype=np.float32)[:, None]
    z0[:, :, 4] = np.arange(W, dtype=np.float32)[None, :]
    z0 = z0.reshape(-1, 5)

    if fixed_density_approx is None:
        rng = np.random.default_rng(seed)
        count = min(sample_count, len(z0))
        indices = rng.choice(len(z0), size=count, replace=False)
        fixed_density_approx = z0[indices].copy()

    density = np.asarray(fixed_density_approx, dtype=np.float32)

    oldz = z0.copy()
    converged = np.zeros(len(oldz), dtype=np.bool_)

    for iteration in range(max_iter):
        newz = _mean_shift_update(oldz, density, converged, hr, hs)

        delta = newz - oldz
        displacement = np.sqrt(np.sum(delta * delta, axis=1))

        converged |= displacement < eps
        oldz = newz

        remaining = np.count_nonzero(~converged)
        print(f"Iteration {iteration + 1}: {remaining} pixels unconverged")

        if remaining == 0:
            break

    return oldz.reshape(H, W, 5)