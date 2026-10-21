import numpy as np

def make_label_image(features, color_tol=50, spatial_tol=50):
    H, W = features.shape[:2]
    data = features.reshape(-1, 5).astype(np.float32)

    scaled = data.copy()
    scaled[:, :3] /= color_tol
    scaled[:, 3:5] /= spatial_tol

    quantized = np.round(scaled, decimals=1)

    unique_modes, inverse = np.unique(quantized, axis=0, return_inverse=True)

    mode_labels = np.zeros(len(unique_modes), dtype=np.int32)
    representatives = []
    num_labels = 0

    for i, mode in enumerate(unique_modes):
        if mode_labels[i] != 0:
            continue

        num_labels += 1
        representatives.append(mode)
        mode_labels[i] = num_labels

        remaining = np.flatnonzero(mode_labels == 0)
        if len(remaining) > 0:
            distances = np.linalg.norm(unique_modes[remaining] - mode, axis=1)
            close = remaining[distances <= 1.0]
            mode_labels[close] = num_labels

    labels = mode_labels[inverse].reshape(H, W)
    return labels

def draw_black_boundaries(image, labels):
    result = image.copy()
    boundary = np.zeros(labels.shape, dtype=bool)

    diff = labels[:, 1:] != labels[:, :-1]
    boundary[:, 1:] |= diff
    boundary[:, :-1] |= diff

    diff = labels[1:, :] != labels[:-1, :]
    boundary[1:, :] |= diff
    boundary[:-1, :] |= diff

    result[boundary] = 0
    return result



        