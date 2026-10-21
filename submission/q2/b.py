import numpy as np

def bilateral_filter( image, sigma_spatial=3.0, sigma_range=25.0, radius=9):
    image = image.astype(np.float64)
    grayscale = image.ndim == 2
    if grayscale:
        image = image[:, :, None]

    height, width, channels = image.shape

    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect",)

    weighted_sum = np.zeros_like(image, dtype=np.float64)
    weight_sum = np.zeros((height, width, 1), dtype=np.float64)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):

            spatial_distance_sq = dx * dx + dy * dy
            spatial_weight = np.exp(-spatial_distance_sq / (2.0 * sigma_spatial**2))
        
            neighbour = padded[radius + dy : radius + dy + height, radius + dx : radius + dx + width, :]
            difference = neighbour - image
            range_distance_sq = np.sum(difference * difference, axis=2, keepdims=True)
            range_weight = np.exp(-range_distance_sq / (2.0 * sigma_range**2))

            weight = spatial_weight * range_weight
            weighted_sum += weight * neighbour
            weight_sum += weight

    result = weighted_sum / np.maximum(weight_sum, 1e-12)
    result = np.clip(result, 0, 255).astype(np.uint8)

    if grayscale:
        result = result[:, :, 0]

    return result

