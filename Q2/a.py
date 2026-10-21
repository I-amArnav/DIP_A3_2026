import numpy as np

def anisotropic_diffusion(image,iterations=20,kappa=25.0,step=0.20,conduction="reciprocal"):
    
    if not 0 < step <= 0.25:
        raise ValueError("step must satisfy 0 < step <= 0.25")

    if kappa <= 0:
        raise ValueError("kappa must be positive")

    u = image.astype(np.float64).copy()

    for _ in range(iterations):
        padded = np.pad(u, ((1, 1), (1, 1)) + (((0, 0),) if u.ndim == 3 else ()), mode="edge")

        center = padded[1:-1, 1:-1, ...]
        north = padded[:-2, 1:-1, ...]
        south = padded[2:, 1:-1, ...]
        west = padded[1:-1, :-2, ...]
        east = padded[1:-1, 2:, ...]

        differences = [ north - center, south - center, west - center, east - center]

        update = np.zeros_like(u)

        for diff in differences:
            if u.ndim == 3:
                gradient_sq = np.sum(diff * diff, axis=2, keepdims=True)
            else:
                gradient_sq = diff * diff

            if conduction == "exponential":
                conductance = np.exp(-gradient_sq / (kappa * kappa))
            else:
                conductance = 1.0 / ( 1.0 + gradient_sq / (kappa * kappa))

            update += conductance * diff

        u += step * update

    return np.clip(u, 0, 255).astype(np.uint8)



