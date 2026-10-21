
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from a import anisotropic_diffusion
from b import bilateral_filter

def process_image(path, stem, sigma_spatial=5.0, sigma_range=40.0, radius=9, iterations=50, kappa=45.0, step=0.20, conduction="exponential"):
    image = iio.imread(path)

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    anisotropic = anisotropic_diffusion( image, iterations=iterations, kappa=kappa, step=step, conduction=conduction)

    bilateral = bilateral_filter( image, sigma_spatial=sigma_spatial, sigma_range=sigma_range, radius=radius)

    panels = [("Input image", image),("Anisotropic diffusion", anisotropic),("Bilateral filtering", bilateral)]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    for ax, (title, img) in zip(axes, panels):
        ax.set_title(title)
        ax.axis("off")

        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)

    fig.suptitle(stem)
    fig.tight_layout()
    fig.savefig(f"outputs/{stem}_comparison.png",bbox_inches="tight")
    plt.close(fig)

    h, w = image.shape[:2]
    crop_h = min(200, h)
    crop_w = min(200, w)
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    for ax, (title, img) in zip(axes, panels):
        ax.set_title(title + " (zoom)")
        ax.axis("off")

        crop = img[y0:y0 + crop_h, x0:x0 + crop_w, ...]
        if crop.ndim == 2:
            ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(crop)

    fig.suptitle(stem + ": enlarged central crop")
    fig.tight_layout()
    fig.savefig(f"outputs/{stem}_zoom.png",bbox_inches="tight",)
    plt.close(fig)

    iio.imwrite(f"outputs/{stem}_anisotropic.png", anisotropic)
    iio.imwrite(f"outputs/{stem}_bilateral.png", bilateral)

    print("Finished:", stem)


process_image("../data/diffusion/colorCheckerNoisy.png", "colorChecker", sigma_spatial=7.0, sigma_range=60.0, radius=11, iterations=40, kappa=15.0, step=0.20, conduction="reciprocal")
process_image("../data/diffusion/rajaRaviVarma.png", "rajaRaviVarma", sigma_spatial=5.0, sigma_range=60.0, radius=11, iterations=40, kappa=17.0, step=0.20, conduction="reciprocal")
process_image("../data/diffusion/sagarmaathaa.png", "sagarmaathaa", sigma_spatial=3.0, sigma_range=50.0, radius=9, iterations=30, kappa=26.0, step=0.20, conduction="exponential")
process_image("../data/diffusion/racing.png", "racing", sigma_spatial=5.0, sigma_range=60.0, radius=9, iterations=20, kappa=20.0, step=0.20, conduction="reciprocal")
