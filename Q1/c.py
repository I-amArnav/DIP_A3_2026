import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from a import my_mean_shift_segmentation
from b import make_label_image, draw_black_boundaries



def process_image(image_path, hr, hs, color_tol, spatial_tol, stem):
    np.random.seed(0)
    
    image = iio.imread(image_path)

    features = my_mean_shift_segmentation(image, hr, hs, sample_count=300, max_iter=300, eps=1e-3)
    labels = make_label_image(features, color_tol=color_tol, spatial_tol=spatial_tol)
    boundary_image = draw_black_boundaries(image, labels)
    label_visualization = ((labels.astype(np.float32) - 1) / max(int(labels.max()) - 1, 1)* 255).astype(np.uint8)

    iio.imwrite(f"outputs/{stem}_label_map.png", label_visualization)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(labels, cmap="tab10", vmin=1, vmax=max(int(labels.max())+1, 2), interpolation="nearest")
    axes[1].set_title(f"Mean-shift segmentation ({labels.max()} segments)")
    axes[1].axis("off")
    
    fig.suptitle(f"Mean-shift segmentation (hr={hr}, hs={hs})", fontsize=15)
    fig.tight_layout()
    fig.savefig(f"outputs/{stem}_segmentation.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(boundary_image)
    ax.set_title("Input image with black segment boundaries")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"outputs/{stem}_boundary_overlay.png", bbox_inches="tight")


process_image("../data/segment/bison.png", hr=40, hs=250, color_tol=70, spatial_tol=70, stem="bison")
process_image("../data/segment/ghats.png", hr=25, hs=300, color_tol=50, spatial_tol=50, stem="ghats")
process_image("../data/segment/leopard.png", hr=25, hs=300, color_tol=50, spatial_tol=70, stem="leopard")
