
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from c import inpainting

original_path = "../data/texture/lohagad.png"
marked_path = "lohagad_marked.png"

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

original = cv2.imread(original_path, cv2.IMREAD_COLOR)
marked = cv2.imread(marked_path, cv2.IMREAD_COLOR)

if original is None:
    raise FileNotFoundError(f"Could not load original image: {original_path}")

if marked is None:
    raise FileNotFoundError(f"Could not load marked image: {marked_path}")

if original.shape != marked.shape:
    raise ValueError(
        f"Original and marked images must have identical dimensions. "
        f"Original: {original.shape}, marked: {marked.shape}"
    )

white_threshold = 250

mask = np.all(marked >= white_threshold, axis=2)
mask = mask.astype(np.uint8) * 255

if not np.any(mask):
    raise ValueError(
        "No white pixels detected. Paint the text using pure white "
        "(#FFFFFF) and save the image as PNG."
    )

dilation_iterations = 1
kernel = np.ones((3, 3), dtype=np.uint8)

if dilation_iterations > 0:
    mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

mask_path = os.path.join(output_dir, "lohagad_mask.png")
hole_path = os.path.join(output_dir, "lohagad_with_hole.png")

cv2.imwrite(mask_path, mask)

with_hole = original.copy()
with_hole[mask > 0] = 0

cv2.imwrite(hole_path, with_hole)

filled = inpainting(original, mask, window_size=9, candidate_stride=3, max_candidates=2500, random_seed=42)

filled_path = os.path.join(output_dir, "lohagad_inpainted.png")
cv2.imwrite(filled_path, filled)

original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
with_hole_rgb = cv2.cvtColor(with_hole, cv2.COLOR_BGR2RGB)
filled_rgb = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
mask_rgb = cv2.cvtColor(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].imshow(original_rgb)
axes[0, 0].set_title("Original Image")

axes[0, 1].imshow(mask_rgb)
axes[0, 1].set_title("Generated Mask")

axes[1, 0].imshow(with_hole_rgb)
axes[1, 0].set_title("Image with Hole")

axes[1, 1].imshow(filled_rgb)
axes[1, 1].set_title("Texture-Based Inpainting")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

comparison_path = os.path.join(output_dir, "lohagad_comparison.png")

plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved outputs:")
print(mask_path)
print(hole_path)
print(filled_path)
print(comparison_path)
