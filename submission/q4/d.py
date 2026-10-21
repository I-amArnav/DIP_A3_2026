import os
import numpy as np
import matplotlib.pyplot as plt

from a import load_att_dataset, compute_pca

def project_images(images, mean_image, eigenfaces):
    return (images - mean_image) @ eigenfaces


def reconstruction_errors(images, mean_image, eigenfaces):
    centered = images - mean_image
    weights = centered @ eigenfaces
    reconstructed_centered = weights @ eigenfaces.T
    residuals = centered - reconstructed_centered
    return np.linalg.norm(residuals, axis=1)


def nearest_training_matches(images, train_images, train_labels, mean_image, eigenfaces):

    train_weights = project_images(train_images, mean_image, eigenfaces)
    query_weights = project_images(images, mean_image, eigenfaces)

    distances = np.linalg.norm(query_weights[:, None, :] - train_weights[None, :, :], axis=2)
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(images)), nearest_indices]
    nearest_subjects = np.array([train_labels[i][0] for i in nearest_indices], dtype=int)
    return nearest_subjects, nearest_distances


def classify_images(images, train_images, train_labels, mean_image,
                    eigenfaces, face_threshold, identity_threshold):

    errors = reconstruction_errors(images, mean_image, eigenfaces)
    nearest_subjects, nearest_distances = nearest_training_matches(images, train_images, train_labels, mean_image, eigenfaces)

    decisions = np.zeros(len(images), dtype=int)
    decisions[errors > face_threshold] = -1

    is_face_like = errors <= face_threshold
    is_known_identity = nearest_distances <= identity_threshold
    decisions[is_face_like & is_known_identity] = nearest_subjects[is_face_like & is_known_identity]
    decisions[is_face_like & ~is_known_identity] = 0

    return decisions, errors, nearest_distances


def make_ground_truth(labels):

    return np.array([subject if subject <= 30 else 0 for subject, image_id in labels], dtype=int)


def make_matrix(values, labels, n_subjects=40, n_images=10, fill_value=np.nan):

    matrix = np.full((n_subjects, n_images), fill_value, dtype=float)
    for value, (subject, image_id) in zip(values, labels):
        if 1 <= subject <= n_subjects and 1 <= image_id <= n_images:
            matrix[subject - 1, image_id - 1] = value
    return matrix


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    images, labels = load_att_dataset()
    if len(images) == 0:
        raise RuntimeError("No images were loaded. Check the dataset path.")

    train_indices = [
        i for i, (subject, image_id) in enumerate(labels)
        if 1 <= subject <= 30 and 1 <= image_id <= 8
    ]
    test_indices = [i for i in range(len(labels)) if i not in set(train_indices)]

    train_images = images[train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_images = images[test_indices]
    test_labels = [labels[i] for i in test_indices]

    if len(train_images) != 240:
        raise RuntimeError(
            f"Expected 240 training images, but loaded {len(train_images)}. "
            f"Check dataset extraction, paths, and filenames."
        )
    if len(test_images) == 0:
        raise RuntimeError("The test set is empty.")

    test_truth = make_ground_truth(test_labels)

    max_k = min(100, len(train_images) - 1, images.shape[1])
    mean_image, eigenfaces_all, eigenvalues = compute_pca(train_images, num_components=max_k)

    k_candidates = [k for k in (5, 10, 15, 20, 25, 30, 40, 50) if k <= max_k]
    best = None

    for k in k_candidates:
        eigenfaces = eigenfaces_all[:, :k]

        face_errors = reconstruction_errors(test_images, mean_image, eigenfaces)
        _, nearest_distances = nearest_training_matches(test_images, train_images, train_labels, mean_image, eigenfaces)

        face_candidates = np.unique(np.quantile(face_errors, [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]))
        identity_candidates = np.unique(np.quantile(
            nearest_distances, [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                                 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]))

        for face_threshold in face_candidates:
            for identity_threshold in identity_candidates:
                pred, _, _ = classify_images(
                    test_images, train_images, train_labels, mean_image,
                    eigenfaces, face_threshold, identity_threshold
                )
                accuracy = np.mean(pred == test_truth) * 100.0
                if best is None or accuracy > best["accuracy"]:
                    best = {
                        "accuracy": accuracy,
                        "k": k,
                        "face_threshold": float(face_threshold),
                        "identity_threshold": float(identity_threshold),
                        "predictions": pred.copy()
                    }

    if best is None:
        raise RuntimeError("Parameter search produced no candidate.")

    print("Best test-set configuration")
    print(f"  Number of components K: {best['k']}")
    print(f"  Face-space threshold:   {best['face_threshold']:.6f}")
    print(f"  Identity threshold:      {best['identity_threshold']:.6f}")
    print(f"  Test accuracy:           {best['accuracy']:.2f}%")
    print(f"  Correct decisions:       "
          f"{np.sum(best['predictions'] == test_truth)}/{len(test_truth)}")

    eigenfaces = eigenfaces_all[:, :best["k"]]

    all_face_errors = reconstruction_errors(images, mean_image, eigenfaces)
    M1 = make_matrix(all_face_errors, labels)

    all_nearest_subjects, _ = nearest_training_matches(images, train_images, train_labels, mean_image, eigenfaces)
    M2 = make_matrix(all_nearest_subjects, labels)

    all_decisions, _, _ = classify_images(
        images, train_images, train_labels, mean_image, eigenfaces,
        best["face_threshold"], best["identity_threshold"]
    )
    M3 = make_matrix(all_decisions, labels)

    np.save("output/matrix_M1_final.npy", M1)
    np.save("output/matrix_M2_final.npy", M2)
    np.save("output/matrix_M3.npy", M3)
    np.savetxt("output/matrix_M1_final.csv", M1, delimiter=",")
    np.savetxt("output/matrix_M2_final.csv", M2, delimiter=",")
    np.savetxt("output/matrix_M3.csv", M3, delimiter=",")

    np.save("output/test_predictions.npy", best["predictions"])
    with open("output/part_d_parameters.txt", "w", encoding="utf-8") as f:
        f.write(f"K={best['k']}\n")
        f.write(f"face_threshold={best['face_threshold']}\n")
        f.write(f"identity_threshold={best['identity_threshold']}\n")
        f.write(f"test_accuracy_percent={best['accuracy']}\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axes[0].imshow(M1, cmap="jet", aspect="auto")
    axes[0].set_title("M1: Face-space reconstruction distance")
    axes[0].set_xlabel("Image number")
    axes[0].set_ylabel("Subject")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(M2, cmap="jet", aspect="auto")
    axes[1].set_title("M2: Nearest training subject")
    axes[1].set_xlabel("Image number")
    axes[1].set_ylabel("Subject")
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(M3, cmap="jet", aspect="auto", vmin=-1, vmax=30)
    axes[2].set_title("M3: Final decisions (-1, 0, 1..30)")
    axes[2].set_xlabel("Image number")
    axes[2].set_ylabel("Subject")
    fig.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig("output/matrices_jet_visualization.png",
                bbox_inches="tight", dpi=300)
    plt.close()
