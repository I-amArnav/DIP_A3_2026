import os
import numpy as np
from a import load_att_dataset, compute_pca

def compute_face_space_distances(images, mean_image, eigenfaces):

    scores = np.zeros(len(images))
    for i, img in enumerate(images):
        centered = img - mean_image
        weights = np.dot(centered, eigenfaces)
        reconstructed = mean_image + np.dot(weights, eigenfaces.T)
        scores[i] = np.linalg.norm(img - reconstructed)
    return scores

if __name__ == "__main__":

    os.makedirs("output", exist_ok=True)
    images, labels = load_att_dataset()
    
    train_indices = [idx for idx, (subj, img_id) in enumerate(labels) if 1 <= subj <= 30 and 1 <= img_id <= 8]
    train_images = images[train_indices]
    
    mean_image, eigenfaces, _ = compute_pca(train_images, num_components=20)
    all_distances = compute_face_space_distances(images, mean_image, eigenfaces)
    
    M1 = np.zeros((40, 10))
    for idx, (subj, img_id) in enumerate(labels):
        M1[subj - 1, img_id - 1] = all_distances[idx]
        
    np.save("output/matrix_M1.npy", M1)
    np.savetxt("output/matrix_M1.csv", M1, delimiter=",")