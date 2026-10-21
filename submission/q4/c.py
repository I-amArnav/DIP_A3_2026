import os
import numpy as np
from a import load_att_dataset, compute_pca

def recognize_faces(images, train_images, train_labels, mean_image, eigenfaces):

    train_centered = train_images - mean_image
    train_weights = np.dot(train_centered, eigenfaces)
    
    predictions = []
    for img in images:
        centered = img - mean_image
        weights = np.dot(centered, eigenfaces)
        distances = np.linalg.norm(train_weights - weights, axis=1)
        min_idx = np.argmin(distances)
        pred_subject = train_labels[min_idx][0]
        predictions.append(pred_subject)
        
    return predictions

if __name__ == "__main__":

    os.makedirs("output", exist_ok=True)
    images, labels = load_att_dataset()
    
    train_indices = []
    train_labels = []
    for idx, (subj, img_id) in enumerate(labels):
        if 1 <= subj <= 30 and 1 <= img_id <= 8:
            train_indices.append(idx)
            train_labels.append((subj, img_id))
            
    train_images = images[train_indices]
    mean_image, eigenfaces, _ = compute_pca(train_images, num_components=20)
    
    preds = recognize_faces(images, train_images, train_labels, mean_image, eigenfaces)
    
    M2 = np.zeros((40, 10))
    for idx, (subj, img_id) in enumerate(labels):
        M2[subj - 1, img_id - 1] = preds[idx]
        
    np.save("output/matrix_M2.npy", M2)
    np.savetxt("output/matrix_M2.csv", M2, delimiter=",")