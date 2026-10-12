import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_att_dataset(dataset_path="../data/faces/ATT_Faces/ATT_Faces_Dataset", img_shape=(112, 92)):
    images = []
    labels = []
    for subject_id in range(1, 41):
        subj_folder = os.path.join(dataset_path, f"s{subject_id}")
        if not os.path.exists(subj_folder):
            continue                
        for img_id in range(1, 11):
            if subject_id <= 37:
                img_name = f"{img_id}.pgm"
            elif subject_id == 38:
                img_name = f"{img_id}_fl.pgm"
            elif subject_id == 39:
                img_name = f"{img_id}_sp.pgm"
            elif subject_id == 40:
                img_name = f"{img_id}_neg.pgm"
            else:
                img_name = f"{img_id}.pgm"
                
            img_path = os.path.join(subj_folder, img_name)
            
            if not os.path.exists(img_path):
                matches = [f for f in os.listdir(subj_folder) if f.startswith(f"{img_id}_") or f == f"{img_id}.pgm"]
                if matches:
                    img_path = os.path.join(subj_folder, matches[0])
                    
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (img_shape[1], img_shape[0]))
                    images.append(img_resized.flatten())
                    labels.append((subject_id, img_id))
                    
    return np.array(images, dtype=np.float32), labels

def compute_pca(train_images, num_components=10):

    mean_image = np.mean(train_images, axis=0)
    centered_data = train_images - mean_image
    
    N = centered_data.shape[0]
    gram_matrix = np.dot(centered_data, centered_data.T) / N
    evals, evecs = np.linalg.eigh(gram_matrix)
    
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    eigenfaces = np.dot(centered_data.T, evecs)
    for i in range(eigenfaces.shape[1]):
        norm = np.linalg.norm(eigenfaces[:, i])
        if norm > 0:
            eigenfaces[:, i] /= norm
            
    selected_eigenfaces = eigenfaces[:, :num_components]
    return mean_image, selected_eigenfaces, evals[:num_components]

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    images, labels = load_att_dataset()
    
    train_indices = [idx for idx, (subj, img_id) in enumerate(labels) if 1 <= subj <= 30 and 1 <= img_id <= 8]
    train_images = images[train_indices]
    mean_image, eigenfaces, evals = compute_pca(train_images, num_components=10)
    
    np.save("output/mean_image.npy", mean_image)
    np.save("output/eigenfaces.npy", eigenfaces)
    np.save("output/eigenvalues.npy", evals)
    
    img_h, img_w = 112, 92
    
    plt.figure(figsize=(4, 4))
    plt.imshow(mean_image.reshape(img_h, img_w), cmap='gray')
    plt.title("Mean Face Image")
    plt.axis('off')
    plt.savefig("output/mean_image.png", bbox_inches='tight')
    plt.close()
    
    alpha = 50.0
    fig, axes = plt.subplots(10, 3, figsize=(9, 25))
    for i in range(10):
        ef = eigenfaces[:, i].reshape(img_h, img_w)
        mean_plus = (mean_image + alpha * eigenfaces[:, i]).reshape(img_h, img_w)
        mean_minus = (mean_image - alpha * eigenfaces[:, i]).reshape(img_h, img_w)
        
        axes[i, 0].imshow(ef, cmap='gray')
        axes[i, 0].set_title(f"Eigenface {i+1}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mean_plus, cmap='gray')
        axes[i, 1].set_title(f"Mean + alpha * EF")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(mean_minus, cmap='gray')
        axes[i, 2].set_title(f"Mean - alpha * EF")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig("output/principal_modes.png", bbox_inches='tight')
    plt.close()
    print("Part A outputs saved successfully to output/")