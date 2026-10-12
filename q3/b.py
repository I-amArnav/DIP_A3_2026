import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from a import synthesize_texture

def display_and_save_sample_and_synthesized(sample_paths, output_dir="outputs"):

    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(len(sample_paths), 2, figsize=(6, 3 * len(sample_paths)))
    
    if len(sample_paths) == 1:
        axes = np.array([axes])
        
    for i, path in enumerate(sample_paths):
        sample = cv2.imread(path)
        if sample is None:
            print(f"Could not load {path}")
            continue
            
        synth = synthesize_texture(sample, output_size=(256, 256), window_size=11, tol_factor=1.1)
        
        file_name = os.path.basename(path)
        save_path = os.path.join(output_dir, f"synthesized_{file_name}")
        cv2.imwrite(save_path, synth)
        print(f"Saved synthesized texture to {save_path}")
        
        axes[i, 0].imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Sample: {file_name}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(cv2.cvtColor(synth, cv2.COLOR_BGR2RGB) if len(synth.shape) == 3 else synth, cmap='gray' if len(synth.shape) == 2 else None)
        axes[i, 1].set_title("Synthesized (256x256)")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "texture_synthesis_comparison.png")
    plt.savefig(comparison_path, dpi=300)
    print(f"Saved comparison figure to {comparison_path}")
    plt.show()

if __name__ == "__main__":
    paths = [
        "../data/texture/textureSample81.png",
        "../data/texture/textureSample110.png",
        "../data/texture/textureSample207.png"
    ]
    display_and_save_sample_and_synthesized(paths)