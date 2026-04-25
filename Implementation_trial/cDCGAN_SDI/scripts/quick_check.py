import os
import random
from PIL import Image
import matplotlib.pyplot as plt

BASE = "/Users/wangtiles/DSCI601/DSCI601-generative-ai-for-sustainability/Implementation_trial/cDCGAN_SDI"

normal_dir = os.path.join(BASE, "A_ok")
scratch_dirs = [
    os.path.join(BASE, "A_nok", "train", "scratches"),
    os.path.join(BASE, "A_nok", "val", "scratches"),
    os.path.join(BASE, "A_nok", "test", "scratches"),
]
spot_dirs = [
    os.path.join(BASE, "A_nok", "train", "spots"),
    os.path.join(BASE, "A_nok", "val", "spots"),
    os.path.join(BASE, "A_nok", "test", "spots"),
]

def get_images(folder_list):
    files = []
    for folder in folder_list if isinstance(folder_list, list) else [folder_list]:
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                files.append(os.path.join(folder, f))
    return sorted(files)

normal_files = get_images(normal_dir)
scratch_files = get_images(scratch_dirs)
spot_files = get_images(spot_dirs)

print("Normals:", len(normal_files))
print("Scratches:", len(scratch_files))
print("Spots:", len(spot_files))

def show_samples(files, title, n=10):
    chosen = random.sample(files, min(n, len(files)))
    plt.figure(figsize=(15, 2))
    for i, path in enumerate(chosen):
        img = Image.open(path)
        plt.subplot(1, len(chosen), i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

show_samples(normal_files, "Normal samples")
show_samples(scratch_files, "Scratch samples")
show_samples(spot_files, "Spot samples")