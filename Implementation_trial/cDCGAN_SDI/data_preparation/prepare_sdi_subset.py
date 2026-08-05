import os
import random
import shutil

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(BASE, "prepared_A")

random.seed(42)

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

sampled_normals = random.sample(normal_files, 700)

targets = {
    "normal": sampled_normals,
    "scratches": scratch_files,
    "spots": spot_files,
}

for cls, files in targets.items():
    out_dir = os.path.join(OUT, cls)
    os.makedirs(out_dir, exist_ok=True)
    for i, src in enumerate(files):
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(out_dir, f"{cls}_{i:05d}{ext}")
        shutil.copy2(src, dst)

print("Prepared subset created at:", OUT)
print("normal:", len(sampled_normals))
print("scratches:", len(scratch_files))
print("spots:", len(spot_files))