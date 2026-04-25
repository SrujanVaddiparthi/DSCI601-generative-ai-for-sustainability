# AFTER THE OOM related corrections
import os
import random
import json
import gc
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =========================
# CONFIG
# =========================
LATENT_DIM = 100
N_CLASSES = 3
IMG_SIZE = 128
IMG_SHAPE = (128, 128, 1)

SEED = 42
BATCH_SIZE = 32
N_EPOCHS = 82

# Relative path is intentional; update on cluster if needed
BASE_DIR = "/shared/rc/defgengan/data/prepared_A"

NORMAL_DIR = os.path.join(BASE_DIR, "normal")
SCRATCH_DIR = os.path.join(BASE_DIR, "scratches")
SPOT_DIR = os.path.join(BASE_DIR, "spots")

# Output structure
OUT_ROOT = "/shared/rc/defgengan/outputs/exp3_3class_fullrun"
GENERATED_DIR = os.path.join(OUT_ROOT, "generated_samples")
CHECKPOINT_DIR = os.path.join(OUT_ROOT, "checkpoints")
FINAL_MODEL_DIR = os.path.join(OUT_ROOT, "final_models")
LOSS_DIR = os.path.join(OUT_ROOT, "losses")

# Balanced baseline targets
TARGET_COUNTS = {
    0: 700,  # normal
    1: 700,  # scratch
    2: 700   # spot
}

# Save fewer checkpoints to reduce overhead
CHECKPOINT_EPOCHS = [21, 42, 82]

# Save sample grid every epoch; if memory still grows, change to 2
SAVE_SAMPLE_EVERY = 5

LABEL_NAMES = {
    0: "normal",
    1: "scratch",
    2: "spot"
}


# =========================
# REPRODUCIBILITY
# =========================
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================
# UTILITIES
# =========================
def ensure_dirs() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    os.makedirs(LOSS_DIR, exist_ok=True)


def list_images(folder: str) -> list[str]:
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            files.append(os.path.join(folder, f))
    return sorted(files)


def oversample_to_target(file_list: list[str], target_count: int) -> list[str]:
    result = []
    while len(result) < target_count:
        result.extend(file_list)
    return result[:target_count]


def load_and_preprocess(path: str, label: int) -> tuple[np.ndarray, int]:
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype("float32")

    # Scale [0,255] -> [-1,1]
    img = (img / 127.5) - 1.0

    # Add channel dimension: (128,128) -> (128,128,1)
    img = np.expand_dims(img, axis=-1)
    return img, label


# =========================
# DATA PREP
# =========================
def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    normal_files = list_images(NORMAL_DIR)
    scratch_files = list_images(SCRATCH_DIR)
    spot_files = list_images(SPOT_DIR)

    print("Original class counts:")
    print("Normal:", len(normal_files))
    print("Scratches:", len(scratch_files))
    print("Spots:", len(spot_files))

    all_files = normal_files + scratch_files + spot_files
    all_labels = (
        [0] * len(normal_files) +
        [1] * len(scratch_files) +
        [2] * len(spot_files)
    )

    class_to_files = defaultdict(list)
    for path, label in zip(all_files, all_labels):
        class_to_files[label].append(path)

    balanced_files = []
    balanced_labels = []

    for label in [0, 1, 2]:
        files = class_to_files[label]
        target_count = TARGET_COUNTS[label]

        if len(files) >= target_count:
            selected = random.sample(files, target_count)
        else:
            selected = oversample_to_target(files, target_count)

        balanced_files.extend(selected)
        balanced_labels.extend([label] * len(selected))

    print("\nBalanced class counts:")
    for label in [0, 1, 2]:
        print(label, balanced_labels.count(label))

    X, y = [], []
    for path, label in zip(balanced_files, balanced_labels):
        img, lab = load_and_preprocess(path, label)
        X.append(img)
        y.append(lab)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print("\nDataset tensors:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X min/max:", X.min(), X.max())
    print("Unique labels:", np.unique(y, return_counts=True))

    return X, y


# =========================
# MODELS
# =========================
def build_generator(latent_dim: int = 100, n_classes: int = 3) -> keras.Model:
    label_input = layers.Input(shape=(1,), name="Generator-Label-Input")
    label_embedding = layers.Embedding(n_classes, 50, name="Generator-Label-Embedding")(label_input)
    label_dense = layers.Dense(4 * 4 * 1, name="Generator-Label-Dense")(label_embedding)
    label_reshape = layers.Reshape((4, 4, 1), name="Generator-Label-Reshape")(label_dense)

    latent_input = layers.Input(shape=(latent_dim,), name="Generator-Latent-Input")
    latent_dense = layers.Dense(4 * 4 * 256, name="Generator-Latent-Dense")(latent_input)
    latent_act = layers.LeakyReLU(0.2)(latent_dense)
    latent_reshape = layers.Reshape((4, 4, 256), name="Generator-Latent-Reshape")(latent_act)

    merge = layers.Concatenate(name="Generator-Combine")([latent_reshape, label_reshape])

    x = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same", name="G_Deconv_8")(merge)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", name="G_Deconv_16")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", name="G_Deconv_32")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", name="G_Deconv_64")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", name="G_Deconv_128")(x)
    x = layers.LeakyReLU(0.2)(x)

    output = layers.Conv2D(1, kernel_size=7, activation="tanh", padding="same", name="Generator-Output")(x)

    return keras.Model([latent_input, label_input], output, name="Generator")


def build_discriminator(in_shape: tuple[int, int, int] = (128, 128, 1), n_classes: int = 3) -> keras.Model:
    label_input = layers.Input(shape=(1,), name="Discriminator-Label-Input")
    label_embedding = layers.Embedding(n_classes, 50, name="Discriminator-Label-Embedding")(label_input)
    label_dense = layers.Dense(in_shape[0] * in_shape[1], name="Discriminator-Label-Dense")(label_embedding)
    label_reshape = layers.Reshape((in_shape[0], in_shape[1], 1), name="Discriminator-Label-Reshape")(label_dense)

    image_input = layers.Input(shape=in_shape, name="Discriminator-Image-Input")
    merge = layers.Concatenate(name="Discriminator-Combine")([image_input, label_reshape])

    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", name="D_Conv_64")(merge)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", name="D_Conv_32")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", name="D_Conv_16")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same", name="D_Conv_8")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same", name="D_Conv_4")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1, activation="sigmoid", name="Discriminator-Output")(x)

    return keras.Model([image_input, label_input], output, name="Discriminator")


def build_models() -> tuple[keras.Model, keras.Model, keras.Model]:
    generator = build_generator(LATENT_DIM, N_CLASSES)
    discriminator = build_discriminator(IMG_SHAPE, N_CLASSES)

    # 1) Compile discriminator once as standalone trainable model
    opt_d = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=opt_d,
        metrics=["accuracy"]
    )

    # 2) Freeze discriminator only while building GAN model
    discriminator.trainable = False

    noise_input = layers.Input(shape=(LATENT_DIM,), name="GAN-Noise-Input")
    label_input = layers.Input(shape=(1,), name="GAN-Label-Input")

    generated_img = generator([noise_input, label_input])
    gan_output = discriminator([generated_img, label_input])

    gan_model = keras.Model([noise_input, label_input], gan_output, name="cDCGAN")

    opt_g = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan_model.compile(
        loss="binary_crossentropy",
        optimizer=opt_g
    )

    # 3) Re-enable discriminator flag for clarity in standalone usage
    discriminator.trainable = True

    print("\nTrainable weights:")
    print("Generator:", len(generator.trainable_weights))
    print("Discriminator:", len(discriminator.trainable_weights))
    print("GAN model:", len(gan_model.trainable_weights))

    return generator, discriminator, gan_model


# =========================
# TRAINING HELPERS
# =========================
def real_samples(X: np.ndarray, y: np.ndarray, n: int):
    idx = np.random.randint(0, X.shape[0], n)
    images = X[idx]
    labels = y[idx].reshape(-1, 1)
    targets = np.ones((n, 1), dtype=np.float32)
    return [images, labels], targets


def latent_vector(latent_dim: int, n: int):
    z = np.random.randn(n, latent_dim).astype(np.float32)
    labels = np.random.randint(0, N_CLASSES, n).reshape(-1, 1).astype(np.int32)
    return z, labels


def fake_samples(generator: keras.Model, latent_dim: int, n: int):
    z, labels = latent_vector(latent_dim, n)
    images = generator([z, labels], training=False).numpy()
    targets = np.zeros((n, 1), dtype=np.float32)
    return [images, labels], targets


# =========================
# VISUALIZATION
# =========================
def save_generated_grid(generator: keras.Model, epoch: int, latent_dim: int = 100, n_per_class: int = 5,
                        save_dir: str = GENERATED_DIR) -> None:
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(3, n_per_class, figsize=(12, 6))

    for row, label in enumerate([0, 1, 2]):
        z = np.random.randn(n_per_class, latent_dim).astype(np.float32)
        labels = np.full((n_per_class, 1), label, dtype=np.int32)

        gen_imgs = generator([z, labels], training=False).numpy()

        for col in range(n_per_class):
            axes[row, col].imshow((gen_imgs[col].squeeze() + 1) / 2.0, cmap="gray")
            axes[row, col].axis("off")

        axes[row, 0].set_title(LABEL_NAMES[label], fontsize=12, pad=8)

    plt.subplots_adjust(left=0.05, wspace=0.05, hspace=0.18)
    filepath = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close(fig)

    del gen_imgs, z, labels
    print(f"Saved sample grid to: {filepath}")


def save_loss_plot(d_losses_real, d_losses_fake, g_losses) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses_real, label="D real loss", marker='o', markersize=2, alpha=0.7)
    plt.plot(d_losses_fake, label="D fake loss", marker='x', markersize=2, alpha=0.7)
    plt.plot(g_losses, label="G loss", marker='s', markersize=2, alpha=0.7)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("cDCGAN Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_plot_path = os.path.join(LOSS_DIR, "loss_plot.png")
    plt.savefig(loss_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved loss plot to: {loss_plot_path}")


def save_loss_arrays(d_losses_real, d_losses_fake, g_losses) -> None:
    np.save(os.path.join(LOSS_DIR, "d_losses_real.npy"), np.array(d_losses_real))
    np.save(os.path.join(LOSS_DIR, "d_losses_fake.npy"), np.array(d_losses_fake))
    np.save(os.path.join(LOSS_DIR, "g_losses.npy"), np.array(g_losses))

    summary = {
        "epochs": len(g_losses),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "img_size": IMG_SIZE,
        "target_counts": TARGET_COUNTS,
        "checkpoint_epochs": CHECKPOINT_EPOCHS,
        "save_sample_every": SAVE_SAMPLE_EVERY
    }
    with open(os.path.join(LOSS_DIR, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved loss arrays and run summary.")


# =========================
# TRAIN LOOP
# =========================
def train_cdcgan(generator: keras.Model, discriminator: keras.Model, gan_model: keras.Model,
                 X: np.ndarray, y: np.ndarray, latent_dim: int = 100,
                 n_epochs: int = 100, batch_size: int = 16,
                 save_dir: str = GENERATED_DIR):

    bat_per_epo = X.shape[0] // batch_size
    half_batch = batch_size // 2

    d_losses_real, d_losses_fake, g_losses = [], [], []

    for epoch in range(1, n_epochs + 1):
        d_loss_real_epoch = []
        d_loss_fake_epoch = []
        g_loss_epoch = []

        for _ in range(bat_per_epo):
            # Train D on real
            [X_real, labels_real], y_real = real_samples(X, y, half_batch)
            d_loss_real, _ = discriminator.train_on_batch([X_real, labels_real], y_real)

            # Train D on fake
            [X_fake, labels_fake], y_fake = fake_samples(generator, latent_dim, half_batch)
            d_loss_fake, _ = discriminator.train_on_batch([X_fake, labels_fake], y_fake)

            # Train G through frozen-D GAN model
            z_input, labels_input = latent_vector(latent_dim, batch_size)
            y_gan = np.ones((batch_size, 1), dtype=np.float32)
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)

            d_loss_real_epoch.append(float(d_loss_real))
            d_loss_fake_epoch.append(float(d_loss_fake))
            g_loss_epoch.append(float(g_loss))

            del X_real, labels_real, y_real
            del X_fake, labels_fake, y_fake
            del z_input, labels_input, y_gan

        d_losses_real.append(np.mean(d_loss_real_epoch))
        d_losses_fake.append(np.mean(d_loss_fake_epoch))
        g_losses.append(np.mean(g_loss_epoch))

        print(
            f"Epoch {epoch}/{n_epochs} | "
            f"D_real: {d_losses_real[-1]:.4f} | "
            f"D_fake: {d_losses_fake[-1]:.4f} | "
            f"G: {g_losses[-1]:.4f}"
        )

        if epoch % SAVE_SAMPLE_EVERY == 0:
            save_generated_grid(generator, epoch, latent_dim=latent_dim, n_per_class=5, save_dir=save_dir)

        if epoch in CHECKPOINT_EPOCHS:
            generator.save(os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch:03d}.keras"))
            print(f"Saved generator checkpoint at epoch {epoch}")

        gc.collect()

    return d_losses_real, d_losses_fake, g_losses


# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices("GPU"))

    X, y = build_dataset()
    generator, discriminator, gan_model = build_models()

    d_losses_real, d_losses_fake, g_losses = train_cdcgan(
        generator,
        discriminator,
        gan_model,
        X, y,
        latent_dim=LATENT_DIM,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir=GENERATED_DIR
    )

    # Save final models
    generator.save(os.path.join(FINAL_MODEL_DIR, "generator_final.keras"))
    discriminator.save(os.path.join(FINAL_MODEL_DIR, "discriminator_final.keras"))
    gan_model.save(os.path.join(FINAL_MODEL_DIR, "gan_model_final.keras"))
    print("Saved final models.")

    # Save training artifacts
    save_loss_plot(d_losses_real, d_losses_fake, g_losses)
    save_loss_arrays(d_losses_real, d_losses_fake, g_losses)

    print("Training complete.")


if __name__ == "__main__":
    main()

