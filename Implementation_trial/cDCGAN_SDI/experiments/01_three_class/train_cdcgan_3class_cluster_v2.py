import os
import random
import json
import gc
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =========================================================
# CONFIG
# =========================================================
LATENT_DIM = 100
N_CLASSES = 3
IMG_SIZE = 128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)

SEED = 42
# BATCH_SIZE = 16
BATCH_SIZE = 32
N_EPOCHS = 82


# Cluster data paths
BASE_DIR = "/shared/rc/defgengan/data/prepared_A"
NORMAL_DIR = os.path.join(BASE_DIR, "normal")
SCRATCH_DIR = os.path.join(BASE_DIR, "scratches")
SPOT_DIR = os.path.join(BASE_DIR, "spots")

# Output structure
OUT_ROOT = "/shared/rc/defgengan/outputs/exp3_3class_fullrun_v2"
GENERATED_DIR = os.path.join(OUT_ROOT, "generated_samples")
CHECKPOINT_DIR = os.path.join(OUT_ROOT, "checkpoints")
FINAL_MODEL_DIR = os.path.join(OUT_ROOT, "final_models")
LOSS_DIR = os.path.join(OUT_ROOT, "losses")

# Balanced target counts
TARGET_COUNTS = {
    0: 700,  # normal
    1: 700,  # scratch
    2: 700   # spot
}

# Save only generator checkpoints during training
CHECKPOINT_EPOCHS = [21, 42, 82]

# Save generated grids every few epochs
SAVE_SAMPLE_EVERY = 5

LABEL_NAMES = {
    0: "normal",
    1: "scratch",
    2: "spot"
}

AUTOTUNE = tf.data.AUTOTUNE


# =========================================================
# REPRODUCIBILITY
# =========================================================
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# DIRECTORY SETUP
# =========================================================
def ensure_dirs() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    os.makedirs(LOSS_DIR, exist_ok=True)


# =========================================================
# FILE LISTING + BALANCING
# =========================================================
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


def build_balanced_file_lists():
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

    return balanced_files, balanced_labels


# =========================================================
# TF.DATA PIPELINE
# =========================================================
def load_and_preprocess_tf(path, label):
    # Read raw bytes from disk
    img = tf.io.read_file(path)

    # Decode image; channels=1 forces grayscale
    img = tf.io.decode_image(img, channels=1, expand_animations=False)

    # Convert to float32 in [0,1]
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Resize to target shape
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    # Scale [0,1] -> [-1,1]
    img = (img * 2.0) - 1.0

    # Cast label to int32
    label = tf.cast(label, tf.int32)

    return img, label


def build_tf_dataset(file_paths, labels, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # Shuffle before map so the reading order changes
    ds = ds.shuffle(buffer_size=len(file_paths), seed=SEED, reshuffle_each_iteration=True)

    # Load and preprocess lazily
    ds = ds.map(load_and_preprocess_tf, num_parallel_calls=AUTOTUNE)

    # Batch with drop_remainder=True so shapes stay fixed
    ds = ds.batch(batch_size, drop_remainder=True)

    # Prefetch for pipeline efficiency
    ds = ds.prefetch(AUTOTUNE)

    return ds


# =========================================================
# MODEL DEFINITIONS
# =========================================================
def build_generator(latent_dim: int = 100, n_classes: int = 3) -> keras.Model:
    # Label branch
    label_input = layers.Input(shape=(1,), name="Generator-Label-Input")
    label_embedding = layers.Embedding(n_classes, 50, name="Generator-Label-Embedding")(label_input)
    label_dense = layers.Dense(4 * 4 * 1, name="Generator-Label-Dense")(label_embedding)
    label_reshape = layers.Reshape((4, 4, 1), name="Generator-Label-Reshape")(label_dense)

    # Latent branch
    latent_input = layers.Input(shape=(latent_dim,), name="Generator-Latent-Input")
    latent_dense = layers.Dense(4 * 4 * 256, name="Generator-Latent-Dense")(latent_input)
    latent_act = layers.LeakyReLU(0.2)(latent_dense)
    latent_reshape = layers.Reshape((4, 4, 256), name="Generator-Latent-Reshape")(latent_act)

    # Merge label + latent feature maps
    merge = layers.Concatenate(name="Generator-Combine")([latent_reshape, label_reshape])

    # Upsampling path
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


def build_discriminator(in_shape=(128, 128, 1), n_classes=3) -> keras.Model:
    # Label branch
    label_input = layers.Input(shape=(1,), name="Discriminator-Label-Input")
    label_embedding = layers.Embedding(n_classes, 50, name="Discriminator-Label-Embedding")(label_input)
    label_dense = layers.Dense(in_shape[0] * in_shape[1], name="Discriminator-Label-Dense")(label_embedding)
    label_reshape = layers.Reshape((in_shape[0], in_shape[1], 1), name="Discriminator-Label-Reshape")(label_dense)

    # Image branch
    image_input = layers.Input(shape=in_shape, name="Discriminator-Image-Input")

    # Merge image + label map
    merge = layers.Concatenate(name="Discriminator-Combine")([image_input, label_reshape])

    # Downsampling conv path
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


# =========================================================
# LOSSES + OPTIMIZERS
# =========================================================
bce = keras.losses.BinaryCrossentropy(from_logits=False)

generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)


def generator_loss(fake_output):
    return bce(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


# =========================================================
# TRAIN STEP
# =========================================================
@tf.function
def train_step(real_images, labels):
    current_batch_size = tf.shape(real_images)[0]

    # Sample fresh noise each step
    noise = tf.random.normal([current_batch_size, LATENT_DIM])

    # labels shape should be (batch,1) for embedding input
    labels = tf.reshape(labels, (-1, 1))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        generated_images = generator([noise, labels], training=True)

        # D evaluates real and fake
        real_output = discriminator([real_images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        # Compute losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Compute gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


# =========================================================
# SAMPLE GENERATION FOR MONITORING
# =========================================================
def save_generated_grid(generator_model, epoch, latent_dim=100, n_per_class=5, save_dir=GENERATED_DIR):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(3, n_per_class, figsize=(12, 6))

    for row, label in enumerate([0, 1, 2]):
        z = tf.random.normal([n_per_class, latent_dim])
        labels = tf.constant([[label]] * n_per_class, dtype=tf.int32)

        gen_imgs = generator_model([z, labels], training=False).numpy()

        for col in range(n_per_class):
            axes[row, col].imshow((gen_imgs[col].squeeze() + 1) / 2.0, cmap="gray")
            axes[row, col].axis("off")

        axes[row, 0].set_title(LABEL_NAMES[label], fontsize=12, pad=8)

    plt.subplots_adjust(left=0.05, wspace=0.05, hspace=0.18)
    filepath = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved sample grid to: {filepath}")


# =========================================================
# LOSS PLOTTING / SAVING
# =========================================================
def save_loss_plot(d_losses, g_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="D loss", marker='o', markersize=2, alpha=0.7)
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


def save_loss_arrays(d_losses, g_losses):
    np.save(os.path.join(LOSS_DIR, "d_losses.npy"), np.array(d_losses))
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


# =========================================================
# MAIN TRAINING LOOP
# =========================================================
def train(dataset):
    d_losses = []
    g_losses = []

    for epoch in range(1, N_EPOCHS + 1):
        epoch_d_losses = []
        epoch_g_losses = []

        for real_images, labels in dataset:
            gen_loss, disc_loss = train_step(real_images, labels)
            epoch_g_losses.append(float(gen_loss.numpy()))
            epoch_d_losses.append(float(disc_loss.numpy()))

        mean_g = float(np.mean(epoch_g_losses))
        mean_d = float(np.mean(epoch_d_losses))

        g_losses.append(mean_g)
        d_losses.append(mean_d)

        print(f"Epoch {epoch}/{N_EPOCHS} | D: {mean_d:.4f} | G: {mean_g:.4f}")

        if epoch % SAVE_SAMPLE_EVERY == 0:
            save_generated_grid(generator, epoch, latent_dim=LATENT_DIM, n_per_class=5, save_dir=GENERATED_DIR)

        if epoch in CHECKPOINT_EPOCHS:
            generator.save(os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch:03d}.keras"))
            print(f"Saved generator checkpoint at epoch {epoch}")

        gc.collect()

    return d_losses, g_losses


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dirs()

    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices("GPU"))

    file_paths, labels = build_balanced_file_lists()
    dataset = build_tf_dataset(file_paths, labels, batch_size=BATCH_SIZE)

    global generator, discriminator
    generator = build_generator(LATENT_DIM, N_CLASSES)
    discriminator = build_discriminator(IMG_SHAPE, N_CLASSES)

    print("\nTrainable weights:")
    print("Generator:", len(generator.trainable_weights))
    print("Discriminator:", len(discriminator.trainable_weights))

    d_losses, g_losses = train(dataset)

    # Save final models
    generator.save(os.path.join(FINAL_MODEL_DIR, "generator_final.keras"))
    discriminator.save(os.path.join(FINAL_MODEL_DIR, "discriminator_final.keras"))
    print("Saved final models.")

    save_loss_plot(d_losses, g_losses)
    save_loss_arrays(d_losses, g_losses)

    print("Training complete.")


if __name__ == "__main__":
    main()