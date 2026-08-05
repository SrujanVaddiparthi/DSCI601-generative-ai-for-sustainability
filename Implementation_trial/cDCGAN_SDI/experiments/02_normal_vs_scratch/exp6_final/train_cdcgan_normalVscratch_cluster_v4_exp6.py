import os
import random
import json
import gc

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =========================================================
# CONFIG
# =========================================================
LATENT_DIM = 100
N_CLASSES = 2
IMG_SIZE = 128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)

SEED = 42
BATCH_SIZE = 16
HALF_BATCH = BATCH_SIZE // 2
N_EPOCHS = 200

BASE_DIR = "/shared/rc/defgengan/data/prepared_A"
NORMAL_DIR = os.path.join(BASE_DIR, "normal")
SCRATCH_DIR = os.path.join(BASE_DIR, "scratches")

OUT_ROOT = "/shared/rc/defgengan/outputs/exp6_normal_vs_scratch_final"
GENERATED_DIR = os.path.join(OUT_ROOT, "generated_samples")
CHECKPOINT_DIR = os.path.join(OUT_ROOT, "checkpoints")
FINAL_MODEL_DIR = os.path.join(OUT_ROOT, "final_models")
LOSS_DIR = os.path.join(OUT_ROOT, "losses")

TARGET_COUNTS = {
    0: 700,  # normal
    1: 700   # scratch
}

CHECKPOINT_EPOCHS = [21, 42, 82, 120, 160, 200]
SAVE_SAMPLE_EVERY = 5

LABEL_NAMES = {
    0: "normal",
    1: "scratch"
}

AUTOTUNE = tf.data.AUTOTUNE

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# DIRECTORY SETUP
# =========================================================
def ensure_dirs():
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    os.makedirs(LOSS_DIR, exist_ok=True)


# =========================================================
# FILE LISTING + BALANCING
# =========================================================
def list_images(folder):
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            files.append(os.path.join(folder, f))
    return sorted(files)


def oversample_to_target(file_list, target_count):
    result = []
    while len(result) < target_count:
        result.extend(file_list)
    return result[:target_count]


def build_balanced_class_lists():
    normal_files = list_images(NORMAL_DIR)
    scratch_files = list_images(SCRATCH_DIR)

    print("Original class counts:")
    print("Normal:", len(normal_files))
    print("Scratch:", len(scratch_files))

    if len(normal_files) >= TARGET_COUNTS[0]:
        normal_balanced = random.sample(normal_files, TARGET_COUNTS[0])
    else:
        normal_balanced = oversample_to_target(normal_files, TARGET_COUNTS[0])

    if len(scratch_files) >= TARGET_COUNTS[1]:
        scratch_balanced = random.sample(scratch_files, TARGET_COUNTS[1])
    else:
        scratch_balanced = oversample_to_target(scratch_files, TARGET_COUNTS[1])

    print("\nBalanced class counts:")
    print("Normal:", len(normal_balanced))
    print("Scratch:", len(scratch_balanced))

    return normal_balanced, scratch_balanced


# =========================================================
# AUGMENTATION / ENHANCEMENT
# =========================================================
def mild_rotate(img, max_degrees=8):
    angle = np.random.uniform(-max_degrees, max_degrees)
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)


def augment_normal_pil(img_pil):
    """
    Normal gets only mild rotation.
    """
    img = img_pil
    img = mild_rotate(img, max_degrees=6)
    return img


def enhance_and_augment_scratch_pil(img_pil):
    """
    Scratch gets:
    - mild random contrast
    - mild random sharpness
    - optional sharpen
    - mild random rotation
    - optional horizontal flip
    """
    img = img_pil

    # Random but milder than before
    contrast_factor = np.random.uniform(1.15, 1.7)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    sharpness_factor = np.random.uniform(1.0, 1.8)
    img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)

    if np.random.rand() < 0.4:
        img = img.filter(ImageFilter.SHARPEN)

    # Mild rotation to encourage orientation diversity
    img = mild_rotate(img, max_degrees=12)

    # Occasional horizontal flip
    if np.random.rand() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    return img


# =========================================================
# TF.DATA PREPROCESSING
# =========================================================
def load_and_preprocess_tf(path, label):
    path_str = path.numpy().decode("utf-8")
    label_int = int(label.numpy())

    img = Image.open(path_str).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    if label_int == 0:
        img = augment_normal_pil(img)
    elif label_int == 1:
        img = enhance_and_augment_scratch_pil(img)

    img = np.array(img, dtype=np.float32)

    # Scale [0,255] -> [-1,1]
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, axis=-1)

    return img.astype(np.float32), np.int32(label_int)


def tf_wrapper(path, label):
    img, lab = tf.py_function(
        func=load_and_preprocess_tf,
        inp=[path, label],
        Tout=[tf.float32, tf.int32]
    )
    img.set_shape((IMG_SIZE, IMG_SIZE, 1))
    lab.set_shape(())
    return img, lab


# =========================================================
# STRATIFIED TF.DATA PIPELINE
# =========================================================
def combine_batches(normal_batch, scratch_batch):
    normal_imgs, normal_labels = normal_batch
    scratch_imgs, scratch_labels = scratch_batch

    images = tf.concat([normal_imgs, scratch_imgs], axis=0)
    labels = tf.concat([normal_labels, scratch_labels], axis=0)

    idx = tf.random.shuffle(tf.range(tf.shape(labels)[0]))
    images = tf.gather(images, idx)
    labels = tf.gather(labels, idx)

    return images, labels


def build_stratified_tf_dataset(normal_files, scratch_files, batch_size=BATCH_SIZE):
    assert batch_size % 2 == 0, "Batch size must be even for 50/50 stratified batching."
    half_batch = batch_size // 2

    normal_labels = [0] * len(normal_files)
    scratch_labels = [1] * len(scratch_files)

    normal_ds = tf.data.Dataset.from_tensor_slices((normal_files, normal_labels))
    normal_ds = normal_ds.shuffle(len(normal_files), seed=SEED, reshuffle_each_iteration=True)
    normal_ds = normal_ds.repeat()
    normal_ds = normal_ds.map(tf_wrapper, num_parallel_calls=AUTOTUNE)
    normal_ds = normal_ds.batch(half_batch, drop_remainder=True)

    scratch_ds = tf.data.Dataset.from_tensor_slices((scratch_files, scratch_labels))
    scratch_ds = scratch_ds.shuffle(len(scratch_files), seed=SEED, reshuffle_each_iteration=True)
    scratch_ds = scratch_ds.repeat()
    scratch_ds = scratch_ds.map(tf_wrapper, num_parallel_calls=AUTOTUNE)
    scratch_ds = scratch_ds.batch(half_batch, drop_remainder=True)

    ds = tf.data.Dataset.zip((normal_ds, scratch_ds))
    ds = ds.map(combine_batches, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)

    steps_per_epoch = min(len(normal_files) // half_batch, len(scratch_files) // half_batch)
    return ds, steps_per_epoch


# =========================================================
# MODELS
# =========================================================
def build_generator(latent_dim=100, n_classes=2):
    label_input = layers.Input(shape=(1,), name="Generator-Label-Input")
    label_embedding = layers.Embedding(n_classes, 50, name="Generator-Label-Embedding")(label_input)
    label_dense = layers.Dense(4 * 4 * 1, name="Generator-Label-Dense")(label_embedding)
    label_reshape = layers.Reshape((4, 4, 1), name="Generator-Label-Reshape")(label_dense)

    latent_input = layers.Input(shape=(latent_dim,), name="Generator-Latent-Input")
    latent_dense = layers.Dense(4 * 4 * 256, name="Generator-Latent-Dense")(latent_input)
    latent_act = layers.LeakyReLU(0.2)(latent_dense)
    latent_reshape = layers.Reshape((4, 4, 256), name="Generator-Latent-Reshape")(latent_act)

    merge = layers.Concatenate(name="Generator-Combine")([latent_reshape, label_reshape])

    x = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")(merge)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)

    output = layers.Conv2D(1, kernel_size=7, activation="tanh", padding="same")(x)

    return keras.Model([latent_input, label_input], output, name="Generator")


def build_discriminator(in_shape=(128, 128, 1), n_classes=2):
    label_input = layers.Input(shape=(1,), name="Discriminator-Label-Input")
    label_embedding = layers.Embedding(n_classes, 50, name="Discriminator-Label-Embedding")(label_input)
    label_dense = layers.Dense(in_shape[0] * in_shape[1], name="Discriminator-Label-Dense")(label_embedding)
    label_reshape = layers.Reshape((in_shape[0], in_shape[1], 1), name="Discriminator-Label-Reshape")(label_dense)

    image_input = layers.Input(shape=in_shape, name="Discriminator-Image-Input")
    merge = layers.Concatenate()([image_input, label_reshape])

    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(merge)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1, activation="sigmoid")(x)

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
    noise = tf.random.normal([current_batch_size, LATENT_DIM])
    labels = tf.reshape(labels, (-1, 1))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, labels], training=True)

        real_output = discriminator([real_images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


# =========================================================
# VISUALIZATION
# =========================================================
def save_generated_grid(generator_model, epoch, latent_dim=100, n_per_class=5, save_dir=GENERATED_DIR):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, n_per_class, figsize=(12, 5))

    for row, label in enumerate([0, 1]):
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
# TRAIN LOOP
# =========================================================
def train(dataset, steps_per_epoch):
    d_losses = []
    g_losses = []

    for epoch in range(1, N_EPOCHS + 1):
        epoch_d_losses = []
        epoch_g_losses = []

        for real_images, labels in dataset.take(steps_per_epoch):
            gen_loss, disc_loss = train_step(real_images, labels)
            epoch_g_losses.append(float(gen_loss.numpy()))
            epoch_d_losses.append(float(disc_loss.numpy()))

        mean_g = float(np.mean(epoch_g_losses))
        mean_d = float(np.mean(epoch_d_losses))

        g_losses.append(mean_g)
        d_losses.append(mean_d)

        print(f"Epoch {epoch}/{N_EPOCHS} | D: {mean_d:.4f} | G: {mean_g:.4f}")

        if epoch % SAVE_SAMPLE_EVERY == 0:
            try:
                save_generated_grid(generator, epoch, latent_dim=LATENT_DIM, n_per_class=5, save_dir=GENERATED_DIR)
            except Exception as e:
                print(f"WARNING: failed to save generated grid at epoch {epoch}: {e}")

        if epoch in CHECKPOINT_EPOCHS:
            try:
                generator.save(os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch:03d}.keras"))
                print(f"Saved generator checkpoint at epoch {epoch}")
            except Exception as e:
                print(f"WARNING: failed to save checkpoint at epoch {epoch}: {e}")

        gc.collect()

    return d_losses, g_losses


# =========================================================
# LOSS PLOTTING / SAVING
# =========================================================
def save_loss_plot(d_losses, g_losses):
    import matplotlib.pyplot as plt
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


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dirs()

    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices("GPU"))

    normal_files, scratch_files = build_balanced_class_lists()
    dataset, steps_per_epoch = build_stratified_tf_dataset(normal_files, scratch_files, batch_size=BATCH_SIZE)

    global generator, discriminator
    generator = build_generator(LATENT_DIM, N_CLASSES)
    discriminator = build_discriminator(IMG_SHAPE, N_CLASSES)

    print("\nTrainable weights:")
    print("Generator:", len(generator.trainable_weights))
    print("Discriminator:", len(discriminator.trainable_weights))
    print("Steps per epoch:", steps_per_epoch)

    d_losses, g_losses = train(dataset, steps_per_epoch)

    generator.save(os.path.join(FINAL_MODEL_DIR, "generator_final.keras"))
    discriminator.save(os.path.join(FINAL_MODEL_DIR, "discriminator_final.keras"))
    print("Saved final models.")

    np.save(os.path.join(LOSS_DIR, "d_losses.npy"), np.array(d_losses))
    np.save(os.path.join(LOSS_DIR, "g_losses.npy"), np.array(g_losses))

    with open(os.path.join(LOSS_DIR, "run_summary.json"), "w") as f:
        json.dump({
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "half_batch": HALF_BATCH,
            "latent_dim": LATENT_DIM,
            "labels": LABEL_NAMES,
            "target_counts": TARGET_COUNTS,
            "stratified_batching": True,
            "randomized_scratch_enhancement": True,
            "mild_geometric_augmentation": True,
            "normal_mild_rotation_only": True
        }, f, indent=2)

    save_loss_plot(d_losses=d_losses, g_losses=g_losses)
    print("Training complete.")


if __name__ == "__main__":
    main()