#!/usr/bin/env python3
"""
Controlled normal-vs-scratch cDCGAN experiment.

This script deliberately preserves the earlier binary cDCGAN architecture and
training logic while changing the data domain:

- 128x128 grayscale
- two deliberately imbalanced classes: 100 normal and 700 scratch images
- scratch-heavy stratified batches: 2 normal and 14 scratch images
- latent dimension 100
- same generator/discriminator architecture
- no defect-specific enhancement
- no rotation or flip augmentation
- fixed-z checkpoint grids for controlled progression
- random-z grids for diversity inspection
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


LATENT_DIM = 100
N_CLASSES = 2
IMG_SIZE = 128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)

SEED = 42
BATCH_SIZE = 16
NORMAL_PER_BATCH = 2
SCRATCH_PER_BATCH = 14

DEFAULT_EPOCHS = 82
DEFAULT_NORMAL_COUNT = 100
DEFAULT_SCRATCH_COUNT = 700
SAVE_SAMPLE_EVERY = 5

LABEL_NAMES = {0: "normal", 1: "scratch"}
AUTOTUNE = tf.data.AUTOTUNE

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/shared/rc/defgengan/data/simple_scratch_v1"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/shared/rc/defgengan/outputs/exp8_simple_scratch_100N_700S"),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--normal-count",
        type=int,
        default=DEFAULT_NORMAL_COUNT,
    )
    parser.add_argument(
        "--scratch-count",
        type=int,
        default=DEFAULT_SCRATCH_COUNT,
    )
    parser.add_argument(
        "--normal-per-batch",
        type=int,
        default=NORMAL_PER_BATCH,
    )
    parser.add_argument(
        "--scratch-per-batch",
        type=int,
        default=SCRATCH_PER_BATCH,
    )
    return parser.parse_args()


def ensure_dirs(out_root: Path) -> dict[str, Path]:
    paths = {
        "generated_fixed": out_root / "generated_samples" / "fixed_z",
        "generated_random": out_root / "generated_samples" / "random_z",
        "checkpoints": out_root / "checkpoints",
        "final_models": out_root / "final_models",
        "losses": out_root / "losses",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def list_images(folder: Path) -> list[str]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing class directory: {folder}")
    files = sorted(
        str(path)
        for path in folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not files:
        raise ValueError(f"No images found in: {folder}")
    return files


def select_exact_count(files: list[str], target_count: int, class_name: str) -> list[str]:
    if len(files) < target_count:
        raise ValueError(
            f"{class_name} contains {len(files)} images, but {target_count} are required. "
            "Regenerate the controlled dataset instead of oversampling."
        )
    if len(files) == target_count:
        return files
    rng = random.Random(SEED)
    return sorted(rng.sample(files, target_count))


def load_and_preprocess_tf(path: tf.Tensor, label: tf.Tensor):
    path_str = path.numpy().decode("utf-8")
    label_int = int(label.numpy())

    image = Image.open(path_str).convert("L")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    image = np.asarray(image, dtype=np.float32)
    image = (image / 127.5) - 1.0
    image = np.expand_dims(image, axis=-1)

    return image.astype(np.float32), np.int32(label_int)


def tf_wrapper(path: tf.Tensor, label: tf.Tensor):
    image, lab = tf.py_function(
        func=load_and_preprocess_tf,
        inp=[path, label],
        Tout=[tf.float32, tf.int32],
    )
    image.set_shape((IMG_SIZE, IMG_SIZE, 1))
    lab.set_shape(())
    return image, lab


def combine_batches(normal_batch, scratch_batch):
    normal_images, normal_labels = normal_batch
    scratch_images, scratch_labels = scratch_batch

    images = tf.concat([normal_images, scratch_images], axis=0)
    labels = tf.concat([normal_labels, scratch_labels], axis=0)

    indices = tf.random.shuffle(tf.range(tf.shape(labels)[0]))
    return tf.gather(images, indices), tf.gather(labels, indices)


def build_scratch_heavy_tf_dataset(
    normal_files: list[str],
    scratch_files: list[str],
    batch_size: int,
    normal_per_batch: int,
    scratch_per_batch: int,
):
    if normal_per_batch <= 0 or scratch_per_batch <= 0:
        raise ValueError("Per-class batch counts must both be positive.")

    if normal_per_batch + scratch_per_batch != batch_size:
        raise ValueError(
            "normal_per_batch + scratch_per_batch must equal batch_size."
        )

    normal_ds = tf.data.Dataset.from_tensor_slices(
        (normal_files, [0] * len(normal_files))
    )
    normal_ds = normal_ds.shuffle(
        len(normal_files), seed=SEED, reshuffle_each_iteration=True
    )
    normal_ds = normal_ds.repeat()
    normal_ds = normal_ds.map(tf_wrapper, num_parallel_calls=AUTOTUNE)
    normal_ds = normal_ds.batch(normal_per_batch, drop_remainder=True)

    scratch_ds = tf.data.Dataset.from_tensor_slices(
        (scratch_files, [1] * len(scratch_files))
    )
    scratch_ds = scratch_ds.shuffle(
        len(scratch_files), seed=SEED, reshuffle_each_iteration=True
    )
    scratch_ds = scratch_ds.repeat()
    scratch_ds = scratch_ds.map(tf_wrapper, num_parallel_calls=AUTOTUNE)
    scratch_ds = scratch_ds.batch(scratch_per_batch, drop_remainder=True)

    dataset = tf.data.Dataset.zip((normal_ds, scratch_ds))
    dataset = dataset.map(combine_batches, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(AUTOTUNE)

    normal_steps = len(normal_files) // normal_per_batch
    scratch_steps = len(scratch_files) // scratch_per_batch
    steps_per_epoch = min(normal_steps, scratch_steps)

    if steps_per_epoch <= 0:
        raise ValueError(
            "Not enough images for the requested per-batch class counts."
        )

    return dataset, steps_per_epoch


def build_generator(latent_dim: int = 100, n_classes: int = 2):
    label_input = layers.Input(shape=(1,), name="Generator-Label-Input")
    label_embedding = layers.Embedding(
        n_classes, 50, name="Generator-Label-Embedding"
    )(label_input)
    label_dense = layers.Dense(
        4 * 4 * 1, name="Generator-Label-Dense"
    )(label_embedding)
    label_reshape = layers.Reshape(
        (4, 4, 1), name="Generator-Label-Reshape"
    )(label_dense)

    latent_input = layers.Input(
        shape=(latent_dim,), name="Generator-Latent-Input"
    )
    latent_dense = layers.Dense(
        4 * 4 * 256, name="Generator-Latent-Dense"
    )(latent_input)
    latent_act = layers.LeakyReLU(negative_slope=0.2)(latent_dense)
    latent_reshape = layers.Reshape(
        (4, 4, 256), name="Generator-Latent-Reshape"
    )(latent_act)

    merged = layers.Concatenate(name="Generator-Combine")(
        [latent_reshape, label_reshape]
    )

    x = layers.Conv2DTranspose(256, 4, strides=2, padding="same")(merged)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    output = layers.Conv2D(
        1, kernel_size=7, activation="tanh", padding="same"
    )(x)

    return keras.Model(
        [latent_input, label_input],
        output,
        name="Generator",
    )


def build_discriminator(in_shape=(128, 128, 1), n_classes: int = 2):
    label_input = layers.Input(shape=(1,), name="Discriminator-Label-Input")
    label_embedding = layers.Embedding(
        n_classes, 50, name="Discriminator-Label-Embedding"
    )(label_input)
    label_dense = layers.Dense(
        in_shape[0] * in_shape[1],
        name="Discriminator-Label-Dense",
    )(label_embedding)
    label_reshape = layers.Reshape(
        (in_shape[0], in_shape[1], 1),
        name="Discriminator-Label-Reshape",
    )(label_dense)

    image_input = layers.Input(
        shape=in_shape, name="Discriminator-Image-Input"
    )
    merged = layers.Concatenate()([image_input, label_reshape])

    x = layers.Conv2D(64, 4, strides=2, padding="same")(merged)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(
        [image_input, label_input],
        output,
        name="Discriminator",
    )


bce = keras.losses.BinaryCrossentropy(from_logits=False)


def generator_loss(fake_output):
    return bce(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def make_train_step(generator, discriminator, generator_optimizer, discriminator_optimizer):
    @tf.function
    def train_step(real_images, labels):
        current_batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([current_batch_size, LATENT_DIM])
        labels_2d = tf.reshape(labels, (-1, 1))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(
                [noise, labels_2d],
                training=True,
            )
            real_output = discriminator(
                [real_images, labels_2d],
                training=True,
            )
            fake_output = discriminator(
                [generated_images, labels_2d],
                training=True,
            )

            gen_loss_value = generator_loss(fake_output)
            disc_loss_value = discriminator_loss(real_output, fake_output)

        generator_gradients = gen_tape.gradient(
            gen_loss_value, generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss_value, discriminator.trainable_variables
        )

        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

        return gen_loss_value, disc_loss_value

    return train_step


def save_grid(
    generator_model,
    epoch: int,
    z: tf.Tensor,
    save_path: Path,
    title_suffix: str,
):
    n_per_class = int(z.shape[0])
    fig, axes = plt.subplots(2, n_per_class, figsize=(12, 5))

    for row, label in enumerate([0, 1]):
        labels = tf.constant([[label]] * n_per_class, dtype=tf.int32)
        generated = generator_model([z, labels], training=False).numpy()

        for col in range(n_per_class):
            axes[row, col].imshow(
                (generated[col].squeeze() + 1.0) / 2.0,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, col].axis("off")

        axes[row, 0].set_title(
            LABEL_NAMES[label],
            fontsize=11,
            pad=7,
        )

    fig.suptitle(f"Epoch {epoch}: {title_suffix}", fontsize=12)
    plt.subplots_adjust(left=0.05, wspace=0.05, hspace=0.20, top=0.88)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid: {save_path}")


def checkpoint_epochs(total_epochs: int) -> set[int]:
    candidates = {5, 10, 21, 42, 82, 120, 160, 200, total_epochs}
    return {epoch for epoch in candidates if 1 <= epoch <= total_epochs}


def save_loss_plot(d_losses, g_losses, loss_dir: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="D loss", marker="o", markersize=2, alpha=0.7)
    plt.plot(g_losses, label="G loss", marker="s", markersize=2, alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Exp8 scratch-heavy simple-scratch cDCGAN losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = loss_dir / "loss_plot.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved loss plot: {path}")


def train(
    dataset,
    steps_per_epoch: int,
    epochs: int,
    generator,
    discriminator,
    train_step,
    fixed_z: tf.Tensor,
    paths: dict[str, Path],
):
    discriminator_losses: list[float] = []
    generator_losses: list[float] = []
    checkpoints = checkpoint_epochs(epochs)

    for epoch in range(1, epochs + 1):
        epoch_d_losses = []
        epoch_g_losses = []

        for real_images, labels in dataset.take(steps_per_epoch):
            gen_loss_value, disc_loss_value = train_step(real_images, labels)
            epoch_g_losses.append(float(gen_loss_value.numpy()))
            epoch_d_losses.append(float(disc_loss_value.numpy()))

        mean_g = float(np.mean(epoch_g_losses))
        mean_d = float(np.mean(epoch_d_losses))
        generator_losses.append(mean_g)
        discriminator_losses.append(mean_d)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"D: {mean_d:.4f} | G: {mean_g:.4f}"
        )

        if epoch % SAVE_SAMPLE_EVERY == 0 or epoch in checkpoints:
            fixed_path = paths["generated_fixed"] / f"epoch_{epoch:03d}.png"
            save_grid(
                generator,
                epoch,
                fixed_z,
                fixed_path,
                "fixed latent inputs",
            )

            random_z = tf.random.normal([int(fixed_z.shape[0]), LATENT_DIM])
            random_path = paths["generated_random"] / f"epoch_{epoch:03d}.png"
            save_grid(
                generator,
                epoch,
                random_z,
                random_path,
                "fresh random latent inputs",
            )

        if epoch in checkpoints:
            checkpoint_path = (
                paths["checkpoints"] / f"generator_epoch_{epoch:03d}.keras"
            )
            generator.save(checkpoint_path)
            print(f"Saved generator checkpoint: {checkpoint_path}")

        gc.collect()

    return discriminator_losses, generator_losses


def main() -> None:
    args = parse_args()
    paths = ensure_dirs(args.out_root)

    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices("GPU"))

    normal_files = select_exact_count(
        list_images(args.data_dir / "normal"),
        args.normal_count,
        "normal",
    )
    scratch_files = select_exact_count(
        list_images(args.data_dir / "scratches"),
        args.scratch_count,
        "scratch",
    )

    print("Controlled class counts:")
    print("Normal:", len(normal_files))
    print("Scratch:", len(scratch_files))
    print(
        "Batch composition:",
        f"{args.normal_per_batch} normal + "
        f"{args.scratch_per_batch} scratch",
    )

    dataset, steps_per_epoch = build_scratch_heavy_tf_dataset(
        normal_files,
        scratch_files,
        batch_size=args.batch_size,
        normal_per_batch=args.normal_per_batch,
        scratch_per_batch=args.scratch_per_batch,
    )

    generator = build_generator(LATENT_DIM, N_CLASSES)
    discriminator = build_discriminator(IMG_SHAPE, N_CLASSES)

    generator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002,
        beta_1=0.5,
    )
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002,
        beta_1=0.5,
    )
    train_step = make_train_step(
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
    )

    fixed_z = tf.random.stateless_normal(
        [5, LATENT_DIM],
        seed=[SEED, 2026],
    )
    np.save(
        args.out_root / "generated_samples" / "fixed_z.npy",
        fixed_z.numpy(),
    )

    with (args.out_root / "generator_summary.txt").open(
        "w",
        encoding="utf-8",
    ) as handle:
        generator.summary(print_fn=lambda line: handle.write(line + "\n"))

    with (args.out_root / "discriminator_summary.txt").open(
        "w",
        encoding="utf-8",
    ) as handle:
        discriminator.summary(print_fn=lambda line: handle.write(line + "\n"))

    print("Generator parameters:", generator.count_params())
    print("Discriminator parameters:", discriminator.count_params())
    print("Steps per epoch:", steps_per_epoch)

    d_losses, g_losses = train(
        dataset=dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        generator=generator,
        discriminator=discriminator,
        train_step=train_step,
        fixed_z=fixed_z,
        paths=paths,
    )

    generator.save(paths["final_models"] / "generator_final.keras")
    discriminator.save(paths["final_models"] / "discriminator_final.keras")

    np.save(paths["losses"] / "d_losses.npy", np.asarray(d_losses))
    np.save(paths["losses"] / "g_losses.npy", np.asarray(g_losses))
    save_loss_plot(d_losses, g_losses, paths["losses"])

    summary = {
        "experiment": "exp8_simple_scratch_100N_700S",
        "research_question": (
            "Does increasing scratch exposure improve scratch-class consistency "
            "and geometric diversity in the same cDCGAN?"
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "normal_count": len(normal_files),
        "scratch_count": len(scratch_files),
        "normal_per_batch": args.normal_per_batch,
        "scratch_per_batch": args.scratch_per_batch,
        "steps_per_epoch": steps_per_epoch,
        "normal_presentations_per_epoch": (
            steps_per_epoch * args.normal_per_batch
        ),
        "scratch_presentations_per_epoch": (
            steps_per_epoch * args.scratch_per_batch
        ),
        "latent_dim": LATENT_DIM,
        "labels": LABEL_NAMES,
        "stratified_batching": True,
        "class_balance": "100 normal / 700 scratch",
        "batch_composition": "2 normal / 14 scratch",
        "scratch_enhancement": False,
        "geometric_augmentation": False,
        "fixed_z_checkpoint_evaluation": True,
        "random_z_diversity_grids": True,
        "data_dir": str(args.data_dir),
    }
    with (paths["losses"] / "run_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    print("Training complete.")
    print("Output root:", args.out_root)


if __name__ == "__main__":
    main()