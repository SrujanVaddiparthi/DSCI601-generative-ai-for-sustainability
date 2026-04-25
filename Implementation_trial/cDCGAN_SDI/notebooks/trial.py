import matplotlib.pyplot as plt
import numpy as np
import os

d_losses = np.load("/Users/wangtiles/DSCI601/DSCI601-generative-ai-for-sustainability/Implementation_trial/cDCGAN_SDI/scripts/batch16_v2_normalVscratch_epoch200/exp4_normal_vs_scratch/losses/d_losses.npy")
g_losses = np.load("/Users/wangtiles/DSCI601/DSCI601-generative-ai-for-sustainability/Implementation_trial/cDCGAN_SDI/scripts/batch16_v2_normalVscratch_epoch200/exp4_normal_vs_scratch/losses/g_losses.npy")

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

    loss_plot_path = os.path.join("/Users/wangtiles/DSCI601/DSCI601-generative-ai-for-sustainability/Implementation_trial/cDCGAN_SDI/scripts/batch16_v2_normalVscratch_epoch200/exp4_normal_vs_scratch/losses", "loss_plot.png")
    plt.savefig(loss_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved loss plot to: {loss_plot_path}")

def main():
    save_loss_plot(d_losses, g_losses)


if __name__ == "__main__":
    main()