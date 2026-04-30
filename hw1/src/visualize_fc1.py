import os
import argparse
import numpy as np


def _minmax01(x, eps=1e-8):
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    return (x - mn) / (mx - mn + eps)


def _to_rgb_img(w_flat, h=64, w=64, c=3):
    img = w_flat.reshape(h, w, c)
    # per-filter minmax normalization for visualization
    return _minmax01(img)


def save_fc1_weight_grid(
    ckpt_path: str,
    out_path: str,
    num_show: int = 36,
    grid_cols: int = 6,
    img_hw: int = 64,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt = np.load(ckpt_path, allow_pickle=True)
    if "fc1.W" not in ckpt:
        raise KeyError(f"'{ckpt_path}' missing key 'fc1.W'. Available keys: {ckpt.files}")

    W = ckpt["fc1.W"].astype(np.float32)  # (in_dim=12288, h1)
    in_dim, h1 = W.shape
    expected = img_hw * img_hw * 3
    if in_dim != expected:
        raise ValueError(f"fc1.W has in_dim={in_dim}, expected {expected} for {img_hw}x{img_hw}x3.")

    k = min(int(num_show), int(h1))
    cols = int(grid_cols)
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < k:
            w_flat = W[:, i]
            img = _to_rgb_img(w_flat, h=img_hw, w=img_hw, c=3)
            ax.imshow(img)
            ax.set_title(f"unit {i}", fontsize=8)

    fig.suptitle(f"fc1 weights reshaped to {img_hw}x{img_hw}x3 (show {k}/{h1})", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_fc1_color_bias_stats(ckpt_path: str, out_path: str, img_hw: int = 64):
    """
    Simple quantitative summary:
    For each hidden unit, compute mean weight in R/G/B channels (over spatial dims),
    then report top units with strongest positive/negative channel bias.
    """
    ckpt = np.load(ckpt_path, allow_pickle=True)
    W = ckpt["fc1.W"].astype(np.float32)  # (12288, h1)
    in_dim, h1 = W.shape
    expected = img_hw * img_hw * 3
    if in_dim != expected:
        raise ValueError(f"fc1.W has in_dim={in_dim}, expected {expected} for {img_hw}x{img_hw}x3.")

    Wimg = W.T.reshape(h1, img_hw, img_hw, 3)  # (h1,H,W,3)
    ch_mean = Wimg.mean(axis=(1, 2))  # (h1,3)
    ch_std = Wimg.std(axis=(1, 2))   # (h1,3)

    def topk(arr, k=10, largest=True):
        idx = np.argsort(arr)
        if largest:
            idx = idx[::-1]
        return idx[:k]

    lines = []
    lines.append("fc1 hidden-unit channel bias stats (mean±std over spatial positions)\n")
    for ch, name in enumerate(["R", "G", "B"]):
        vals = ch_mean[:, ch]
        lines.append(f"## Channel {name}: top positive means")
        for i in topk(vals, k=10, largest=True):
            lines.append(f"unit {i:4d}: mean={vals[i]: .6f}, std={ch_std[i, ch]: .6f}")
        lines.append("")
        lines.append(f"## Channel {name}: top negative means")
        for i in topk(vals, k=10, largest=False):
            lines.append(f"unit {i:4d}: mean={vals[i]: .6f}, std={ch_std[i, ch]: .6f}")
        lines.append("")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best_single_strong.npz")
    p.add_argument("--out", default="checkpoints/fc1_weight_grid.png")
    p.add_argument("--num_show", type=int, default=36)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--img_hw", type=int, default=64)
    p.add_argument("--stats_out", default="checkpoints/fc1_color_bias.txt")
    args = p.parse_args()

    save_fc1_weight_grid(
        ckpt_path=args.ckpt,
        out_path=args.out,
        num_show=args.num_show,
        grid_cols=args.cols,
        img_hw=args.img_hw,
    )
    save_fc1_color_bias_stats(
        ckpt_path=args.ckpt,
        out_path=args.stats_out,
        img_hw=args.img_hw,
    )
    print(f"saved grid to {args.out}")
    print(f"saved stats to {args.stats_out}")


if __name__ == "__main__":
    main()

