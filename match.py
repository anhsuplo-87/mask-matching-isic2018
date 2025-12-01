import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as ssim
import argparse
from enum import StrEnum

mask_folder = "./ISIC2018_Task1_Validation_GroundTruth"
img_folder = "./ISIC2018_Task1-2_Validation_Input"
input_mask_path = "./mask.png"
resize_shape = (256, 256)


class METRIC(StrEnum):
    CHECK_SUM = "check_sum"
    PIXEL_WISE_DIFF = "pixel_wise_diff"
    DICE = "dice"
    IOU = "iou"
    SSIM = "ssim"
    HAUSDORFF = "hausdorff"


def read_mask(mask_path, mode):
    img = Image.open(mask_path).convert(mode)
    return img


def get_arr(img):
    return np.array(img, dtype=np.float64)


def transform(img, resize_shape):
    # resize
    if resize_shape != (-1, -1):
        img = img.resize(resize_shape)
    return img


def match_metric(input_mask, candidate_mask, metric_choice):
    def check_sum(input_mask, candidate_mask):
        return np.abs(np.sum(input_mask) - np.sum(candidate_mask))

    # Pixel-wise(L1) difference
    def pixel_wise_diff(input_mask, candidate_mask):
        return np.sum(np.abs(input_mask.astype(np.float32) -
                             candidate_mask.astype(np.float32)))

    # Dice coefficient (very common for masks)
    def dice_coefficient(input_mask, candidate_mask):
        im = input_mask.astype(bool)
        cm = candidate_mask.astype(bool)

        intersection = np.logical_and(im, cm).sum()
        return 1 - (2 * intersection) / (im.sum() + cm.sum() + 1e-8)

    # Intersection over Union (IoU / Jaccard)
    def iou(input_mask, candidate_mask):
        im = input_mask.astype(bool)
        cm = candidate_mask.astype(bool)

        intersection = np.logical_and(im, cm).sum()
        union = np.logical_or(im, cm).sum()
        return 1 - intersection / (union + 1e-8)

    # Structural Similarity Index (SSIM)
    def ssim_metric(input_mask, candidate_mask):
        score, _ = ssim(input_mask, candidate_mask, full=True,
                        data_range=np.max(input_mask))
        return 1 - score

    # Hausdorff distance (boundary-aware)
    def hausdorff_metric(input_mask, candidate_mask):
        p = np.column_stack(np.where(input_mask > 0))
        q = np.column_stack(np.where(candidate_mask > 0))

        return max(
            directed_hausdorff(p, q)[0],
            directed_hausdorff(q, p)[0]
        )

    if metric_choice == METRIC.CHECK_SUM:
        return check_sum(input_mask, candidate_mask)

    if metric_choice == METRIC.PIXEL_WISE_DIFF:
        return pixel_wise_diff(input_mask, candidate_mask)

    if metric_choice == METRIC.DICE:
        return dice_coefficient(input_mask, candidate_mask)

    if metric_choice == METRIC.IOU:
        return iou(input_mask, candidate_mask)

    if metric_choice == METRIC.SSIM:
        return ssim_metric(input_mask, candidate_mask)

    if metric_choice == METRIC.HAUSDORFF:
        return hausdorff_metric(input_mask, candidate_mask)


def plot_mask_matching(mask_path, candidate_mask_path, image_path, match_error, resize_shape):
    """
    Draws a figure with three subplots:
      1. mask_path image (grayscale)
      2. candidate_mask_path image (grayscale)
      3. image_path (RGB)
    Displays the basename of each path as subplot title.
    Shows match_error clearly on the figure.
    """
    mask = np.array(transform(Image.open(
        mask_path).convert("L"), resize_shape))
    cand = np.array(
        transform(Image.open(candidate_mask_path).convert("L"), resize_shape))
    img = np.array(
        transform(Image.open(image_path).convert("RGB"), resize_shape))

    # mask = np.array(Image.open(mask_path).convert("L"))
    # cand = np.array(Image.open(candidate_mask_path).convert("L"))
    # img = np.array(Image.open(image_path).convert("RGB"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title(os.path.basename(mask_path))
    axes[0].axis('off')

    axes[1].imshow(cand, cmap='gray')
    axes[1].set_title(os.path.basename(candidate_mask_path))
    axes[1].axis('off')

    axes[2].imshow(img)
    axes[2].set_title(os.path.basename(image_path))
    axes[2].axis('off')

    fig.suptitle(
        f"Matching Mask • Error Value: {match_error}", fontsize=16, fontweight='bold')

    fig.text(
        0.5, -0.02,
        f"Absolute pixel-sum difference: {match_error}",
        ha='center', fontsize=12
    )

    plt.tight_layout()
    plt.show()


def plot_mask_matching_slider(mask_path, candidates, resize_shape):
    mask = np.array(transform(Image.open(
        mask_path).convert("L"), resize_shape))
    # mask = np.array(Image.open(mask_path).convert("L"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(bottom=0.2)

    ax_mask, ax_cand, ax_img = axes

    def update_display(idx):
        cand_mask_path, img_path, err = candidates[idx]

        cand_mask = np.array(
            transform(Image.open(cand_mask_path).convert("L"), resize_shape))
        img = np.array(
            transform(Image.open(img_path).convert("RGB"), resize_shape))

        # cand_mask = np.array(Image.open(cand_mask_path).convert("L"))
        # img = np.array(Image.open(img_path).convert("RGB"))

        ax_mask.clear()
        ax_cand.clear()
        ax_img.clear()

        ax_mask.imshow(mask, cmap='gray')
        ax_mask.set_title(os.path.basename(mask_path))
        ax_mask.axis('off')

        ax_cand.imshow(cand_mask, cmap='gray')
        ax_cand.set_title(os.path.basename(cand_mask_path))
        ax_cand.axis('off')

        ax_img.imshow(img)
        ax_img.set_title(os.path.basename(img_path))
        ax_img.axis('off')

        fig.suptitle(
            f"Mask Matching • Candidate {idx} • Error = {err}",
            fontsize=16, fontweight='bold'
        )

        fig.canvas.draw_idle()

    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.05])
    slider = Slider(
        ax=ax_slider,
        label="Candidate Index",
        valmin=0,
        valmax=len(candidates) - 1,
        valinit=0,
        valstep=1
    )

    update_display(0)

    def on_slider_change(value):
        update_display(int(value))

    slider.on_changed(on_slider_change)

    plt.show()


if __name__ == "__main__":

    # ------------------- Argument Parsing -------------------
    parser = argparse.ArgumentParser(
        description="Browse mask-matching results with an interactive slider."
    )

    parser.add_argument(
        "--input_mask_path",
        type=str,
        default=input_mask_path,
        help="Path to the reference mask you want to match against."
    )

    parser.add_argument(
        "--mask_folder",
        type=str,
        default=mask_folder,
        help="Folder containing all candidate mask PNG files."
    )

    parser.add_argument(
        "--img_folder",
        type=str,
        default=img_folder,
        help="Folder with RGB images corresponding to candidate masks."
    )

    parser.add_argument(
        "--resize_shape",
        nargs=2,
        type=int,
        default=resize_shape,
        help="Resize the mask image in matching pharse. Default = (256, 256)."
    )

    parser.add_argument(
        "--metric_choice",
        type=str,
        default=METRIC.CHECK_SUM,
        help="Matching metric available: [check_sum, pixel_wise_diff, dice, iou, ssim, hausdorff]. Default = check_sum"
    )

    args = parser.parse_args()

    # Overwrite defaults with user-provided values
    input_mask_path = args.input_mask_path
    mask_folder = args.mask_folder
    img_folder = args.img_folder
    resize_shape = tuple(args.resize_shape)
    metric_choice = args.metric_choice
    # ---------------------------------------------------------

    print("Input:")
    print(f"Input Mask path = {input_mask_path}")
    print(f"(Candidate) Mask folder = {mask_folder}")
    print(f"Image folder = {img_folder}")
    print(f"Resize shape = {resize_shape}")
    print(f"Metric choice = {metric_choice}")
    print()

    mask_arr = read_mask(input_mask_path, "L")
    mask_arr = transform(mask_arr, resize_shape)
    mask_arr = get_arr(mask_arr)

    # print(mask_arr)
    # print(mask_arr.shape)

    # exit()

    candidate_mask_paths = [
        os.path.join(mask_folder, path)
        for path in os.listdir(mask_folder)
        if path.endswith(".png")
    ]

    match_errors = []

    with tqdm(total=len(candidate_mask_paths), ncols=128, unit="mask") as pbar:
        for candidate_mask_path in candidate_mask_paths:
            mask_name = os.path.basename(candidate_mask_path)
            img_name = mask_name.replace(
                "_segmentation", "").replace(".png", ".jpg")

            pbar.set_description(f"Checking #{mask_name}")

            candidate_arr = read_mask(candidate_mask_path, "L")
            candidate_arr = transform(candidate_arr, resize_shape)
            candidate_arr = get_arr(candidate_arr)

            match_errors.append(
                (
                    os.path.join(mask_folder, mask_name),
                    os.path.join(img_folder, img_name),
                    match_metric(mask_arr, candidate_arr, metric_choice),
                )
            )

            pbar.update()

    print()

    match_errors = sorted(match_errors, key=lambda e: e[-1])

    print("TOP 3 matching image:")
    for id in range(3):
        print("+", match_errors[id])
    print("...")

    print("Displaying matching results:")
    plot_mask_matching_slider(input_mask_path, match_errors, resize_shape)
