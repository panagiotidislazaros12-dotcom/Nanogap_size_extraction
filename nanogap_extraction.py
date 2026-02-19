"""nanogap_method3.py

Method 3 (Kano-like) nanogap width estimation:
    mean width w ≈ A / (P/2) = 2A/P

Where:
- A is the segmented nanogap area (pixels^2)
- P is the nanogap perimeter (pixels)

Dependencies:
    pip install opencv-python scikit-image numpy
"""

import argparse
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.measure import label, regionprops, perimeter as sk_perimeter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nanogap width estimation using Method 3 (2A/P).")
    p.add_argument("--image", required=True, help="Path to input image (jpg/png).")
    p.add_argument(
        "--nm_per_px",
        type=float,
        required=True,
        help="Calibration factor (nm per pixel). Example: if 221 px = 500 nm, nm_per_px = 500/221.",
    )
    p.add_argument(
        "--roi",
        nargs=4,
        type=int,
        default=None,
        metavar=("x", "y", "w", "h"),
        help="Optional crop ROI as x y w h to exclude scale bar/labels.",
    )
    p.add_argument(
        "--gap_is_darker",
        type=int,
        default=1,
        help="1 if gap is darker than background (default), 0 otherwise.",
    )
    p.add_argument(
        "--min_obj_size",
        type=int,
        default=300,
        help="Remove connected components smaller than this (pixels).",
    )
    p.add_argument(
        "--close_radius",
        type=int,
        default=2,
        help="Morphological closing radius in pixels (fills small breaks).",
    )
    return p.parse_args()


def load_image_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def crop_roi(img: np.ndarray, roi) -> np.ndarray:
    if roi is None:
        return img
    x, y, w, h = roi
    return img[y : y + h, x : x + w]


def segment_gap(
    img: np.ndarray,
    gap_is_darker: bool = True,
    min_obj_size: int = 300,
    close_radius: int = 2,
) -> np.ndarray:
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    t = threshold_otsu(img_blur)
    mask = (img_blur < t) if gap_is_darker else (img_blur > t)

    mask = binary_closing(mask, disk(close_radius))
    mask = remove_small_objects(mask, min_size=min_obj_size)

    lab = label(mask)
    props = regionprops(lab)
    if not props:
        raise RuntimeError(
            "No gap region found. Try adjusting ROI, gap_is_darker, min_obj_size, or close_radius."
        )
    largest = max(props, key=lambda r: r.area)
    gap = lab == largest.label
    return gap


def compute_method3_width(gap_mask: np.ndarray, nm_per_px: float) -> dict:
    A = float(gap_mask.sum())  # area in pixels^2
    P = float(sk_perimeter(gap_mask, neighborhood=8))  # perimeter in pixels

    if A <= 0 or P <= 0:
        raise RuntimeError("Invalid area/perimeter. The segmented mask may be empty or corrupted.")

    width_px = (2.0 * A) / P
    width_nm = width_px * nm_per_px

    return {
        "area_px2": A,
        "perimeter_px": P,
        "width_px": width_px,
        "width_nm": width_nm,
    }


def main() -> None:
    args = parse_args()

    img = load_image_gray(args.image)
    img = crop_roi(img, args.roi)

    gap = segment_gap(
        img,
        gap_is_darker=bool(args.gap_is_darker),
        min_obj_size=args.min_obj_size,
        close_radius=args.close_radius,
    )

    res = compute_method3_width(gap, args.nm_per_px)

    print(f"Image: {args.image}")
    if args.roi:
        print(f"ROI: x={args.roi[0]} y={args.roi[1]} w={args.roi[2]} h={args.roi[3]}")
    print(f"Calibration: {args.nm_per_px:.6f} nm/px")
    print(f"Gap area A: {res['area_px2']:.0f} px^2")
    print(f"Gap perimeter P: {res['perimeter_px']:.2f} px")
    print(f"Method 3 mean width: {res['width_px']:.2f} px  ->  {res['width_nm']:.2f} nm")


if __name__ == "__main__":
    main()
