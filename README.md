# Nanogap width estimation — (2A/P)

This repository provides a small Python script to estimate the **mean nanogap width** from an SEM/AFM image using a *Kano-like* geometric metric:

\[
w \approx \frac{A}{P/2} = \frac{2A}{P}
\]

where:

- **A** is the segmented gap **area** (in pixels²)
- **P** is the segmented gap **perimeter** (in pixels)

The script segments the nanogap by thresholding and then keeps the **largest connected component** as the gap region.

## Files

- `nanogap_method3.py` — the main script (Method 3 only)

## Requirements

Install dependencies:

```bash
pip install opencv-python scikit-image numpy
```

## Usage

Basic:

```bash
python nanogap_method3.py --image Picture1.jpg --nm_per_px 2.262443
```

With an ROI crop to exclude scale bar / labels (recommended):

```bash
python nanogap_method3.py --image Picture1.jpg --nm_per_px 2.262443 --roi 0 0 927 560
```

### Parameters

- `--image` : path to input image (`.jpg`, `.png`, etc.)
- `--nm_per_px` : calibration factor (nm per pixel). Example: if **221 px = 500 nm**, then `nm_per_px = 500/221 ≈ 2.262443`.
- `--roi x y w h` : crop region in pixels (top-left x,y and width,height)
- `--gap_is_darker` : `1` if the gap is darker than background (default), `0` otherwise
- `--min_obj_size` : removes tiny connected components (noise)
- `--close_radius` : fills small breaks in the segmented gap

## Notes

- For best accuracy, crop tightly around the nanogap and avoid including the scale bar text/legends.
- If segmentation fails, adjust `--roi`, `--min_obj_size`, `--close_radius`, or set `--gap_is_darker 0`.

## Citation

If you use this script in a publication, please cite the corresponding methods paper(s) and/or include a brief acknowledgement of this implementation in your methods section.
