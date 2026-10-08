# FenotIT documentation

Technical documentation of the methods implemented in FenotIT. It is written to support the
methods section of the software paper. Every figure is produced by
[`make_figures.py`](make_figures.py) with FenotIT's own pipeline functions on the test images,
so the pictures show what the software actually computes.

```
python docs/make_figures.py      # regenerates docs/img/*.png
```

## Processing chain

Each photo goes through the same ordered chain. Every block is optional except segmentation
and labelling, and the parameters used are written to `metadata.csv` on export.

| Step | Section |
|---|---|
| 1. Image corrections (lens distortion → perspective → color → scale) and working resolution | [01 Corrections](methods/01_corrections.md) |
| 2. Segmentation (color-space threshold or Otsu), analysis areas and cleaning | [02 Segmentation](methods/02_segmentation.md) |
| 3. Objects: labelling, separation of touching objects, filters and numbering | [03 Objects](methods/03_objects.md) |
| 4. Size and shape descriptors (morphometry) | [04 Morphometry](methods/04_morphometry.md) |
| 5. Outline shape: elliptic Fourier descriptors, alignment and mean shape | [05 Shape](methods/05_shape.md) |
| 6. Color: mean color and dominant colors (k-means), CIELAB | [06 Color](methods/06_color.md) |
| 7. Spatial pattern: neighbor distances and Clark–Evans index | [07 Distances](methods/07_distances.md) |

All references are in [references.md](methods/references.md).

## Conventions

- **Units.** Measurements are computed in pixels. If a scale is set, they are converted with
  $s$ = length unit per pixel (mm/px by default): lengths × $s$, areas × $s^2$. The unit is
  part of every column name (`area_mm2`, `length_mm`); without a scale the columns end in `px`.
- **Image coordinates.** $x$ grows to the right and $y$ downwards; the origin is the top-left pixel.
- **Connectivity.** Objects are 8-connected regions of the binary mask.
- **Reproducibility.** Random steps (k-means) use a fixed seed, so the same photo and the same
  parameters always give the same numbers. `tests/reference.py` checks this on every change.
