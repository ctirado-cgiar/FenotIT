# FenotIT methods

Description of the methods implemented in FenotIT. The figures are made with photos of the
Alliance bean program by [`make_figures.py`](../make_figures.py), which calls FenotIT's own functions:

```
python docs/make_figures.py      # regenerates docs/img/*.png
```

It needs the example photos in `tests/data/` (folders `Cgiar_*`) and, for the detection figure,
the id of an installed detection model in `FENOTIT_DOCS_DETECTOR`.

## Processing chain

Each photo goes through the same chain; the parameters used are written to `metadata.csv`.

| Step | Section |
|---|---|
| 1. Image corrections (lens distortion, perspective, color, scale) and working resolution | [01 Corrections](01_corrections.md) |
| 2. Segmentation: threshold, Otsu, classification by examples, MobileSAM regions; analysis areas; refining the mask; area classification | [02 Segmentation](02_segmentation.md) |
| 3. Objects: labelling, separation of touching objects, filters and numbering | [03 Objects](03_objects.md) |
| 4. Size and shape descriptors | [04 Morphometry](04_morphometry.md) |
| 5. Outline shape: elliptic Fourier descriptors, alignment and mean shape | [05 Shape](05_shape.md) |
| 6. Color: mean color, dominant colors and their number | [06 Color](06_color.md) |
| 7. Detection models: boxes as objects, detections per object; bruchid damage in bean seeds | [07 Detection](07_detection.md) |

References are listed in [references.md](references.md).

## Conventions

- **Units.** Measurements are computed in pixels. With a scale (length per pixel, mm/px by
  default), lengths are multiplied by the scale and areas by the scale squared. The unit is part
  of each column name (`area_mm2`, `length_mm`); without a scale the columns end in `px`.
- **Image coordinates.** x grows to the right and y downwards, from the top-left pixel.
- **Connectivity.** Objects are 8-connected regions of the mask.
- **Reproducibility.** Random steps (k-means, random forests) use a fixed seed;
  `tests/reference.py` checks that the numbers stay the same.
