# 4. Morphometry

Size and shape descriptors are computed for each object from its outer outline (Suzuki & Abe,
1985). Objects from a split region (`touching = 1`) are measured when "Measure touching objects"
is on.

![Morphometry](../img/morphometry.png)

*Figure 4.1. Geometry of one mottled bean seed (bruchid trial, section 7.5). (a) Outline (white),
smallest enclosing rectangle (blue) and centroid (cross). (b) Convex hull (dashed blue) over the
outline. (c) Fitted ellipse (blue) and the shortest (solid) and longest (dashed) radius.*

## 4.1 What is measured

- **Area:** number of pixels of the object.
- **Perimeter:** length of the outline simplified with the Douglas–Peucker algorithm at 1 px
  (Douglas & Peucker, 1973), which removes the steps of the pixel grid.
- **Polygon area:** area enclosed by the simplified outline, used in the ratios with the perimeter.
- **Convex hull:** smallest convex polygon containing the outline (Sklansky, 1982).
- **Length and width:** sides of the smallest enclosing rectangle at any angle (Freeman & Shapira,
  1975; rotating calipers, Toussaint, 1983).
- **Ellipse axes:** axes of the least-squares ellipse fitted to the outline (Fitzgibbon & Fisher,
  1995).
- **Centroid and radii:** mean position of the pixels and distances from it to the outline.

## 4.2 Exported descriptors

`<u>` is the unit of the scale (`mm`, `um`, `cm`, `m` or `px`). Ratios have no unit.

| Column | Definition | Values |
|---|---|---|
| `area_<u>2` | Area | > 0 |
| `length_<u>`, `width_<u>` | Length and width of the smallest enclosing rectangle | > 0 |
| `perimeter_<u>` | Perimeter | > 0 |
| `aspect_ratio` | length / width | 1 for an object as wide as long, larger for longer objects |
| `elongation` | 1 − width / length | 0 for an object as wide as long, close to 1 for a thin one |
| `circularity` | 4π × polygon area / perimeter², at most 1 (Cox, 1927) | 1 for a disk, lower for long or irregular outlines |
| `solidity` | polygon area / convex hull area | 1 for a convex object, lower with notches and concavities |
| `convexity` | convex hull perimeter / perimeter, at most 1 | 1 for a smooth convex outline, lower for rough or lobed ones |
| `rectangularity` | polygon area / (length × width) | 1 for a rectangle, about 0.785 for an ellipse |
| `eccentricity` | √(1 − (minor axis / major axis)²) of the fitted ellipse | 0 for a circle, close to 1 for a thin ellipse |
| `major_axis_<u>`, `minor_axis_<u>` | Ellipse axes | > 0 |
| `radius_min_<u>`, `radius_mean_<u>`, `radius_max_<u>` | Shortest, mean and longest radius | > 0 |
| `radius_ratio` | shortest radius / longest radius | 1 for a disk |
| `centroid_x_px`, `centroid_y_px` | Centroid, in pixels from the top-left corner | |

Definitions follow Russ & Neal (2016), as in ImageJ (Schneider et al., 2012) and CellProfiler
(Stirling et al., 2021).

## 4.3 Values per photo

| Column | Meaning |
|---|---|
| `n_objects` | Objects counted (status ok or highlighted) |
| `n_touching` | Objects that come from a split region |
| `n_measured` | Objects with size and shape measurements |
| `n_excluded_manual`, `n_highlighted` | Objects excluded or highlighted in the inspector |

Legends and charts give the mean and the sample standard deviation (n − 1).
