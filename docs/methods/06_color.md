# 6. Color

Each object reports its mean color and its dominant colors, found with k-means. The number of
dominant colors is fixed or chosen per object.

![Color](../img/color.png)

*Figure 6.1. Color of the seed of Figure 4.1. (a) Edge ring removed before measuring (blue).
(b) Pixels painted with the color of their group, k = 3. (c) The three colors, their share and
their CIELAB values.*

## 6.1 Pixels used

A ring is removed from the edge of each object by erosion with a 3 × 3 square, where object and
background mix. Its width is 1.5 % of the square root of the object area, rounded and kept between
1 and 3 px. Objects with fewer than 20 remaining pixels use all their pixels.

## 6.2 Mean color

Mean of the remaining pixels in RGB (`mean_R`, `mean_G`, `mean_B`) and CIELAB (`mean_L`, `mean_a`,
`mean_b`).

## 6.3 Dominant colors

Pixels are grouped in RGB with k-means (Lloyd, 1982), k-means++ starts (Arthur & Vassilvitskii,
2007), three repetitions and a fixed seed, on at most 3000 pixels; every pixel is then assigned to
its closest group color. Groups are listed by share (`pct`). By default $k$ is 3 for objects and 5
for whole photos.

- **Each object** (`object`): each object gets its own colors.
- **All objects together** (`pooled`): one set of colors for the photo, and each object is
  described by the share of each common color.

The colors of the whole photo are exported in `image_colors`.

## 6.4 Number of colors per object

With "Automatic", pixels are grouped in CIELAB for $k = 1, 2, 3, \dots$ up to the maximum, and
two quantities are computed with CIEDE2000 (ΔE₀₀; Sharma et al., 2005) on 1500 pixels: the mean
ΔE₀₀ between each pixel and its group color (fit) and the ΔE₀₀ between the two closest group colors
(separation). The number chosen is the smallest $k$ with a fit of 5 ΔE₀₀ or less, stopping earlier
if the next $k$ gives two colors closer than 10 ΔE₀₀.

![Number of colors](../img/color_k.png)

*Figure 6.2. Number of colors for the seed of Figure 6.1. Top: the seed with 1 to 6 colors (the
chosen number in blue). (a) Fit, with the limit of 5 (dashed). (b) Separation, with the limit of
10 (dashed).*

For this seed the fit is 11.0 ΔE₀₀ with one color, 6.4 with two and 5.5 with three; with four it
reaches 4.8 but two colors are 6.9 ΔE₀₀ apart, so three are chosen: seed coat (57 %), an
intermediate tone (26 %) and the light mottles (17 %). Among the 15 seeds of the photo, 2 colors
were chosen for one seed, 3 for five, 4 for eight and 5 for one.

## 6.5 Reporting and charts

Color is reported in CIELAB by default (CIE, 2004), converted with OpenCV assuming sRGB and D65;
charts can also use RGB and HSV. Charts show the a\*–b\* plane with equal axes or a 3D view, with
one point per color, object or photo; object and photo points sit at the area-weighted mean color
and are drawn as pies of their dominant colors.
