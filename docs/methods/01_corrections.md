# 1. Image corrections

Corrections are optional and independent. When several are active they are applied in a
fixed order, always on the full-resolution photo and in memory (the original file is never
modified):

**lens distortion → perspective → color → scale → working resolution**

A photo that cannot be corrected with an active correction (missing ArUco marker, color card
not found, different aspect ratio than the calibration) is skipped and reported as
`status = skipped`, so corrected and uncorrected photos are never mixed in one data set.

## 1.1 Lens distortion

**Model.** The camera is described by the pinhole model with the Brown–Conrady distortion
terms used by OpenCV (Brown, 1966; Bradski, 2000): an intrinsic matrix

$$K = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}$$

and the coefficients $(k_1, k_2, p_1, p_2, k_3)$. For a normalized image point $(x, y)$ with
$r^2 = x^2 + y^2$:

$$x_d = x\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 x y + p_2 (r^2 + 2x^2)$$
$$y_d = y\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2y^2) + 2 p_2 x y$$

**Calibration.** $K$ and the coefficients are estimated with Zhang's method (Zhang, 2000) from
10–40 photos of a planar chessboard taken with the same camera, lens and zoom. Inner corners
are detected on a copy of at most 1600 px and refined to sub-pixel accuracy on the full photo
(`cornerSubPix`). The per-photo reprojection error is reported; photos whose error exceeds
max(2.5 × median, 0.5 px) are flagged. At most 40 photos are used, spread evenly over the set.

**Correction.** The photo is remapped with the optimal new camera matrix for $\alpha = 1$
(all source pixels kept) and cropped to the valid region of interest, as in the original
calibration scripts of the Bean Physiology team. A calibration made at a resolution
$W_c \times H_c$ is applied to a photo $W \times H$ with the same aspect ratio by scaling $K$:
$K' = \mathrm{diag}(W/W_c,\ H/H_c,\ 1)\,K$.

## 1.2 Perspective (ArUco markers)

Four ArUco markers (dictionary `DICT_4X4_50`, Garrido-Jurado et al., 2014) with IDs 0–3 are
placed at the corners of the working surface (0 top-left, 1 top-right, 2 bottom-right,
3 bottom-left). Their centers $\mathbf{p}_0 \dots \mathbf{p}_3$ define a quadrilateral that is
mapped to a rectangle by a plane homography $H$ (Hartley & Zisserman, 2004), computed from the
four correspondences and applied with bilinear interpolation.

The target rectangle keeps the real proportions instead of stretching the photo:

- **Without measurements**: width $w = \tfrac12(\lVert\mathbf p_1-\mathbf p_0\rVert + \lVert\mathbf p_2-\mathbf p_3\rVert)$ and height $h = \tfrac12(\lVert\mathbf p_3-\mathbf p_0\rVert + \lVert\mathbf p_2-\mathbf p_1\rVert)$ in pixels.
- **With the real distances** $W_{mm}$ (ID0→ID1) and $H_{mm}$ (ID0→ID3):

$$\rho = \tfrac12\left(\frac{w}{W_{mm}} + \frac{h}{H_{mm}}\right)\ \text{px/mm},\qquad
\text{target} = (W_{mm}\,\rho) \times (H_{mm}\,\rho),\qquad s = 1/\rho\ \text{mm/px}$$

so every photo carries its own scale, robust to small changes in camera height.

## 1.3 Color (color card)

**Detection.** The card is located on a copy of at most 1600 px: smooth, nearly square
regions bounded by Canny edges in the three CIELAB channels are kept as candidate patches
(aspect ≤ 1.35, fill ≥ 0.85); the largest cluster of neighbors (distance < 1.7 × patch pitch)
is the card. The grid orientation is the dominant direction of nearest-neighbor vectors
modulo 90°, $\theta = \tfrac14 \arg \sum_i e^{4 i \phi_i}$, which gives the row/column index
of every patch at any rotation. A RANSAC homography from grid indices to image positions
(tolerance 0.25 × pitch) recovers the patches that were not detected.

**Measurement.** The color of each patch is the mean of its central 40 % (avoids the borders).
The card may appear in any of four 90° rotations; the rotation whose affine fit to the
reference has the smallest mean error is used.

**Correction model.** Colors are corrected with an affine transform in RGB scaled to [0, 1],
as in PlantCV (Berry et al., 2018):

$$\begin{pmatrix} r' & g' & b' \end{pmatrix} = \begin{pmatrix} r & g & b & 1 \end{pmatrix} A,\qquad A \in \mathbb{R}^{4 \times 3}$$

$A$ is the least-squares solution $A = S^{+} T$ ($S$: measured patches with a column of ones,
$T$: reference values, $^{+}$: Moore–Penrose pseudo-inverse). Patches affected by glare or
shadow are removed iteratively (at most 4) while the largest residual exceeds
max(3 × median residual, 0.08). The reference is either the published values of a 24-patch
ColorChecker (McCamy et al., 1976), the values given by the card maker, or the colors measured
on a reference photo (all photos are then matched to that photo).

**White balance only.** With a white or gray card, a diagonal gain per channel makes the
marked area neutral (or equal to the reference photo): $g_c = \bar{m} / m_c$.

## 1.4 Scale

The scale $s$ (length unit per pixel) has four sources, in this priority for each photo:

1. **The photo's own scale**, set with two points: $s = d_{real} / \lVert \mathbf{q}_1 - \mathbf{q}_2 \rVert$.
2. **ArUco** with real distances (section 1.2).
3. **The scale for all photos** (two points or a typed value).
4. **None**: results stay in pixels.

## 1.5 Working resolution

After the corrections, photos larger than $M$ megapixels (default 50 MP) are reduced by
$k = \sqrt{M \cdot 10^6 / (W H)}$ with area interpolation, and the scale is adjusted
($s \leftarrow s / k$), so measurements in real units do not change. This keeps very large
phone photos (e.g. 200 MP) within normal memory. The factor is exported as `work_scale`.
In a 69 MP → 20 MP test, object areas in mm² differ by less than 2 %.
