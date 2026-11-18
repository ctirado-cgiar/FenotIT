# 5. Outline shape: elliptic Fourier descriptors

The descriptors of section 4 summarize an outline in a few numbers. To compare outlines as a
whole, FenotIT describes each one with elliptic Fourier descriptors (EFD; Kuhl & Giardina,
1982), a common method for seed and leaf outlines in plant biology (Iwata & Ukai, 2002).

![EFD](../img/efd_harmonics.png)

*Figure 5.1. Outline of the seed of Figure 4.1 (gray) and its reconstruction (blue) with 1, 2,
4, 8 and 20 harmonics. The first harmonic is an ellipse; each further harmonic adds finer
detail. FenotIT uses 20 harmonics by default.*

## 5.1 The outline as a sum of ellipses

Walking once around the outline, the x and y coordinates of the point reached are two periodic
curves. Each of them can be written as a sum of cosines and sines of increasing frequency
(a Fourier series). Harmonic $n$ goes around $n$ times while the outline is walked once, and
it is described by four numbers $a_n$, $b_n$, $c_n$, $d_n$: $a_n$ and $b_n$ for x, and $c_n$
and $d_n$ for y. Each harmonic traces an ellipse, and adding the harmonics in order rebuilds
the outline (Figure 5.1).

Let the outline have $K$ points and let $t$ be the distance walked along it from the first
point, so that $t$ goes from 0 to the perimeter $T$. Point $p$ is reached at distance $t_p$,
and the step from point $p - 1$ to point $p$ has horizontal size $\Delta x_p$, vertical size
$\Delta y_p$ and length $\Delta t_p$. The coefficients of harmonic $n$ are

$$a_n = \frac{T}{2 n^2 \pi^2} \sum_{p=1}^{K} \frac{\Delta x_p}{\Delta t_p}\left[\cos\frac{2\pi n t_p}{T} - \cos\frac{2\pi n t_{p-1}}{T}\right],\qquad
b_n = \frac{T}{2 n^2 \pi^2} \sum_{p=1}^{K} \frac{\Delta x_p}{\Delta t_p}\left[\sin\frac{2\pi n t_p}{T} - \sin\frac{2\pi n t_{p-1}}{T}\right]$$

and $c_n$, $d_n$ are given by the same formulas with $\Delta y_p$ in place of $\Delta x_p$.
With $N$ harmonics, the rebuilt outline is

$$x(t) = \sum_{n=1}^{N} \left( a_n \cos\frac{2\pi n t}{T} + b_n \sin\frac{2\pi n t}{T} \right),\qquad
y(t) = \sum_{n=1}^{N} \left( c_n \cos\frac{2\pi n t}{T} + d_n \sin\frac{2\pi n t}{T} \right)$$

around the centroid of the outline. An outline needs at least $2N + 2$ points to be described
with $N$ harmonics; objects with fewer points have empty coefficients.

## 5.2 Removing position, size, rotation and starting point

The coefficients change when the same outline is walked from another starting point, rotated
or scaled. They are normalized with the ellipse of the first harmonic (Kuhl & Giardina, 1982):

1. **Starting point.** The start is moved to the end of the major axis of the first ellipse;
   every harmonic is shifted in phase accordingly.
2. **Rotation.** The outline is rotated so that the major axis of the first ellipse lies along
   x.
3. **Size.** All coefficients are divided by the semi-major axis of the first ellipse.

After this, the first harmonic is always $a_1 = 1$, $b_1 = c_1 = 0$, and $|d_1|$ is the ratio of
the minor to the major axis of the first ellipse. The $4N$ normalized coefficients of each
object are exported in the table `object_shape` (`efd_a1` to `efd_dN`).

## 5.3 Alignment and mean shape

A seed turned by 180°, or seen from its other face (a mirror image), gives different
coefficients after normalization. Within each group compared (a photo, a batch or a genotype),
every outline is replaced by whichever of four versions (as it is, turned by 180°, and its two
mirror images) is closest to the current mean of the group, measured as the Euclidean distance
between coefficient lists. The mean starts as the first object; the choice and the mean are
updated three times. The mean shape is the mean of the aligned coefficients, drawn by rebuilding
its outline.

**Difference from the mean.** For each object the inspector shows the Euclidean distance between
its aligned coefficients and those of the mean shape, next to the median distance in the photo.
Broken or deformed seeds have distances well above the median.
