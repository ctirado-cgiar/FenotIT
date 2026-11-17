# 3. Objects

The mask from section 2 is turned into numbered objects: connected regions are labelled,
touching objects are separated, small noise and cut objects are removed, and the remaining
objects are numbered in reading order.

## 3.1 Labelling

Each 8-connected region of the mask (pixels that touch by a side or a corner) is one object;
its outline is traced with the algorithm of Suzuki & Abe (1985). This is the method used when
the option "Objects touch" is off. It keeps irregular shapes, such as flowers or lobed leaves,
in one piece.

## 3.2 Separation of touching objects

When "Objects touch" is on, one region of the mask may contain several objects. FenotIT splits
it with the distance transform and a watershed started from markers (Beucher & Meyer, 1993;
Soille, 2003; scikit-image, van der Walt et al., 2014), and then applies rules that join parts
cut by mistake and cut objects left joined.

![Separation](../img/separation.png)

*Figure 3.1. Separation of four touching bean seeds (crop of a seed tray photo, drought trial).
(a) The seeds. (b) In the mask they form one region. (c) Distance from each pixel to the
background, with one peak per seed (crosses). (d) Watershed started from the peaks. (e) Final
objects after the joining rules.*

**Distance to the background.** For each pixel of the mask, the distance transform gives the
distance to the nearest background pixel (Euclidean, computed with a 5 × 5 mask; Borgefors,
1986). It is largest near the middle of each object, and its value there is half the local
thickness of the object (Figure 3.1c).

**Markers.** The markers are the peaks of the distance that are at least $d$ pixels apart. The
spacing $d$ is about the typical object radius: a first search with peaks 5 px apart gives the
distance value at each peak, and $d$ is 0.9 times the median of those values (at least 3 px). A
peak is kept only if its value is at least 0.2 times the highest value in its own region, so a
large object elsewhere in the photo does not hide the peaks of thin objects.

**Watershed.** Starting from the markers, each pixel of the region is assigned to the marker it
reaches first when the distance map is flooded from its peaks downwards (Figure 3.1d).

**Shadows between light objects.** In regions at least 1.6 times the median region area, dark
pixels (Otsu threshold on L\* inside the mask) that touch the background and are either large
(at least 0.1 times the typical area) or line shaped (square of the longest bounding-box side
divided by the area ≥ 4) are treated as shadows between objects. The distance is computed
without them, and parts separated by a shadow are never joined again.

**Joining parts.** The watershed tends to cut elongated or curved objects in several parts. Two
neighboring parts are joined when any of these holds:

- *No narrowing between them.* Along their common boundary, the distance to the background
  stays at least 0.9 times the peak distance of the thinner part.
- *No notches.* Two touching objects leave a notch (a concavity of the outline) at both ends
  of the line where they touch; a cut through a single object does not. FenotIT measures how
  deep each end of the cut lies inside the convex hull of the two parts together. If one end is
  shallower than 0.2 times the typical radius (at least 2 px) and the connection is wide (at
  least 0.6 times the peak distance of the thinner part), the parts are joined.
- *Small part.* A part smaller than 0.35 times the typical area of an isolated object is joined
  to the neighbor with which it shares the longest boundary (for example, the tip of a pod).

A region smaller than 1.5 times the typical area of an isolated object is left as one object.

**Cutting at notches.** Two seeds lying side by side can form a single ridge in the distance
map and stay joined after the watershed. Objects at least 1.6 times the median area are
therefore tested for straight cuts between two notches of their outline (notches deeper than
0.25 times the typical radius), or between one notch and the opposite side. Each candidate cut
is scored by three terms: the solidity (area divided by convex hull area) of its least convex
part, which should be high; how far the areas of the parts are from the typical area, which
should be small; and the cut length, which should be short:

$$\text{score} = \text{lowest solidity of the parts} \;-\; 0.1 \times \sum_{\text{parts}} \left| \frac{\text{part area}}{\text{typical area}} - 1 \right| \;-\; 0.01 \times \frac{\text{cut length}}{\text{typical radius}}$$

The cut with the highest score is made if both parts have a solidity of at least 0.85 and at
least 0.03 above the solidity of the whole object. Up to three cuts are made one after another.

**Flag.** Objects that come from a split region carry `touching = 1`. They are counted, and by
default they are left out of the size and shape measurements (section 4), since part of their
outline is a cut.

## 3.3 Filters

- **Noise.** Objects smaller than 0.01 % of the photo and smaller than 5 % of the typical object
  are removed. The typical object area is the median weighted by area, so many specks of noise
  do not lower it. Both limits are relative, so the rule gives the same result at any
  resolution.
- **Border.** Objects that touch the edge of the photo are cut by it; they can be excluded
  (option "Exclude objects on the border").
- **Size.** Optional minimum and maximum length and width (section 4), in the unit of the scale;
  0 means no limit.
- **Manual exclusion.** An object excluded by the user in the inspector keeps its number but is
  neither counted nor measured (`status = excluded`). The choice is stored as a point in
  relative coordinates and applied again each time the photo is analyzed.

## 3.4 Numbering

Objects are numbered from 1 in reading order. The row of an object is its centroid height
divided by the median height of the objects' bounding boxes, rounded; within a row, objects go
from left to right. The number identifies the object in the image, in the tables and in the
export.

## 3.5 Objects in analysis areas

When analysis areas are drawn, each object gets the area that contains its centroid (column
`analysis_area`, with the area name or number); if the centroid falls outside every area, the
area that contains most of its pixels. A table `areas` gives the number of objects and the size
of each area.
