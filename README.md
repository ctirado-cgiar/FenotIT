# FenotIT — Digital Plant Phenotyping Platform

Desktop software for image-based plant phenotyping. It segments objects (seeds, leaves, roots…) and measures their shape, color and number, one image at a time or in batches.

Developed by Cristian Tirado-Murcia (software development and analysis methods), in collaboration with the Physiology Team of the Bean Program, Alliance of Bioversity International & CIAT (Palmira, Colombia).

> **Status:** under active development towards v1.0. Results and file formats may still change.

## Features

- **One analysis, several measurements:** segment once, then tick what to measure
  - **Count:** one dot per object, to check at a glance what was counted
  - **Morphometry:** area, length, width, perimeter, circularity, solidity and more per object
  - **Shape:** elliptic Fourier descriptors per object, and mean shape by image, batch or group
  - **Color:** dominant colors (KMeans) per object and per image, in RGB and CIELab
- **Touching objects:** optional declumping (distance peaks + watershed + notch cuts); touching objects are counted but not measured
- **Segmentation** by manual or automatic (Otsu) threshold on any channel of BGR, HSV, LAB, YCrCb, HLS, XYZ, YUV or LUV
- **Analysis areas and excluded areas** (rectangle or polygon), several per photo, for one photo or all
- **Scale per photo or for all** (two points, typed value or ArUco markers)
- **Optional corrections:** lens distortion (chessboard), perspective (ArUco), color card and scale
- **Projects** (`.fnt`) that keep images, thresholds, parameters, areas and scale
- **Export** to CSV and Excel, one table per level (object, image), with the result images
- **Interface in English and Spanish**
- 100 % local processing, no internet required

## Installation (from source)

Requires Python 3.11+ (3.12 recommended). Windows 10/11 is the tested platform.

```bash
git clone https://github.com/ctirado-cgiar/FenotIT.git
cd FenotIT
conda create -n fenotit python=3.12 -y
conda activate fenotit
pip install -r requirements.txt
```

## Usage

```bash
python main.py                      # open the app
python main.py MyTrial.fnt      # open a project
python -m fenotit --version
```

Typical workflow: add images → set the segmentation (channel and threshold, or Otsu) → optionally draw the analysis area → tick the measurements → **▶ Run** (this image or all) → **File → Export**. **Help → Shortcuts** lists the keyboard shortcuts.

## Methods

How every measurement is computed (equations, references and figures): [`docs/methods`](docs/methods/README.md).

## Development

Code layout, how to add a step or an analysis, tests and code rules: [`docs/desarrollo.md`](docs/desarrollo.md).

Application logs are written to `%LOCALAPPDATA%\FenotIT\logs\fenotit.log` (Windows) or `~/.fenotit/logs/` (other systems).

## Citation

See [`CITATION.cff`](CITATION.cff).

## Contact

Cristian Tirado-Murcia — c.tirado@cgiar.org
Alliance of Bioversity International and CIAT
