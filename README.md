# FenotIT — Digital Phenotyping from Images

Desktop software for image-based plant phenotyping. It segments objects (seeds, leaves, roots…) and measures their shape, color and number, one image at a time or in batches.

Developed by the Physiology team of the Bean Breeding Program, Alliance of Bioversity International and CIAT (Palmira, Colombia).

> **Status:** under active development towards v1.0. Results and file formats may still change.

## Features

- **Morphometry:** area, length, width, perimeter and shape descriptors per object
- **Color:** dominant colors with KMeans, reported in RGB, CIELab, luminance and % of area
- **Object counting** with distance transform and local peaks
- **Segmentation** by thresholding any channel of BGR, HSV, LAB, YCrCb, HLS, XYZ, YUV or LUV
- **Regions of interest:** rectangle, square, polygon, holes and exclusion zones
- **Optional corrections:** lens distortion (chessboard), perspective (ArUco), color card and scale
- **Projects** (`.fenotit`) that keep images, thresholds, parameters, ROI and scale
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

Color-card correction also needs PlantCV: `pip install -r requirements-optional.txt`.

## Usage

```bash
python main.py                      # open the app
python main.py MyTrial.fenotit      # open a project
python -m fenotit --version
```

Typical workflow: load an image or a folder → choose color space, channel and threshold → draw a ROI (optional) → pick an analysis and its parameters → run on one image or on the whole batch → export. Save everything with **File → Save project**.

## Development

- Code layout and how to add a new analysis: [`README_ESTRUCTURA.md`](README_ESTRUCTURA.md)
- After any change, check that results did not change unexpectedly:

```bash
python tests/reference.py        # compares against tests/reference/
python tests/test_project.py
python tests/test_i18n.py
```

Application logs are written to `%LOCALAPPDATA%\FenotIT\logs\fenotit.log` (Windows) or `~/.fenotit/logs/` (other systems).

## Citation

See [`CITATION.cff`](CITATION.cff), or use **Cite this repository** on GitHub.

## License

[CC BY-NC-SA 4.0](LICENCE.md) — free for research and education, no commercial use.

## Contact

Cristian Tirado-Murcia — Alliance of Bioversity International and CIAT
