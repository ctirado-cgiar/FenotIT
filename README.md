# FenotIT: Digital Plant Phenotyping Platform

Desktop software for image-based plant phenotyping. It finds the objects in a photo (seeds, pods, leaves, lesions, insects) and measures their number, size, shape, color and arrangement, one photo at a time or in batches, without programming.

Developed by Cristian Tirado-Murcia (software development and analysis methods), in collaboration with the Physiology Team of the Bean Program, Alliance of Bioversity International and CIAT (Palmira, Colombia).

> **Status:** in development towards version 1.0. Results and file formats may still change.

## Features

- **Analyses:** Objects (count and measure each object), Detection (objects found by a detection model), Classify areas (share and area of each class) and Image color.
- **Segmentation:** threshold on one channel of eight color spaces, manual or automatic (Otsu); classification by examples painted on the photo (random forest); regions from the MobileSAM model, grouped into categories with clicks.
- **Detection models:** YOLO and RF-DETR models trained by the user, run on the CPU; boxes as objects or counted inside each object (for example, holes per seed).
- **Measurements per object:** area, length, width, perimeter and shape ratios; elliptic Fourier descriptors and mean shape; mean and dominant colors (fixed or automatic number) in RGB and CIELAB; area of each class.
- **Touching objects:** optional separation (distance peaks, watershed and cuts at notches).
- **Analysis areas and excluded zones** (rectangle or polygon), per photo or for all photos, with results per area.
- **Corrections:** lens distortion (chessboard), perspective (ArUco markers), color card and scale (two points, typed value or ArUco markers), each optional.
- **Projects** (`.fnt`) and **presets** (`.fntp`) that keep the corrections, segmentation and analysis of a capture setup.
- **Export** to CSV and Excel, one table per level (object, photo, class, area), with the result images and a metadata table with every parameter.
- **Interface** in English and Spanish. All processing is local, without internet.

## Installation (from source)

Requires Python 3.11 or later (3.12 recommended). Windows 10 and 11 are the tested platforms.

```bash
git clone https://github.com/ctirado-cgiar/FenotIT.git
cd FenotIT
conda create -n fenotit python=3.12 -y
conda activate fenotit
pip install -r requirements.txt
```

## Usage

```bash
python main.py                  # open the app
python main.py MyTrial.fnt      # open a project
python -m fenotit --version
```

Typical workflow: add photos, choose an analysis, set the segmentation (or the detection model), draw the analysis area if needed, tick the measurements, run on this photo or on all, and export with **Project → Export**. **Help → Shortcuts** lists the keyboard shortcuts.

## Methods

How each measurement is computed, with equations, references and figures made with real photos: [`docs/methods`](docs/methods/README.md).

## Development

Code layout, how to add a step or an analysis, tests and code rules: [`docs/desarrollo.md`](docs/desarrollo.md).

Application logs are written to `%LOCALAPPDATA%\FenotIT\logs\fenotit.log` (Windows) or `~/.fenotit/logs/` (other systems).

## Citation

See [`CITATION.cff`](CITATION.cff).

## Contact

Cristian Tirado-Murcia, c.tirado@cgiar.org
Alliance of Bioversity International and CIAT
