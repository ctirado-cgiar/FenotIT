"""Segmentación por umbral: espacio de color, canal, rango, Otsu e histograma."""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.gui.widgets.help import HelpIcon
from fenotit.gui.window.common import CHANNEL_NAMES
from fenotit.i18n import t


class ThresholdMixin:
    def _update_channel_names(self):
        cs    = self.cs_var.get()
        names = CHANNEL_NAMES.get(cs, ["0","1","2"])
        for i, rb in enumerate(self._ch_radios):
            rb.config(text=names[i])

    def _on_cs_change(self, _=None):
        self._update_channel_names()
        self.ch_var.set(0)
        self._on_slider_change()

    def _build_threshold_box(self, box):
        """Espacio de color, canal y rango (o Otsu), con vista previa en vivo."""
        bg, small = COLORS["bg_panel"], FONTS["small"]
        self._seg_box = box
        row = tk.Frame(box, bg=bg)
        row.pack(fill=tk.X, pady=SPACE["xs"])
        tk.Label(row, text=t("controls.space"), bg=bg, fg=COLORS["text_muted"],
                 font=small).pack(side=tk.LEFT)
        self.cs_var = tk.StringVar(value="BGR")
        cs_combo = ttk.Combobox(row, textvariable=self.cs_var, values=list(CHANNEL_NAMES.keys()),
                                state="readonly", width=7, font=small)
        cs_combo.pack(side=tk.LEFT, padx=(SPACE["sm"], SPACE["md"]))
        cs_combo.bind("<<ComboboxSelected>>", self._on_cs_change)
        self.ch_var = tk.IntVar(value=0)
        self._ch_radios = []
        for i in range(3):
            rb = tk.Radiobutton(row, text="", variable=self.ch_var, value=i,
                                command=self._on_slider_change, bg=bg, fg=COLORS["text"],
                                selectcolor=COLORS["bg_card"], activebackground=bg, font=small)
            rb.pack(side=tk.LEFT)
            self._ch_radios.append(rb)
        self._update_channel_names()

        self.auto_var = tk.BooleanVar(value=False)
        arow = tk.Frame(box, bg=bg)
        arow.pack(anchor="w", pady=(SPACE["xs"], 0))
        tk.Checkbutton(arow, text=t("seg.auto"), variable=self.auto_var, command=self._on_auto_change,
                       bg=bg, fg=COLORS["text"], selectcolor=COLORS["bg_card"],
                       activebackground=bg, font=small, padx=0).pack(side=tk.LEFT)
        HelpIcon(arow, t("seg.auto"), t("seg.auto_help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))

        from fenotit.gui.widgets.histogram import ChannelHistogram, RangeSlider
        rrow = tk.Frame(box, bg=bg)
        rrow.pack(fill=tk.X, pady=(SPACE["md"], 0))
        tk.Label(rrow, text=t("seg.range"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        HelpIcon(rrow, t("seg.range"), t("seg.range_help"), bg=bg).pack(side=tk.LEFT, padx=(SPACE["sm"], 0))
        self.range_slider = RangeSlider(box, on_change=lambda lo, hi: self._on_slider_moved(), bg=bg)
        self.range_slider.pack(fill=tk.X)
        self.min_slider = self.range_slider.end("lo")
        self.max_slider = self.range_slider.end("hi")
        self.histogram = ChannelHistogram(box, on_drag=self._on_hist_drag, bg=bg)
        self.histogram.pack(fill=tk.X)

    def _on_auto_change(self):
        """Otsu se ve de inmediato y deja Mín/Máx en su rango; moverlos después lo apaga."""
        if self.auto_var.get():
            self._show_preview()
        else:
            self._on_slider_change()

    def _on_slider_moved(self):
        """El command de Scale también llega cuando el valor se pone por código (y tarde):
        solo cuenta como cambio del usuario si difiere del último valor puesto."""
        cur = (int(self.min_slider.get()), int(self.max_slider.get()))
        prev = getattr(self, "_slider_state", None)
        if cur == prev:
            return
        if cur[0] > cur[1]:                     # Mín no pasa a Máx ni Máx baja de Mín
            cur = (cur[1], cur[1]) if prev is None or cur[0] != prev[0] else (cur[0], cur[0])
            self._sync_sliders(*cur)
        self._slider_state = cur
        self.histogram.set_range(*cur)
        if self.auto_var.get():                 # el usuario toma el control: umbral manual
            self.auto_var.set(False)
        self._on_slider_change()

    def _sync_sliders(self, lo: int, hi: int):
        self._slider_state = (int(lo), int(hi))
        self.min_slider.set(lo)
        self.max_slider.set(hi)
        if hasattr(self, "histogram"):
            self.histogram.set_range(lo, hi)

    def _on_hist_drag(self, lo: int, hi: int):
        """Arrastrar una línea del histograma = mover el deslizador (umbral manual)."""
        self.auto_var.set(False)
        self._sync_sliders(lo, hi)
        self._on_slider_change()

    def _update_histogram(self, img: np.ndarray, inc, exc):
        """Histograma del canal elegido, solo dentro del área de análisis (sin excluidas)."""
        key = (id(img), self.cs_var.get(), self.ch_var.get(),
               json.dumps(self.project.roi_for(self.current_image_path), sort_keys=True, default=str))
        if getattr(self, "_hist_key", None) != key:
            self._hist_key = key
            from fenotit.core import colorspaces
            space, sel = self.cs_var.get(), int(self.ch_var.get())
            chans = colorspaces.channels(img, space)
            region = None
            if inc is not None or exc:
                region = (inc > 0) if inc is not None else np.ones(chans[0].shape, bool)
                if exc:
                    region = region.astype(np.uint8)
                    cv2.fillPoly(region, [pts for pts, _ in exc], 0)
                    region = region > 0
            vals = [c[region] if region is not None else c[::2, ::2] for c in chans]
            colors = colorspaces.CHANNEL_COLORS.get(space, [COLORS["accent"]] * 3)
            sel = min(sel, len(vals) - 1)
            self.histogram.set_data(vals[sel], colors[sel],
                                    [(v, colors[i]) for i, v in enumerate(vals) if i != sel])
            self.range_slider.set_track(colors[sel])
        self.histogram.set_range(int(self.min_slider.get()), int(self.max_slider.get()))
