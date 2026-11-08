"""Segmentación con MobileSAM: modos, regiones, categorías por clic y clics en objetos."""
from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from fenotit import log, settings
from fenotit.gui.style.theme import COLORS, FONTS
from fenotit.gui.widgets.help import HelpIcon
from fenotit.i18n import t


_log = log.get("gui.window.model")


class ModelMixin:
    def _grid(self) -> int:
        return next((k for k, v in self._grid_names.items() if v == self.grid_var.get()), 16)

    def _build_model_box(self, box):
        bg, small = COLORS["bg_panel"], FONTS["small"]
        head = tk.Frame(box, bg=bg)
        head.pack(fill=tk.X)
        tk.Label(head, text="MobileSAM", bg=bg, fg=COLORS["text"], font=FONTS["body_bold"]).pack(side=tk.LEFT)
        HelpIcon(head, "MobileSAM", t("sam.help"), bg=bg).pack(side=tk.LEFT, padx=(4, 0))
        self.model_mode = tk.StringVar(value="auto")
        for key in ("auto", "points"):
            tk.Radiobutton(box, text=t(f"sam.mode.{key}"), variable=self.model_mode, value=key, bg=bg,
                           fg=COLORS["text"], selectcolor=COLORS["bg_card"], activebackground=bg, font=small,
                           command=self._on_model_change).pack(anchor="w")
        grow = tk.Frame(box, bg=bg)
        tk.Label(grow, text=t("sam.grid"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        self._grid_names = {8: t("sam.grid.8"), 16: t("sam.grid.16"), 24: t("sam.grid.24"), 32: t("sam.grid.32")}
        self.grid_var = tk.StringVar(value=self._grid_names[16])
        cb = ttk.Combobox(grow, textvariable=self.grid_var, values=list(self._grid_names.values()),
                          state="readonly", width=16, font=small)
        cb.pack(side=tk.LEFT, padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())
        HelpIcon(grow, t("sam.grid"), t("sam.grid_help"), bg=bg).pack(side=tk.LEFT)
        self._grid_row = grow
        arow = tk.Frame(box, bg=bg)
        self._sam_actions = arow
        self._sam_try = tk.Button(arow, text=t("sam.try"), command=self._sam_try_now, bg=COLORS["btn_bg"],
                                  fg=COLORS["accent"], relief="flat", font=small, cursor="hand2", padx=8)
        self._sam_clear = tk.Label(arow, text=t("sam.clear"), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2")
        self._sam_clear.bind("<Button-1>", lambda e: self._clear_prompts())
        # categorías de las regiones (solo Automático)
        cat = tk.Frame(box, bg=bg)
        chead = tk.Frame(cat, bg=bg)
        chead.pack(fill=tk.X, pady=(8, 2))
        tk.Label(chead, text=t("cat.title"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        HelpIcon(chead, t("cat.title"), t("cat.help"), bg=bg).pack(side=tk.LEFT, padx=(4, 0))
        from fenotit.gui.widgets.segment import CategoryList
        self.category_list = CategoryList(cat, self._on_categories_change, lambda i: None)
        self.category_list.set(self._default_categories())
        self.category_list.pack(fill=tk.X)
        links = tk.Frame(cat, bg=bg)
        links.pack(fill=tk.X, pady=(4, 0))
        tk.Label(links, text=t("sam.regions_photo"), bg=bg, fg=COLORS["text_muted"], font=small).pack(side=tk.LEFT)
        for key, fn in (("sam.all", lambda: self._set_all_regions(True)),
                        ("sam.none", lambda: self._set_all_regions(False))):
            lb = tk.Label(links, text=t(key), bg=bg, fg=COLORS["accent"], font=small, cursor="hand2")
            lb.pack(side=tk.LEFT, padx=(6, 0))
            lb.bind("<Button-1>", lambda e, f=fn: f())
        self._cat_box = cat
        self.sam_status = tk.StringVar(value="")
        self._sam_status_lbl = tk.Label(box, textvariable=self.sam_status, bg=bg, fg=COLORS["text_muted"],
                                        font=small, justify="left", wraplength=240, anchor="w")
        self._on_model_change(quiet=True)

    @staticmethod
    def _default_categories() -> list[dict]:
        return [{"name": t("cat.default"), "color": COLORS["accent"]}]

    def _on_model_change(self, quiet: bool = False):
        auto = self.model_mode.get() == "auto"
        for w in (self._grid_row, self._sam_actions, self._sam_status_lbl, self._sam_try, self._sam_clear,
                  self._cat_box):
            w.pack_forget()
        if auto:
            self._grid_row.pack(fill=tk.X, pady=(4, 0))
        self._sam_actions.pack(fill=tk.X, pady=(6, 0))
        (self._sam_try if auto else self._sam_clear).pack(side=tk.LEFT)
        self._sam_status_lbl.pack(fill=tk.X, pady=(4, 0))
        if auto:
            self._cat_box.pack(fill=tk.X)
        self.sam_status.set(t("sam.hint_auto") if auto else t("sam.hint_points"))
        if not quiet:
            self._settings_changed()
            self._update_stroke_tool()
            self._on_slider_change()

    def _regions_for(self, path) -> dict:
        return self.project.regions.get(self.project.key(path)) or {"default": 1, "picks": []}

    def _sam_labels(self):
        """Las regiones detectadas en la foto actual (Automático), o None."""
        cached = getattr(self, "_sam_cache", None)
        full = self.scaler_left.original
        if cached is None or full is None or cached[0] != (id(full), "auto", self._grid()):
            return None
        return cached[1]

    def _on_region_click(self, nx: float, ny: float):
        from fenotit.core.segmentation import sam
        labels = self._sam_labels()
        path = self.current_image_path
        if labels is None or not path:
            self.sam_status.set(t("sam.pick_first"))
            return
        h, w = labels.shape
        r = int(labels[min(h - 1, int(ny * h)), min(w - 1, int(nx * w))])
        if not r:
            return
        reg = self._regions_for(path)
        if self.project.key(path) not in self.project.regions:
            reg = self._learned_entry(labels) or reg         # se parte de lo que se ve
        cats = sam.categorize(labels, reg["default"], reg["picks"])
        cur = self.category_list.current + 1
        new = 0 if cats[r] == cur else cur                 # otro clic con la misma categoría: sin usar
        picks = [p for p in reg["picks"]
                 if labels[min(h - 1, int(p[1] * h)), min(w - 1, int(p[0] * w))] != r]
        picks.append([round(nx, 5), round(ny, 5), new])
        self._remember("undo.regions")
        self.project.regions[self.project.key(path)] = {"default": reg["default"], "picks": picks}
        self._store_region_examples(path)
        self._on_slider_change()

    def _learned_entry(self, labels):
        """La clasificación aprendida de esta foto como clics: la categoría más común es la
        de partida y cada región distinta lleva un clic en un punto interior."""
        model = self._region_model()
        if model is None:
            return None
        from fenotit.core.segmentation import regions
        from fenotit.core.pipeline.views import inside_point
        from scipy.ndimage import find_objects
        cats = regions.predict(model, self.scaler_left.original, labels)
        default = int(np.bincount(cats[1:]).argmax()) if len(cats) > 1 else 1
        h, w = labels.shape
        picks = []
        for i, sl in enumerate(find_objects(labels), 1):
            if sl is None or cats[i] == default:
                continue
            x, y = inside_point((labels[sl] == i).astype(np.uint8))
            picks.append([round((x + sl[1].start + 0.5) / w, 5), round((y + sl[0].start + 0.5) / h, 5), int(cats[i])])
        return {"default": default, "picks": picks}

    def _set_all_regions(self, on: bool):
        path = self.current_image_path
        if not path:
            return
        self._remember("undo.regions")
        self.project.regions[self.project.key(path)] = {
            "default": self.category_list.current + 1 if on else 0, "picks": []}
        self._store_region_examples(path)
        self._on_slider_change()

    def _region_file(self, key: str):
        import hashlib
        return settings.data_dir() / "regions" / (hashlib.blake2b(key.encode(), digest_size=10).hexdigest() + ".npz")

    def _store_region_examples(self, path):
        """Rasgos y categoría de cada región de esta foto: ejemplos para clasificar las
        regiones de las fotos donde no se hizo clic."""
        from fenotit.core.segmentation import regions, sam
        labels, full = self._sam_labels(), self.scaler_left.original
        if labels is None:
            return
        key = self.project.key(path)
        reg = self._regions_for(path)
        X, y = regions.examples(full, labels, sam.categorize(labels, reg["default"], reg["picks"]))
        if not hasattr(self, "_region_ex"):
            self._region_ex = {}
        self._region_ex[key] = (X, y)
        try:
            f = self._region_file(key)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, X=X, y=y)
        except OSError:
            _log.exception("Ejemplos de regiones")

    def _region_examples(self):
        """Todos los ejemplos de las fotos con regiones clasificadas, o None."""
        mem = getattr(self, "_region_ex", {})
        Xs, ys = [], []
        for p in self.project.images:
            key = self.project.key(p)
            if key not in self.project.regions:
                continue
            ex = mem.get(key)
            if ex is None and self._region_file(key).exists():
                with np.load(self._region_file(key)) as d:
                    ex = (d["X"], d["y"])
                mem[key] = ex
            if ex is not None and len(ex[1]):
                Xs.append(ex[0]); ys.append(ex[1])
        if not ys:
            return None
        return np.concatenate(Xs), np.concatenate(ys)

    def _region_model(self):
        from fenotit.core.segmentation import classifier as C, regions
        ex = self._region_examples()
        return None if ex is None else regions.train(*ex, C.fingerprint(*ex))

    def _on_categories_change(self, removed: int | None = None):
        if removed is not None:                          # las regiones de esa categoría quedan sin usar
            k = removed + 1
            for reg in self.project.regions.values():
                fix = lambda c: 0 if c == k else (c - 1 if c > k else c)
                reg["default"] = fix(reg.get("default", 1))
                reg["picks"] = [[x, y, fix(c)] for x, y, c in reg.get("picks", [])]
        self._settings_changed()
        self._on_slider_change()

    def _on_prompts_change(self, points: list):
        path = self.current_image_path
        if not path:
            return
        self._remember("undo.prompts")
        key = self.project.key(path)
        if points:
            self.project.prompts[key] = points
        else:
            self.project.prompts.pop(key, None)
        self._on_slider_change()

    def _clear_prompts(self):
        path = self.current_image_path
        if path and self.project.prompts.get(self.project.key(path)):
            self._remember("undo.prompts")
            self.project.prompts.pop(self.project.key(path), None)
            self.prompts.set_points([])
            self._on_slider_change()

    def _sam_try_now(self):
        self._sam_try_requested = True
        self._show_preview()

    def _model_preview(self, img, inc, exc):
        """Con clics: al instante (el codificador corre una vez por foto). Automático: solo
        al pedirlo con «Probar en esta foto» (tarda decenas de segundos)."""
        from fenotit.core.segmentation import sam
        full = self.scaler_left.original
        auto = self.model_mode.get() == "auto"
        pts = self.project.prompts.get(self.project.key(self.current_image_path), [])
        key = (id(full), "auto", self._grid()) if auto else (id(full), "points", json.dumps(pts))
        cached = getattr(self, "_sam_cache", None)
        if cached is not None and cached[0] == key:
            self._render_instances(img, cached[1], inc, exc)
            return
        self.zoom_ctrl.redraw_with_overlay(overlay_left=img)
        self.preview_var.set("")
        if auto and not getattr(self, "_sam_try_requested", False):
            return
        if not auto and not any(p[2] for p in pts):
            self.sam_status.set(t("sam.hint_points"))
            return
        self._sam_try_requested = False
        self._sam_gen = getattr(self, "_sam_gen", 0) + 1
        gen, grid = self._sam_gen, self._grid()
        self.sam_status.set(t("sam.working"))
        self._sam_try.config(state=tk.DISABLED)

        def work():
            t0 = time.perf_counter()
            try:
                labels = sam.automatic(full, grid=grid) if auto else sam.from_points(full, pts)
                secs = time.perf_counter() - t0
                self.root.after(0, lambda: self._on_sam_done(gen, key, labels, secs))
            except Exception as e:
                _log.exception("MobileSAM")
                msg = str(e)
                self.root.after(0, lambda: (self.sam_status.set(t("sam.failed", error=msg)),
                                            self._sam_try.config(state=tk.NORMAL)))
        threading.Thread(target=work, daemon=True).start()

    def _on_sam_done(self, gen, key, labels, secs):
        self._sam_try.config(state=tk.NORMAL)
        if gen != getattr(self, "_sam_gen", 0):
            return
        self._sam_cache = (key, labels)
        self.sam_status.set(t("sam.done", n=int(labels.max()), s=f"{secs:.1f}"))
        self._show_preview()

    def _render_instances(self, img, labels, inc, exc):
        """Con clics: cada objeto con su color. Automático: cada región con el color de su
        categoría y el borde de todas (las sin usar, solo el borde)."""
        lab = cv2.resize(labels.astype(np.int32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.model_mode.get() != "auto":
            mask = self._finish_mask(((lab > 0) * 255).astype(np.uint8), img, inc, exc)
            rng = np.random.default_rng(7)
            pal = rng.integers(60, 255, (int(lab.max()) + 1, 3)).astype(np.uint8)
            colored = img.copy()
            inside = mask > 0
            colored[inside] = (0.45 * img[inside] + 0.55 * pal[lab[inside]]).astype(np.uint8)
            self._show_mask_overlay(img, mask, colored)
            return
        from fenotit.core.segmentation import sam
        key = self.project.key(self.current_image_path)
        model = None if key in self.project.regions else self._region_model()
        if model is not None:                              # sin clics aquí: lo aprendido de las otras fotos
            from fenotit.core.segmentation import regions
            cats = regions.predict(model, self.scaler_left.original, labels)
            self.sam_status.set(t("sam.learned"))
        else:
            reg = self._regions_for(self.current_image_path)
            cats = sam.categorize(lab, reg["default"], reg["picks"])
        cat_img = cats[lab]
        mask = self._finish_mask(((cat_img > 0) * 255).astype(np.uint8), img, inc, exc)
        pal = np.zeros((len(self.category_list.cats) + 1, 3), np.uint8)
        for i, c in enumerate(self.category_list.cats, 1):
            h = c["color"].lstrip("#")
            pal[i] = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))
        colored = (img * 0.75).astype(np.uint8)            # sin usar: un poco más oscuras
        inside = mask > 0
        colored[inside] = (0.5 * img[inside] + 0.5 * pal[np.clip(cat_img[inside], 0, len(pal) - 1)]).astype(np.uint8)
        edges = cv2.morphologyEx(lab.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        colored[edges] = (255, 255, 255)
        self._show_mask_overlay(img, mask, colored)
