"""Pantalla de inicio (sin fotos): franja con el nombre de la app a la izquierda; a la
derecha "Nuevo proyecto" (al tocarlo: agregar fotos o carpeta), "Abrir proyecto" y los
recientes con miniatura de su primera foto. Se pone encima de Entrada/Resultado."""
from __future__ import annotations

import datetime as dt
import tkinter as tk
from pathlib import Path

import yaml
from PIL import Image, ImageTk

from fenotit import APP_NAME, __version__, settings
from fenotit.gui.toolbar import icon
from fenotit.i18n import t

_THUMB = (88, 66)
_SERIF = "Georgia"          # títulos en cursiva; si no existe, Tk usa una parecida


def _project_info(path: Path) -> tuple[int, Path | None]:
    """Número de fotos y la primera foto de un proyecto (sin cargarlo entero)."""
    try:
        with open(path, encoding="utf-8") as f:
            images = (yaml.safe_load(f) or {}).get("images") or []
    except (OSError, yaml.YAMLError):
        return 0, None
    if not images:
        return 0, None
    first = Path(images[0])
    return len(images), first if first.is_absolute() else path.parent / first


def _thumbnail(path: Path | None):
    if not path or not path.exists():
        return None
    try:
        with Image.open(path) as im:
            im.draft("RGB", (_THUMB[0] * 2, _THUMB[1] * 2))       # JPEG: decodifica en pequeño
            im = im.convert("RGB")
            im.thumbnail(_THUMB)
            canvas = Image.new("RGB", _THUMB, (235, 238, 242))
            canvas.paste(im, ((_THUMB[0] - im.width) // 2, (_THUMB[1] - im.height) // 2))
            return ImageTk.PhotoImage(canvas)
    except Exception:
        return None


class StartScreen(tk.Frame):

    def __init__(self, parent, colors: dict, fonts: dict, on_photos, on_folder, on_open, on_recent,
                 links=()):
        super().__init__(parent, bg=colors["bg"])
        self.c = colors
        self._links = links
        self._cb = (on_photos, on_folder, on_open, on_recent)
        self._img = {n: ImageTk.PhotoImage(icon(n, 30, colors["accent"])) for n in ("photo", "folder", "project")}
        self._img["photo_w"] = ImageTk.PhotoImage(icon("photo", 22, "#FFFFFF"))
        self._img["folder_w"] = ImageTk.PhotoImage(icon("folder", 22, "#FFFFFF"))
        self._thumbs: list = []
        self._build_hero()
        self._main = tk.Frame(self, bg=colors["bg"])
        self._main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.refresh()

    def _build_hero(self):
        c = self.c
        hero = tk.Frame(self, bg=c["bg_topbar"], width=300)
        hero.pack(side=tk.LEFT, fill=tk.Y)
        hero.pack_propagate(False)
        inner = tk.Frame(hero, bg=c["bg_topbar"])
        inner.place(relx=0.5, rely=0.45, anchor="center")
        try:
            from fenotit.gui.main_window import _assets
            logo = Image.open(_assets() / "logo.ico").convert("RGBA").resize((72, 72), Image.LANCZOS)
            self._img["logo"] = ImageTk.PhotoImage(logo)
            tk.Label(inner, image=self._img["logo"], bg=c["bg_topbar"]).pack(pady=(0, 10))
        except Exception:
            pass
        tk.Label(inner, text=APP_NAME, bg=c["bg_topbar"], fg="#FFFFFF", font=("Segoe UI", 26, "bold")).pack()
        tk.Label(inner, text=t("start.subtitle"), bg=c["bg_topbar"], fg="#E3EDF9", font=(_SERIF, 12, "italic"),
                 wraplength=240, justify="center").pack(pady=(4, 18))
        tk.Label(inner, text=t("start.tagline"), bg=c["bg_topbar"], fg="#B9CFEA", font=("Segoe UI", 8),
                 wraplength=230, justify="center").pack()
        tk.Label(hero, text=f"v{__version__}  ·  Alliance Bioversity & CIAT", bg=c["bg_topbar"], fg="#9FBBE0",
                 font=("Segoe UI", 7)).pack(side=tk.BOTTOM, pady=(4, 10))
        links = tk.Frame(hero, bg=c["bg_topbar"])          # idioma y acerca de (sin la barra de menús)
        links.pack(side=tk.BOTTOM)
        for i, (text, cmd) in enumerate(self._links):
            lbl = tk.Label(links, text=text.strip("… ").split("  ")[-1], bg=c["bg_topbar"], fg="#FFFFFF",
                           font=("Segoe UI", 8, "underline"), cursor="hand2")
            lbl.pack(side=tk.LEFT, padx=8)
            lbl.bind("<Button-1>", lambda e, f=cmd: f())

    def refresh(self):
        """Vuelve a armar la parte derecha (al mostrarse: recientes al día)."""
        c, m = self.c, self._main
        for w in m.winfo_children():
            w.destroy()
        self._thumbs = []
        on_photos, on_folder, on_open, on_recent = self._cb
        body = tk.Frame(m, bg=c["bg"])
        body.place(relx=0.5, rely=0.42, anchor="center")

        tk.Label(body, text=t("start.begin"), bg=c["bg"], fg=c["text_muted"],
                 font=(_SERIF, 14, "italic")).pack(anchor="w", pady=(0, 10))
        row = tk.Frame(body, bg=c["bg"])
        row.pack(anchor="w")
        self._new_tile(row, on_photos, on_folder)
        self._tile(row, "project", t("start.open"), t("start.open_hint"), on_open)

        tk.Label(body, text=t("start.recent"), bg=c["bg"], fg=c["text_muted"],
                 font=(_SERIF, 12, "italic")).pack(anchor="w", pady=(22, 6))
        recent = settings.recent_projects()
        if not recent:
            tk.Label(body, text=t("start.no_recent"), bg=c["bg"], fg=c["text_muted"],
                     font=("Segoe UI", 9)).pack(anchor="w")
        grid = tk.Frame(body, bg=c["bg"])
        grid.pack(anchor="w")
        for i, path in enumerate(recent[:6]):
            self._recent_card(grid, Path(path), on_recent).grid(row=i // 2, column=i % 2, padx=(0, 10), pady=(0, 8),
                                                              sticky="w")

    # ── tarjetas ─────────────────────────────────────────────────────────────

    def _card(self, parent, width, height):
        c = self.c
        card = tk.Frame(parent, bg=c["bg_card"], highlightthickness=1, highlightbackground=c["border"],
                        width=width, height=height, cursor="hand2")
        card.pack_propagate(False)
        return card

    def _hover(self, card, widgets):
        c = self.c

        def paint(on):
            card.config(highlightbackground=c["accent"] if on else c["border"])
        for w in [card] + list(widgets):
            w.bind("<Enter>", lambda e: paint(True), add="+")
            w.bind("<Leave>", lambda e: paint(False), add="+")

    def _tile(self, parent, ico, title, hint, cmd):
        c = self.c
        card = self._card(parent, 210, 120)
        card.pack(side=tk.LEFT, padx=(0, 12))
        parts = [tk.Label(card, image=self._img[ico], bg=c["bg_card"]),
                 tk.Label(card, text=title, bg=c["bg_card"], fg=c["text"], font=("Segoe UI", 10, "bold")),
                 tk.Label(card, text=hint, bg=c["bg_card"], fg=c["text_muted"], font=("Segoe UI", 8),
                          wraplength=190, justify="center")]
        parts[0].pack(pady=(18, 4))
        parts[1].pack()
        parts[2].pack()
        self._hover(card, parts)
        for w in [card] + parts:
            w.bind("<Button-1>", lambda e: cmd())

    def _new_tile(self, parent, on_photos, on_folder):
        """Nuevo proyecto: al tocarlo aparecen "Agregar fotos" y "Agregar carpeta"."""
        c = self.c
        card = self._card(parent, 210, 120)
        card.pack(side=tk.LEFT, padx=(0, 12))
        front = [tk.Label(card, image=self._img["photo"], bg=c["bg_card"]),
                 tk.Label(card, text=t("start.new"), bg=c["bg_card"], fg=c["text"], font=("Segoe UI", 10, "bold")),
                 tk.Label(card, text=t("start.new_hint"), bg=c["bg_card"], fg=c["text_muted"], font=("Segoe UI", 8),
                          wraplength=190, justify="center")]
        front[0].pack(pady=(18, 4))
        front[1].pack()
        front[2].pack()
        choices = tk.Frame(card, bg=c["bg_card"])
        for ico, text, cmd in (("photo_w", t("start.photos"), on_photos), ("folder_w", t("start.folder"), on_folder)):
            tk.Button(choices, text=f"  {text}", image=self._img[ico], compound="left", command=cmd,
                      bg=c["accent"], fg="#FFFFFF", activebackground=c["bg_topbar"], activeforeground="#FFFFFF",
                      relief="flat", bd=0, font=("Segoe UI", 9, "bold"), anchor="w", padx=12, pady=6,
                      cursor="hand2").pack(fill=tk.X, padx=14, pady=(10, 0))

        def open_choices(_e=None):
            for w in front:
                w.pack_forget()
            choices.pack(fill=tk.BOTH, expand=True)
        self._hover(card, front)
        for w in [card] + front:
            w.bind("<Button-1>", open_choices)

    def _recent_card(self, parent, path: Path, cmd):
        c = self.c
        card = self._card(parent, 300, 82)
        n, first = _project_info(path)
        thumb = _thumbnail(first)
        self._thumbs.append(thumb)
        pic = tk.Label(card, bg="#EBEEF2", width=_THUMB[0], height=_THUMB[1])
        if thumb:
            pic.config(image=thumb)
        pic.place(x=7, y=7, width=_THUMB[0], height=_THUMB[1])
        try:
            when = dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
        except OSError:
            when = ""
        texts = [tk.Label(card, text=path.stem, bg=c["bg_card"], fg=c["accent"], font=("Segoe UI", 10, "bold"),
                          anchor="w"),
                 tk.Label(card, text=t("start.recent_info", n=n, date=when), bg=c["bg_card"], fg=c["text"],
                          font=("Segoe UI", 8), anchor="w"),
                 tk.Label(card, text=str(path.parent), bg=c["bg_card"], fg=c["text_muted"], font=("Segoe UI", 7),
                          anchor="w")]
        for i, w in enumerate(texts):
            w.place(x=_THUMB[0] + 18, y=10 + i * 21, width=300 - _THUMB[0] - 26)
        self._hover(card, [pic] + texts)
        for w in [card, pic] + texts:
            w.bind("<Button-1>", lambda e: cmd(str(path)))
        return card
