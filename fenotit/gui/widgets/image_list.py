"""Lista de imágenes del panel izquierdo: buscador, vista de lista o de cuadrícula con
miniaturas. Las miniaturas se leen en un hilo aparte (solo Pillow; los PhotoImage se
crean en el hilo principal) y primero las que están a la vista."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import ttk
from typing import Callable

from PIL import Image, ImageOps, ImageTk

from fenotit import log
from fenotit.gui.widgets.toolbar import IconButton
from fenotit.i18n import t
from fenotit.gui.style.theme import COLORS, FONTS, SPACE

_log = log.get("gui.images")
THUMB = 42                       # lado máximo de la miniatura (px): 3 columnas en el panel por defecto
SEARCH_FROM = 6                  # con menos fotos no hace falta buscador
MAX_SHARE = 0.45                 # tope por defecto (fracción del panel): debajo van CAPAS e inspector
MIN_ROWS = 2                     # si otros necesitan espacio, la lista cede hasta dejar ver estas filas
MARKS = {"board": ("▦", COLORS["text_faint"]), "done": ("✓", COLORS["success"]), "error": ("⚠", COLORS["warning"]), "pending": ("…", COLORS["text_faint"])}


def _read_thumb(path: str) -> Image.Image | None:
    try:
        with Image.open(path) as picture:
            picture.draft("RGB", (THUMB * 2, THUMB * 2))          # JPEG: decodifica en pequeño
            picture = ImageOps.exif_transpose(picture).convert("RGB")
            picture.thumbnail((THUMB, THUMB))
            return picture
    except Exception:
        _log.debug("miniatura: %s", path, exc_info=True)
        return None


class _ThumbLoader:
    """Hilo que lee miniaturas en el orden pedido (lo visible va primero)."""

    def __init__(self):
        self._todo: deque[str] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self.done: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def request(self, paths, first: bool = False):
        with self._lock:
            for path in (reversed(paths) if first else paths):
                if path in self._todo:
                    self._todo.remove(path)
                (self._todo.appendleft if first else self._todo.append)(path)
        self._wake.set()

    def _run(self):
        while True:
            self._wake.wait()
            with self._lock:
                path = self._todo.popleft() if self._todo else None
                if path is None:
                    self._wake.clear()
                    continue
            self.done.put((path, _read_thumb(path)))


class ImageList(tk.Frame):

    def __init__(self, parent, colors: dict, fonts: dict, on_select: Callable[[int], None],
                 on_delete: Callable[[], None], mode: str = "list", on_mode: Callable[[str], None] | None = None,
                 max_height: Callable[[], int] | None = None):
        super().__init__(parent, bg=colors["bg_panel"])
        self.c, self.f = colors, fonts
        self.on_select, self.on_delete, self.on_mode = on_select, on_delete, on_mode
        self.paths: list[str] = []
        self.marks: dict[str, str] = {}
        self.current = -1
        self.visible: list[int] = []
        self.mode = mode if mode in ("list", "grid") else "list"
        self._thumbs: dict[str, ImageTk.PhotoImage | None] = {}
        self._loader = _ThumbLoader()
        self._fit_job = None
        self.search_text: Callable[[str], str] | None = None    # texto extra que encuentra el buscador
        self.max_height = max_height or (lambda: int(self.master.winfo_height() * MAX_SHARE))
        self._build()
        self.pack_propagate(False)
        parent.bind("<Configure>", lambda event: self._schedule_fit(), add="+")
        self._poll()

    # ── interfaz ──────────────────────────────────────────────────────────────

    def _build(self):
        colors = self.c
        top = self._top = tk.Frame(self, bg=colors["bg_panel"])
        top.pack(fill=tk.X, padx=SPACE["md"], pady=(SPACE["xs"], SPACE["sm"]))
        kwargs = dict(bg=colors["bg_panel"], hover=colors["btn_hover"], active=colors["accent_light"], size=14, color=colors["accent"])
        self._btn = {mode: IconButton(top, mode, lambda mode=mode: self.set_mode(mode), t(f"images.view_{mode}"), **kwargs)
                     for mode in ("grid", "list")}
        for button_widget in self._btn.values():                   # primero los botones: el buscador toma lo que sobra
            button_widget.pack(side=tk.RIGHT, padx=(SPACE["xs"], 0))
        box = self._box = tk.Frame(top, bg=colors["bg_card"], highlightthickness=1, highlightbackground=colors["border"],
                       highlightcolor=colors["accent"])
        self.query = tk.StringVar()
        self._entry = tk.Entry(box, textvariable=self.query, bd=0, relief="flat", bg=colors["bg_card"],
                               fg=colors["text"], font=self.f["small"], insertwidth=1, width=6)
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(SPACE["md"], 0), ipady=3)
        self._clear = tk.Label(box, text="×", bg=colors["bg_card"], fg=colors["text_muted"], cursor="hand2",
                               font=FONTS["large"])
        self._clear.bind("<Button-1>", lambda tk_event: (self.query.set(""), self._entry.focus_set()))
        self._placeholder(True)
        self._entry.bind("<FocusIn>", lambda tk_event: self._placeholder(False))
        self._entry.bind("<FocusOut>", lambda tk_event: self._placeholder(not self.query.get()))
        self._entry.bind("<Escape>", lambda tk_event: (self.query.set(""), self.focus_set()))
        self.query.trace_add("write", lambda *_: self._filter())

        self._body = tk.Frame(self, bg=colors["bg_panel"])
        self._body.pack(fill=tk.BOTH, expand=True, padx=SPACE["md"], pady=(0, SPACE["sm"]))
        scrollbar = ttk.Scrollbar(self._body, orient="vertical")
        self._sb = scrollbar
        self.listbox = tk.Listbox(self._body, bg=colors["bg_card"], fg=colors["text"], font=self.f["small"],
                                  selectbackground=colors["bg_table_sel"], selectforeground=COLORS["text"], borderwidth=1,
                                  relief="flat", highlightthickness=1, highlightcolor=colors["border"],
                                  highlightbackground=colors["border"], activestyle="none", exportselection=False)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)
        self._touched = 0.0                       # el usuario movió la lista (no seguir el lote)
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>", "<Button-1>"):
            self.listbox.bind(event, self._touch, add="+")
        self.grid = tk.Canvas(self._body, bg=colors["bg_card"], highlightthickness=1,
                              highlightbackground=colors["border"], yscrollincrement=16)
        self.grid.bind("<Configure>", lambda tk_event: (self._draw_grid(), self._schedule_fit()))
        self.grid.bind("<Button-1>", self._on_grid_click)
        self.grid.bind("<Motion>", self._hover)
        self.grid.bind("<Leave>", lambda tk_event: self._tip_hide())
        for widget in (self.grid,):
            widget.bind("<MouseWheel>", self._wheel)
            widget.bind("<Button-4>", lambda tk_event: self._scroll(-3))
            widget.bind("<Button-5>", lambda tk_event: self._scroll(3))
        for widget in (self.listbox, self.grid):
            widget.bind("<Delete>", lambda tk_event: (self.on_delete(), "break")[1])
        self._empty = tk.Label(self._body, text=t("images.no_match"), bg=colors["bg_card"], fg=colors["text_muted"],
                               font=self.f["small"])
        self._show_mode()

    def _placeholder(self, active: bool):
        colors = self.c
        if active and not self.query.get():
            self._ph = True                       # antes de escribir: el buscador lo ignora
            self._entry.config(fg=colors["text_muted"])
            self._entry.insert(0, t("images.search"))
        elif getattr(self, "_ph", False):
            self._entry.delete(0, tk.END)
            self._ph = False
            self._entry.config(fg=colors["text"])

    def set_mode(self, mode: str):
        self.mode = mode
        self._show_mode()
        if self.on_mode:
            self.on_mode(mode)

    def _show_mode(self):
        for key, value in self._btn.items():
            value.select(key == self.mode)
        for widget in (self.listbox, self.grid, self._sb, self._empty):
            widget.pack_forget()
            widget.place_forget()
        self._scroll_on = False
        if self.mode == "list":
            self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.listbox.config(yscrollcommand=self._sb.set)
            self._sb.config(command=lambda *scroll_args: (self._touch(), self.listbox.yview(*scroll_args)))
            self._fill_list()
        else:
            self.grid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.grid.config(yscrollcommand=self._sb.set)
            self._sb.config(command=self._grid_yview)
            self._draw_grid()
        self._show_empty()
        self._schedule_fit()

    # ── alto según el contenido ─────────────────────────────────────────────────

    def refit(self):
        self._schedule_fit()

    def _schedule_fit(self):
        if self._fit_job is None:
            self._fit_job = self.after_idle(self._fit)

    def _row_height(self) -> int:
        if self.mode == "grid":
            return THUMB + 8
        if not getattr(self, "_lb_row", None):          # lo que Tk suma por cada fila
            listbox, saved_height = self.listbox, int(self.listbox.cget("height"))
            listbox.config(height=1)
            one_row = listbox.winfo_reqheight()
            listbox.config(height=2)
            self._lb_row = listbox.winfo_reqheight() - one_row
            listbox.config(height=saved_height)
        return self._lb_row

    def _fit(self):
        """Alto justo para las fotos que hay (sin barra ni buscador si no hacen falta),
        hasta MAX_SHARE del panel; desde ahí, barra de desplazamiento."""
        self._fit_job = None
        count = len(self.paths)
        searching = bool(self._query())
        if count >= SEARCH_FROM or searching:
            if not self._box.winfo_ismapped():
                self._box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        elif self._box.winfo_ismapped():
            self._box.pack_forget()
        self._top.update_idletasks()                 # el alto del encabezado cambia con el buscador
        row_height = self._row_height()
        if self.mode == "grid":
            width = (self.winfo_width() or 200) - 12
            cols = max(1, (width - 4) // (THUMB + 8))
            rows = (max(count, 1) + cols - 1) // cols
            content = rows * row_height + 6
        else:
            if int(self.listbox.cget("height")) != max(count, 1):
                self.listbox.config(height=max(count, 1))
            content = self.listbox.winfo_reqheight()             # Tk mide las filas exactas
        head = self._top.winfo_reqheight() + 6 + 4
        min_rows = self.min_rows() if callable(getattr(self, "min_rows", None)) else MIN_ROWS
        floor = head + (min_rows if count >= min_rows else 1) * row_height + 6
        limit = max(floor, self.max_height())
        want = head + content
        scroll = want > limit
        if scroll:                                   # filas enteras, sin una cortada abajo
            extra = content - rows * row_height if self.mode == "grid" else content - max(count, 1) * row_height
            height = head + extra + max(1, (limit - head - extra) // row_height) * row_height
        else:
            height = want
        if scroll != self._scroll_on:
            self._scroll_on = scroll
            if scroll:
                self._sb.pack(side=tk.RIGHT, fill=tk.Y, before=self.listbox if self.mode == "list" else self.grid)
            else:
                self._sb.pack_forget()
        if int(self.cget("height")) != height:
            self.config(height=height)
            self.after_idle(lambda: self.set_current(self.current))

    # ── datos ─────────────────────────────────────────────────────────────────

    def set_items(self, paths: list[str], marks: dict[str, str], current: int):
        if list(paths) == self.paths:               # solo cambian las marcas: sin mover la vista
            moved = current != self.current
            self.marks, self.current = dict(marks), current
            if self.mode == "list":
                self._fill_list(keep_view=not moved)
            else:
                self._draw_grid()
                if moved:
                    self._see(current)
            return
        new = [path for path in paths if path not in self._thumbs]
        self.paths, self.marks, self.current = list(paths), dict(marks), current
        for path in list(self._thumbs):
            if path not in self.paths:
                del self._thumbs[path]
        if new:
            self._loader.request(new)
        self._filter()
        self._schedule_fit()

    def set_current(self, index: int):
        self.current = index
        if self.mode == "list":
            self.listbox.selection_clear(0, tk.END)
            if index in self.visible:
                row = self.visible.index(index)
                self.listbox.selection_set(row)
                if getattr(self, "_scroll_on", True):
                    self.listbox.see(row)
                else:                                   # cabe todo: nada corrido
                    self.listbox.yview_moveto(0)
        else:
            self._draw_grid()
            self._see(index)

    def step(self, index: int, delta: int) -> int | None:
        """La imagen anterior/siguiente entre las que se ven (respeta el buscador)."""
        if not self.visible:
            return None
        if index not in self.visible:
            return self.visible[0]
        position = self.visible.index(index) + delta
        return self.visible[position] if 0 <= position < len(self.visible) else None

    def _query(self) -> str:
        return "" if getattr(self, "_ph", False) else self.query.get().strip().lower()

    def _filter(self):
        other = self._query()
        terms = other.split()
        extra = self.search_text or (lambda path: "")
        self.visible = [i for i, path in enumerate(self.paths)
                        if all(term in f"{Path(path).name} {extra(path)}".lower() for term in terms)]
        self._clear.pack(side=tk.RIGHT, padx=(0, SPACE["sm"])) if other else self._clear.pack_forget()
        if self.mode == "list":
            self._fill_list()
        else:
            self.grid.yview_moveto(0)
            self._draw_grid()
        self._show_empty()

    def _show_empty(self):
        if self.paths and not self.visible:
            self._empty.place(relx=0.5, rely=0.3, anchor="center")
        else:
            self._empty.place_forget()

    # ── lista ─────────────────────────────────────────────────────────────────

    def _fill_list(self, keep_view: bool = False):
        label = self.listbox
        top = label.yview()[0]
        label.delete(0, tk.END)
        for row, i in enumerate(self.visible):
            path = self.paths[i]
            sym, color = MARKS.get(self.marks.get(path, ""), ("", None))
            label.insert(tk.END, f"{sym or ' '}  {Path(path).name}")
            if color and self.marks.get(path) in ("error", "pending", "board"):
                label.itemconfig(row, fg=color)
        if keep_view:
            if self.current in self.visible:
                label.selection_set(self.visible.index(self.current))
            label.yview_moveto(top)
        else:
            self.set_current(self.current)

    def _on_list_select(self, _event):
        sel = self.listbox.curselection()
        if sel and sel[0] < len(self.visible) and self.visible[sel[0]] != self.current:
            self.on_select(self.visible[sel[0]])

    # ── cuadrícula ────────────────────────────────────────────────────────────

    def _cell(self):
        width = max(self.grid.winfo_width() - 4, THUMB + 8)
        cols = max(1, width // (THUMB + 8))
        canvas_width = width // cols
        return cols, canvas_width, THUMB + 8

    def _draw_grid(self):
        grid = self.grid
        if self.mode != "grid":
            return
        grid.delete("all")
        cols, canvas_width, canvas_height = self._cell()
        colors = self.c
        self._layout = (cols, canvas_width, canvas_height)
        for position, i in enumerate(self.visible):
            path = self.paths[i]
            x0, y0 = (position % cols) * canvas_width + 2, (position // cols) * canvas_height + 2
            sel = i == self.current
            grid.create_rectangle(x0 + 2, y0 + 2, x0 + canvas_width - 2, y0 + canvas_height - 2, width=2 if sel else 0,
                               outline=colors["accent"], fill=colors["accent_light"] if sel else colors["bg_card"],
                               tags=(f"cell{i}",))
            cx = x0 + canvas_width // 2
            thumb = self._thumbs.get(path)
            if thumb:
                grid.create_image(cx, y0 + 4 + THUMB // 2, image=thumb, tags=(f"img{i}",))
            else:
                grid.create_rectangle(cx - THUMB // 2 + 2, y0 + 8, cx + THUMB // 2 - 2, y0 + THUMB,
                                   fill=COLORS["placeholder"], width=0, tags=(f"img{i}",))
            mark = self.marks.get(path)
            if mark in MARKS:
                sym, color = MARKS[mark]
                box_x, by = x0 + canvas_width - 10, y0 + 9
                grid.create_oval(box_x - 6, by - 6, box_x + 6, by + 6, fill=color, outline=COLORS["white"], width=1.5,
                              tags=("badge",))
                grid.create_text(box_x, by, text=sym, fill=COLORS["white"], font=FONTS["tiny_bold"], tags=("badge",))
        rows = (len(self.visible) + cols - 1) // cols
        grid.config(scrollregion=(0, 0, cols * canvas_width, max(rows * canvas_height + 4, 1)))
        self._request_visible()

    def _index_at(self, event) -> int | None:
        cols, canvas_width, canvas_height = self._cell()
        x, y = self.grid.canvasx(event.x), self.grid.canvasy(event.y)
        position = int(y // canvas_height) * cols + int(x // canvas_width)
        return self.visible[position] if x // canvas_width < cols and 0 <= position < len(self.visible) else None

    def _hover(self, event):
        """El nombre de la foto aparece al pasar el mouse (en la cuadrícula no hay espacio)."""
        i = self._index_at(event)
        if i is None:
            self._tip_hide()
            return
        name = Path(self.paths[i]).name
        tip = getattr(self, "_tip", None)
        if tip is None:
            self._tip = tip = tk.Toplevel(self)
            tip.wm_overrideredirect(True)
            self._tip_lbl = tk.Label(tip, bg=COLORS["bg_tooltip"], fg=COLORS["text"], relief="solid", bd=1,
                                     font=FONTS["small"], padx=SPACE["md"], pady=SPACE["xs"])
            self._tip_lbl.pack()
        self._tip_lbl.config(text=name)
        tip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 14}")

    def _tip_hide(self):
        if getattr(self, "_tip", None) is not None:
            self._tip.destroy()
            self._tip = None

    def _touch(self, *_):
        import time
        self._touched = time.monotonic()

    def reveal(self, index: int, quiet: float = 4.0):
        """Mostrar esa foto (sin elegirla) mientras corre el lote, salvo que el usuario haya
        movido la lista hace poco."""
        import time
        if time.monotonic() - self._touched < quiet or index not in self.visible:
            return
        if self.mode == "list":
            self.listbox.see(self.visible.index(index))
        else:
            cols, _, canvas_height = self._cell()
            rows = max(1, (len(self.visible) + cols - 1) // cols)
            row = self.visible.index(index) // cols
            top, bottom = self.grid.yview()
            if not (top * rows <= row and row + 1 <= bottom * rows):
                self.grid.yview_moveto(max(0.0, (row + 1) / rows - (bottom - top)))
                self._request_visible()

    def _grid_yview(self, *args):
        self._touch()
        self.grid.yview(*args)
        self._request_visible()

    def _scroll(self, units: int):
        self.grid.yview_scroll(units, "units")
        self._request_visible()

    def _wheel(self, event):
        self._touch()
        self._scroll(-int(event.delta / 40) or (-1 if event.delta > 0 else 1))

    def _visible_range(self):
        cols, _, canvas_height = self._cell()
        top, bottom = self.grid.yview()
        total = max(1, (len(self.visible) + cols - 1) // cols) * canvas_height
        first_row, last_row = int(top * total // canvas_height), int(bottom * total // canvas_height) + 1
        return self.visible[first_row * cols:(last_row + 1) * cols]

    def _request_visible(self):
        want = [self.paths[i] for i in self._visible_range() if self.paths[i] not in self._thumbs]
        if want:
            self._loader.request(want, first=True)

    def _see(self, index: int):
        if not getattr(self, "_scroll_on", True):
            self.grid.yview_moveto(0)
            return
        if index not in self.visible:
            return
        cols, _, canvas_height = self._cell()
        row = self.visible.index(index) // cols
        rows = max(1, (len(self.visible) + cols - 1) // cols)
        top, bottom = self.grid.yview()
        if not (top * rows * canvas_height <= row * canvas_height and (row + 1) * canvas_height <= bottom * rows * canvas_height):
            self.grid.yview_moveto(max(0.0, row / rows - 0.1))
            self._request_visible()

    def _on_grid_click(self, event):
        self.grid.focus_set()
        i = self._index_at(event)
        if i is not None and i != self.current:
            self.on_select(i)

    def _poll(self):
        """Pasa al canvas las miniaturas que ya leyó el hilo."""
        try:
            while True:
                path, picture = self._loader.done.get_nowait()
                if path in self.paths:
                    self._thumbs[path] = ImageTk.PhotoImage(picture) if picture else None
                    if self.mode == "grid" and picture:
                        self._place_thumb(self.paths.index(path))
        except queue.Empty:
            pass
        self.after(120, self._poll)

    def _place_thumb(self, i: int):
        """Pone una miniatura recién leída sin redibujar toda la cuadrícula."""
        if i not in self.visible or not getattr(self, "_layout", None):
            return
        cols, canvas_width, canvas_height = self._layout
        position = self.visible.index(i)
        x0, y0 = (position % cols) * canvas_width + 2, (position // cols) * canvas_height + 2
        self.grid.delete(f"img{i}")
        self.grid.create_image(x0 + canvas_width // 2, y0 + 4 + THUMB // 2, image=self._thumbs[self.paths[i]],
                               tags=(f"img{i}",))
        self.grid.tag_raise("badge")
