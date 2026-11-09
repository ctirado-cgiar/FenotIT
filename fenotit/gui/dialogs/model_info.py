"""Ficha de un modelo de IA: para qué sirve, propietario, licencia, fuente, cita y nombres de
las clases. Se llena al agregar un modelo; los de la aplicación solo se leen."""
import tkinter as tk

from fenotit.gui.dialogs.scale import BaseDialog
from fenotit.gui.style.components import button
from fenotit.gui.style.theme import COLORS, FONTS, SPACE
from fenotit.i18n import t

# (clave, alto en líneas; 1 = una línea)
FIELDS = (("name", 1), ("description", 3), ("owner", 1), ("license", 1), ("source", 1), ("citation", 2))
REQUIRED = ("name", "description", "owner", "license")


class ModelInfoDialog(BaseDialog):
    """`on_save(cambios)` guarda; devuelve None o el texto de un error. Sin `on_save`, solo
    lectura. `on_share(ventana)` guarda el modelo como paquete para otro equipo."""
    W, H = 520, 640

    def __init__(self, parent, model: dict, on_save=None, on_share=None):
        self._model, self._on_save, self._on_share = model, on_save, on_share
        super().__init__(parent, t("minfo.title"))
        bg = COLORS["bg_card"]
        wrap = self.W - 70
        if on_save:
            tk.Label(self.body, text=t("minfo.intro"), bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                     justify="left", wraplength=wrap, anchor="w").pack(fill=tk.X, pady=(0, SPACE["sm"]))
        facts = [t(f"model.out.{model['output']}", n=len(model["classes"])) + "."]
        if model.get("added"):
            facts.append(t("minfo.added", date=model["added"])
                         + (f" ({t('minfo.converted', tool=model['tool'])})" if model.get("tool") else "") + ".")
        tk.Label(self.body, text=" ".join(facts), bg=bg, fg=COLORS["text_muted"], font=FONTS["small"],
                 justify="left", wraplength=wrap, anchor="w").pack(fill=tk.X, pady=(0, SPACE["md"]))
        self._widgets = {}
        for key, lines in FIELDS + ((("classes", 4),) if model["output"] == "boxes" else ()):
            mark = " *" if on_save and key in REQUIRED else ""
            tk.Label(self.body, text=t(f"minfo.{key}") + mark, bg=bg, fg=COLORS["text"], font=FONTS["small_bold"],
                     anchor="w").pack(fill=tk.X, pady=(SPACE["sm"], 0))
            value = "\n".join(model["classes"]) if key == "classes" else model.get(key, "")
            box = tk.Text(self.body, height=lines, font=FONTS["body"], relief="flat", bg=COLORS["bg"],
                          fg=COLORS["text"], wrap="word", padx=SPACE["sm"], pady=SPACE["xs"], highlightthickness=1,
                          highlightbackground=COLORS["border"], highlightcolor=COLORS["accent"])
            box.insert("1.0", value)
            if not on_save:
                box.config(state=tk.DISABLED)
            box.pack(fill=tk.X)
            self._widgets[key] = box
        if model["output"] == "boxes":
            tk.Label(self.body, text=t("minfo.classes_help", n=len(model["classes"])), bg=bg,
                     fg=COLORS["text_muted"], font=FONTS["small"], justify="left", wraplength=wrap,
                     anchor="w").pack(fill=tk.X)
        self._error = tk.Label(self.body, text="", bg=bg, fg=COLORS["warning_text"], font=FONTS["small"],
                               anchor="w", justify="left", wraplength=wrap)
        self._error.pack(fill=tk.X, pady=(SPACE["sm"], 0))
        self.update_idletasks()                           # alto según el contenido, dentro de la pantalla
        h = min(self.winfo_reqheight(), self.winfo_screenheight() - 80)
        self.geometry(f"{self.W}x{h}+{(self.winfo_screenwidth() - self.W) // 2}+{max(0, (self.winfo_screenheight() - h) // 2 - 20)}")

    def _build_buttons(self, parent):
        if self._on_share:
            button(parent, text=t("minfo.share"), command=lambda: self._on_share(self), kind="secondary").pack(
                side=tk.LEFT)
        if self._on_save:
            button(parent, text=t("common.save"), command=self._save, kind="primary").pack(side=tk.RIGHT)
            button(parent, text=t("common.cancel"), command=self.destroy, kind="quiet").pack(
                side=tk.RIGHT, padx=(0, SPACE["md"]))
        else:
            button(parent, text=t("common.close"), command=self.destroy, kind="secondary").pack(side=tk.RIGHT)

    def values(self) -> dict:
        out = {k: w.get("1.0", "end").strip() for k, w in self._widgets.items()}
        if "classes" in out:
            out["classes"] = [ln.strip() for ln in out["classes"].splitlines() if ln.strip()]
        return out

    def _save(self):
        v = self.values()
        empty = [t(f"minfo.{k}") for k in REQUIRED if not v.get(k)]
        if empty:
            self._error.config(text=t("minfo.required", fields=", ".join(empty)))
            return
        if "classes" in v and len(v["classes"]) != len(self._model["classes"]):
            self._error.config(text=t("det.names_count", got=len(v["classes"]), n=len(self._model["classes"])))
            return
        err = self._on_save(v)
        if err:
            self._error.config(text=err)
            return
        self.destroy()
