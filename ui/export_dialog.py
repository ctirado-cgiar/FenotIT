"""
ui/export_dialog.py
Diálogo de exportación de FenotIT.
Permite elegir qué exportar: análisis individual o todo.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from ui.theme import COLORS, FONTS


def _assets() -> Path:
    return Path(__file__).parent.parent / "assets"


class ExportDialog(tk.Toplevel):
    W, H = 480, 400

    def __init__(self, parent,
                 available_analyses: list[str],
                 exporter,
                 output_root: str | None = None):
        super().__init__(parent)
        self.title("Exportar resultados")
        self.configure(bg=COLORS["bg_card"])
        self.resizable(False, False)
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(
            f"{self.W}x{self.H}+{(sw-self.W)//2}+{(sh-self.H)//2}")
        try:
            self.iconbitmap(str(_assets() / "logo.ico"))
        except Exception:
            pass
        self.grab_set()

        self._exporter      = exporter
        self._output_root   = output_root
        self._available     = available_analyses

        tk.Frame(self, bg=COLORS["accent"], height=4).pack(fill=tk.X)

        body = tk.Frame(self, bg=COLORS["bg_card"])
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)

        tk.Label(body, text="Exportar resultados",
                 bg=COLORS["bg_card"], fg=COLORS["accent"],
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")

        tk.Frame(body, bg=COLORS["border"],
                 height=1).pack(fill=tk.X, pady=8)

        # Carpeta de salida
        folder_f = tk.Frame(body, bg=COLORS["bg_card"])
        folder_f.pack(fill=tk.X, pady=4)
        tk.Label(folder_f, text="Carpeta de salida:",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=FONTS["body"], width=18,
                 anchor="w").pack(side=tk.LEFT)
        self._folder_var = tk.StringVar(value=output_root or "")
        tk.Entry(folder_f, textvariable=self._folder_var,
                 width=24, bg=COLORS["bg_panel"],
                 fg=COLORS["text"], relief="solid", bd=1,
                 font=FONTS["small"]).pack(side=tk.LEFT, padx=4)
        tk.Button(folder_f, text="…",
                  command=self._pick_folder,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", cursor="hand2",
                  font=FONTS["body"]).pack(side=tk.LEFT)

        tk.Frame(body, bg=COLORS["border"],
                 height=1).pack(fill=tk.X, pady=6)

        # Qué exportar
        tk.Label(body, text="¿Qué exportar?",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0,4))

        self._export_all_var = tk.BooleanVar(value=True)
        tk.Radiobutton(body,
                       text="Todo (Excel con pestañas + CSVs individuales)",
                       variable=self._export_all_var, value=True,
                       command=self._on_mode_change,
                       bg=COLORS["bg_card"], fg=COLORS["text"],
                       selectcolor=COLORS["bg_panel"],
                       activebackground=COLORS["bg_card"],
                       font=FONTS["body"]).pack(anchor="w")

        tk.Radiobutton(body,
                       text="Solo un análisis específico (CSV)",
                       variable=self._export_all_var, value=False,
                       command=self._on_mode_change,
                       bg=COLORS["bg_card"], fg=COLORS["text"],
                       selectcolor=COLORS["bg_panel"],
                       activebackground=COLORS["bg_card"],
                       font=FONTS["body"]).pack(anchor="w", pady=(2,0))

        # Selector de análisis individual
        sel_f = tk.Frame(body, bg=COLORS["bg_card"])
        sel_f.pack(fill=tk.X, pady=4)
        tk.Label(sel_f, text="  Análisis:",
                 bg=COLORS["bg_card"], fg=COLORS["text_muted"],
                 font=FONTS["small"]).pack(side=tk.LEFT)
        self._analysis_var = tk.StringVar(
            value=available_analyses[0] if available_analyses else "")
        self._analysis_combo = ttk.Combobox(
            sel_f, textvariable=self._analysis_var,
            values=available_analyses,
            state="disabled", width=22,
            font=FONTS["body"])
        self._analysis_combo.pack(side=tk.LEFT, padx=6)

        tk.Frame(body, bg=COLORS["border"],
                 height=1).pack(fill=tk.X, pady=6)

        # Opciones adicionales
        tk.Label(body, text="Opciones:",
                 bg=COLORS["bg_card"], fg=COLORS["text"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0,4))

        self._excel_var    = tk.BooleanVar(value=True)
        self._img_var      = tk.BooleanVar(value=True)
        self._resumen_var  = tk.BooleanVar(value=True)

        for var, label in [
            (self._excel_var,   "Generar Excel con pestañas"),
            (self._img_var,     "Guardar imágenes de resultados"),
            (self._resumen_var, "Incluir CSV de resumen (medias por imagen)"),
        ]:
            tk.Checkbutton(body, variable=var, text=label,
                           bg=COLORS["bg_card"], fg=COLORS["text"],
                           selectcolor=COLORS["bg_panel"],
                           activebackground=COLORS["bg_card"],
                           font=FONTS["small"]).pack(anchor="w")

        # Log de resultado
        self._log_var = tk.StringVar(value="")
        tk.Label(body, textvariable=self._log_var,
                 bg=COLORS["bg_card"], fg=COLORS["accent2"],
                 font=FONTS["small"], anchor="w",
                 wraplength=420).pack(fill=tk.X, pady=(8,0))

        # Botones
        tk.Frame(self, bg=COLORS["border"],
                 height=1).pack(fill=tk.X)
        btn_row = tk.Frame(self, bg=COLORS["bg_card"])
        btn_row.pack(fill=tk.X, padx=20, pady=10)

        tk.Button(btn_row, text="Cerrar",
                  command=self.destroy,
                  bg=COLORS["btn_bg"], fg=COLORS["accent"],
                  relief="flat", font=FONTS["body"],
                  cursor="hand2", padx=12).pack(side=tk.RIGHT, padx=4)

        tk.Button(btn_row, text="💾  Exportar",
                  command=self._do_export,
                  bg=COLORS["accent"], fg="#FFFFFF",
                  relief="flat",
                  font=("Segoe UI", 9, "bold"),
                  cursor="hand2", padx=12).pack(side=tk.RIGHT, padx=4)

    def _pick_folder(self):
        d = filedialog.askdirectory(
            title="Carpeta de exportación", parent=self)
        if d:
            self._folder_var.set(d)

    def _on_mode_change(self):
        if self._export_all_var.get():
            self._analysis_combo.config(state="disabled")
        else:
            self._analysis_combo.config(state="readonly")

    def _do_export(self):
        folder = self._folder_var.get()
        if not folder:
            messagebox.showwarning("Sin carpeta",
                                   "Selecciona una carpeta de salida.",
                                   parent=self)
            return

        if self._exporter is None:
            messagebox.showwarning("Sin datos",
                                   "No hay resultados para exportar.",
                                   parent=self)
            return

        # Actualizar output_root del exporter
        from pathlib import Path as _Path
        self._exporter.output_root = _Path(folder)
        self._exporter.results_dir = _Path(folder) / "resultados"

        try:
            if self._export_all_var.get():
                generated = self._exporter.export_all(
                    generate_excel=self._excel_var.get())
            else:
                name = self._analysis_var.get()
                generated = self._exporter.export_all(
                    analyses_to_include=[name],
                    generate_excel=self._excel_var.get())

            # Mostrar resultado
            files = [Path(v).name for v in generated.values()
                     if isinstance(v, str) and Path(v).exists()]
            msg = f"✓ Exportado en:\n{folder}\n\n" + \
                  "\n".join(f"  • {f}" for f in files)
            self._log_var.set(
                f"✓ {len(files)} archivos generados en {folder}")
            messagebox.showinfo("Exportación completada",
                                msg, parent=self)

        except Exception as e:
            self._log_var.set(f"✗ Error: {e}")
            messagebox.showerror("Error al exportar",
                                 str(e), parent=self)
