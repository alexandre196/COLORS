import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

# ─── Palette & style ────────────────────────────────────────────────────────
BG        = "#0f1117"
PANEL     = "#1a1d27"
CARD      = "#22263a"
ACCENT    = "#4f8ef7"
RED_COL   = "#e05a5a"
GREEN_COL = "#4ecb71"
BLUE_COL  = "#4f8ef7"
TEXT      = "#e8eaf6"
MUTED     = "#6b7280"
BORDER    = "#2d3148"

FONT_TITLE = ("Courier New", 18, "bold")
FONT_LABEL = ("Courier New", 10, "bold")
FONT_SMALL = ("Courier New", 9)
FONT_BTN   = ("Courier New", 10, "bold")


def cv2_to_photoimage(img_bgr, size=(260, 200)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb).resize(size, Image.LANCZOS)
    return ImageTk.PhotoImage(pil)


class ColorDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HSV Color Detector")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(1050, 700)

        self.image_path      = tk.StringVar(value="")
        self.image_bgr       = None
        self._photos         = {}
        self._pending        = None
        self._pipette_target = None
        self._disp_w         = 1
        self._disp_h         = 1

        # Valeurs HSV par défaut — vert élargi pour couvrir kaki/olive
        defaults = {
            "r1_lo": [0,   120,  70],  "r1_hi": [10,  255, 255],
            "r2_lo": [170, 120,  70],  "r2_hi": [180, 255, 255],
            "g_lo":  [25,   30,  30],  "g_hi":  [95,  255, 255],
            "b_lo":  [90,   50,  70],  "b_hi":  [128, 255, 255],
        }
        self.hsv_vars = {}
        for key, vals in defaults.items():
            self.hsv_vars[key] = [tk.StringVar(value=str(v)) for v in vals]

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        header = tk.Frame(self, bg=BG, pady=18)
        header.pack(fill="x", padx=30)
        tk.Label(header, text="◈  HSV COLOR DETECTOR", font=FONT_TITLE,
                 bg=BG, fg=ACCENT).pack(side="left")
        self.pipette_lbl = tk.Label(header, text="", font=FONT_SMALL,
                                    bg=BG, fg=GREEN_COL)
        self.pipette_lbl.pack(side="right", padx=10)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        body.columnconfigure(0, weight=0, minsize=300)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=PANEL)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self._build_controls(left)

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_results(right)

    def _build_controls(self, parent):
        sec = self._section(parent, "IMAGE SOURCE")
        tf = tk.Frame(sec, bg=PANEL)
        tf.pack(fill="x", pady=(0, 6))
        tk.Entry(tf, textvariable=self.image_path, font=FONT_SMALL,
                 bg=CARD, fg=TEXT, relief="flat", bd=0,
                 insertbackground=TEXT, width=22
                 ).pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 6))
        self._btn(tf, "Parcourir", self._browse).pack(side="right")
        self._btn(sec, "▶  ANALYSER", self._run, big=True).pack(fill="x", pady=(8, 0))

        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=10)

        sec2 = self._section(parent, "PARAMÈTRES HSV")
        tk.Label(sec2, text="🖱 Clic sur l'image originale → lit la valeur HSV",
                 font=FONT_SMALL, bg=PANEL, fg=MUTED,
                 wraplength=260, justify="left").pack(fill="x", pady=(0, 6))

        nb = ttk.Notebook(sec2)
        nb.pack(fill="both", expand=True)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",     background=PANEL, borderwidth=0)
        style.configure("TNotebook.Tab", background=CARD, foreground=MUTED,
                        font=FONT_SMALL, padding=[10, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#fff")])

        colors_cfg = [
            ("Rouge", RED_COL, [
                ("Plage 1 — inf", "r1_lo"), ("Plage 1 — sup", "r1_hi"),
                ("Plage 2 — inf", "r2_lo"), ("Plage 2 — sup", "r2_hi"),
            ]),
            ("Vert", GREEN_COL, [("Inf", "g_lo"), ("Sup", "g_hi")]),
            ("Bleu", BLUE_COL,  [("Inf", "b_lo"), ("Sup", "b_hi")]),
        ]
        for name, col, params in colors_cfg:
            frame = tk.Frame(nb, bg=CARD)
            nb.add(frame, text=f"  {name}  ")
            for label, key in params:
                self._hsv_row(frame, label, key, col)
            # Boutons pipette
            pf = tk.Frame(frame, bg=CARD, pady=8)
            pf.pack(fill="x", padx=10)
            lo_key = params[0][1]
            hi_key = params[-1][1]
            self._pip_btn(pf, "⊙ Pipette → Inf", lo_key, col).pack(side="left", padx=(0, 6))
            self._pip_btn(pf, "⊙ Pipette → Sup", hi_key, col).pack(side="left")

        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=8)
        self.pixel_var = tk.StringVar(value="HSV pixel : —")
        tk.Label(parent, textvariable=self.pixel_var, font=FONT_SMALL,
                 bg=PANEL, fg=GREEN_COL, anchor="w", padx=14).pack(fill="x")

    def _hsv_row(self, parent, label, key, col):
        row = tk.Frame(parent, bg=CARD, pady=4)
        row.pack(fill="x", padx=10)
        tk.Label(row, text=label, font=FONT_SMALL, bg=CARD,
                 fg=MUTED, width=14, anchor="w").pack(side="left")
        for i, ch in enumerate(["H", "S", "V"]):
            tk.Label(row, text=ch, font=FONT_SMALL, bg=CARD,
                     fg=col, width=2).pack(side="left")
            tk.Entry(row, textvariable=self.hsv_vars[key][i],
                     font=FONT_SMALL, bg=BG, fg=TEXT,
                     relief="flat", bd=0, width=4,
                     insertbackground=TEXT, justify="center"
                     ).pack(side="left", padx=(0, 6), ipady=4)

    def _pip_btn(self, parent, text, key, col):
        b = tk.Label(parent, text=text, font=FONT_SMALL,
                     bg=CARD, fg=col, cursor="crosshair",
                     relief="solid", bd=1, padx=6, pady=4)
        b.bind("<Button-1>", lambda e: self._activate_pipette(key, col))
        return b

    def _build_results(self, parent):
        parent.columnconfigure((0, 1), weight=1, uniform="col")
        parent.rowconfigure((0, 1), weight=1, uniform="row")

        titles  = ["Image originale", "Rouge détecté", "Vert détecté", "Bleu détecté"]
        accents = [ACCENT, RED_COL, GREEN_COL, BLUE_COL]
        self.img_labels = {}

        for idx, (title, col) in enumerate(zip(titles, accents)):
            r, c = divmod(idx, 2)
            card = tk.Frame(parent, bg=CARD)
            card.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
            card.rowconfigure(2, weight=1)
            card.columnconfigure(0, weight=1)

            tk.Frame(card, bg=col, height=3).grid(row=0, column=0, sticky="ew")
            tk.Label(card, text=title, font=FONT_LABEL,
                     bg=CARD, fg=col, pady=6).grid(row=1, column=0)

            cur = "crosshair" if idx == 0 else "arrow"
            lbl = tk.Label(card, bg=CARD, text="— aucune image —",
                           font=FONT_SMALL, fg=MUTED, cursor=cur)
            lbl.grid(row=2, column=0, padx=8, pady=(0, 8), sticky="nsew")
            self.img_labels[idx] = lbl

            if idx == 0:
                lbl.bind("<Motion>",   self._on_img_motion)
                lbl.bind("<Button-1>", self._on_img_click)

        self.status_var = tk.StringVar(value="Chargez une image et cliquez sur Analyser.")
        bar = tk.Frame(parent, bg=PANEL)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        tk.Label(bar, textvariable=self.status_var, font=FONT_SMALL,
                 bg=PANEL, fg=MUTED, anchor="w", padx=10).pack(fill="x", ipady=5)

    # ── Helpers UI ────────────────────────────────────────────────────────────
    def _section(self, parent, title):
        tk.Frame(parent, bg=PANEL, height=10).pack(fill="x")
        tk.Label(parent, text=title, font=FONT_SMALL, bg=PANEL,
                 fg=MUTED, anchor="w", padx=12).pack(fill="x")
        frame = tk.Frame(parent, bg=PANEL, padx=12, pady=4)
        frame.pack(fill="x")
        return frame

    def _btn(self, parent, text, cmd, color=ACCENT, big=False):
        f    = FONT_BTN if big else FONT_SMALL
        pady = 10 if big else 6
        b = tk.Label(parent, text=text, font=f, bg=color, fg="#fff",
                     pady=pady, cursor="hand2", relief="flat")
        b.bind("<Button-1>", lambda e: cmd())
        b.bind("<Enter>",    lambda e: b.configure(bg=self._darken(color)))
        b.bind("<Leave>",    lambda e: b.configure(bg=color))
        return b

    @staticmethod
    def _darken(hex_col, factor=0.85):
        r, g, b = int(hex_col[1:3],16), int(hex_col[3:5],16), int(hex_col[5:7],16)
        return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"

    # ── Pipette ───────────────────────────────────────────────────────────────
    def _activate_pipette(self, key, col):
        if self.image_bgr is None:
            messagebox.showinfo("Info", "Analysez d'abord une image.")
            return
        self._pipette_target = key
        self.pipette_lbl.configure(
            text=f"⊙ Cliquez sur l'image originale → {key}", fg=col)

    def _img_coords(self, event):
        if self.image_bgr is None or self._disp_w <= 1:
            return None, None
        lbl  = self.img_labels[0]
        lw, lh = lbl.winfo_width(), lbl.winfo_height()
        off_x = (lw - self._disp_w) // 2
        off_y = (lh - self._disp_h) // 2
        px = int((event.x - off_x) / self._disp_w * self.image_bgr.shape[1])
        py = int((event.y - off_y) / self._disp_h * self.image_bgr.shape[0])
        H, W = self.image_bgr.shape[:2]
        if 0 <= px < W and 0 <= py < H:
            return px, py
        return None, None

    def _on_img_motion(self, event):
        if self.image_bgr is None:
            return
        px, py = self._img_coords(event)
        if px is None:
            self.pixel_var.set("HSV pixel : —")
            return
        bgr = self.image_bgr[py, px]
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        self.pixel_var.set(f"HSV pixel : H={hsv[0]}  S={hsv[1]}  V={hsv[2]}")

    def _on_img_click(self, event):
        if self._pipette_target is None or self.image_bgr is None:
            return
        px, py = self._img_coords(event)
        if px is None:
            return
        bgr = self.image_bgr[py, px]
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        key = self._pipette_target
        for i, v in enumerate(hsv):
            self.hsv_vars[key][i].set(str(int(v)))
        self.pipette_lbl.configure(
            text=f"✓ {key} ← H={hsv[0]} S={hsv[1]} V={hsv[2]}", fg=GREEN_COL)
        self._pipette_target = None
        self.after(2500, lambda: self.pipette_lbl.configure(text=""))

    # ── Analyser ─────────────────────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("Tous", "*.*")])
        if path:
            self.image_path.set(path)

    def _run(self):
        path = self.image_path.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Erreur", "Veuillez sélectionner un fichier image valide.")
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Erreur", "Impossible de lire l'image.")
            return
        self.image_bgr = img

        def get(key):
            try:
                return np.array([int(v.get()) for v in self.hsv_vars[key]])
            except ValueError:
                messagebox.showerror("Erreur", f"Valeurs HSV invalides pour '{key}'.")
                return None

        vals = {k: get(k) for k in self.hsv_vars}
        if any(v is None for v in vals.values()):
            return

        hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_r = (cv2.inRange(hsv, vals["r1_lo"], vals["r1_hi"]) |
                  cv2.inRange(hsv, vals["r2_lo"], vals["r2_hi"]))
        mask_g = cv2.inRange(hsv, vals["g_lo"], vals["g_hi"])
        mask_b = cv2.inRange(hsv, vals["b_lo"], vals["b_hi"])

        res_r = cv2.bitwise_and(img, img, mask=mask_r)
        res_g = cv2.bitwise_and(img, img, mask=mask_g)
        res_b = cv2.bitwise_and(img, img, mask=mask_b)

        self._pending = [img, res_r, res_g, res_b]
        self.after(50, self._display_images)

        h, w = img.shape[:2]
        pct_r = mask_r.sum() // 255 / (h * w) * 100
        pct_g = mask_g.sum() // 255 / (h * w) * 100
        pct_b = mask_b.sum() // 255 / (h * w) * 100
        self.status_var.set(
            f"Image : {w}×{h}px  |  Rouge : {pct_r:.1f}%  "
            f"Vert : {pct_g:.1f}%  Bleu : {pct_b:.1f}%")

    def _display_images(self):
        images = self._pending
        src_h, src_w = images[0].shape[:2]
        src_ratio = src_w / src_h

        lbl0   = self.img_labels[0]
        cell_w = max(240, lbl0.winfo_width())
        cell_h = max(180, lbl0.winfo_height())

        if cell_w / cell_h > src_ratio:
            fit_h = cell_h - 4
            fit_w = int(fit_h * src_ratio)
        else:
            fit_w = cell_w - 4
            fit_h = int(fit_w / src_ratio)

        fit_w = max(fit_w, 100)
        fit_h = max(fit_h, 80)
        self._disp_w, self._disp_h = fit_w, fit_h

        for idx, im in enumerate(images):
            photo = cv2_to_photoimage(im, (fit_w, fit_h))
            self._photos[idx] = photo
            self.img_labels[idx].configure(image=photo, text="")


if __name__ == "__main__":
    app = ColorDetectorApp()
    app.mainloop()