"""Lupa que sigue al cursor sobre un ZoomableCanvas: muestra ampliada la zona bajo el cursor,
con una cruz en el punto exacto, para marcar puntos con precisión."""
import tkinter as tk

import cv2
from PIL import Image, ImageDraw, ImageTk

# Valores del tema (fenotit/assets/themes): COLORS (colores).
from fenotit.gui.style.theme import COLORS


class Loupe:
    """Círculo que se dibuja en el mismo canvas, debajo y a la derecha del cursor; cerca de un
    borde pasa al otro lado para quedar siempre dentro del canvas y sin tapar el punto que se marca."""

    SIZE = 96                # diámetro del círculo (px de pantalla)
    MAGNIFICATION = 4        # aumento sobre el zoom actual del canvas
    MIN_PIXEL_SIZE = 1.5     # tamaño mínimo en pantalla de un píxel de la foto dentro de la lupa
    GAP = 18                 # separación entre el cursor y el círculo
    CROSS_GAP = 6            # hueco de la cruz alrededor del centro
    TAG = "loupe"

    def __init__(self, canvas):
        self.canvas = canvas
        self._photo = None
        canvas.bind("<Motion>", self._on_motion, add="+")
        canvas.bind("<Leave>", lambda event: self.hide(), add="+")
        canvas.bind("<B1-Motion>", lambda event: self.hide(), add="+")

    def _circle_mask(self):
        mask = Image.new("L", (self.SIZE, self.SIZE), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, self.SIZE - 1, self.SIZE - 1), fill=255)
        return mask

    def hide(self):
        self.canvas.delete(self.TAG)

    def _on_motion(self, event):
        image = self.canvas.image
        self.hide()
        if image is None:
            return
        image_x, image_y = self.canvas.to_image(event.x, event.y)
        height, width = image.shape[:2]
        if not (0 <= image_x < width and 0 <= image_y < height):
            return
        screen_per_pixel = max(self.canvas.zoom * self.MAGNIFICATION, self.MIN_PIXEL_SIZE)
        half_span = self.SIZE / screen_per_pixel / 2          # media zona de la foto que entra en la lupa
        left, top = int(round(image_x - half_span)), int(round(image_y - half_span))
        span = max(1, int(round(2 * half_span)))
        crop = cv2.copyMakeBorder(image[max(top, 0):top + span, max(left, 0):left + span],
                                  max(-top, 0), max(top + span - height, 0),
                                  max(-left, 0), max(left + span - width, 0), cv2.BORDER_CONSTANT, value=(40, 40, 40))
        sharp_pixels = screen_per_pixel >= 3                  # píxeles nítidos cuando ya se distinguen uno a uno
        enlarged = cv2.resize(crop, (self.SIZE, self.SIZE),
                              interpolation=cv2.INTER_NEAREST if sharp_pixels else cv2.INTER_LINEAR)
        round_view = Image.fromarray(cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB))
        round_view.putalpha(self._circle_mask())
        self._photo = ImageTk.PhotoImage(round_view)

        box_left = event.x + self.GAP
        box_top = event.y + self.GAP
        if box_left + self.SIZE > self.canvas.winfo_width():
            box_left = event.x - self.GAP - self.SIZE
        if box_top + self.SIZE > self.canvas.winfo_height():
            box_top = event.y - self.GAP - self.SIZE
        center_x, center_y = box_left + self.SIZE / 2, box_top + self.SIZE / 2
        canvas, tag = self.canvas, self.TAG
        canvas.create_image(box_left, box_top, image=self._photo, anchor=tk.NW, tags=tag)
        canvas.create_oval(box_left, box_top, box_left + self.SIZE, box_top + self.SIZE,
                           outline=COLORS["white"], width=3, tags=tag)
        canvas.create_oval(box_left, box_top, box_left + self.SIZE, box_top + self.SIZE,
                           outline=COLORS["accent"], width=1, tags=tag)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            start = (center_x + dx * self.CROSS_GAP, center_y + dy * self.CROSS_GAP)
            end = (center_x + dx * self.SIZE / 2, center_y + dy * self.SIZE / 2)
            canvas.create_line(*start, *end, fill=COLORS["highlight"], width=1, tags=tag)
