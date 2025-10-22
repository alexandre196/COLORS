import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Charger l'image en niveaux de gris ===
image = cv2.imread("genoux_01.png", cv2.IMREAD_GRAYSCALE)

# === Améliorer le contraste ===
equalized = cv2.equalizeHist(image)

# === Détection des contours ===
edges = cv2.Canny(equalized, 50, 150)

# === Seuillage (binaire) ===
_, thresh = cv2.threshold(equalized, 127, 255, cv2.THRESH_BINARY)

# === Affichage ===
plt.figure(figsize=(10, 7))

# Image originale
plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Image originale")
plt.axis("off")

# Contraste amélioré
plt.subplot(2, 2, 2)
plt.imshow(equalized, cmap="gray")
plt.title("Contraste amélioré")
plt.axis("off")

# Contours détectés
plt.subplot(2, 2, 3)
plt.imshow(edges, cmap="gray")
plt.title("Contours (Canny)")
plt.axis("off")

# Seuillage
plt.subplot(2, 2, 4)
plt.imshow(thresh, cmap="gray")
plt.title("Seuillage binaire")
plt.axis("off")

plt.tight_layout()
plt.show()
