import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image DICOM
ds = pydicom.dcmread("genou001.dcm")  # remplace par ton fichier
img = ds.pixel_array

# Normalisation et conversion en uint8
img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
img_norm = np.uint8(img_norm)

# Contraste
img_eq = cv2.equalizeHist(img_norm)

# Contours
edges = cv2.Canny(img_eq, 50, 150)

# Seuillage
_, thresh = cv2.threshold(img_eq, 127, 255, cv2.THRESH_BINARY)

# Affichage
plt.figure(figsize=(15, 9))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Réference")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(img_eq, cmap="gray")
plt.title("Contraste")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(edges, cmap="gray")
plt.title("Contours")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(thresh, cmap="gray")
plt.title("Seuillage")
plt.axis("off")

plt.tight_layout()
plt.show()
