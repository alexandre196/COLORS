import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Charger l'image ===
image = cv2.imread("colors.jpg")  #  chemin de l'image
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# === Définir les plages HSV pour chaque couleur ===

# Rouge
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Vert
lower_green = np.array([36, 50, 70])
upper_green = np.array([89, 255, 255])

# Bleu
lower_blue = np.array([90, 50, 70])
upper_blue = np.array([128, 255, 255])

# === Créer les masques ===
mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# === Appliquer les masques ===
res_red = cv2.bitwise_and(image, image, mask=mask_red)
res_green = cv2.bitwise_and(image, image, mask=mask_green)
res_blue = cv2.bitwise_and(image, image, mask=mask_blue)

# === Affichage avec matplotlib /// avec Réglage de la taille de la fenêtre ===
plt.figure(figsize=(11, 7))

# Image originale
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Image de Référence")
plt.axis("off")

# Rouge détecté
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(res_red, cv2.COLOR_BGR2RGB))
plt.title("Rouge détecté")
plt.axis("off")

# Vert détecté
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(res_green, cv2.COLOR_BGR2RGB))
plt.title("Vert détecté")
plt.axis("off")

# Bleu détecté
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(res_blue, cv2.COLOR_BGR2RGB))
plt.title("Bleu détecté")
plt.axis("off")

plt.tight_layout()
plt.show()
