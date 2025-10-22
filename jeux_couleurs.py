import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# === Charger l'image ===
image = cv2.imread("image1.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# === Définir les plages de couleurs (HSV) ===
color_ranges = {
    "Rouge": [(np.array([0, 120, 70]), np.array([10, 255, 255])),
              (np.array([170, 120, 70]), np.array([180, 255, 255]))],
    "Vert": [(np.array([36, 50, 70]), np.array([89, 255, 255]))],
    "Bleu": [(np.array([90, 50, 70]), np.array([128, 255, 255]))]
}

# === Choisir une couleur au hasard ===
target_color = random.choice(list(color_ranges.keys()))
print(f"👉 Trouve le {target_color.upper()} dans l'image !")

# === Fonction de vérification du clic ===
def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    
    x, y = int(event.xdata), int(event.ydata)
    pixel_hsv = hsv[y, x]

    found = False
    for lower, upper in color_ranges[target_color]:
        if cv2.inRange(np.uint8([[pixel_hsv]]), lower, upper) == 255:
            found = True
            break
    
    if found:
        print("🎉 Bravo ! Tu as trouvé la bonne couleur !")
    else:
        print("❌ Raté ! Essaie encore.")

# === Afficher l'image et attendre le clic ===
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"Clique sur : {target_color}")
plt.axis("off")

cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()
