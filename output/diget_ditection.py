import cv2
import numpy as np

# טען את התמונה עם הפינג
frame = cv2.imread("internal_20250928_122843.png", cv2.IMREAD_GRAYSCALE)

# חתוך את האזור שבו מוצג המספר (נניח ידוע)
roi = frame[30:60, 820:880]  # דוגמה - להתאים למיקום שלך

# טען תבניות של כל ספרה
templates = {str(i): cv2.imread(f"digits/{i}.png", cv2.IMREAD_GRAYSCALE) for i in range(10)}

# חפש כל ספרה בתור תבנית
results = {}
for digit, tmpl in templates.items():
    res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    results[digit] = max_val

best_digit = max(results, key=results.get)
print("Detected digit:", best_digit)
import cv2
import numpy as np
from collections import deque

def region_grow_same_color(img_bgr, seed_xy, tol=25, min_area=20, connectivity=8):
    x0, y0 = seed_xy  # seed: (x,y)
    h, w = img_bgr.shape[:2]
    seed = img_bgr[y0, x0].astype(np.int16)

    # מסכה של פיקסלים דומים בצבע (טולרנס באיבי־ג'י־אר; אפשר גם HSV)
    diff = np.abs(img_bgr.astype(np.int16) - seed[None, None, :]).sum(axis=2)
    similar = (diff <= tol).astype(np.uint8)

    # רכיבים מחוברים על המסכה
    num, labels, stats, _ = cv2.connectedComponentsWithStats(similar, connectivity)
    # מצא את תווית הרכיב שמכיל את הפיקסל־זרע
    seed_label = labels[y0, x0]
    # שמור רק אם מספיק גדול
    keep = (labels == seed_label) & (stats[seed_label, cv2.CC_STAT_AREA] >= min_area)

    out = np.zeros_like(img_bgr)
    out[keep] = img_bgr[keep]
    return out, keep.astype(np.uint8)*255