import cv2
import cv2.aruco as aruco

# Choose a dictionary
dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

# IDs to generate
marker_ids = [1, 2, 3,
              4, 5, 6, 
              7, 8, 9,
              10, 11, 12,
              13, 14, 15]  # match your marker_to_book mapping
marker_size = 200  # pixels

names = ["03_KWJ", "11_PSJ", "12_JSEJ", 
        "16_SSMG", "14_CCWJ", "13_DGJ",
        "06_JHHRJ", "19_JWCJ", "17_HBJ",
        "05_OGJJ", "07_SCJ", "10_BJBJ",
        "08_SGYS_M", "09_GBUJ", "18_SGYS_S"]

for marker_id, name in zip(marker_ids, names):
    marker_image = aruco.generateImageMarker(dictionary, marker_id, marker_size)
    filename = f"{name}.png"
    cv2.imwrite(filename, marker_image)
    print(f"Saved {filename}")
