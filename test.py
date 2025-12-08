# import cv2
# import numpy as np
# import os  # <-- 이 줄이 꼭 필요합니다!

# # 1. 현재 이 파이썬 파일(main.py)이 있는 진짜 위치를 알아냄
# current_path = os.path.dirname(os.path.abspath(__file__))

# # 2. 이미지 파일명 설정 (파일 확장자가 png인지 jpg인지 꼭 확인하세요!)
# # 만약 파일이 jpg라면 .png를 .jpg로 바꿔주세요.
# targets = {
#     'Circle': 'circle.png',
#     'Triangle': 'triangle.png',
#     'Rectangle': 'rectangle.png',
#     'Star': 'star.png'
# }

# # ORB 검출기 생성
# orb = cv2.ORB_create()
# reference_features = {}

# print(f"Current working file: {current_path}") # 확인용 출력

# for name, filename in targets.items():
#     # 경로를 합쳐서 절대 경로를 만듦 (예: /Users/pyo/project/circle.png)
#     full_path = os.path.join(current_path, filename)
    
#     img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print(f"⚠️ Warning: Can't find file -> {full_path}")
#         # 혹시 확장자가 다를 수 있으니 확인하라는 메시지
#         continue
    
#     kp, des = orb.detectAndCompute(img, None)
#     reference_features[name] = des

# print("Image Load Finish!")

# # Variables to manage the stack state
# book_stack = []
# last_detected_book = None
# stability_counter = 0  # To prevent flickering

# # --- Camera Setup ---
# # 0 is usually the built-in MacBook cam. 
# # Your ABKO camera might be index 1 or 2. Change this number if needed.
# cap = cv2.VideoCapture(0) 

# print("Camera started. Press 'q' to quit.")
# # --- REPLACEMENT CODE FOR THE WHILE LOOP ---

# # --- SETTINGS ---
# MIN_MATCH_COUNT = 25      # Increased from 8 to 25 (Ignores noise)
# REQUIRED_FRAMES = 20      # Must see the same book for 20 frames to confirm

# # --- STATE VARIABLES ---
# current_stable_book = None
# frame_counter = 0
# candidate_book = None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

#     best_match_name = None
#     max_good_matches = 0

#     # 1. MATCHING PROCESS
#     if des_frame is not None:
#         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#         for name, des_ref in reference_features.items():
#             if des_ref is None: continue
            
#             matches = bf.match(des_ref, des_frame)
#             matches = sorted(matches, key=lambda x: x.distance)
            
#             # Stricter distance filter (only very similar points)
#             good_matches = [m for m in matches if m.distance < 60]
            
#             if len(good_matches) > max_good_matches:
#                 max_good_matches = len(good_matches)
#                 best_match_name = name

#     # 2. DEBUG VISUALS
#     # Draw green dots to see what the camera is looking at
#     cv2.drawKeypoints(frame, kp_frame, frame, color=(0,255,0), flags=0)
    
#     # Show the raw numbers (Helpful for tuning)
#     status_text = f"Raw: {best_match_name} ({max_good_matches})"
#     color = (0, 0, 255) # Red (Bad match)
    
#     # 3. STABILITY LOGIC (The Anti-Flicker Filter)
#     if max_good_matches >= MIN_MATCH_COUNT:
#         color = (0, 255, 255) # Yellow (Potential match)
        
#         if best_match_name == candidate_book:
#             # If it's the same book as the last frame, count up
#             frame_counter += 1
#         else:
#             # If it changed, reset counter
#             candidate_book = best_match_name
#             frame_counter = 0
            
#         # If we saw the same book enough times...
#         if frame_counter > REQUIRED_FRAMES:
#             color = (0, 255, 0) # Green (Confirmed!)
            
#             # Only add to stack if it's different from the *last confirmed* book
#             if candidate_book != current_stable_book:
#                 current_stable_book = candidate_book
#                 book_stack.append(current_stable_book)
#                 print(f"Confirmed Book: {current_stable_book}")
                
#             status_text = f"CONFIRMED: {current_stable_book}"
            
#     else:
#         # If matches drop below threshold, reset the counter
#         frame_counter = 0
#         candidate_book = None

#     # Draw Text
#     cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
#     stack_str = " -> ".join(book_stack[-5:])
#     cv2.putText(frame, f"Stack: {stack_str}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#     cv2.imshow('Book Stacking Cam', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import cv2.aruco as aruco
import numpy as np
import os

# ---------------- CONFIGURATION ----------------
current_path = os.path.dirname(os.path.abspath(__file__))

# ORB reference images
targets = {
    'Circle': 'circle.png',
    'Triangle': 'triangle.png',
    'Rectangle': 'rectangle.png',
    'Star': 'star.png'
}

# ArUco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
marker_to_book = {
    1: '03_KWJ', 
    2: '11_PSJ', 
    3: '12_JSEJ', 
    4: '16_SSMG', 
    5: '14_CCWJ', 
    6: '13_DGJ', 
    7: '06_JHHRJ', 
    8: '19_JWCJ',
    9: '17_HBJ',
    10: '05_OGJJ',
    11: '07_SCJ',
    12: '10_BJBJ',
    13: '08_SGYS_M',
    14: '09_GBUJ',
    15: '18_SGYS_S',
    }  # marker ID mapping

# ORB detector
orb = cv2.ORB_create()
reference_features = {}

for name, filename in targets.items():
    full_path = os.path.join(current_path, filename)
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Warning: Can't find file -> {full_path}")
        continue
    kp, des = orb.detectAndCompute(img, None)
    reference_features[name] = des

print("Image Load Finish!")

# ---------------- STACK STATE ----------------
book_stack = []
current_stable_book = None
frame_counter = 0
candidate_book = None

# ---------------- CAMERA SETUP ----------------
cap = cv2.VideoCapture(0)
MIN_MATCH_COUNT = 25
REQUIRED_FRAMES = 20

print("Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    # ---------------- ORB MATCHING ----------------
    best_match_name = None
    max_good_matches = 0
    if des_frame is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        for name, des_ref in reference_features.items():
            if des_ref is None: continue
            matches = bf.match(des_ref, des_frame)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 60]
            if len(good_matches) > max_good_matches:
                max_good_matches = len(good_matches)
                best_match_name = name

    # ---------------- ARUCO DETECTION ----------------
    corners, ids, _ = aruco.detectMarkers(gray_frame, aruco_dict)
    aruco_book_name = None
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for marker_id in ids.flatten():
            if marker_id in marker_to_book:
                aruco_book_name = marker_to_book[marker_id]
                # Prioritize ArUco if detected
                best_match_name = aruco_book_name
                max_good_matches = MIN_MATCH_COUNT + 10  # ensures stability logic triggers

    # ---------------- DEBUG VISUALS ----------------
    cv2.drawKeypoints(frame, kp_frame, frame, color=(0,255,0), flags=0)
    status_text = f"Raw: {best_match_name} ({max_good_matches})"
    color = (0, 0, 255)

    # ---------------- STABILITY LOGIC ----------------
    if max_good_matches >= MIN_MATCH_COUNT:
        color = (0, 255, 255)
        if best_match_name == candidate_book:
            frame_counter += 1
        else:
            candidate_book = best_match_name
            frame_counter = 0

        if frame_counter > REQUIRED_FRAMES:
            color = (0, 255, 0)
            if candidate_book != current_stable_book:
                current_stable_book = candidate_book
                book_stack.append(current_stable_book)
                print(f"Confirmed Book: {current_stable_book}")
            status_text = f"CONFIRMED: {current_stable_book}"
    else:
        frame_counter = 0
        candidate_book = None

    # ---------------- DISPLAY ----------------
    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    stack_str = " -> ".join(book_stack[-5:])
    cv2.putText(frame, f"Stack: {stack_str}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Book Stacking Cam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
