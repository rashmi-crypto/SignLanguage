import os
import pickle
import cv2
import mediapipe as mp
# This is the secret sauce for Windows
from mediapipe.python.solutions import hands as mp_hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

print("Starting extraction in the clean folder...")

if not os.path.exists(DATA_DIR):
    print(f"Error: Could not find 'data' folder at {os.path.abspath(DATA_DIR)}")
else:
    for dir_ in os.listdir(DATA_DIR):
        if dir_.startswith('.'): continue
        print(f"Processing category: {dir_}")
        
        category_path = os.path.join(DATA_DIR, dir_)
        for img_path in os.listdir(category_path):
            img = cv2.imread(os.path.join(category_path, img_path))
            if img is None: continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                hand_landmarks = results.multi_hand_landmarks[0]
                
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)
                
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))
                
                data.append(data_aux)
                labels.append(dir_)

    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("SUCCESS! data.pickle created.")