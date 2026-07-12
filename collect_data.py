import cv2
import os

# 1. Define your specific signs
signs = ['help', 'water', 'food', 'yes', 'no', 'bathroom']
num_images = 100  # Captures 100 photos per sign
DATA_DIR = './data'

# Create the main data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(0)

for sign in signs:
    # Create a subfolder for each sign (e.g., ./data/help)
    sign_path = os.path.join(DATA_DIR, sign)
    if not os.path.exists(sign_path):
        os.makedirs(sign_path)

    print(f'PREPARING TO RECORD: "{sign.upper()}"')

    # PHASE 1: Wait for you to get in position
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready for "{sign}"? Press "S" to start', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Collector', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # PHASE 2: Automatic capture (100 images)
    counter = 0
    while counter < num_images:
        ret, frame = cap.read()
        cv2.imshow('Collector', frame)
        
        # Save the image to the specific folder
        img_name = os.path.join(sign_path, f'{counter}.jpg')
        cv2.imwrite(img_name, frame)
        
        counter += 1
        cv2.waitKey(50) # Tiny delay so you can move your hand slightly
        print(f'Captured {counter}/{num_images} for {sign}')

print("\nSUCCESS: All signs collected!")
cap.release()
cv2.destroyAllWindows()