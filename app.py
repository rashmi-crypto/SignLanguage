import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# 1. Load the "Brain"
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# 2. Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Process frame for MediaPipe
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_, y_ = [], []
            
            # Use the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            # Predict the sign
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = str(prediction[0])

            # Draw the label on the frame for the website
            # Shadow (Black)
            cv2.putText(frame, predicted_character, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8, cv2.LINE_AA)
# Main Text (Blue)
            cv2.putText(frame, predicted_character, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 145, 59), 4, cv2.LINE_AA)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 3. Routes to serve your HTML files
@app.route('/')
def index():
    return render_template('index.html')

# Add extra routes for your other HTML files if needed
# @app.route('/about')
# def about():
#     return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5000)