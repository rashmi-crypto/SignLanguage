What it does: It recognizes few common hand patterns for sign language such as Hello, Thank You, Sorry, Water, I'm Hungry and Toilet.

Tech Stack:
Python 3.11
MediaPipe 0.10.11 — hand landmark detection (21 keypoints per hand)
OpenCV — webcam capture, frame encoding
scikit-learn 1.8.0, specifically RandomForestClassifier
Flask 3.1.3 — serves a live web app with an MJPEG video stream (/video_feed route), not just a local script

Tested in a live demo 

Built by following a guided tutorial on hand-gesture recognition using MediaPipe and scikit-learn, as a way to learn the end-to-end ML pipeline, from raw data collection through feature extraction, training, and real-time inference
