from io import BytesIO

import cv2
import mediapipe as mp
import math
import face_recognition
import numpy as np
from datetime import datetime
import requests
import time
import base64
from PIL import Image

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

SIGN_HOLD_THRESHOLD = 1.0  # seconds
RESEND_THRESHOLD = 30.0  # seconds before same sign can be sent again
BACKEND_URL = "http://localhost:8080/api/update"

person_sign_start_times = {}
last_sent_times = {}  # Maps (name, sign) -> timestamp

PRIORITY_SIGNS = {
    "New Question", "Reply", "Joke", "Technical",  "Clarification", "Ask for Clarification"
}
REACTION_SIGNS = {
    "Like", "Dislike", "Speak Louder", "Discussion in 2", "Speak Quieter", "Abberation"
}

def base64_to_np(base64_string):
    decoded = base64.b64decode(base64_string)
    image = Image.open(BytesIO(decoded)).convert('RGB')
    return np.array(image)

def load_known_faces():
    """
    Load known face images and names from the backend MySQL database.
    """
    url = "http://localhost:8080/api/facePhotos"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for person in data:
            image_data = base64_to_np(person['photoBase64'])
            name = person['name']
            encodings = face_recognition.face_encodings(image_data)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
        print("Loaded known faces from DB.")
    else:
        print("Failed to load face data. Status:", response.status_code)

# Load known faces
known_face_encodings = []
known_face_names = []

load_known_faces()

# # Example: Add known face
# person1_image = face_recognition.load_image_file("lucas.jpg")
# person1_encoding = face_recognition.face_encodings(person1_image)[0]
# known_face_encodings.append(person1_encoding)
# known_face_names.append("Lucas")

# Helper functions for hand state
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

def get_pinky_state(hand_landmarks):
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    return 'up' if tip.y < pip.y - 0.03 else 'down' if tip.y > pip.y + 0.03 else 'folded'

def get_thumb_state(hand_landmarks):
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    if abs(tip.x - mcp.x) > abs(tip.y - mcp.y):
        return 'up' if tip.x < mcp.x else 'down'
    else:
        if tip.y < ip.y < mcp.y:
            return 'up'
        elif tip.y > ip.y and ip.y > mcp.y:
            return 'down'
        else:
            return 'folded'

def get_finger_state(hand_landmarks, tip, pip):
    tip_y = hand_landmarks.landmark[tip].y
    pip_y = hand_landmarks.landmark[pip].y
    return 'up' if tip_y < pip_y - 0.05 else 'down' if tip_y > pip_y + 0.05 else 'folded'

def get_hand_orientation(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    return 'up' if middle_mcp.y < wrist.y else 'down'

def expand_roi(x, y, w, h, frame_width, frame_height):
    scale = 3
    new_w = int(w * scale)
    new_h = int(h * scale)
    x_new = max(0, x - (new_w - w) // 2)
    y_new = max(0, y - (new_h - h) // 2)
    x_new = min(x_new, frame_width - new_w)
    y_new = min(y_new, frame_height - new_h)
    return x_new, y_new, new_w, new_h

# Main loop
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=2) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
        face_results = face_detection.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        now = datetime.now()

        if face_results.detections:
            for detection in face_results.detections:
                box = detection.location_data.relative_bounding_box
                x, y = int(box.xmin * frame_width), int(box.ymin * frame_height)
                w, h = int(box.width * frame_width), int(box.height * frame_height)

                face_roi = frame[y:y + h, x:x + w]
                name = "Unknown"

                if face_roi.size > 0:
                    rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_face)
                    if encodings:
                        face_encoding = encodings[0]
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                x_roi, y_roi, w_roi, h_roi = expand_roi(x, y, w, h, frame_width, frame_height)
                roi = image[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]

                if roi.size > 0:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    hand_results = hands.process(roi_rgb)

                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            for landmark in hand_landmarks.landmark:
                                landmark.x = (landmark.x * w_roi + x_roi) / frame_width
                                landmark.y = (landmark.y * h_roi + y_roi) / frame_height

                            thumb = get_thumb_state(hand_landmarks)
                            index = get_finger_state(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
                            middle = get_finger_state(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
                            ring = get_finger_state(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
                            pinky = get_pinky_state(hand_landmarks)
                            hand_orientation = get_hand_orientation(hand_landmarks)

                            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            thumb_index_dist = calculate_distance(thumb_tip, index_tip)

                            sign = None
                            if hand_orientation == 'up' and index == 'up' and all(f != 'up' for f in [middle, ring, pinky]):
                                sign = 'New Question'
                            elif hand_orientation == 'up' and index == 'up' and middle == 'up' and all(f != 'up' for f in [ring, pinky]):
                                sign = 'Reply'
                            elif hand_orientation == 'up' and thumb_index_dist < 0.05 and all(f == 'up' for f in [middle, ring, pinky]):
                                sign = 'Joke'
                            elif hand_orientation == 'up' and pinky == 'up' and all(f != 'up' for f in [index, middle, ring]):
                                sign = 'Technical'
                            elif hand_orientation == 'up' and index == 'up' and pinky == 'up' and all(f != 'up' for f in [middle, ring]):
                                sign = 'Discussion in 2'
                            elif hand_orientation == 'up' and middle == 'up' and all(f != 'up' for f in [index, ring, pinky]):
                                sign = 'Speak Louder'
                            elif hand_orientation == 'down' and middle == 'down' and all(f != 'down' for f in [index, ring, pinky]):
                                sign = 'Speak Quieter'
                            elif hand_orientation == 'up' and all(f == 'up' for f in [index, middle, ring, pinky]) and thumb != 'down':
                                sign = 'Ask for Clarification'
                            elif hand_orientation == 'up' and thumb == 'up' and all(f != 'up' for f in [index, middle, ring, pinky]):
                                sign = 'Like'
                            elif hand_orientation == 'down' and thumb == 'down' and all(f != 'down' for f in [index, middle, ring, pinky]):
                                sign = 'Dislike'
                            elif thumb_index_dist > 0.08 and all(f == "folded" for f in [index, middle, ring, pinky]):
                                sign = 'Clarification'
                            elif hand_orientation == 'up' and index == 'up' and middle == 'up' and ring == 'up' and pinky != 'up':
                                sign = 'Abberation'

                            if sign:
                                key = (name, sign)
                                if key not in person_sign_start_times:
                                    person_sign_start_times[key] = now
                                elif (now - person_sign_start_times[key]).total_seconds() >= SIGN_HOLD_THRESHOLD:
                                    current_time = time.time()
                                    last_sent = last_sent_times.get(key, 0)
                                    if current_time - last_sent >= RESEND_THRESHOLD:
                                        try:
                                            payload = { "name": name, "sign": sign, "time": now.strftime("%H:%M:%S"), "type": "priority" if sign in PRIORITY_SIGNS else "reaction"}
                                            response = requests.post(BACKEND_URL, json=payload)
                                            if response.status_code == 200:
                                                last_sent_times[key] = current_time
                                                print(f"Sent to backend: {payload}")
                                            else:
                                                print(f"Failed to send {payload}: {response.status_code}")
                                        except Exception as e:
                                            print(f"Error sending {payload}: {e}")

                            cv2.putText(image, sign if sign else '', (x_roi, y_roi - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Sign Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
