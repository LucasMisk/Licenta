import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)


def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)


def is_finger_up(hand_landmarks, finger_tip, finger_pip):
    return hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_pip].y


def is_finger_down(hand_landmarks, finger_tip, finger_pip, finger_mcp=None):
    # For thumb (which doesn't have a proper MCP), we'll use IP joint as reference
    if finger_mcp is None:
        return hand_landmarks.landmark[finger_tip].y > hand_landmarks.landmark[finger_pip].y
    else:
        # For other fingers, check if tip is below both PIP and MCP
        return (hand_landmarks.landmark[finger_tip].y > hand_landmarks.landmark[finger_pip].y and
                hand_landmarks.landmark[finger_tip].y > hand_landmarks.landmark[finger_mcp].y)


def expand_roi(x, y, w, h, frame_width, frame_height, expansion=1.0):
    """Expand the ROI around the face (double the area by default)"""
    scale_factor = 3  # Triple the dimensions for larger search area
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    x_new = max(0, int(x - (new_w - w) / 2))
    y_new = max(0, int(y - (new_h - h) / 2))

    if x_new + new_w > frame_width:
        x_new = frame_width - new_w
    if y_new + new_h > frame_height:
        y_new = frame_height - new_h

    return max(0, x_new), max(0, y_new), min(new_w, frame_width), min(new_h, frame_height)


with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection, \
        mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape

        # First detect faces
        face_results = face_detection.process(image)

        # Convert back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if face_results.detections:
            for detection in face_results.detections:
                # Get face bounding box
                box = detection.location_data.relative_bounding_box
                x = int(box.xmin * frame_width)
                y = int(box.ymin * frame_height)
                w = int(box.width * frame_width)
                h = int(box.height * frame_height)

                # Draw original face bounding box (green)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Expand the ROI around the face
                x_roi, y_roi, w_roi, h_roi = expand_roi(x, y, w, h, frame_width, frame_height)

                # Draw expanded ROI (blue)
                cv2.rectangle(image, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (255, 0, 0), 2)

                # Extract the ROI for hand detection
                roi = image[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]

                if roi.size > 0:
                    # Process the ROI for hand detection
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    hand_results = hands.process(roi_rgb)

                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            # Draw hand landmarks on the original image
                            for landmark in hand_landmarks.landmark:
                                landmark.x = (landmark.x * w_roi + x_roi) / frame_width
                                landmark.y = (landmark.y * h_roi + y_roi) / frame_height

                            mp_drawing.draw_landmarks(
                                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                            # Get the landmarks for the hand
                            landmarks = hand_landmarks.landmark

                            # Check which fingers are up or down
                            thumb_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.THUMB_TIP,
                                                    mp_hands.HandLandmark.THUMB_IP)
                            thumb_down = is_finger_down(hand_landmarks, mp_hands.HandLandmark.THUMB_TIP,
                                                        mp_hands.HandLandmark.THUMB_IP)

                            index_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                                    mp_hands.HandLandmark.INDEX_FINGER_PIP)
                            middle_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                                     mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
                            middle_down = is_finger_down(hand_landmarks,
                                                         mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                                         mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                                                         mp_hands.HandLandmark.MIDDLE_FINGER_MCP)

                            ring_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP,
                                                   mp_hands.HandLandmark.RING_FINGER_PIP)
                            pinky_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP,
                                                    mp_hands.HandLandmark.PINKY_PIP)

                            thumb_index_distance = calculate_distance(landmarks[mp_hands.HandLandmark.THUMB_TIP],
                                                                      landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP])

                            # Detect gestures
                            if index_up and not middle_up and not ring_up and not pinky_up:
                                cv2.putText(image, 'New Question', (x_roi, y_roi - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            elif index_up and middle_up and not ring_up and not pinky_up:
                                cv2.putText(image, 'Reply', (x_roi, y_roi - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            elif thumb_index_distance < 0.05 and middle_up and ring_up and pinky_up:
                                cv2.putText(image, 'Bullshit', (x_roi, y_roi - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            elif not thumb_up and not index_up and not middle_up and not ring_up and pinky_up:
                                cv2.putText(image, 'Technical', (x_roi, y_roi - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            elif index_up and pinky_up and not middle_up and not ring_up:
                                cv2.putText(image, 'Discussion in 2', (x_roi, y_roi - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            elif middle_up and not index_up and not ring_up and not pinky_up:
                                cv2.putText(image, 'Speak Louder', (x_roi, y_roi - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            # Revised Speak Quieter gesture - middle finger extended but pointing down
                            elif (middle_down and
                                  not index_up and
                                  not ring_up and
                                  not pinky_up and
                                  not thumb_up):
                                cv2.putText(image, 'Speak Quieter', (x_roi, y_roi - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
                                cv2.putText(image, 'Like', (x_roi, y_roi - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            # Revised Dislike gesture - thumb extended but pointing down
                            elif (thumb_down and
                                  not index_up and
                                  not middle_up and
                                  not ring_up and
                                  not pinky_up):
                                cv2.putText(image, 'Dislike', (x_roi, y_roi - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            elif not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
                                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                                middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                                ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
                                pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

                                if (calculate_distance(thumb_tip, index_tip) < 0.1 and
                                        calculate_distance(index_tip, middle_tip) < 0.1 and
                                        calculate_distance(middle_tip, ring_tip) < 0.1 and
                                        calculate_distance(ring_tip, pinky_tip) < 0.1):
                                    cv2.putText(image, 'Clarify', (x_roi, y_roi - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                            elif thumb_up and index_up and middle_up and ring_up and pinky_up:
                                cv2.putText(image, 'Ask for Clarification', (x_roi, y_roi - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Face and Hand Gesture Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()