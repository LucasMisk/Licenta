import cv2
import mediapipe as mp
import math


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)


def is_finger_up(hand_landmarks, finger_tip, finger_pip):
    return hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_pip].y


def is_finger_down(hand_landmarks, finger_tip, finger_pip):
    return hand_landmarks.landmark[finger_tip].y > hand_landmarks.landmark[finger_pip].y


with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark

                thumb_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP)
                index_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
                middle_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
                ring_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
                pinky_up = is_finger_up(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)

                middle_down = is_finger_down(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)

                thumb_index_distance = calculate_distance(landmarks[mp_hands.HandLandmark.THUMB_TIP],
                                                         landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP])

                if index_up and not middle_up and not ring_up and not pinky_up:
                    cv2.putText(image, 'New Question', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif index_up and middle_up and not ring_up and not pinky_up:
                    cv2.putText(image, 'Reply', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif thumb_index_distance < 0.05 and middle_up and ring_up and pinky_up:
                    cv2.putText(image, 'Bullshit', (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif not thumb_up and not index_up and not middle_up and not ring_up and pinky_up:
                    cv2.putText(image, 'Technical', (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif index_up and pinky_up and not middle_up and not ring_up:
                    cv2.putText(image, 'Discussion in 2', (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif middle_up and not index_up and not ring_up and not pinky_up:
                    cv2.putText(image, 'Speak Louder', (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif middle_down and not index_up and not middle_up and not ring_up and not pinky_up:
                    cv2.putText(image, 'Speak Quieter', (10, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
                    cv2.putText(image, 'Like', (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
                    cv2.putText(image, 'Dislike', (10, 270),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
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
                        cv2.putText(image, 'Clarify', (10, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif thumb_up and index_up and middle_up and ring_up and pinky_up:
                    cv2.putText(image, 'Ask for Clarification', (10, 330),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Gesture Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()