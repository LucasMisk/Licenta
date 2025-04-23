import cv2
import mediapipe as mp
import math

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_face_center = None
SMOOTHING_FACTOR = 0.5

def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

def fingers_grouped(hand_landmarks, finger_tips):
    tips = [hand_landmarks.landmark[tip] for tip in finger_tips]
    avg_x = sum(tip.x for tip in tips) / len(tips)
    avg_y = sum(tip.y for tip in tips) / len(tips)
    max_distance = 0.08
    for tip in tips:
        if math.sqrt((tip.x - avg_x) ** 2 + (tip.y - avg_y) ** 2) > max_distance:
            return False
    return True

def get_pinky_state(hand_landmarks):
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    if tip.y < pip.y - 0.03:
        return 'up'
    elif tip.y > pip.y + 0.03:
        return 'down'
    else:
        return 'folded'
def get_thumb_state(hand_landmarks):
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]

    # Orientation-aware thumb check (horizontal direction matters more)
    if abs(tip.x - mcp.x) > abs(tip.y - mcp.y):  # Thumb is extended sideways
        if tip.x < mcp.x:
            return 'up'  # Right hand, thumb pointing left
        else:
            return 'down'  # Right hand, thumb pointing right
    else:
        if tip.y < ip.y < mcp.y:
            return 'up'
        elif tip.y > ip.y and ip.y > mcp.y:
            return 'down'
        else:
            return 'folded'
def get_finger_state(hand_landmarks, finger_tip, finger_pip):
    tip = hand_landmarks.landmark[finger_tip]
    pip = hand_landmarks.landmark[finger_pip]

    if tip.y < pip.y - 0.05:
        return 'up'
    elif tip.y > pip.y + 0.05:
        return 'down'
    else:
        return 'folded'
def get_hand_orientation(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    return 'up' if middle_mcp.y < wrist.y else 'down'
def stabilize_face_position(current_center):
    global prev_face_center
    if prev_face_center is None:
        prev_face_center = current_center
        return current_center
    smoothed_x = SMOOTHING_FACTOR * prev_face_center[0] + (1 - SMOOTHING_FACTOR) * current_center[0]
    smoothed_y = SMOOTHING_FACTOR * prev_face_center[1] + (1 - SMOOTHING_FACTOR) * current_center[1]
    smoothed_center = (smoothed_x, smoothed_y)
    prev_face_center = smoothed_center
    return smoothed_center

def expand_roi(x, y, w, h, frame_width, frame_height):
    scale = 3
    new_w = int(w * scale)
    new_h = int(h * scale)
    x_new = max(0, x - (new_w - w) // 2)
    y_new = max(0, y - (new_h - h) // 2)
    if x_new + new_w > frame_width:
        x_new = frame_width - new_w
    if y_new + new_h > frame_height:
        y_new = frame_height - new_h
    return x_new, y_new, new_w, new_h

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

        if face_results.detections:
            for detection in face_results.detections:
                box = detection.location_data.relative_bounding_box
                x, y = int(box.xmin * frame_width), int(box.ymin * frame_height)
                w, h = int(box.width * frame_width), int(box.height * frame_height)

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                x_roi, y_roi, w_roi, h_roi = expand_roi(x, y, w, h, frame_width, frame_height)
                cv2.rectangle(image, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (255, 0, 0), 2)

                roi = image[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]
                if roi.size > 0:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    hand_results = hands.process(roi_rgb)

                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            for landmark in hand_landmarks.landmark:
                                landmark.x = (landmark.x * w_roi + x_roi) / frame_width
                                landmark.y = (landmark.y * h_roi + y_roi) / frame_height

                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                            thumb = get_thumb_state(hand_landmarks)
                            index = get_finger_state(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
                            middle = get_finger_state(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
                            ring = get_finger_state(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
                            pinky = get_pinky_state(hand_landmarks)

                            hand_orientation = get_hand_orientation(hand_landmarks)

                            finger_tips = [
                                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                mp_hands.HandLandmark.RING_FINGER_TIP,
                                mp_hands.HandLandmark.PINKY_TIP
                            ]

                            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            thumb_index_dist = calculate_distance(thumb_tip, index_tip)

                            if hand_orientation == 'up' and index == 'up' and all(f != 'up' for f in [middle, ring, pinky]):
                                cv2.putText(image, 'New Question', (x_roi, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            elif hand_orientation == 'up' and index == 'up' and middle == 'up' and all(f != 'up' for f in [ring, pinky]):
                                cv2.putText(image, 'Reply', (x_roi, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            elif hand_orientation == 'up' and index == 'up' and middle == 'up' and ring == 'up' and pinky != 'up':
                                cv2.putText(image, 'Abberation', (x_roi, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2)
                            elif hand_orientation == 'up' and thumb_index_dist < 0.05 and all(f == 'up' for f in [middle, ring, pinky]):
                                cv2.putText(image, 'Bullshit', (x_roi, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            elif hand_orientation == 'up' and pinky == 'up' and all(f != 'up' for f in [index, middle, ring]):
                                cv2.putText(image, 'Technical', (x_roi, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            elif hand_orientation == 'up' and index == 'up' and pinky == 'up' and all(f != 'up' for f in [middle, ring]):
                                cv2.putText(image, 'Discussion in 2', (x_roi, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            elif hand_orientation == 'up' and middle == 'up' and all(f != 'up' for f in [index, ring, pinky]):
                                cv2.putText(image, 'Speak Louder', (x_roi, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            elif hand_orientation == 'down' and middle == 'down' and all(f != 'down' for f in [index, ring, pinky]):
                                cv2.putText(image, 'Speak Quieter', (x_roi, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            elif hand_orientation == 'up' and all(f == 'up' for f in [index, middle, ring, pinky]) and thumb != 'down':
                                cv2.putText(image, 'Ask for Clarification', (x_roi, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            elif hand_orientation == 'up' and thumb == 'up' and all(f != 'up' for f in [index, middle, ring, pinky]):
                                cv2.putText(image, 'Like', (x_roi, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            elif hand_orientation == 'down' and thumb == 'down' and all(f != 'down' for f in [index, middle, ring, pinky]):
                                cv2.putText(image, 'Dislike', (x_roi, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            elif thumb_index_dist > 0.08 and all( f == "folded" for f in [index, middle, ring, pinky]):
                                cv2.putText(image, 'Clarify', (x_roi, y_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            print(f"Thumb: {thumb}, Index: {index}, Middle: {middle}, Ring: {ring}, Pinky: {pinky}, Hand: {hand_orientation}")
        cv2.imshow('Gesture Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
