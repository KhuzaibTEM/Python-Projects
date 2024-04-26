import mediapipe as mp
import cv2

#Cv2 is a good library
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
capture = cv2.VideoCapture(0)


with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:

    while capture.isOpened():

        ret, frame = capture.read()
        frame = cv2.resize(frame, (900, 600))
        frame = cv2.flip(frame, 1)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = img.shape

        # Face Detection
        mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(230, 222, 14), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))

        # Right Hand
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(230, 222, 14), thickness=2, circle_radius=2))

        # Left Hand
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(230, 222, 14), thickness=2, circle_radius=2))

        # Pose
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(24, 222, 240), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(97, 62, 6), thickness=2, circle_radius=2))

        cv2.imshow('Model Detection', img)

        if cv2.waitKey(10) & 0xFF == ord('\x1b'):
            break


capture.release()
cv2.destroyAllWindows()
