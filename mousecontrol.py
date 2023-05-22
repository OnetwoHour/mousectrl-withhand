import sys
import subprocess

try:
    import pyautogui
    import numpy as np
    import mediapipe as mp
    import cv2
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "-q", "opencv-python", "mediapipe", "numpy", "pyautogui"])


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

rps_gesture = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'None'}

cap = cv2.VideoCapture(0)
IMG_FRAME = 10
angles = []

# sample data
file = np.genfromtxt('data.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)


def main():
    with mp_hands.Hands(max_num_hands=1) as hands:
        frame = 0
        ave = {'Up': 0, 'Down': 0, 'Left': 0, 'Right': 0, 'None': 0}
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks and not frame:
                for hand_landmarks in results.multi_hand_landmarks:
                    joint = np.zeros((21, 3))
                    for i, lm in enumerate(hand_landmarks.landmark):
                        joint[i] = [lm.x, lm.y, lm.z]

                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9,
                                10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                    v2 = joint[range(1, 21), :]
                    v = v2 - v1

                    v /= np.linalg.norm(v, axis=1)[:, np.newaxis]
                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10,
                                                    12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

                    angle = np.degrees(angle)
                    angles.append(angle)

                    data = np.array([angle], dtype=np.float32)
                    ret, result, neighbors, dist = knn.findNearest(data, 3)
                    idx = int(result[0][0])
                    var = rps_gesture[idx]
                    ave[var] += 1

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            cv2.imshow('Hand', cv2.flip(image, 1))
            cv2.waitKey(1)

            frame += 1 if frame < IMG_FRAME else -IMG_FRAME

            if frame == 0:
                var = max(ave, key=ave.get)
                if max(ave.values()) == 0:
                    continue
                elif var == 'Up':
                    pyautogui.moveRel(0, 100, 0.2)
                elif var == 'Down':
                    pyautogui.moveRel(0, -100, 0.2)
                elif var == 'Left':
                    pyautogui.moveRel(-100, 0, 0.2)
                elif var == 'Right':
                    pyautogui.moveRel(100, 0, 0.2)
                for v in ave.keys():
                    ave[v] = 0
                yield var

            """ making train data 
            
            if cv2.waitKey() & 0xFF == 27 or len(angles) == 10000:
                np.savetxt('angle.csv', np.array(angles), delimiter=',')
                break
            elif cv2.waitKey() & 0xFF == 32 and results.multi_hand_landmarks:
                angles.append(angle)
                print(f'NO.{print(np.array(angles).shape[0])} ADD COMPLETE!')
            """

    cap.release()


if __name__ == "__main__":
    r = main()
    while True:
        next(r)
