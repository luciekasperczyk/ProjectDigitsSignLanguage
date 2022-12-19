"""
'CollectData' FILE
Subject : module project in Computer Science
Semester : F2022
Authors :
    - Nasrdin Ahmed Aden
    - Zainab Ahmad
    - Lucie Kasperczyk
    - Hermann Yunus Knudsen
Code inspirations :
    - Camera :
        * video capture : https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
        * video capture : https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
        * flipping the image : https://techtutorialsx.com/2019/04/21/python-opencv-flipping-an-image/
    - Landmarks :
        * landmarks detection : https://google.github.io/mediapipe/solutions/hands.html
        * landmarks detection : https://www.analyticsvidhya.com/blog/2022/03/hand-landmarks-detection-on-an-image-using-mediapipe/
    - CSV file :
        * write : https://www.learnbyexample.org/reading-and-writing-csv-files-in-python/
        * dialects : https://docs.python.org/3/library/csv.html#csv-fmt-params
        * flatten : https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
"""

import mediapipe as mp
import cv2
import csv
import numpy as np

# define a video capture object
cap = cv2.VideoCapture(0)
#initialize the hands class and store it in a variable
mediapipeHands = mp.solutions.hands
#set the drawing function of the hand landmarks on the image
mediapipeDrawing = mp.solutions.drawing_utils
# set the hands function that will hold the landmarks points
hands = mediapipeHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands = 1)


num_coords_for_each_digit = 21
landmarks = ['class']
for num in range(1, num_coords_for_each_digit + 1):
    landmarks += [f'x{num}', f'y{num}', f'z{num}']

with open('Coordinates_9.csv', mode='w', newline='') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(landmarks)

class_name = "Digit_nine"

while True:
    ret, img = cap.read()  # create and open the camera window
    img = cv2.flip(img, 1)  # flip the image so it is like a mirror
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img.flags.writeable = False
    results = hands.process(img)
    img.flags.writeable = True
    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks:
            mediapipeDrawing.draw_landmarks(img, landmark, mediapipeHands.HAND_CONNECTIONS)

        hand = enumerate(results.multi_hand_landmarks[0].landmark)
        hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for _, landmark in hand]).flatten())
        hand_row.insert(0, class_name)
        with open('Coordinates_9.csv', mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(hand_row)


    cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(10) & 0xff == ord('q'):
        break




