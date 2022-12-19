"""
'Main' FILE
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
        * put text : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    - Random forest :
        * random forest classifier : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        * predictions : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict
        * predictions : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
        * accuracy : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    - CSV
        * read the file : https://www.w3schools.com/python/pandas/pandas_csv.asp
        * write : https://www.learnbyexample.org/reading-and-writing-csv-files-in-python/
        * dialects : https://docs.python.org/3/library/csv.html#csv-fmt-params
        * flatten : https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
    - Landmarks :
        * landmarks detection : https://google.github.io/mediapipe/solutions/hands.html
        * landmarks detection : https://www.analyticsvidhya.com/blog/2022/03/hand-landmarks-detection-on-an-image-using-mediapipe/
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# define a video capture object
cap = cv2.VideoCapture(0)
# initialize the hands class and store it in a variable
mediapipeHands = mp.solutions.hands
# set the hands function that will hold the landmarks points
hands = mediapipeHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
# set the drawing function of the hand landmarks on the image
mediapipeDrawing = mp.solutions.drawing_utils




dataCoord = pd.read_csv("./merged.csv")
print(dataCoord)
#x corresponds to all the coordinates
#.iloc[all the rows, the column starting at x0 to z20]
x = dataCoord.iloc[:,2:65]
print(x)
#y corresponds to the label : the different digits
y = dataCoord.iloc[:,1]
print(y)

#---------------Tests and Train part-----------------#
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0) #test size = 30%
classifier = RandomForestClassifier(n_estimators=20, random_state=0)#20 decision trees
classifier.fit(xTrain, yTrain)
yPrediction = classifier.predict(xTest)

#------------------------Accuracy----------------------------#
print(classification_report(yTest, yPrediction))
print(accuracy_score(yTest, yPrediction))


while True:
    ret, img = cap.read()  # create and open the camera window
    img = cv2.flip(img, 1)  # flip the image so it is like a mirror

    # create a list of all the different fingertips from 0 to 20
    fingerTips = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # create an empty list for the future hand's landmarks
    landMarkList = []
    results = hands.process(img)

    # if there is a hand on the screen
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # draw the landmarks as connections (lines)
            # draw the fingertips as connectors (circles)
            mediapipeDrawing.draw_landmarks(img, hand_landmark, mediapipeHands.HAND_CONNECTIONS,
                                            mediapipeDrawing.DrawingSpec((9, 52, 28), 2, 2),
                                            mediapipeDrawing.DrawingSpec((9, 52, 28), 2, 2))


    try:
        hand = enumerate(results.multi_hand_landmarks[0].landmark)
        hand_row = list(np.array([[landmark.x, landmark.y, landmark.z] for _, landmark in hand]).flatten())
        x = pd.DataFrame([hand_row])
        prediction = classifier.predict(x)
        prediction = prediction[0]
        cv2.putText(img, prediction, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception as e:
        pass

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break


