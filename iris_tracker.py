# TODO: Get the right eye as well, and quantify position of pupil relative to the other eye keypoints


import numpy as np
from imutils import face_utils
from imutils.video import VideoStream
import dlib
import time
import screeninfo
from numpy.polynomial import Polynomial as P
import cv2
from ctypes import windll, Structure, c_long, byref

user32 = windll.user32

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
vs = VideoStream(src=0).start()
time.sleep(2.0)
numerator = 0
denominator = 0

# create dlib faces detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# number of calibration points needed
NB_KEYPOINTS = 16


class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]


def get_screen_metrics():
    screens = screeninfo.get_monitors()

    min_x = float("+inf")
    min_y = float("+inf")

    max_x = float("-inf")
    max_y = float("-inf")

    for screen in screens:
        min_x = min(min_x, screen.x)
        min_y = min(min_y, screen.y)

        max_x = max(max_x, (screen.x + screen.width))
        max_y = max(max_y, (screen.y + screen.height))

    return min_x, max_x, min_y, max_y


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # calculates and return the eye aspect ratio
    return (A + B) / (2.0 * C)


def create_poly_model(keypoints, y_x, y_y, degree: int=3):
    """Returns a set of polynom to determine the x and y coordinates of the cursor
    based on the position of some sample points y_x and y_y"""

    x_x = []
    x_y = []

    for i in range(len(keypoints)):
        lx, ly = keypoints[i][0]
        rx, ry = keypoints[i][1]
        x_x.append(complex(lx, rx))
        x_y.append(complex(ly, ry))

    x_predict = P.fit(x_x, y_x, degree)
    y_predict = P.fit(x_y, y_y, degree)

    # print(x_predict)
    # print(y_predict)
    # print(x_x, x_y)

    return x_predict, y_predict


def query_mouse_position():
    pt = POINT()
    user32.GetCursorPos(byref(pt))
    return pt.x, pt.y


keypoints: list[list[tuple[float, float]]] = []
y_x: list[int] = []
y_y: list[int] = []

while True:
    key = cv2.waitKey(1) & 0xFF

    frame = vs.read()
    frame = cv2.flip(frame, 1)  # flips the frame

    roi = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect dlib face rectangles in the grayscale frame
    dlib_faces = detector(gray, 0)

    # loop through each face
    for face in dlib_faces:

        # store 2 eyes here
        eyes = []

        # convert dlib rect to a bounding box
        (x, y, w, h) = face_utils.rect_to_bb(face)
        # print(x,y,w,h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # get the landmarks from dlib, convert to np array
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]  # indexes for left eye key points
        rightEye = shape[rStart:rEnd]  # indexes for right eye key points

        eyes.append(leftEye)  # wrap in a list
        eyes.append(rightEye)

        eye_centers: list[tuple[float, float]] = []

        # loop through both eyes
        # index 0 : left eye, index 1 : right eye
        for index, eye in enumerate(eyes):
            pupil_found = False

            eye_EAR = eye_aspect_ratio(eye)

            left_side_eye = eye[0]  # left edge of eye
            right_side_eye = eye[3]  # right edge of eye
            top_side_eye = eye[1]  # top side of eye
            bottom_side_eye = eye[4]  # bottom side of eye

            # calculate height and width of dlib eye keypoints
            eye_width = right_side_eye[0] - left_side_eye[0]
            eye_height = bottom_side_eye[1] - top_side_eye[1]

            # create bounding box with buffer around keypoints
            eye_x1 = int(left_side_eye[0] - 0 * eye_width)  # .25 works well too
            eye_x2 = int(right_side_eye[0] + 0 * eye_height)  # .75 works well too

            eye_y1 = int(top_side_eye[1] - 1 * eye_height)
            eye_y2 = int(bottom_side_eye[1] + 1 * eye_height)

            # draw bounding box around eye roi
            cv2.rectangle(frame, (eye_x1, eye_y1), (eye_x2, eye_y2), (0, 255, 0), 2)

            # draw the circles for the eye landmarks
            for i in eye:
                cv2.circle(frame, tuple(i), 3, (0, 0, 255), -1)

            # ------------ estimation of distance of the human from camera--------------#
            # d=10920.0/float(w)

            roi = frame[eye_y1:eye_y2, eye_x1:eye_x2]

            #  ---------    check if eyes open   -------------  #

            # the eye is opened
            if eye_EAR > 0.25:

                #  ---------    find center of pupil   -------------  #

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # grey scale convert
                blur = cv2.medianBlur(gray, 5)  # blue image to find the iris better
                equ = cv2.equalizeHist(
                    blur
                )  # ie, improve contrast by spreading the range over the same window of intensity
                thres = cv2.inRange(
                    equ, 0, 15
                )  # threshold the contour edges, higher number means more will be black
                kernel = np.ones((3, 3), np.uint8)  # placeholder

                #     #/------- removing small noise inside the white image ---------/#
                dilation = cv2.dilate(thres, kernel, iterations=2)
                #     #/------- decreasing the size of the white region -------------/#
                erosion = cv2.erode(dilation, kernel, iterations=3)
                #     #/-------- finding the contours -------------------------------/#
                # image, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                contours, hierarchy = cv2.findContours(
                    erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )
                #     #--------- checking for 2 contours found or not ----------------#

                if len(contours) == 2:
                    # print('2 contours found')
                    pupil_found = True

                    img = cv2.drawContours(roi, contours, 1, (0, 255, 0), 3)
                    # ------ finding the centroid of the contour ----------------#
                    M = cv2.moments(contours[1])

                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        cv2.line(
                            roi, (int(cx), int(cy)), (int(cx), int(cy)), (0, 0, 255), 3
                        )
                # -------- checking for one contour present --------------------#

                if len(contours) == 1:
                    pupil_found = True
                    # print('only 1 contour found ------- ')

                    img = cv2.drawContours(roi, contours, 0, (0, 255, 0), 3)

                    # ------- finding centroid of the contour ----#
                    M = cv2.moments(contours[0])
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        cv2.line(
                            roi, (int(cx), int(cy)), (int(cx), int(cy)), (0, 0, 255), 3
                        )

                if pupil_found:
                    eye_centers.append((cx, cy))

            else:
                pass  # eye is closed

        if len(eye_centers) == 2:
            if key == ord("p"):
                if len(keypoints) < NB_KEYPOINTS:
                    keypoints.append(eye_centers)
                    x, y = query_mouse_position()
                    y_x.append(x)
                    y_y.append(y)
                else:
                    x_poly, y_poly = create_poly_model(keypoints, y_x, y_y)

                    x = x_poly(complex(eye_centers[0][0], eye_centers[1][0]))
                    y = y_poly(complex(eye_centers[0][1], eye_centers[1][1]))
                    print(x, y)
                    # print(query_mouse_position())
                    user32.SetCursorPos(int(x), int(y))

    cv2.imshow("frame", frame)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
vs.release()
# print("accurracy=",(float(numerator)/float(numerator+denominator))*100)
cv2.destroyAllWindows()
