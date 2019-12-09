# Auther : Rishikesh
# Date	: 29 Dec 2017
# import the necessary packages

# python classLiveness.py --shape-predictor shape_predictor_68_face_landmarks.dat

import argparse
import time
from scipy.spatial.distance import cdist
from imutils import face_utils
import dlib
import numpy as np
import cv2


# construct the argument parser and parse the arguments
#AP = argparse.ArgumentParser()
#AP.add_argument("-p", "--shape-predictor", required=True,
#                help="path to facial landmark predictor")
#ARGS = vars(AP.parse_args())

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions


class LivenessDetector:

    start_time = time.time()

    def __init__(self, predictor, detector):

        self.predictor = predictor
        self.detector = detector
        self.eye_thresh = 0.25
        self.ratio_thresh = 0.0015
        self.eye_cosec_frames = 2
        # initialize the frame counters and the total number of blinks
        self.counter = 0
        self.total = 0
        self.live = None
        self.prev_shape = []

    @staticmethod
    def eye_aspect_ratio(eye):
        
    	# compute the euclidean distances between the two sets of
    	# vertical eye landmarks (x, y)-coordinates
        vect_a = cdist(eye[1].reshape(2, 1), eye[5].reshape(2, 1), 'euclidean')
        vect_b = cdist(eye[2].reshape(2, 1), eye[4].reshape(2, 1), 'euclidean')
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        vect_c = cdist(eye[0].reshape(2, 1), eye[3].reshape(2, 1), 'euclidean')
        # compute the eye aspect ratio
        e_a_r = (vect_a.diagonal().sum() + vect_b.diagonal().sum()) / \
                (2.0 * vect_c.diagonal().sum())

        # return the eye aspect ratio
        return e_a_r

    def face_aspect(self, frame, rectangle):

        shape = self.predictor(frame, rectangle)
        shape = face_utils.shape_to_np(shape)    
        # 0.6 right and 1.6 left threshold
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        return shape, ear

    def blinks_counter(self, eye_aspect_ratio):
        if eye_aspect_ratio < self.eye_thresh:
            self.counter += 1
                # otherwise, the eye aspect ratio is not below the blink
		        # threshold:
        else:
            # if the eyes were closed for a sufficient number of
        	# then increment the total number of blinks
            if 1 <= self.counter <= self.eye_cosec_frames:
                self.total += 1
                # reset the eye frame counter
                self.counter = 0
            self.counter = 0
            if time.time() - self.start_time < 8: #and self.total >= 1:
                self.live = True
            elif time.time() - self.start_time > 8:
                self.start_time = time.time()
                self.total = 0

    def steady_face(self, face_shape):

        if len(self.prev_shape) == 0:
            self.prev_shape = face_shape[0:17]

        steadyness = self.prev_shape - face_shape[0:17]
        self.prev_shape = face_shape[0:17]

        return np.mean(abs(steadyness))
        
    def detect_liveness(self, frame):
        rects = self.detector(frame, 1)
        if len(rects) != 0:
            big_box_face = 0
            steady = 100
            for (i, rect) in enumerate(rects):
    	        # determine the facial landmarks for the face region, then
    	        # convert the facial landmark (x, y)-coordinates to a NumPy
    	        # array of face
                shape, ear = self.face_aspect(frame, rect)
    
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                # top, bottom, left, right = (y, y+h, x, x+w)
    
                # ear_ratio = ear / w
                
                face_box_diag = np.sqrt(w**2 + h**2)
                
                if big_box_face < face_box_diag:
                    big_box_face = face_box_diag
                    steady = self.steady_face(shape)
    
                if steady <= 5 and big_box_face > 150:
                #print("steadyness", np.mean(abs(diff)))
                    self.blinks_counter(ear)
                else:
                    self.live = False
                    self.start_time = time.time()
                    self.total = 0
            return self.live
        
        self.live = False
        return self.live
