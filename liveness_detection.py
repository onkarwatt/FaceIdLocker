from scipy.spatial import distance as dist
#used to cal eucledian dist to 
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import dlib
import time
import cv2
import random

class LivenessDetection:
    def __init__(self, shape_predictor_path, eye_ar_thresh=0.22, eye_ar_consec_frames=(3, 5), success_threshold=5):
        self.EYE_AR_THRESH = eye_ar_thresh
        self.EYE_AR_CONSEC_FRAMES = random.randint(*eye_ar_consec_frames)
        self.COUNTER = 0
        self.TOTAL = 0
        self.SUCCESS_THRESHOLD = success_threshold
        self.LOOK_THRESHOLD = 10

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        print("[INFO] starting video stream...")
        self.vs = VideoStream(src=0).start()
        time.sleep(1.0)

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_blinks(self, shape):
        leftEye = shape[self.lStart:self.lEnd]
        rightEye = shape[self.rStart:self.rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < self.EYE_AR_THRESH:
            self.COUNTER += 1
        else:
            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                self.TOTAL += 1
            self.COUNTER = 0

        return ear, self.TOTAL

    def detect_look_left(self, shape):
        left_eye_center = np.mean(shape[self.lStart:self.lEnd], axis=0)
        right_eye_center = np.mean(shape[self.rStart:self.rEnd], axis=0)
        nose = shape[30]  # Nose tip
        left_eye_distance = np.linalg.norm(nose - left_eye_center)
        right_eye_distance = np.linalg.norm(nose - right_eye_center)
        if left_eye_distance < right_eye_distance -self.LOOK_THRESHOLD:
            return True
        return False

    def detect_look_right(self, shape):
        left_eye_center = np.mean(shape[self.lStart:self.lEnd], axis=0)
        right_eye_center = np.mean(shape[self.rStart:self.rEnd], axis=0)
        nose = shape[30]  # Nose tip
        left_eye_distance = np.linalg.norm(nose - left_eye_center)
        right_eye_distance = np.linalg.norm(nose - right_eye_center)
        if right_eye_distance < left_eye_distance - self.LOOK_THRESHOLD:
            return True
        return False

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            ear, blinks = self.detect_blinks(shape)
            is_looking_left = self.detect_look_left(shape)
            is_looking_right = self.detect_look_right(shape)

            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        return frame, rects

    def start_detection(self):
        actions = ['blink','look_left', 'look_right']
        num_actions = random.randint(4, 6)
        action_sequence = []

        while len(action_sequence) < num_actions:
            action = random.choice(actions)
            if len(action_sequence) == 0 or action_sequence[-1] != action:
                action_sequence.append(action)

        current_action_index = 0
        blinks_required = random.randint(3, 5)
        success_actions = 0

        while True:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=450)
            frame, rects = self.process_frame(frame)

            if current_action_index < len(action_sequence):
                current_action = action_sequence[current_action_index]

                if current_action == 'blink':
                    cv2.putText(frame, f"Please blink ({blinks_required - self.TOTAL} left)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if self.TOTAL >= blinks_required:
                        success_actions += 1
                        current_action_index += 1
                        self.TOTAL = 0
                        blinks_required = random.randint(3, 5)

                elif current_action == 'look_left':
                    cv2.putText(frame, "Please look left", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if len(rects) > 0:
                        shape = face_utils.shape_to_np(self.predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), rects[0]))
                        if self.detect_look_left(shape):
                            success_actions += 1
                            current_action_index += 1

                elif current_action == 'look_right':
                    cv2.putText(frame, "Please look right", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if len(rects) > 0:
                        shape = face_utils.shape_to_np(self.predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), rects[0]))
                        if self.detect_look_right(shape):
                            success_actions += 1
                            current_action_index += 1

            else:
                cv2.putText(frame, "All actions completed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"Progress: {success_actions}/{num_actions}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if success_actions >= num_actions:
                print("Liveness Detection Successful")
                cv2.putText(frame, "Liveness Detection Successful", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Frame", frame)
                cv2.waitKey(3000)
                break

        cv2.destroyAllWindows()
        self.vs.stop()
        return True

if __name__ == "__main__":
    liveness_detection = LivenessDetection("shape_predictor_68_face_landmarks.dat")
    result = liveness_detection.start_detection()
    print(f"Detection Result: {result}")
