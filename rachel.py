import csv 
import cv2 as cv
import mediapipe as mp
import argparse
from collections import deque



def initiModules(): 
    global mp_hands
    mp_hands = mp.solutions.hands #solutions.hands is the points that mediapipe makes I think? ??/  ? / ?
    
    global mp_drawing
    mp_drawing = mp.solutions.drawing_utils #drawing the landparks from mp.solutions
    
    global hands
    hands = mp_hands.Hands(
        static_image_mode=False, #false because otherwise, the lib will treat every frame of the video as a seperate image
        #so it will REdetect the hands every frame, not really tracking every single position (THIS IS SUPER SLOW)
        #we need it to detect the hands relative to a frame, meaning IT WILL DETECT ONCE, then track where that same object lies in future hands
        #I guess this is like after you found the object, you don't need to find it again, you can track it after you've found it. 
        max_num_hands=4, #need cos were gunna have to halves 
        min_detection_confidence=0.9, #mediapipe will ONLY detect the hand if it measures that its 90% confident that the hand is there. If the hand is obstructed and the landmarks
        #are slightly invisible, then the confidence goes down, and if it goes down to 10% then it just wont draw a landmark 
        min_tracking_confidence=0.9) #minimum tracking score means it will track ACROSS MULTIPLE FRAMES. 
        #if the confidence for tracking drops below this point then it REDETECTS THE HANDS (RE FINDS) 
        
        #If hands keep disappearing, lower this to 0.3 so Mediapipe tracks hands more aggressively (without stopping).
        #If hands flicker, increase it to 0.7 for more stable tracking.
    
    #OUTPUT OF THE DETECTION (in hands var) will be given in X, Y, and Z coords

    #gunna use two types of calssifiers: 
        #KEYPOINT (BASED ON A STATIC POSITION OF THE HAND)
        #POINT HISTORY (BASED ON THE MOVEMENT OF THE HAND, MEANING THAT THE GESTURE IS AN ACTION RATHER THAN STATIC IMAGE)

   # keypoint_classifier = KeyPointClassifier() #MLM that takes hand landmarks and maps it to keypoint classiiers
    #point_history_classifier = PointHistoryClassifier() #MLM for point history classfifiers
    
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', #change the points (xyz) into actual classfifiers, it just maps it
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]
        
    historyLen = 20
    global pointHistory
    pointHistory = deque(mazlen=historyLen) #BRO WTF IS DEQUE
    #double-ended queue that manages old data efficinetly
    finger_gesture_history = deque(maxlen=historyLen)
    
    mode = 0
    
    
        
        
def videoCapture():
    args = get_args()
    
    cap_device = args.device #is 0
    cap_width = args.width
    cap_height = args.height
    
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True
    
    cap = cv.VideoCapture(cap_device) #pretty sure this 0 just means capture from default webcam (built in)
    
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    
    while True: #capturing the frames for mediapipe continuously in a loop
        print("Eeeeee")
        #wait are we gunna have to multithread? 
        capture, frame = cap.read() #cap.read() returns two values, first being a boolean value of whether cv captures the frame from webcam.
        if not capture: #if capture boolean is true then it will keep capturing. 
            break 
        #frame  IS THE ACTUAL IMAGE or some shit IDK but its stored as a NumPy array representing pixel values????? 
        #oh its like the RGB values. The screen is scanned by rows, and so its a 2D array
        #then the size would be (# horizontal pixels (# vertical pixels (3)))
        
        frame2 = cv.flip(frame, 1) #i have no clue wtf 1 means but this inverts the image to be mirror
        
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #this converts bgr to rgb (the coolors are inverted)
        
        #NOW the frame aactually needs to be processed by mediapipe
        
        results = hands.process(rgb_frame) #hands is the mediapipe object and it will process the rgb frame
        frame.flags.writeable = True
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(frame, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(frame, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(frame, pointHistory)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)
                
                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)
                
                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

        
        
        cv.imshow("SKIBIDI HAND GESTURE RECO", frame2) #DISPLAYS THE FRAME
        if cv.waitKey(1) & 0xFF ==ord('q'): #TO QUIT PRESS Q
            break
        
    cap.release()
    cv.destroyAllWindows()

#will break the loop either if frame catured is not valid, or if you press q

def get_args(): #this gets the args of the screen for displaying
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args



def main():
    initiModules()
    videoCapture()
    
main()

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]
