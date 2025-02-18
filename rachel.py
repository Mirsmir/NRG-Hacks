import csv 
import cv2 as cv
import mediapipe as mp
import argparse



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
        
        frame = cv.flip(frame, 1) #i have no clue wtf 1 means but this inverts the image to be mirror
        
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #this converts bgr to rgb (the coolors are inverted)
        
        #NOW the frame aactually needs to be processed by mediapipe
        
        result_from_mp = hands.process(rgb_frame) #hands is the mediapipe object and it will process the rgb frame
        
        if result_from_mp.multi_hand_landmarks:
            for hand_landmarks in result_from_mp.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        
        cv.imshow("SKIBIDI HAND GESTURE RECO", frame) #DISPLAYS THE FRAME
        if cv.waitKey(1) & 0xFF ==ord('q'): #TO QUIT PRESS Q
            break
        
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
