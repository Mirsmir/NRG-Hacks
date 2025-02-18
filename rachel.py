import csv 
import cv2 as cv
import mediapipe as mp

def videoCapture():
    cap = cv2.VideoCapture(0) #pretty sure this 0 just means capture from default webcam (built in)
    
    while cap.isOpened(): #capturing the frames for mediapipe continuously in a loop
        #wait are we gunna have to multithread? 
        ret, frame = cap.read()
        if not ret:
            break


def initiModules(): 
    mp_hands = mp.solutions.hands #solutions.hands is the points that mediapipe makes I think? ??/  ? / ?
    
    mp_drawing = mp.solutions.drawing_utils #drawing the landparks from mp.solutions
    
    hands = mp_hands.Hands(
        static_image_mode=False, #false because otherwise, the lib will treat every frame of the video as a seperate image
        #so it will REdetect the hands every frame, not really tracking every single position (THIS IS SUPER SLOW)
        #we need it to detect the hands relative to a frame, meaning IT WILL DETECT ONCE, then track where that same object lies in future hands
        #I guess this is like after you found the object, you don't need to find it again, you can track it after you've found it. 
        max_num_hands=2, #need cos were gunna have to halves 
        min_detection_confidence=0.9, #mediapipe will ONLY detect the hand if it measures that its 90% confident that the hand is there. If the hand is obstructed and the landmarks
        #are slightly invisible, then the confidence goes down, and if it goes down to 10% then it just wont draw a landmark 
        min_tracking_confidence=0.9) #minimum tracking score means it will track ACROSS MULTIPLE FRAMES. 
        #if the confidence for tracking drops below this point then it REDETECTS THE HANDS (RE FINDS) 
        
        #If hands keep disappearing, lower this to 0.3 so Mediapipe tracks hands more aggressively (without stopping).
        #If hands flicker, increase it to 0.7 for more stable tracking.
    
    #OUTPUT OF THE DETECTION (in hands var) will be given in X, Y, and Z coords
    

