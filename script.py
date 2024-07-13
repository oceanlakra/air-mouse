import cv2
import mediapipe as mp
import pyautogui
import time

#setting scaling factor for calibrating sensitivity
scaling_factor = 1.2

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
#using Phone camera as webcam for better quality and faster detection
cap = cv2.VideoCapture(2)

# Screen size
screen_width, screen_height = pyautogui.size()

#initializing the relative flag to toggle relative mode
relative = True

#Variables to store previous position of the palm
prev_x, prev_y = 0, 0
hand_present = False

# Tap detection variables
tap_threshold = 0.02
last_tap_time = 0
tap_delay = 0.9
tap_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hand_detected = False
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get coordinates of palm (wrist)
            palm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Calculate the distance between thumb and index finger tips
            distance = ((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2) ** 0.5
            
            #checking if relative moment or absolute
            if relative:
                #Get relative moment
                if hand_present:
                    delta_x = (palm.x - prev_x)*scaling_factor
                    delta_y = (palm.y - prev_y)*scaling_factor

                    #move the mouse by relative movement
                    pyautogui.moveRel(delta_x*1000, delta_y*1000)
                
                prev_x, prev_y = palm.x, palm.y
                hand_present = True
                hand_detected = True
            else:
                # Get screen coordinates, multiplying scaling_factor for calibration of display
                screen_x = int(palm.x * screen_width * scaling_factor)
                screen_y = int(palm.y * screen_height * scaling_factor)
                
                # Move the mouse
                pyautogui.moveTo(screen_x, screen_y)
            
            # Detect tap
            if distance < tap_threshold:
                current_time = time.time()
                if current_time - last_tap_time < tap_delay:
                    tap_count += 1
                else:
                    tap_count = 1
                last_tap_time = current_time
                
                # Perform single click for one tap and double click for two taps
                if tap_count == 1:
                    pyautogui.click()
                elif tap_count == 2:
                    pyautogui.doubleClick()
                    tap_count = 0  # Reset tap count after double click
        if relative:
            if not hand_detected:
                hand_present = False  
    else:
        hand_present = False
    
    # Display the frame
    cv2.imshow('Hand Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
