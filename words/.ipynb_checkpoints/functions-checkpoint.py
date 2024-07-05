import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          ) 
    # Draw pose connections
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                             # mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             # mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             # ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
                             # mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=4), 
                             # mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             # ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
                             # mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             # mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             # )
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    # return np.concatenate([pose, face, lh, rh])
    return np.concatenate([pose, lh, rh])

def extract_specific_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    if(results.pose_landmarks):
        landmarks = results.pose_landmarks.landmark
        if(mp_holistic.PoseLandmark.NOSE):
            nose = np.array([landmarks[mp_holistic.PoseLandmark.NOSE].x, landmarks[mp_holistic.PoseLandmark.NOSE].y, landmarks[mp_holistic.PoseLandmark.NOSE].z, landmarks[mp_holistic.PoseLandmark.NOSE].visibility]).flatten()
        else:
            nose = np.zeros(4)
        if(mp_holistic.PoseLandmark.LEFT_SHOULDER):
            left_shoulder = np.array([landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].z, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].visibility]).flatten()
        else:
            left_shoulder = np.zeros(4)
        if(mp_holistic.PoseLandmark.RIGHT_SHOULDER):
            right_shoulder = np.array([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].z, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].visibility]).flatten()
        else:
            right_shoulder = np.zeros(4)
        if(mp_holistic.PoseLandmark.RIGHT_INDEX):
            right_index = np.array([landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX].x, landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX].y, landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX].z, landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX].visibility]).flatten()
        else:
            right_index = np.zeros(4)
    return np.concatenate([nose, left_shoulder, right_shoulder, right_index, lh, rh])
        
def important_features(image):

    list =[]

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic

    # Function to calculate 3D Euclidean distance
    def euclidean_distance_3d(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # Function to calculate slope of a line
    def calculate_slope(point1, point2):
        return (point2[1] - point1[1]) / (point2[0] - point1[0])

    # Process image with MediaPipe Holistic
    with mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the image
        results = holistic.process(image_rgb)

        if results.pose_landmarks:
            # Get pose landmarks
            landmarks = results.pose_landmarks.landmark

            # Extract nose and right palm landmarks with x, y, and z coordinates
            nose_center = [landmarks[mp_holistic.PoseLandmark.NOSE].x * image.shape[1], landmarks[mp_holistic.PoseLandmark.NOSE].y * image.shape[0]]
            right_index = [landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX].x * image.shape[1], landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX].y * image.shape[0]]
            right_index_array = np.array([landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX].x * image.shape[1],
                                  landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX].y * image.shape[0]])
            
            # Get coordinates of right and left shoulder landmarks
            right_shoulder = (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0])
            left_shoulder = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1], landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0])

            # Calculate 3D distance
            distance_3d = euclidean_distance_3d(nose_center, right_index)
            
            list.append(distance_3d)

            shoulder_slope = calculate_slope(right_shoulder, left_shoulder)
            # print(right_index[1], shoulder_slope * (right_index[0] - right_shoulder[0]) + right_shoulder[1])

            if right_index[1] > shoulder_slope * (right_index[0] - right_shoulder[0]) + right_shoulder[1]:
                list.append(0)
            else:
                list.append(1)

        else:
            list.append(0)
            list.append(0)

        if results.right_hand_landmarks:
            # Get hand landmarks
            landmarks = results.right_hand_landmarks.landmark

            # Display z-coordinate of index finger tip
            list.append(landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].z)

            if landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP] and landmarks[mp_holistic.HandLandmark.PINKY_TIP]:
                # Extract landmarks of interest
                # wrist = np.array([landmarks[mp_holistic.HandLandmark.WRIST].x * image.shape[1],
                #                   landmarks[mp_holistic.HandLandmark.WRIST].y * image.shape[0]])
                index_finger_tip = np.array([landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1],
                                             landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]])
                pinky_tip = np.array([landmarks[mp_holistic.HandLandmark.PINKY_TIP].x * image.shape[1],
                                      landmarks[mp_holistic.HandLandmark.PINKY_TIP].y * image.shape[0]])

                # Calculate vectors representing fingers
                vector_index_finger = index_finger_tip - right_index_array
                vector_pinky = pinky_tip - right_index_array

                # Calculate dot product and magnitudes
                dot_product = np.dot(vector_index_finger, vector_pinky)
                magnitude_index_finger = np.linalg.norm(vector_index_finger)
                magnitude_pinky = np.linalg.norm(vector_pinky)

                # Calculate cosine of the angle between fingers
                cosine_angle = dot_product / (magnitude_index_finger * magnitude_pinky)

                # Calculate angle in radians and then convert to degrees
                angle_radians = np.arccos(cosine_angle)
                angle_degrees = np.degrees(angle_radians)

                list.append(angle_degrees)

            else:
                list.append(0)


        else:
            list.insert(2,0)
            list.append(0)

    return list

actions = np.array(['hello!', 'how', "you", "good", "bye"])