import numpy as np
import cv2

def correct_keypoints_with_optical_flow(prev_frame, curr_frame, keypoints, frame_height, frame_width):
    # Convert frames to grayscale for optical flow calculation
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Prepare the keypoints for optical flow calculation
    points_prev = np.float32([
        [keypoint['x'] * frame_width, keypoint['y'] * frame_height]
        for key, keypoint in keypoints.items()
        if keypoint['confidence'] > 0.0
    ]).reshape(-1, 1, 2)
    # Check if there are points to track
    if points_prev.shape[0] == 0:
        return keypoints  # No keypoints to correct
    
    # Parameters for lucas kanade optical flow 
    lk_params = dict( winSize = (15, 15), 
                    maxLevel = 4, 
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                                10, 0.03)) 
        
    # Calculate optical flow (Lucas-Kanade method)
    points_curr, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points_prev, None, **lk_params)
    
    # Update keypoints positions based on optical flow
    corrected_keypoints = {}
    i = 0
    for key, keypoint in keypoints.items():
        if keypoint['confidence'] > 0.0 and status[i] == 1:
            corrected_x = points_curr[i][0][0] / frame_width
            corrected_y = points_curr[i][0][1] / frame_height
            corrected_keypoints[key] = {'x': corrected_x, 'y': corrected_y, 'confidence': keypoint['confidence']}
        else:
            corrected_keypoints[key] = keypoint  # Keep original if flow calculation failed
        i += 1
    
    return corrected_keypoints
