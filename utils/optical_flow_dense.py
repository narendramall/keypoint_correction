import numpy as np
import cv2

def correct_keypoints_with_optical_flow(prev_frame, curr_frame, keypoints, frame_height, frame_width):
    # Convert frames to grayscale for optical flow calculation
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 5, 15, 15, 7, 1.5, 0)
    
    # Update keypoints positions based on optical flow
    corrected_keypoints = {}
    for key, keypoint in keypoints.items():
        if keypoint['confidence'] > 0.0:
            original_position = np.array([keypoint['x'] * frame_width, keypoint['y'] * frame_height], dtype=np.float32)
            displacement = flow[int(original_position[1]), int(original_position[0])]
            
            corrected_x = (original_position[0] + displacement[0]) / frame_width
            corrected_y = (original_position[1] + displacement[1]) / frame_height
            
            # Ensure corrected positions are within frame boundaries
            corrected_x = min(max(corrected_x, 0), 1)
            corrected_y = min(max(corrected_y, 0), 1)
            
            corrected_keypoints[key] = {'x': corrected_x, 'y': corrected_y, 'confidence': keypoint['confidence']}
        else:
            corrected_keypoints[key] = keypoint  # Keep original if confidence is low
    
    return corrected_keypoints
