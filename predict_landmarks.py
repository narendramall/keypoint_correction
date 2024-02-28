import cv2
import json
import argparse
from utils.optical_flow_sparse import correct_keypoints_with_optical_flow 
# from utils.optical_flow_dense import correct_keypoints_with_optical_flow 


# Function to draw landmarks on a frame
def draw_landmarks(frame, keypoints, frame_height, frame_width, color = (0, 255, 0)):
    for key, point in keypoints.items():
        if point['confidence'] >= 0.0:  # Draw only if confidence is high enough
            x = int(point['x'] * frame_width)
            y = int(point['y'] * frame_height)
            cv2.circle(frame, (x, y), 5, color, -1)  # Draw green circle
    return frame

# Function to draw lines between joints for hand and shoulder area skeleton
def draw_lines(frame, keypoints, frame_height, frame_width, color = (255, 0, 0)):
    # Updated joint pairs to include hand and shoulder skeleton
    joint_pairs = [
        ('neck_1_joint', 'right_shoulder_1_joint'),
        ('neck_1_joint', 'left_shoulder_1_joint'),
        ('right_shoulder_1_joint', 'right_forearm_joint'),  # Assuming 'right_hand_joint' is a key in your JSON
        ('left_shoulder_1_joint', 'left_forearm_joint'),  # Assuming 'left_hand_joint' is a key in your JSON
        ('right_forearm_joint', 'right_hand_joint'),  # Connect right hand to right hand
        ('left_forearm_joint', 'left_hand_joint'),  # Connect left hand to left hand
        # Additional connections for a more detailed skeleton, if needed
    ]

    for pair in joint_pairs:
        start_point, end_point = pair
        if start_point in keypoints and end_point in keypoints:
            start = keypoints[start_point]
            end = keypoints[end_point]
            if start['confidence'] >= 0.0 and end['confidence'] >= 0.0:
                start_x = int(start['x'] * frame_width)
                start_y = int(start['y'] * frame_height)
                end_x = int(end['x'] * frame_width)
                end_y = int(end['y'] * frame_height)
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)  # Draw blue line
    return frame

# Function to process the video and draw landmarks
def process_video(video_path, keypoints_data, output_video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object to write the video
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MP4V'), frame_rate, (frame_width, frame_height))
    
    prev_frame = None
    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if the current frame has keypoints in the JSON data
        if str(frame_no) in keypoints_data:
            frame = draw_landmarks(frame, keypoints_data[str(frame_no)]['pose'], frame_height, frame_width, color = (0, 255, 0))
            frame = draw_lines(frame, keypoints_data[str(frame_no)]['pose'], frame_height, frame_width)

            # Correct keypoints using optical flow if not the first frame
            if prev_frame is not None:
                keypoints_data[str(frame_no)]['pose'] = correct_keypoints_with_optical_flow(prev_frame, frame, keypoints_data[str(frame_no)]['pose'], frame_height, frame_width)
            
            frame = draw_lines(frame, keypoints_data[str(frame_no)]['pose'], frame_height, frame_width, color = (0, 0, 255))
        # Draw the frame number on the top-left corner of the frame
        cv2.putText(frame, f'Frame: {frame_no}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        frame_no += 1

        # Update prev_frame for the next iteration
        prev_frame = frame.copy()

    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if  __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description='Draw landmarks on a video based on keypoints from a JSON file.')
    # Adding arguments
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('json_file_path', type=str, help='Path to the JSON file containing keypoints')
    parser.add_argument('output_video_path', type=str, help='Path to the output video file')

    # Parse arguments
    args = parser.parse_args()
    # Load JSON data from a file
    with open(args.json_file_path) as f:
        json_data = json.load(f)

    # Assuming JSON data is structured with frame numbers as keys
    keypoints_data = {str(item['frameNo']): item for item in json_data if 'frameNo' in item}

    # Process the video
    process_video(args.video_path, keypoints_data, args.output_video_path)
