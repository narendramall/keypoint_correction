## Algorithm Used
### Tracking Algorithm Used
1. Sparse Optical Flow
Sparse feature point have been detected b/w two consicutive key frame
``` cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points_prev, None, **lk_params)```

2. Dense Optical Flow
Dense feature point have been detected b/w two consicutive key frame
``` cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 5, 15, 15, 7, 1.5, 0)```

### Smoothening Algorithm
We can use low pass filter to smoothen the keypoints.
I have implemented One Euro Filter for one of my project but due to time constraint I could not implement it here
(https://gist.github.com/3846masa/5628f711e86fd62bea56b18e32177c60)

## How to run the code
1. Create & activate environment using conda 
``` conda create -n onform python=3.10```
``` conda activate onform```

2. Install dependency
``` pip install -r requirements.txt```

3. Run the prediction file

positional arguments:
  video_path         Path to the input video file
  json_file_path     Path to the JSON file containing keypoints
  output_video_path  Path to the output video file

e.g.
``` python predict_landmarks.py assets/test_data/test2/video.mp4 assets/test_data/test2/Old_JSON.json assets/test_data/test2/output.mp4```

