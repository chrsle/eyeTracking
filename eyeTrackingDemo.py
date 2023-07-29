import cv2
import dlib
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import os
import pyautogui
import sys

sys.path.insert(0, "/Users/chrislee/dev/eyetracking/GazeTracking")
from gaze_tracking import GazeTracking



# Load Dlib's facial landmarks model and MTCNN model
predictor = dlib.shape_predictor("/Users/chrislee/dev/eyetracking/shape_predictor_68_face_landmarks.dat")
detector = MTCNN()

# Initialize the GazeTracking object
gaze = GazeTracking()

# Open a handler for the camera
video_capture = cv2.VideoCapture(1)

# Get the resolution of the webcam
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the heatmap
heatmap = np.zeros((frame_height, frame_width))


# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


output_dir = "/Users/chrislee/dev/eyetracking/asdf"
os.makedirs(output_dir, exist_ok=True) 

output_video_path = os.path.join(output_dir, 'output.mp4')
overlay_output_video_path = os.path.join(output_dir, 'overlay_output.mp4')

out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))
overlay_out = cv2.VideoWriter(overlay_output_video_path, fourcc, 20.0, (frame_width, frame_height))

frame_count = 0

def draw_eye_landmarks(frame, landmarks, start, end):
    points = []
    for i in range(start, end+1):
        point = [landmarks.part(i).x, landmarks.part(i).y]
        points.append(point)
    points = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [points], True, (0, 255, 0), 1)

def detect_eyes(frame):
    global frame_count
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detect_faces(frame)
    overlay_frame = frame.copy()
   
    for result in faces:
        x, y, w, h = result['box']
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = predictor(gray, rect)
        draw_eye_landmarks(frame, landmarks, 36, 41)  # Left eye
        draw_eye_landmarks(frame, landmarks, 42, 47)  # Right eye

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around face

        for i in range(36, 48):  # Draw circles around eyes
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 0, 255), -1)

        # Refresh the gaze object with the new frame
        gaze.refresh(frame)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        print("Left pupil: ", left_pupil)
        print("Right pupil: ", right_pupil)


        if left_pupil is not None and right_pupil is not None:
            print('here')

            print(int((left_pupil[1] + right_pupil[1]) / 2))

            print(int((left_pupil[0] + right_pupil[0]) / 2))
            
        
            # Update the heatmap
            heatmap[int((left_pupil[1] + right_pupil[1]) / 2), int((left_pupil[0] + right_pupil[0]) / 2)] += 1

            print("Updated heatmap at: ", (int((left_pupil[1] + right_pupil[1]) / 2), int((left_pupil[0] + right_pupil[0]) / 2)))

            # Normalize the heatmap for visualization
            heatmap_norm = heatmap / (np.max(heatmap) + 1e-10)

            # Create a colorized version of the heatmap
            heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)

            # Resize the heatmap to match the frame size
            heatmap_img = cv2.resize(heatmap_img, (frame.shape[1], frame.shape[0]))


            # Blend the heatmap and the original frame
            overlay_frame = cv2.addWeighted(frame, 0.5, heatmap_img, 0.5, 0)

            # Specify the paths to the output video files
            output_video_path = os.path.join(output_dir, 'output.mp4')
            overlay_output_video_path = os.path.join(output_dir, 'overlay_output.mp4')

            # Initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))
            overlay_out = cv2.VideoWriter(overlay_output_video_path, fourcc, 20.0, (frame_width, frame_height))

            heatmap_filename = os.path.join(output_dir, 'heatmap' + str(frame_count) + '.png')
            cv2.imwrite(heatmap_filename, heatmap_img)
            frame_count += 1  # Increment frame counter

             

    return frame, overlay_frame


while True:
   
   
    _, frame = video_capture.read()

    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    
    frame, overlay_frame = detect_eyes(frame)

    if frame is None:
        print("Could not capture frame. Ending the loop.")
        continue

    # Write the frames to the output video files
    out.write(frame)
    overlay_out.write(overlay_frame)

    cv2.imshow('Video', overlay_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
overlay_out.release()
cv2.destroyAllWindows()

# Normalize the heatmap for better visualization
heatmap = heatmap / np.max(heatmap)

# Create a colorized version of the heatmap
heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

output_dir = "/Users/chrislee/dev/eyetracking/asdf"

# Specify the path to the heatmap image file
heatmap_image_path = os.path.join(output_dir, 'heatmap.png')

# Save the heatmap image
cv2.imwrite(heatmap_image_path, heatmap_img)
