import cv2
import mediapipe as mp
import json
import os
import urllib.request

def download_model_if_needed(model_name, url):
    if not os.path.exists(model_name):
        print(f"Downloading {model_name}...")
        urllib.request.urlretrieve(url, model_name)
        print("Download complete.")

def process_holistic_single_video(video_path, output_json_path):
    print(f"Processing video: {video_path}")
    
    # Download models if needed (MediaPipe Tasks API requires physical model files)
    hand_model_path = 'hand_landmarker.task'
    pose_model_path = 'pose_landmarker_heavy.task'
    
    download_model_if_needed(hand_model_path, "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    download_model_if_needed(pose_model_path, "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
    
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # Setup Hand tracker
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
        
    # Setup Pose (body) tracker
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)
        
    word_frames = []
    frame_number = 0
    local_timestamp_ms = 0

    cap = cv2.VideoCapture(video_path)
    
    # Run both trackers simultaneously
    with HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
         PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (800, 750))
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            timestamp_ms = local_timestamp_ms
            local_timestamp_ms += 33
            
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            frame_data = {
                "Frame": frame_number,
                "Pose Coordinates": [],
                "Left Hand Coordinates": [],
                "Right Hand Coordinates": []
            }
            
            # Extract Pose (Shoulders, Elbows, Wrists)
            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                landmarks = pose_result.pose_landmarks[0]
                for idx, landmark in enumerate(landmarks):
                    if idx in [11, 12, 13, 14, 15, 16]: 
                        frame_data["Pose Coordinates"].append({
                            "Joint Index": idx,
                            "Coordinates": [landmark.x, landmark.y, landmark.z]
                        })

            # Extract Hands
            if hand_result.hand_landmarks:
                for i in range(len(hand_result.hand_landmarks)):
                    hand_landmarks = hand_result.hand_landmarks[i]
                    handedness = hand_result.handedness[i][0].category_name
                    
                    coords = []
                    for idx, landmark in enumerate(hand_landmarks):
                        coords.append({
                            "Joint Index": idx,
                            "Coordinates": [landmark.x, landmark.y, landmark.z]
                        })
                        
                    if handedness == "Left":
                        frame_data["Left Hand Coordinates"] = coords
                    else:
                        frame_data["Right Hand Coordinates"] = coords
            
            word_frames.append(frame_data)
            frame_number += 1
            
    cap.release()
    
    with open(output_json_path, 'w') as f:
        json.dump(word_frames, f, indent=4)
        
    print(f"Successfully processed {frame_number} frames and saved to {output_json_path}")

if __name__ == '__main__':
    my_video_file = r'dataset\hello\27172.mp4' 
    output_file = 'static/json/single_word_holistic.json'
    
    if os.path.exists(my_video_file):
        process_holistic_single_video(my_video_file, output_file)
    else:
        print(f"Error: Could not find video at {my_video_file}")
                