import cv2
import mediapipe as mp
import json
import os
import urllib.request
import glob
from tqdm import tqdm

def download_model_if_needed(model_name, url):
    if not os.path.exists(model_name):
        print(f"Downloading {model_name}...")
        urllib.request.urlretrieve(url, model_name)
        print("Download complete.")

def process_dataset(dataset_dir, output_json_path):
    print(f"Processing dataset in: {dataset_dir}")
    
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

    # Load existing progress if any
    all_words_data = {}
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r') as f:
                all_words_data = json.load(f)
            print(f"Loaded existing data for {len(all_words_data)} words.")
        except Exception as e:
            print(f"Error loading existing JSON, starting fresh: {e}")

    word_folders = [f.path for f in os.scandir(dataset_dir) if f.is_dir()]
    print(f"Found {len(word_folders)} word folders.")

    with HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
         PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
         
        local_timestamp_ms = 0
        for folder in tqdm(word_folders, desc="Processing Words"):
            word = os.path.basename(folder).lower()
            
            # Skip if already processed
            if word in all_words_data and len(all_words_data[word]) > 0:
                continue

            # Find the first mp4 video in the folder
            video_files = glob.glob(os.path.join(folder, "*.mp4"))
            if not video_files:
                continue
            
            video_path = video_files[0]
            
            cap = cv2.VideoCapture(video_path)
            
            frame_number = 0
            word_frames = []
            
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
            
            if len(word_frames) > 0:
                all_words_data[word] = word_frames
                
                # Save progress periodically (e.g., every 50 words to not slow down too much)
                if len(all_words_data) % 50 == 0:
                    with open(output_json_path, 'w') as f:
                        json.dump(all_words_data, f, indent=4)
                        
    # Final save
    with open(output_json_path, 'w') as f:
        json.dump(all_words_data, f, indent=4)
        
    print(f"Successfully processed {len(all_words_data)} words and saved to {output_json_path}")

if __name__ == '__main__':
    dataset_dir = 'dataset'
    output_file = 'static/json/reference_holistic.json'
    
    if os.path.exists(dataset_dir):
        process_dataset(dataset_dir, output_file)
    else:
        print(f"Error: Could not find dataset directory at {dataset_dir}")
