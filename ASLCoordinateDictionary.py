import mediapipe as mp
import cv2
import numpy as np
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

video_folder = 'dataset/'
json_path = 'static/json/reference.json'

def atomic_json_dump(data_dict, file_path):
    tmp_path = file_path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(data_dict, f)
    os.replace(tmp_path, file_path)

def process_single_word(word_name, word_dir):
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.8,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    mp4_files = [f for f in os.listdir(word_dir) if f.endswith(".mp4")]
    if not mp4_files:
        return word_name, []

    word_frames = []
    frame_number = 0
    local_timestamp_ms = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        video_file = os.path.join(word_dir, mp4_files[0])
        cap = cv2.VideoCapture(video_file)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (800, 750))
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            timestamp_ms = local_timestamp_ms
            local_timestamp_ms += 33

            hand_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if hand_landmarker_result.hand_landmarks:
                hand_coordinates_left = []
                hand_coordinates_right = []
                
                for i in range(len(hand_landmarker_result.hand_landmarks)):
                    hand_landmarks = hand_landmarker_result.hand_landmarks[i]
                    handedness_category = hand_landmarker_result.handedness[i][0]
                    is_left = handedness_category.category_name == "Left"
                    
                    coords = []
                    for joint_id, landmark in enumerate(hand_landmarks):
                        coords.append({
                            "Joint Index": joint_id,
                            "Coordinates": [landmark.x, landmark.y, landmark.z]
                        })
                        
                    if is_left:
                        hand_coordinates_left = coords
                    else:
                        hand_coordinates_right = coords
                
                word_frames.append({
                    "Frame": frame_number,
                    "Left Hand Coordinates": hand_coordinates_left,
                    "Right Hand Coordinates": hand_coordinates_right
                })
                frame_number += 1
        cap.release()

    # Frame gap interpolation
    num_frames = len(word_frames)
    if num_frames > 1:
        interpolated_frames = []
        for i in range(num_frames - 1):
            current_frame = word_frames[i]
            next_frame = word_frames[i + 1]
            if next_frame["Frame"] - current_frame["Frame"] > 1:
                gap = next_frame["Frame"] - current_frame["Frame"]
                for j in range(1, gap):
                    interpolation_ratio = j / gap
                    interpolated_left = []
                    interpolated_right = []
                    
                    for joint_data in current_frame.get("Left Hand Coordinates", []):
                        current_coords = joint_data["Coordinates"]
                        idx_joint = joint_data["Joint Index"]
                        next_left = next_frame.get("Left Hand Coordinates", [])
                        if idx_joint < len(next_left):
                            next_coords = next_left[idx_joint]["Coordinates"]
                            interpolated_left.append({
                                "Joint Index": idx_joint,
                                "Coordinates": [
                                    current_coords[0] + (next_coords[0] - current_coords[0]) * interpolation_ratio,
                                    current_coords[1] + (next_coords[1] - current_coords[1]) * interpolation_ratio,
                                    current_coords[2] + (next_coords[2] - current_coords[2]) * interpolation_ratio
                                ]
                            })
                            
                    for joint_data in current_frame.get("Right Hand Coordinates", []):
                        current_coords = joint_data["Coordinates"]
                        idx_joint = joint_data["Joint Index"]
                        next_right = next_frame.get("Right Hand Coordinates", [])
                        if idx_joint < len(next_right):
                            next_coords = next_right[idx_joint]["Coordinates"]
                            interpolated_right.append({
                                "Joint Index": idx_joint,
                                "Coordinates": [
                                    current_coords[0] + (next_coords[0] - current_coords[0]) * interpolation_ratio,
                                    current_coords[1] + (next_coords[1] - current_coords[1]) * interpolation_ratio,
                                    current_coords[2] + (next_coords[2] - current_coords[2]) * interpolation_ratio
                                ]
                            })
                            
                    interpolated_frames.append({
                        "Frame": current_frame["Frame"] + j,
                        "Left Hand Coordinates": interpolated_left,
                        "Right Hand Coordinates": interpolated_right
                    })
        word_frames.extend(interpolated_frames)
        word_frames.sort(key=lambda x: x["Frame"])

    return word_name, word_frames

def main():
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    all_words = [w for w in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, w))]
    all_words.sort()
    
    remaining_words = [w for w in all_words if w not in data or len(data[w]) == 0]
    total_words = len(all_words)
    already_done = total_words - len(remaining_words)
    
    print(f"Total words: {total_words} | Already completed: {already_done} | Remaining to process: {len(remaining_words)}")

    if not remaining_words:
        print("All words have already been processed into reference.json!")
        return

    workers = min(8, os.cpu_count() or 4)
    print(f"Starting parallel processing using {workers} CPU workers...")

    completed_count = already_done

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_word = {
            executor.submit(process_single_word, word_name, os.path.join(video_folder, word_name)): word_name
            for word_name in remaining_words
        }

        for future in as_completed(future_to_word):
            word_name = future_to_word[future]
            try:
                w_name, frames = future.result()
                data[w_name] = frames
                completed_count += 1
                
                if completed_count % 10 == 0 or completed_count == total_words:
                    print(f"Progress: {completed_count}/{total_words} words completed ({completed_count / total_words * 100:.1f}%) | Latest: {w_name}", flush=True)
                    atomic_json_dump(data, json_path)
            except Exception as e:
                print(f"Error processing word {word_name}: {e}", flush=True)

    atomic_json_dump(data, json_path)
    print("ASLCoordinateDictionary parallel generation complete for all words!")

if __name__ == '__main__':
    main()


