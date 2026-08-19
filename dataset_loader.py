import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ASLDataset(Dataset):
    """
    PyTorch Dataset for ASL landmark sequences.
    Converts 3D hand landmark coordinates (Left + Right hand = 126 dimensions per frame)
    into fixed-length PyTorch tensors for sequence training and fine-tuning.
    If holistic=True, includes 6 pose landmarks (shoulders, elbows, wrists) for 144 dimensions.
    """
    def __init__(self, json_path='static/json/reference.json', max_seq_len=60, holistic=False):
        import time
        self.holistic = holistic
        for attempt in range(5):
            try:
                with open(json_path, 'r') as f:
                    self.raw_data = json.load(f)
                break
            except (json.JSONDecodeError, PermissionError):
                time.sleep(0.2)
        else:
            with open(json_path, 'r') as f:
                self.raw_data = json.load(f)

            
        # Filter out words with empty frame sequences
        self.words = [w for w, frames in self.raw_data.items() if len(frames) > 0]
        self.words.sort()
        
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        self.idx_to_word = {i: word for i, word in enumerate(self.words)}
        self.max_seq_len = max_seq_len
        self.num_classes = len(self.words)
        
        self.samples = []
        self.labels = []
        self._prepare_data()

    def _extract_frame_features(self, frame_data):
        left_coords = np.zeros((21, 3), dtype=np.float32)
        right_coords = np.zeros((21, 3), dtype=np.float32)
        
        left_list = frame_data.get('Left Hand Coordinates', [])
        for joint in left_list:
            idx = joint.get('Joint Index', 0)
            if idx < 21:
                left_coords[idx] = joint.get('Coordinates', [0, 0, 0])
                
        right_list = frame_data.get('Right Hand Coordinates', [])
        for joint in right_list:
            idx = joint.get('Joint Index', 0)
            if idx < 21:
                right_coords[idx] = joint.get('Coordinates', [0, 0, 0])
                
        if getattr(self, 'holistic', False):
            pose_coords = np.zeros((6, 3), dtype=np.float32)
            pose_idx_map = {11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5}
            pose_list = frame_data.get('Pose Coordinates', [])
            for joint in pose_list:
                idx = joint.get('Joint Index', 0)
                if idx in pose_idx_map:
                    pose_coords[pose_idx_map[idx]] = joint.get('Coordinates', [0, 0, 0])
            return np.concatenate([pose_coords.flatten(), left_coords.flatten(), right_coords.flatten()], axis=0) # 144 features
        else:
            return np.concatenate([left_coords.flatten(), right_coords.flatten()], axis=0) # 126 features

    def _prepare_data(self):
        feature_dim = 144 if self.holistic else 126
        for word in self.words:
            sequences = self.raw_data[word]
            
            # Check if this is a single sequence or a list of sequences
            if len(sequences) > 0 and isinstance(sequences[0], dict):
                sequences = [sequences]
                
            for frames in sequences:
                if len(frames) == 0:
                    continue
                    
                seq_feat = []
                for frame in frames:
                    feat = self._extract_frame_features(frame)
                    seq_feat.append(feat)
                    
                seq_feat = np.array(seq_feat, dtype=np.float32) # shape: (num_frames, feature_dim)
                
                # Pad or truncate to max_seq_len
                seq_len = seq_feat.shape[0]
                if seq_len < self.max_seq_len:
                    padding = np.zeros((self.max_seq_len - seq_len, feature_dim), dtype=np.float32)
                    padded_seq = np.vstack([seq_feat, padding])
                else:
                    padded_seq = seq_feat[:self.max_seq_len]
                    
                self.samples.append(padded_seq)
                self.labels.append(self.word_to_idx[word])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def get_dataloader(json_path='static/json/reference_holistic.json', batch_size=32, shuffle=True, max_seq_len=60, holistic=False):
    dataset = ASLDataset(json_path=json_path, max_seq_len=max_seq_len, holistic=holistic)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset

