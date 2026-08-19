# ASL-Transformer: Holistic Text & Speech to Sign Language Animation 🤟

An interactive web application built with Python, Flask, OpenAI Whisper, MediaPipe, PyTorch, and Three.js. It converts spoken audio or input text into fully articulated 3D Sign Language (ASL) avatars in real-time.

![rancher-asl-final](https://github.com/bishal7679/ASL-Transformer/assets/70086051/82273ac1-e7be-4115-9326-c2887591e2ca)

---

## 🌟 Key Features

* 🎙️ **Speech-to-Text Translation**: Uses OpenAI's **Whisper** model to transcribe raw audio input into text.
* ✍️ **Text-to-Sign Translation**: Converts text or transcriptions into matching ASL word sequences using fuzzy text matching (Levenshtein distance).
* 🧍 **Holistic 3D Landmark Mapping**: Maps ASL vocabulary to pre-extracted 3D hand and posture coordinates using MediaPipe Holistic tracking (21-joint coordinates for left & right hands + 6 pose joints = **144 dimensions per frame**).
* 🤖 **PyTorch Transformer Model**: Features sequence dataset loading (dataset_loader.py) and fine-tuning scripts (ine_tune.py) using PyTorch (sl_transformer_model_holistic.pt). Now supports **multi-video processing** to dramatically increase training accuracy!
* 🎨 **Interactive 3D Avatar Rendering**: Renders sign language animations smoothly onto realistic .glb 3D Avatars in the browser using **Three.js**.

---

## 📁 Data Architecture & Dataset Usage

The system works with pre-extracted keypoint coordinates stored in JSON format via Git LFS:

1. **Pre-extracted Holistic Reference Data (static/json/reference_holistic.json)**:
   - Contains 3D joint and pose landmark coordinates generated from ASL dictionary videos.
   - **The web application and 3D rendering run directly off this reference file.**
   - 💡 **Note**: The raw dataset/ folder containing .mp4 video files is **not required** to run the app. You only need dataset/ if you wish to re-extract keypoints from new raw video clips.

2. **Holistic Multi-Video Extractor (process_dataset_holistic.py)**:
   - Uses MediaPipe Hand & Pose Landmarkers to process all raw MP4 videos per word.
   - Outputs both eference_holistic.json (for the web app) and 	raining_reference_holistic.json (a massive multi-video dataset for training the Transformer).

---

## 🧪 Tech Stack

* **Backend & API**: Python 3.9+, Flask
* **Speech Processing**: OpenAI Whisper, FFmpeg, PyTorch
* **Computer Vision**: MediaPipe Hand & Pose Landmarker tasks, OpenCV
* **Model Training**: PyTorch, NumPy, Levenshtein
* **Frontend**: HTML5, CSS3, JavaScript, Three.js (3D Animation), GLTF Models
* **Source Control**: Git Large File Storage (Git LFS)

---

## 💻 Local Setup & Installation

### Prerequisites
* Python 3.9 - 3.11
* FFmpeg (required for Whisper audio processing)
* **Git LFS** (Required to download the large JSON dataset files)

### Installation Steps

1. **Clone the Repository (with LFS):**
   `ash
   git lfs install
   git clone https://github.com/Lokajith-T/Text_To_Sign.git
   cd Text_To_Sign
   `

2. **Create and Activate a Virtual Environment:**
   `ash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   `

3. **Install Dependencies:**
   `ash
   pip install -r requirements.txt
   `

4. **Run the Application:**
   `ash
   python app.py
   `

5. **Access the Web App:** 
   Open your browser and navigate to http://localhost:5000.

---

## 🛠️ Project Structure

`	ext
├── app.py                      # Main Flask application & Whisper integration
├── dataset_loader.py           # PyTorch Dataset & DataLoader implementation
├── fine_tune.py                # Model training & fine-tuning script
├── process_dataset_holistic.py # MediaPipe multi-video holistic coordinate extractor
├── hand_landmarker.task        # MediaPipe hand landmarker task model
├── pose_landmarker_heavy.task  # MediaPipe pose landmarker task model
├── asl_transformer_model_holistic.pt # PyTorch trained model weights
├── requirements.txt            # Python dependencies
├── static/
│   ├── css/styles.css          # Frontend styling
│   ├── model/ManModel.glb      # 3D Avatar model
│   ├── js/                     # Three.js 3D rendering & client scripts
│   └── json/                   # Pre-extracted 3D hand/pose coordinates database (LFS)
└── templates/
    └── index.html              # Web interface template
`

---

## 🔖 What's Next & Future Roadmap

* ⏩ **Animation Speed Control**: Add user sliders for adjustable sign playback speeds.
* 🤲 **Enhanced Bi-Manual Movements**: Improve realistic hand-mesh smoothing when coordinates overlap.
* 🎥 **Reverse Translation**: Implement ASL-to-Text translation via webcam gesture recognition.
* 🎓 **Accessibility & Educational Tools**: Expand support for full sentence ASL grammar and educational phrase practicing.

## 📄 License
This project is open-source and available under the terms specified in the LICENSE file.
