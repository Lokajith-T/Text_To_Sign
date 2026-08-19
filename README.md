# ASL-Transformer: Text & Speech to Sign Language Animation 🤟

An interactive web application built with Python, Flask, OpenAI Whisper, MediaPipe, PyTorch, and Three.js. It converts spoken audio or input text into 3D Sign Language (ASL) hand animations in real-time.

![rancher-asl-final](https://github.com/bishal7679/ASL-Transformer/assets/70086051/82273ac1-e7be-4115-9326-c2887591e2ca)

---

## 🌟 Key Features

* 🎙️ **Speech-to-Text Translation**: Uses OpenAI's **Whisper** model to transcribe raw audio input into text.
* ✍️ **Text-to-Sign Translation**: Converts text or transcriptions into matching ASL word sequences using fuzzy text matching (Levenshtein distance).
* 🖐️ **3D Hand Landmark Mapping**: Maps ASL vocabulary to pre-extracted 3D hand landmark coordinates (MediaPipe 21-joint coordinates for left & right hands = 126 dimensions per frame).
* 🤖 **PyTorch Transformer Model**: Features sequence dataset loading (`dataset_loader.py`) and fine-tuning scripts (`fine_tune.py`) using PyTorch (`asl_transformer_model.pt`).
* 🎨 **Interactive 3D Web Rendering**: Renders sign language animations smoothly in the browser using **Three.js**.

![04-09-2023:22:09:14](https://github.com/bishal7679/ASL-Transformer/assets/70086051/397f0106-0284-4067-a555-3d4a43d9244f)

---

## 📁 Data Architecture & Dataset Usage

The system works with pre-extracted keypoint coordinates stored in JSON format:

1. **Pre-extracted Reference Data (`static/json/reference.json`)**:
   - Contains 3D joint landmark coordinates generated from ASL dictionary videos.
   - **The web application, model fine-tuning, and 3D rendering run directly off this reference file.**
   - 💡 **Note**: The raw `dataset/` folder containing `.mp4` video files is **not required** to run the app or execute PyTorch training. You only need `dataset/` if you wish to run `ASLCoordinateDictionary.py` to re-extract keypoints from new raw video clips.

2. **ASL Keypoint Extractor (`ASLCoordinateDictionary.py`)**:
   - Uses MediaPipe Hand Landmarker (`hand_landmarker.task`) to process raw MP4 videos and output `static/json/reference.json`.

---

## 🧪 Tech Stack

* **Backend & API**: Python 3.9+, Flask
* **Speech Processing**: OpenAI Whisper, FFmpeg, PyTorch
* **Computer Vision**: MediaPipe Hand Landmarker (`hand_landmarker.task`), OpenCV
* **Model Training**: PyTorch, NumPy, Levenshtein
* **Frontend**: HTML5, CSS3, JavaScript, Three.js (3D Animation)
* **Containerization**: Docker

---

## 💻 Local Setup & Installation

### Prerequisites
* Python 3.9 - 3.11
* FFmpeg (required for Whisper audio processing)

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Lokajith-T/Text_To_Sign.git
   cd Text_To_Sign
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the Web App**:
   Open your browser and navigate to `http://localhost:5000`.

---

## 🐳 Docker Setup

You can build and run the application in a Docker container without installing local Python dependencies:

1. **Build the Docker Image**:
   ```bash
   docker build -t asl-transformer .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 5000:5000 asl-transformer
   ```

3. **Access the App**:
   Navigate to `http://localhost:5000` in your web browser.

---

## 🛠️ Project Structure

```
├── app.py                      # Main Flask application & Whisper integration
├── dataset_loader.py           # PyTorch Dataset & DataLoader implementation
├── fine_tune.py                # Model training & fine-tuning script
├── ASLCoordinateDictionary.py # MediaPipe coordinate extractor from MP4 videos
├── hand_landmarker.task        # MediaPipe landmarker task model
├── asl_transformer_model.pt    # PyTorch trained model weights
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
├── static/
│   ├── css/styles.css          # Frontend styling
│   ├── js/                     # Three.js 3D rendering & client scripts
│   └── json/reference.json     # Pre-extracted 3D hand coordinates database
└── templates/
    └── index.html              # Web interface template
```

---

## 🔖 What's Next & Future Roadmap

* ⏩ **Animation Speed Control**: Add user sliders for adjustable sign playback speeds.
* 🤲 **Enhanced Bi-Manual Movements**: Improve realistic hand-mesh smoothing when coordinates overlap.
* 🎥 **Reverse Translation**: Implement ASL-to-Text translation via webcam gesture recognition.
* 🎓 **Accessibility & Educational Tools**: Expand support for full sentence ASL grammar and educational phrase practicing.

---

## 📄 License
This project is open-source and available under the terms specified in the [LICENSE](file:///d:/Text_To_Sign-main/LICENSE) file.
