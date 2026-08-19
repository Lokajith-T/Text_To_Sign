from flask import Flask, render_template, request, jsonify
import numpy as np
from subprocess import CalledProcessError, run
import os

# Handle numba import failure when Windows Application Control blocks compiled DLLs
try:
    import numba
except Exception:
    import sys, types
    _n = types.ModuleType('numba')
    _n.jit = lambda *a, **k: (lambda fn: fn)
    sys.modules['numba'] = _n

import whisper

# libraries for text modification
from Levenshtein import ratio
import re
import json
import time


app = Flask(__name__, template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True

model = whisper.load_model('base')

SAMPLE_RATE = 16000


def get_ffmpeg_cmd():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


# converts byte data to what whisper can use (adapted from https://github.com/openai/whisper/blob/main/whisper/audio.py)
def custom_load_audio(byte_data: bytes, sr=SAMPLE_RATE):
    cmd = [
        get_ffmpeg_cmd(),
        "-nostdin",
        "-threads", "0",
        "-i", "-",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    try:
        out = run(cmd, input=byte_data, capture_output=True, check=True).stdout
    except Exception as e:
        try:
            import io, scipy.io.wavfile as wavfile, scipy.signal as signal
            rate, data = wavfile.read(io.BytesIO(byte_data))
            if data.ndim > 1:
                data = data.mean(axis=1)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0
            if rate != sr:
                num_samples = int(len(data) * sr / rate)
                data = signal.resample(data, num_samples).astype(np.float32)
            return data.astype(np.float32)
        except Exception:
            raise RuntimeError(f"Failed to load audio: {e}")
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def process_audio(audio):
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text


import pickle

_CACHED_REF_DATA = {}
_CACHED_MTIME = 0
_CACHED_HOLISTIC_MTIME = 0
_LAST_LOAD_TIME = 0

def get_reference_data():
    global _CACHED_REF_DATA, _CACHED_MTIME, _CACHED_HOLISTIC_MTIME, _LAST_LOAD_TIME
    json_path = 'static/json/reference.json'
    holistic_path = 'static/json/reference_holistic.json'
    pkl_path = 'static/json/reference.pkl'
    now = time.time()
    try:
        should_reload_ref = False
        if os.path.exists(json_path):
            mtime = os.path.getmtime(json_path)
            if (mtime != _CACHED_MTIME and now - _LAST_LOAD_TIME > 30) or not _CACHED_REF_DATA:
                should_reload_ref = True
                if os.path.exists(pkl_path) and os.path.getmtime(pkl_path) >= mtime:
                    with open(pkl_path, 'rb') as pkl_file:
                        _CACHED_REF_DATA = pickle.load(pkl_file)
                else:
                    with open(json_path, 'r') as json_file:
                        _CACHED_REF_DATA = json.load(json_file)
                    try:
                        with open(pkl_path, 'wb') as pkl_file:
                            pickle.dump(_CACHED_REF_DATA, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception:
                        pass
                _CACHED_MTIME = mtime
                _LAST_LOAD_TIME = now

        # Always try to merge in holistic data if it has been updated
        if os.path.exists(holistic_path):
            holistic_mtime = os.path.getmtime(holistic_path)
            if holistic_mtime != _CACHED_HOLISTIC_MTIME or should_reload_ref:
                try:
                    with open(holistic_path, 'r') as holistic_file:
                        holistic_data = json.load(holistic_file)
                        _CACHED_REF_DATA.update(holistic_data)
                    _CACHED_HOLISTIC_MTIME = holistic_mtime
                except Exception as e:
                    print(f"Error loading holistic data: {e}")
                    
        return _CACHED_REF_DATA
    except Exception:
        pass
    return _CACHED_REF_DATA or {}


# Pre-warm reference data cache in memory on server start
get_reference_data()



def modify_words(text):  # modifies words so all of them are in the dictionary
    ref_data = get_reference_data()
    clean_text = text.lower().strip()
    words = re.findall(r'\b[\w\s]+\b', clean_text)
    
    # Check full phrase or multi-word matches first
    if clean_text in ref_data:
        return clean_text
        
    single_words = re.findall(r'\b\w+\b', clean_text)
    modified_words = []
    
    i = 0
    while i < len(single_words):
        # Try 2-word phrase
        if i + 1 < len(single_words):
            two_word = f"{single_words[i]} {single_words[i+1]}"
            if two_word in ref_data:
                modified_words.append(two_word)
                i += 2
                continue
                
        word = single_words[i]
        if word in ref_data:
            modified_words.append(word)
        else:
            best_match = None
            best_sim = 0.0
            first_char = word[0] if word else ''
            candidates = [w for w in ref_data if w and w[0] == first_char and abs(len(w) - len(word)) <= 2]
            for reference_word in candidates:
                similarity = ratio(word, reference_word)
                if similarity >= 0.8 and similarity > best_sim:
                    best_sim = similarity
                    best_match = reference_word
            if best_match is not None:
                modified_words.append(best_match)

        i += 1
        
    return ' '.join(modified_words)




@app.route("/")
def home():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def upload_file():
    f = request.files['file']
    rawText = process_audio(custom_load_audio(f.read()))
    modText = modify_words(rawText)
    return jsonify({'rawText': rawText, 'modText': modText})

@app.route("/get_sign_data", methods=['POST'])
def get_sign_data():
    req_data = request.get_json(silent=True) or {}
    text = req_data.get('text', '') or request.form.get('text', '')
    mod_text = modify_words(text)
    ref_data = get_reference_data()
    
    words_data = {}
    words_list = []
    
    tokens = re.findall(r'\b\w+\b', mod_text.lower())
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            two_word = f"{tokens[i]} {tokens[i+1]}"
            if two_word in ref_data and len(ref_data[two_word]) > 0:
                words_data[two_word] = ref_data[two_word]
                words_list.append(two_word)
                i += 2
                continue
        word = tokens[i]
        


        if word in ref_data and len(ref_data[word]) > 0:
            words_data[word] = ref_data[word]
            words_list.append(word)
        i += 1
            
    return jsonify({
        'modText': mod_text,
        'data': words_data,
        'words': words_list
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0')


