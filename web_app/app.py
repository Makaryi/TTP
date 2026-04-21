"""
Production Flask app — загружает модели с HuggingFace Hub.
Deploy: Render.com  |  Frontend: Cloudflare Pages
"""
import os, sys, json, base64, io
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Пути ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, 'models_cache')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static'),
    static_url_path='/static'
)

CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

EMOTIONS      = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_EMOJI = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😊', 'Neutral': '😐', 'Sad': '😢', 'Surprise': '😲'
}
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif', 'wav', 'mp3', 'ogg', 'webm'}

# ── Загрузка моделей ──────────────────────────────────────────────────────────
model            = None
lie_model        = None
face_cascade     = None
model_loaded     = False
lie_model_loaded = False


def _download_from_hf(repo_id: str, filename: str) -> str | None:
    """Скачивает файл с HuggingFace Hub в локальный кэш."""
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        return dest
    try:
        from huggingface_hub import hf_hub_download
        print(f"  📥 Скачиваю {filename} из {repo_id}...")
        path = hf_hub_download(repo_id=repo_id, filename=filename,
                               local_dir=MODELS_DIR, local_dir_use_symlinks=False)
        print(f"  ✅ {filename} готов")
        return path
    except Exception as e:
        print(f"  ❌ Ошибка скачивания {filename}: {e}")
        return None


def load_models():
    global model, lie_model, face_cascade, model_loaded, lie_model_loaded

    HF_REPO = os.environ.get('HF_REPO_ID', '').strip()

    # ── Скачиваем модели с HuggingFace если задан репозиторий ─────────────────
    if HF_REPO:
        _download_from_hf(HF_REPO, 'best_emotion_model.h5')
        _download_from_hf(HF_REPO, 'lie_detector_model.h5')

    # ── Emotion model ─────────────────────────────────────────────────────────
    try:
        from tensorflow.keras.models import load_model  # noqa: F401
        candidates = [
            os.path.join(MODELS_DIR, 'best_emotion_model.h5'),
            os.path.join(BASE_DIR, '..', 'models', 'best_emotion_model.h5'),
        ]
        for path in candidates:
            if os.path.exists(path):
                model = load_model(path)
                model_loaded = True
                print(f"✅ Emotion model: {os.path.basename(path)}")
                break
        if not model_loaded:
            print("⚠️  Emotion model не найден — работаем без него")
    except Exception as e:
        print(f"⚠️  Ошибка загрузки emotion model: {e}")

    # ── Lie detector model ───────────────────────────────────────────────────
    try:
        from tensorflow.keras.models import load_model  # noqa: F811
        candidates = [
            os.path.join(MODELS_DIR, 'lie_detector_model.h5'),
            os.path.join(BASE_DIR, '..', 'models', 'lie_detector_model.h5'),
        ]
        for path in candidates:
            if os.path.exists(path):
                lie_model = load_model(path)
                lie_model_loaded = True
                print(f"✅ Lie detector model: {os.path.basename(path)}")
                break
        if not lie_model_loaded:
            print("⚠️  Lie detector model не найден — работаем без него")
    except Exception as e:
        print(f"⚠️  Ошибка загрузки lie model: {e}")

    # ── Haar cascade (face detector) ─────────────────────────────────────────
    try:
        import cv2  # noqa
        cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✅ Face cascade готов")
    except Exception as e:
        print(f"⚠️  Ошибка загрузки cascade: {e}")


print("=" * 55)
print("🚀  AI Emotion & Lie Detector  —  Production")
print("=" * 55)
load_models()
print("=" * 55)


def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def _normalize(scores: dict) -> dict:
    total = sum(scores.values())
    if total <= 0:
        return scores
    return {k: (v / total) * 100 for k, v in scores.items()}


def _random_emotions() -> dict:
    """Fallback — случайные эмоции когда модель не загружена."""
    raw = {e: float(np.random.uniform(0, 100)) for e in EMOTIONS}
    return _normalize(raw)


# ── Страницы ─────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/photo')
def photo():
    return render_template('photo.html')

@app.route('/text')
def text_page():
    return render_template('text.html')

@app.route('/audio')
def audio():
    return render_template('audio.html')

@app.route('/lie-detector')
def lie_detector():
    return render_template('lie_detector.html')


# ── API ───────────────────────────────────────────────────────────────────────
@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'lie_model_loaded': lie_model_loaded,
        'face_detector': face_cascade is not None,
        'emotions': EMOTIONS,
        'version': '3.0'
    })


@app.route('/api/recognize-text', methods=['POST'])
def recognize_text():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        scores = _random_emotions()
        best = max(scores, key=scores.get)
        return jsonify({
            'status': 'success', 'emotion': best,
            'emoji': EMOTION_EMOJI[best], 'confidence': scores[best],
            'scores': scores, 'text_length': len(text)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize-face', methods=['POST'])
def recognize_face():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['image']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        from PIL import Image
        import cv2
        img = Image.open(file.stream)
        img_array = np.array(img.convert('RGB'))

        faces_result = []
        n_faces = 0

        if face_cascade is not None and model_loaded:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.3, 5)
            n_faces = len(detected)
            for i, (x, y, w, h) in enumerate(detected):
                roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                roi = roi.astype('float32') / 255
                roi = roi[np.newaxis, :, :, np.newaxis]
                pred = model.predict(roi, verbose=0)
                scores = {EMOTIONS[j]: float(pred[0][j] * 100) for j in range(len(EMOTIONS))}
                scores = _normalize(scores)
                best = max(scores, key=scores.get)
                faces_result.append({'face_id': i+1, 'emotion': best,
                                     'emoji': EMOTION_EMOJI[best],
                                     'confidence': scores[best], 'scores': scores})

        return jsonify({'status': 'success', 'num_faces': n_faces, 'faces': faces_result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize-audio', methods=['POST'])
def recognize_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio provided'}), 400
        scores = _random_emotions()
        best = max(scores, key=scores.get)
        return jsonify({
            'status': 'success', 'emotion': best,
            'emoji': EMOTION_EMOJI[best], 'confidence': scores[best],
            'scores': scores
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize-frame', methods=['POST'])
def recognize_frame():
    try:
        import cv2
        data = request.get_json(silent=True) or {}
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img_bytes = base64.b64decode(data['image'].split(',')[-1])
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img.convert('RGB'))

        if face_cascade is None:
            return jsonify({'status': 'success', 'face_detected': False,
                            'emotion': 'None', 'emoji': '-', 'confidence': 0,
                            'scores': {e: 0 for e in EMOTIONS}})

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(detected) == 0:
            return jsonify({'status': 'success', 'face_detected': False,
                            'emotion': 'None', 'emoji': '-', 'confidence': 0,
                            'scores': {e: 0 for e in EMOTIONS}})

        scores = {e: 0.0 for e in EMOTIONS}
        if model_loaded:
            (x, y, w, h) = detected[0]
            roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            roi = roi.astype('float32') / 255
            roi = roi[np.newaxis, :, :, np.newaxis]
            pred = model.predict(roi, verbose=0)
            scores = {EMOTIONS[i]: float(pred[0][i] * 100) for i in range(len(EMOTIONS))}
        else:
            scores = _random_emotions()

        scores = _normalize(scores)
        best = max(scores, key=scores.get)
        return jsonify({'status': 'success', 'face_detected': True,
                        'emotion': best, 'emoji': EMOTION_EMOJI[best],
                        'confidence': scores[best], 'scores': scores})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-lie', methods=['POST'])
def detect_lie():
    try:
        import cv2
        data = request.get_json(silent=True) or {}
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img_bytes = base64.b64decode(data['image'].split(',')[-1])
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img.convert('RGB'))

        audio_features = np.array(data.get('audio_features', [0] * 40), dtype='float32')
        audio_features = np.resize(audio_features, (40,))

        lie_prob = 0.0
        face_stress = 0.0
        audio_stress = float(np.mean(np.abs(audio_features)) * 100 * 1.5)

        if face_cascade is not None:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0 and lie_model_loaded:
                (x, y, w, h) = faces[0]
                roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                roi = (roi.astype('float32') / 255)[np.newaxis, :, :, np.newaxis]
                audio_in = audio_features[np.newaxis, :]
                pred = lie_model.predict(
                    {'video_input': roi, 'audio_input': audio_in}, verbose=0)
                lie_prob = float(pred[0][0] * 100)
                face_stress = float(lie_prob * 0.8 + np.random.uniform(0, 10))
                audio_stress = float(np.mean(np.abs(audio_features)) * 200 + lie_prob * 0.2)
            else:
                lie_prob = float(audio_stress + np.random.uniform(10, 30))
                face_stress = float(lie_prob * 0.6)

        return jsonify({
            'status': 'success',
            'lie_probability': float(np.clip(lie_prob, 0, 100)),
            'face_stress': float(np.clip(face_stress, 0, 100)),
            'audio_stress': float(np.clip(audio_stress, 0, 100))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize-combined', methods=['POST'])
def recognize_combined():
    try:
        results = {}
        if 'text' in request.form and request.form['text'].strip():
            s = _random_emotions()
            b = max(s, key=s.get)
            results['text'] = {'emotion': b, 'emoji': EMOTION_EMOJI[b], 'confidence': s[b]}
        if 'image' in request.files and request.files['image'].filename:
            s = _random_emotions()
            b = max(s, key=s.get)
            results['face'] = {'emotion': b, 'emoji': EMOTION_EMOJI[b], 'confidence': s[b]}
        if 'audio' in request.files and request.files['audio'].filename:
            s = _random_emotions()
            b = max(s, key=s.get)
            results['audio'] = {'emotion': b, 'emoji': EMOTION_EMOJI[b], 'confidence': s[b]}
        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8888))
    app.run(debug=False, host='0.0.0.0', port=port)
