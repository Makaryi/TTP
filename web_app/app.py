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

# ── Ключевые слова для анализа текста ────────────────────────────────────────
TEXT_KEYWORDS = {
    'Angry':    ['злой', 'злость', 'гнев', 'бесит', 'раздражает', 'ненавижу', 'достал',
                 'angry', 'rage', 'furious', 'hate', 'annoying', 'mad', 'irritated',
                 'ярость', 'агрессия', 'взбешён', 'в ярости', 'злюсь'],
    'Disgust':  ['отвратительно', 'тошнит', 'мерзко', 'противно', 'гадость', 'фу',
                 'disgusting', 'gross', 'nasty', 'awful', 'horrible', 'revolting',
                 'омерзительно', 'отвращение'],
    'Fear':     ['страшно', 'боюсь', 'страх', 'ужас', 'пугает', 'тревога', 'паника',
                 'fear', 'scary', 'afraid', 'terrified', 'anxious', 'panic', 'worried',
                 'испуган', 'опасаюсь', 'нервничаю', 'беспокоюсь'],
    'Happy':    ['счастлив', 'радость', 'радуюсь', 'весело', 'отлично', 'прекрасно',
                 'замечательно', 'улыбка', 'люблю', 'обожаю', 'ура', 'классно',
                 'happy', 'joy', 'great', 'wonderful', 'love', 'amazing', 'awesome',
                 'excited', 'fantastic', 'excellent', 'супер', 'здорово'],
    'Sad':      ['грустно', 'грусть', 'печаль', 'печально', 'плачу', 'горе', 'тоска',
                 'плохо', 'несчастен', 'уныние', 'депрессия', 'скучаю',
                 'sad', 'unhappy', 'depressed', 'crying', 'miserable', 'lonely',
                 'disappointed', 'heartbroken', 'жаль', 'сожалею'],
    'Surprise': ['удивлён', 'удивление', 'неожиданно', 'вот это да', 'ого', 'ничего себе',
                 'не может быть', 'вау', 'шок', 'неожиданность',
                 'surprised', 'wow', 'omg', 'unexpected', 'amazing', 'unbelievable',
                 'shocked', 'astonished', 'серьёзно', 'правда'],
    'Neutral':  ['нормально', 'окей', 'ладно', 'понятно', 'ясно', 'хорошо', 'ок',
                 'okay', 'fine', 'alright', 'normal', 'neutral', 'whatever',
                 'обычно', 'стандартно'],
}


def analyze_text_emotion(text: str) -> dict:
    """Анализ эмоций в тексте по ключевым словам."""
    text_lower = text.lower()
    scores = {e: 0.0 for e in EMOTIONS}

    for emotion, keywords in TEXT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[emotion] += 1.0

    total = sum(scores.values())
    if total == 0:
        # Нет совпадений — нейтральный
        scores['Neutral'] = 1.0
        total = 1.0

    # Нормализуем в проценты
    for e in scores:
        scores[e] = (scores[e] / total) * 100.0

    return scores


def analyze_audio_emotion(audio_path: str) -> dict:
    """Анализ эмоций в аудио через librosa."""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=22050, duration=10.0)

        # Извлекаем признаки
        energy    = float(np.mean(librosa.feature.rms(y=y)))
        zcr       = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        tempo, _  = librosa.beat.beat_track(y=y, sr=sr)
        tempo     = float(tempo)
        centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(mfccs[0]))  # первый MFCC ≈ громкость/тембр

        # Эвристика по признакам:
        # Высокая энергия + высокий темп → злость или счастье
        # Низкая энергия + низкий темп   → грусть
        # Высокий centroid                → страх или удивление
        # Средние значения               → нейтраль

        scores = {e: 0.0 for e in EMOTIONS}

        # Нормируем признаки к [0,1]
        energy_norm   = min(energy / 0.1, 1.0)
        tempo_norm    = min(tempo / 180.0, 1.0)
        centroid_norm = min(centroid / 4000.0, 1.0)

        scores['Angry']    = energy_norm * 0.4 + tempo_norm * 0.3 + (1 - centroid_norm) * 0.3
        scores['Happy']    = energy_norm * 0.35 + tempo_norm * 0.35 + centroid_norm * 0.3
        scores['Sad']      = (1 - energy_norm) * 0.5 + (1 - tempo_norm) * 0.5
        scores['Fear']     = centroid_norm * 0.4 + (1 - energy_norm) * 0.3 + tempo_norm * 0.3
        scores['Surprise'] = centroid_norm * 0.5 + energy_norm * 0.3 + tempo_norm * 0.2
        scores['Disgust']  = (1 - centroid_norm) * 0.4 + energy_norm * 0.3 + (1 - tempo_norm) * 0.3
        scores['Neutral']  = 1.0 - max(scores.values())
        scores['Neutral']  = max(scores['Neutral'], 0.0)

        # Нормализуем
        total = sum(scores.values())
        if total > 0:
            for e in scores:
                scores[e] = (scores[e] / total) * 100.0

        return scores
    except Exception as ex:
        print(f"  ⚠️  audio analysis error: {ex}")
        # Возвращаем нейтральный если не смогли
        return {e: (100.0 / len(EMOTIONS)) for e in EMOTIONS}


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
        token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
        print(f"  📥 Скачиваю {filename} из {repo_id}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"  ✅ {filename} готов")
        return path
    except Exception as e:
        print(f"  ❌ Ошибка скачивания {filename}: {e}")
        return None


def load_models():
    global model, lie_model, face_cascade, model_loaded, lie_model_loaded

    HF_REPO = os.environ.get('HF_REPO_ID', '').strip()

    # ── Скачиваем модели с HuggingFace ────────────────────────────────────────
    if HF_REPO:
        _download_from_hf(HF_REPO, 'best_emotion_model.h5')
        _download_from_hf(HF_REPO, 'lie_detector_model.h5')

    # ── Emotion model ─────────────────────────────────────────────────────────
    try:
        from tensorflow.keras.models import load_model
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

    # ── Lie detector model ────────────────────────────────────────────────────
    try:
        from tensorflow.keras.models import load_model
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

    # ── Haar cascade ──────────────────────────────────────────────────────────
    try:
        import cv2
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


def _predict_face_emotion(img_array) -> dict | None:
    """Возвращает dict эмоций по numpy RGB-изображению, или None если нет лица/модели."""
    import cv2
    if face_cascade is None or not model_loaded:
        return None

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    detected = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(detected) == 0:
        return None

    (x, y, w, h) = detected[0]
    roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
    roi = roi.astype('float32') / 255
    roi = roi[np.newaxis, :, :, np.newaxis]
    pred = model.predict(roi, verbose=0)
    scores = {EMOTIONS[i]: float(pred[0][i] * 100) for i in range(len(EMOTIONS))}
    return _normalize(scores)


# ── Страницы ──────────────────────────────────────────────────────────────────
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
        'version': '4.0'
    })


@app.route('/api/recognize-text', methods=['POST'])
def recognize_text():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        scores = analyze_text_emotion(text)
        best = max(scores, key=scores.get)
        return jsonify({
            'status': 'success',
            'emotion': best,
            'emoji': EMOTION_EMOJI[best],
            'confidence': scores[best],
            'scores': scores,
            'text_length': len(text)
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
                faces_result.append({
                    'face_id': i + 1,
                    'emotion': best,
                    'emoji': EMOTION_EMOJI[best],
                    'confidence': scores[best],
                    'scores': scores
                })
        elif face_cascade is None:
            return jsonify({'error': 'Face detector not available'}), 503
        else:
            return jsonify({'error': 'Emotion model not loaded'}), 503

        return jsonify({'status': 'success', 'num_faces': n_faces, 'faces': faces_result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize-audio', methods=['POST'])
def recognize_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio provided'}), 400

        file = request.files['audio']
        fname = secure_filename(file.filename or 'audio.wav')
        audio_path = os.path.join(UPLOADS_DIR, fname)
        file.save(audio_path)

        try:
            scores = analyze_audio_emotion(audio_path)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        best = max(scores, key=scores.get)
        return jsonify({
            'status': 'success',
            'emotion': best,
            'emoji': EMOTION_EMOJI[best],
            'confidence': scores[best],
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

        (x, y, w, h) = detected[0]
        scores = {e: 0.0 for e in EMOTIONS}

        if model_loaded:
            roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            roi = roi.astype('float32') / 255
            roi = roi[np.newaxis, :, :, np.newaxis]
            pred = model.predict(roi, verbose=0)
            scores = {EMOTIONS[i]: float(pred[0][i] * 100) for i in range(len(EMOTIONS))}
            scores = _normalize(scores)
        else:
            # Нет модели — возвращаем нейтральный с пометкой
            scores['Neutral'] = 100.0

        best = max(scores, key=scores.get)
        return jsonify({
            'status': 'success',
            'face_detected': True,
            'emotion': best,
            'emoji': EMOTION_EMOJI[best],
            'confidence': scores[best],
            'scores': scores,
            'model_loaded': model_loaded
        })
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

        lie_prob    = 0.0
        face_stress = 0.0
        audio_stress = float(np.clip(np.mean(np.abs(audio_features)) * 100 * 1.5, 0, 100))

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
                lie_prob    = float(np.clip(pred[0][0] * 100, 0, 100))
                face_stress = float(np.clip(lie_prob * 0.8, 0, 100))
            elif len(faces) > 0 and model_loaded:
                # Нет lie модели, но есть emotion — оцениваем стресс по эмоции
                (x, y, w, h) = faces[0]
                roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                roi = (roi.astype('float32') / 255)[np.newaxis, :, :, np.newaxis]
                pred = model.predict(roi, verbose=0)
                scores = {EMOTIONS[i]: float(pred[0][i]) for i in range(len(EMOTIONS))}
                # Стресс = сумма "негативных" эмоций
                stress = scores.get('Angry', 0) + scores.get('Fear', 0) + \
                         scores.get('Disgust', 0) + scores.get('Sad', 0)
                face_stress = float(np.clip(stress * 100, 0, 100))
                lie_prob    = float(np.clip((face_stress + audio_stress) / 2, 0, 100))

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
        import cv2
        results = {}

        # Текст
        text = request.form.get('text', '').strip()
        if text:
            scores = analyze_text_emotion(text)
            best = max(scores, key=scores.get)
            results['text'] = {'emotion': best, 'emoji': EMOTION_EMOJI[best],
                                'confidence': scores[best], 'scores': scores}

        # Изображение / фото
        if 'image' in request.files and request.files['image'].filename:
            from PIL import Image
            img = Image.open(request.files['image'].stream)
            img_array = np.array(img.convert('RGB'))
            face_scores = _predict_face_emotion(img_array)
            if face_scores:
                best = max(face_scores, key=face_scores.get)
                results['face'] = {'emotion': best, 'emoji': EMOTION_EMOJI[best],
                                   'confidence': face_scores[best], 'scores': face_scores}
            else:
                results['face'] = {'emotion': 'Neutral', 'emoji': '😐',
                                   'confidence': 100.0, 'scores': {},
                                   'note': 'Face not detected or model not loaded'}

        # Аудио
        if 'audio' in request.files and request.files['audio'].filename:
            file = request.files['audio']
            fname = secure_filename(file.filename or 'audio.wav')
            audio_path = os.path.join(UPLOADS_DIR, fname)
            file.save(audio_path)
            try:
                scores = analyze_audio_emotion(audio_path)
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            best = max(scores, key=scores.get)
            results['audio'] = {'emotion': best, 'emoji': EMOTION_EMOJI[best],
                                 'confidence': scores[best], 'scores': scores}

        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Error handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8888))
    app.run(debug=False, host='0.0.0.0', port=port)
