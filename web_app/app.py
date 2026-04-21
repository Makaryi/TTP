"""
Production Flask app — загружает модели с HuggingFace Hub.
Deploy: Render.com  |  Frontend: Cloudflare Pages
"""
import os, sys, json, base64, io, re
import numpy as np
from collections import deque
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

# ── Взвешенные ключевые слова (фраза, вес) ──────────────────────────────────
TEXT_KEYWORDS_W = {
    'Angry': [
        ('ненавижу', 3.0), ('убью', 3.0), ('терпеть не могу', 3.0), ('пошёл', 2.5),
        ('злость', 2.0), ('гнев', 2.0), ('ярость', 2.0), ('бесит', 2.0),
        ('достал', 2.0), ('взбешён', 2.0), ('злюсь', 2.0), ('раздражает', 1.5),
        ('агрессия', 1.5), ('злой', 1.5), ('нервирует', 1.0),
        ('hate', 2.5), ('rage', 2.5), ('furious', 2.5), ('angry', 2.0),
        ('mad', 1.5), ('pissed', 2.0), ('irritated', 1.5), ('annoying', 1.5),
        ('can\'t stand', 3.0), ('outraged', 2.0),
    ],
    'Disgust': [
        ('омерзительно', 2.5), ('отвратительно', 2.5), ('мерзко', 2.0),
        ('тошнит', 2.0), ('противно', 2.0), ('гадость', 2.0), ('блевать', 2.5),
        ('фу', 1.5), ('отвращение', 2.0), ('гнусно', 2.0),
        ('disgusting', 2.5), ('gross', 2.0), ('nasty', 2.0), ('revolting', 2.5),
        ('horrible', 1.5), ('awful', 1.5), ('yuck', 2.0), ('eww', 2.0),
        ('repulsive', 2.5), ('vile', 2.5),
    ],
    'Fear': [
        ('ужас', 2.5), ('боюсь', 2.0), ('страх', 2.0), ('паника', 2.5),
        ('пугает', 1.5), ('испуган', 2.0), ('дрожу', 2.0), ('тревога', 1.5),
        ('опасаюсь', 1.5), ('нервничаю', 1.0), ('беспокоюсь', 1.0),
        ('terrified', 3.0), ('horrified', 2.5), ('scared', 2.5), ('afraid', 2.0),
        ('fear', 2.0), ('panic', 2.5), ('anxious', 2.0), ('worried', 1.5),
        ('dread', 2.5), ('phobia', 2.0), ('nightmare', 2.0),
    ],
    'Happy': [
        ('счастлив', 2.5), ('радуюсь', 2.5), ('ликую', 3.0), ('в восторге', 2.5),
        ('обожаю', 2.5), ('восторг', 2.5), ('радость', 2.0), ('улыбаюсь', 2.0),
        ('весело', 2.0), ('замечательно', 2.0), ('прекрасно', 2.0),
        ('кайф', 2.0), ('ура', 2.0), ('люблю', 1.5), ('отлично', 1.5),
        ('классно', 1.5), ('супер', 1.5), ('здорово', 1.5),
        ('ecstatic', 3.0), ('thrilled', 2.5), ('delighted', 2.5), ('joyful', 2.5),
        ('happy', 2.5), ('excited', 2.0), ('fantastic', 2.0), ('wonderful', 2.0),
        ('awesome', 1.5), ('amazing', 1.5), ('great', 1.0), ('love it', 2.0),
        ('yay', 2.0), ('overjoyed', 3.0),
    ],
    'Sad': [
        ('плачу', 2.5), ('рыдаю', 3.0), ('горе', 2.5), ('опустошён', 2.5),
        ('безнадёжно', 2.5), ('несчастен', 2.5), ('уныние', 2.0),
        ('грустно', 2.0), ('грусть', 2.0), ('печаль', 2.0), ('тоска', 2.0),
        ('расстроен', 2.0), ('депрессия', 2.5), ('одинок', 2.0),
        ('скучаю', 1.5), ('плохо', 1.0), ('жаль', 1.5), ('сожалею', 1.5),
        ('devastated', 3.0), ('heartbroken', 3.0), ('miserable', 2.5),
        ('depressed', 2.5), ('crying', 2.5), ('tears', 2.0), ('sad', 2.0),
        ('lonely', 2.0), ('unhappy', 2.0), ('gloomy', 2.0), ('blue', 1.5),
        ('disappointed', 2.0), ('hopeless', 2.5),
    ],
    'Surprise': [
        ('вот это да', 2.5), ('ничего себе', 2.5), ('в шоке', 2.5),
        ('офигеть', 2.0), ('шок', 2.0), ('удивлён', 2.0), ('удивление', 2.0),
        ('неожиданно', 2.0), ('не может быть', 2.5), ('ого', 2.0), ('вау', 2.0),
        ('astonished', 2.5), ('astounded', 2.5), ('shocked', 2.5),
        ('wow', 2.5), ('omg', 2.0), ('no way', 2.5), ('unbelievable', 2.5),
        ('incredible', 2.0), ('mind-blowing', 3.0), ('unexpected', 2.0),
        ('surprised', 2.0), ('jaw-dropping', 3.0),
    ],
    'Neutral': [
        ('нормально', 1.0), ('окей', 1.0), ('ок', 1.0), ('ладно', 1.0),
        ('понятно', 1.0), ('ясно', 1.0), ('обычно', 1.0), ('стандартно', 1.0),
        ('okay', 1.0), ('fine', 1.0), ('alright', 1.0), ('whatever', 1.5),
        ('so-so', 1.0), ('not bad', 1.0), ('as usual', 1.0),
    ],
}

# Слова-отрицания (рядом с ключевым словом инвертируют смысл)
_NEGATIONS = {'не', 'нет', 'ни', 'без', 'никогда', 'никак', 'нисколько',
              'no', 'not', 'never', 'neither', 'nor', "don't", "doesn't",
              "didn't", "won't", "can't", "couldn't", "wouldn't"}

_NEGATION_SWAP = {
    'Happy': 'Sad', 'Sad': 'Happy',
    'Angry': 'Neutral', 'Fear': 'Neutral',
    'Disgust': 'Neutral', 'Surprise': 'Neutral', 'Neutral': 'Neutral',
}


def analyze_text_emotion(text: str) -> dict:
    """Анализ эмоций: взвешенные слова + обнаружение отрицания + фразы."""
    text_lower = text.lower()
    tokens = re.split(r'[\s,\.!?;:()\[\]"]+', text_lower)
    scores = {e: 0.0 for e in EMOTIONS}

    for emotion, kw_list in TEXT_KEYWORDS_W.items():
        for phrase, weight in kw_list:
            if phrase not in text_lower:
                continue
            # Проверяем отрицание: ищем negation-слово в 3 токенах перед фразой
            phrase_pos = text_lower.find(phrase)
            prefix_tokens = re.split(r'\s+', text_lower[:phrase_pos].strip())[-3:]
            negated = any(tok in _NEGATIONS for tok in prefix_tokens)
            if negated:
                target = _NEGATION_SWAP.get(emotion, 'Neutral')
                scores[target] += weight * 0.6
            else:
                scores[emotion] += weight

    total = sum(scores.values())
    if total == 0:
        scores['Neutral'] = 1.0
        total = 1.0

    return {e: (scores[e] / total) * 100.0 for e in EMOTIONS}


def analyze_audio_emotion(audio_path: str) -> dict:
    """Улучшенный анализ: 13 MFCC (mean+std) + RMS + ZCR + centroid + rolloff + tempo."""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=22050, duration=10.0)

        if len(y) < int(sr * 0.3):  # аудио слишком короткое
            return {e: (100.0 / len(EMOTIONS)) for e in EMOTIONS}

        # ── Базовые признаки ──────────────────────────────────────────────────
        energy   = float(np.sqrt(np.mean(y ** 2)))           # RMS
        zcr      = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
        try:
            tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
        except Exception:
            tempo = 100.0

        # ── MFCC: 13 коэффициентов, mean + std ───────────────────────────────
        mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean  = np.mean(mfccs, axis=1)   # (13,)
        mfcc_std   = np.std(mfccs, axis=1)    # (13,) — нестабильность = страх/удивление

        # ── Нормировка признаков к [0, 1] ────────────────────────────────────
        energy_n   = float(np.clip(energy / 0.08, 0, 1))
        zcr_n      = float(np.clip(zcr / 0.12, 0, 1))
        centroid_n = float(np.clip(centroid / 5000.0, 0, 1))
        rolloff_n  = float(np.clip(rolloff / 8000.0, 0, 1))
        tempo_n    = float(np.clip(tempo / 160.0, 0, 1))
        # MFCC1 ≈ общий тембр/громкость; MFCC2 ≈ форманты
        mfcc1_n    = float(np.clip((mfcc_mean[0] + 400) / 800, 0, 1))
        mfcc2_n    = float(np.clip((mfcc_mean[1] + 100) / 200, 0, 1))
        instab     = float(np.clip(np.mean(mfcc_std) / 50, 0, 1))  # нестабильность голоса

        # ── Эвристические правила ─────────────────────────────────────────────
        # Angry : высокая энергия + быстро + низкий centroid (грубый голос)
        s_angry    = energy_n*0.40 + tempo_n*0.25 + (1-centroid_n)*0.20 + zcr_n*0.15
        # Happy  : высокая энергия + быстро + высокий centroid + MFCC2 высокий
        s_happy    = energy_n*0.30 + tempo_n*0.30 + centroid_n*0.25 + mfcc2_n*0.15
        # Sad    : низкая энергия + медленно + стабильный (низкий instab)
        s_sad      = (1-energy_n)*0.40 + (1-tempo_n)*0.35 + (1-zcr_n)*0.15 + (1-instab)*0.10
        # Fear   : нестабильность + высокий ZCR + средняя энергия
        s_fear     = instab*0.35 + zcr_n*0.30 + centroid_n*0.20 + (1-energy_n)*0.15
        # Surprise: высокий rolloff + ZCR + пики энергии
        s_surprise = rolloff_n*0.35 + zcr_n*0.30 + energy_n*0.25 + instab*0.10
        # Disgust : низкий centroid + медленно + умеренная энергия
        s_disgust  = (1-centroid_n)*0.40 + (1-tempo_n)*0.30 + energy_n*0.20 + (1-rolloff_n)*0.10
        # Neutral : «остаток» — всё среднее
        max_val    = max(s_angry, s_happy, s_sad, s_fear, s_surprise, s_disgust)
        s_neutral  = max(0.0, 1.0 - max_val)

        raw = {
            'Angry': s_angry, 'Happy': s_happy, 'Sad': s_sad,
            'Fear': s_fear, 'Surprise': s_surprise, 'Disgust': s_disgust,
            'Neutral': s_neutral,
        }
        total = sum(raw.values())
        return {e: (raw[e] / total) * 100.0 for e in EMOTIONS} if total > 0 \
               else {e: (100.0/len(EMOTIONS)) for e in EMOTIONS}

    except Exception as ex:
        print(f"  ⚠️  audio error: {ex}")
        return {e: (100.0 / len(EMOTIONS)) for e in EMOTIONS}


# ── Загрузка моделей ──────────────────────────────────────────────────────────
model            = None
lie_model        = None
face_cascade     = None
model_loaded     = False
lie_model_loaded = False
_models_loading  = False  # защита от повторной загрузки

# Temporal smoothing для webcam: скользящее среднее по последним 5 кадрам
_frame_history: deque = deque(maxlen=5)


def _download_from_hf(repo_id: str, filename: str) -> str | None:
    """Скачивает файл с HuggingFace Hub напрямую через requests."""
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return dest
    try:
        import requests
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
        headers = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        print(f"  📥 Скачиваю {filename} с {url}...")
        r = requests.get(url, headers=headers, stream=True, timeout=300)
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  ✅ {filename} готов ({size_mb:.1f} MB)")
        return dest
    except Exception as e:
        print(f"  ❌ Ошибка скачивания {filename}: {e}")
        if os.path.exists(dest):
            os.remove(dest)  # удаляем неполный файл
        return None


def _load_cascade():
    """Загружает Haar cascade (быстро, при старте)."""
    global face_cascade
    try:
        import cv2
        cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
        if os.path.exists(cascade_path):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✅ Face cascade готов")
    except Exception as e:
        print(f"⚠️  Ошибка загрузки cascade: {e}")


def load_models():
    """Скачивает и загружает ML-модели. Вызывается лениво при первом запросе."""
    global model, lie_model, model_loaded, lie_model_loaded, _models_loading

    if _models_loading:
        return
    _models_loading = True

    HF_REPO = os.environ.get('HF_REPO_ID', 'Makaryi/emotion-models').strip()
    print(f"\n{'='*55}")
    print(f"🔄 Загрузка моделей из {HF_REPO}")
    print(f"{'='*55}")

    # ── Скачиваем с HuggingFace ───────────────────────────────────────────────
    _download_from_hf(HF_REPO, 'best_emotion_model.h5')
    _download_from_hf(HF_REPO, 'lie_detector_model.h5')

    # ── Патч для совместимости версий Keras ──────────────────────────────────
    try:
        import tensorflow as tf
        _orig_input_init = tf.keras.layers.InputLayer.__init__
        def _patched_input_init(self, *args, **kwargs):
            kwargs.pop('optional', None)
            kwargs.pop('batch_shape', None) if 'input_shape' in kwargs or 'shape' in kwargs else None
            _orig_input_init(self, *args, **kwargs)
        tf.keras.layers.InputLayer.__init__ = _patched_input_init
    except Exception:
        pass

    def _safe_load(path):
        import tensorflow as tf
        # Попытка 1: стандартная загрузка
        try:
            return tf.keras.models.load_model(path, compile=False)
        except Exception:
            pass
        # Попытка 2: через custom_object_scope с патчем InputLayer
        try:
            class _CompatInputLayer(tf.keras.layers.InputLayer):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('optional', None)
                    super().__init__(*args, **kwargs)
            with tf.keras.utils.custom_object_scope({'InputLayer': _CompatInputLayer}):
                return tf.keras.models.load_model(path, compile=False)
        except Exception:
            pass
        # Попытка 3: только веса + пересборка архитектуры вручную
        try:
            import h5py
            with h5py.File(path, 'r') as f:
                model_config = f.attrs.get('model_config', None)
            if model_config:
                import json
                cfg = json.loads(model_config)
                # Убираем 'optional' рекурсивно
                cfg_str = json.dumps(cfg).replace(', "optional": false', '').replace(', "optional": true', '')
                m = tf.keras.models.model_from_json(cfg_str)
                m.load_weights(path)
                return m
        except Exception as e3:
            print(f"  ❌ Все попытки загрузки провалились: {e3}")
        return None

    # ── Emotion model ─────────────────────────────────────────────────────────
    path = os.path.join(MODELS_DIR, 'best_emotion_model.h5')
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        m = _safe_load(path)
        if m is not None:
            model = m
            model_loaded = True
            print(f"✅ Emotion model загружен ({os.path.getsize(path)//1024//1024} MB)")
        else:
            print("⚠️  Emotion model не удалось загрузить")
    else:
        print("⚠️  best_emotion_model.h5 не найден")

    # ── Lie detector model ────────────────────────────────────────────────────
    path = os.path.join(MODELS_DIR, 'lie_detector_model.h5')
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        m = _safe_load(path)
        if m is not None:
            lie_model = m
            lie_model_loaded = True
            print("✅ Lie model загружен")
        else:
            print("⚠️  Lie model не удалось загрузить")
    else:
        print("⚠️  lie_detector_model.h5 не найден")

    print(f"{'='*55}\n")

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
print("🚀  AI Emotion & Lie Detector  —  Production v5.0")
print("=" * 55)
# Cascade грузим при старте (маленький файл ~1MB)
_load_cascade()
# ML-модели — ленивая загрузка при первом API-запросе
print("⏳  ML-модели будут загружены при первом запросе")
print("=" * 55)


def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def _ensure_models():
    """Ленивая загрузка: вызвать перед любым API-запросом, требующим модели."""
    if not model_loaded and not _models_loading:
        load_models()


def _normalize(scores: dict) -> dict:
    total = sum(scores.values())
    if total <= 0:
        return scores
    return {k: (v / total) * 100 for k, v in scores.items()}


def _preprocess_roi(roi_gray):
    """Preprocessing matching FER-2013 training: rescale=1./255, grayscale 48x48."""
    return (roi_gray.astype('float32') / 255.0)[np.newaxis, :, :, np.newaxis]


def _predict_face_emotion(img_array) -> dict | None:
    """Возвращает dict эмоций по numpy RGB-изображению, или None если нет лица/модели."""
    import cv2
    if face_cascade is None or not model_loaded:
        return None

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(detected) == 0:
        return None

    (x, y, w, h) = detected[0]
    ih, iw = gray.shape
    pad = int(max(w, h) * 0.15)
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(iw, x + w + pad); y2 = min(ih, y + h + pad)
    roi = cv2.resize(gray[y1:y2, x1:x2], (48, 48))
    roi = _preprocess_roi(roi)
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
        'version': '5.0'
    })


@app.route('/api/recognize-text', methods=['POST'])
def recognize_text():
    try:
        _ensure_models()
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
        _ensure_models()
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
            ih, iw = gray.shape
            for i, (x, y, w, h) in enumerate(detected):
                pad = int(max(w, h) * 0.15)
                x1 = max(0, x - pad); y1 = max(0, y - pad)
                x2 = min(iw, x + w + pad); y2 = min(ih, y + h + pad)
                roi = cv2.resize(gray[y1:y2, x1:x2], (48, 48))
                roi = _preprocess_roi(roi)
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
        _ensure_models()
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
        _ensure_models()
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
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        if len(detected) == 0:
            _frame_history.clear()  # нет лица — сброс истории
            return jsonify({'status': 'success', 'face_detected': False,
                            'emotion': 'None', 'emoji': '-', 'confidence': 0,
                            'scores': {e: 0 for e in EMOTIONS}})

        # Паддинг: расширяем bbox на 15% для захвата контекста лица
        (x, y, w, h) = detected[0]
        ih, iw = gray.shape
        pad = int(max(w, h) * 0.15)
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(iw, x + w + pad); y2 = min(ih, y + h + pad)

        scores = {e: 0.0 for e in EMOTIONS}

        if model_loaded:
            roi = cv2.resize(gray[y1:y2, x1:x2], (48, 48))
            roi = _preprocess_roi(roi)
            pred = model.predict(roi, verbose=0)
            scores = {EMOTIONS[i]: float(pred[0][i] * 100) for i in range(len(EMOTIONS))}
            scores = _normalize(scores)

            # Temporal smoothing: среднее по последним N предсказаниям
            _frame_history.append(scores)
            if len(_frame_history) >= 2:
                avg = {e: sum(h[e] for h in _frame_history) / len(_frame_history) for e in EMOTIONS}
                scores = _normalize(avg)
        else:
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
        _ensure_models()
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
                roi = _preprocess_roi(roi)
                audio_in = audio_features[np.newaxis, :]
                pred = lie_model.predict(
                    {'video_input': roi, 'audio_input': audio_in}, verbose=0)
                lie_prob    = float(np.clip(pred[0][0] * 100, 0, 100))
                face_stress = float(np.clip(lie_prob * 0.8, 0, 100))
            elif len(faces) > 0 and model_loaded:
                # Нет lie модели, но есть emotion — оцениваем стресс по эмоции
                (x, y, w, h) = faces[0]
                roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                roi = _preprocess_roi(roi)
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
        _ensure_models()
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
