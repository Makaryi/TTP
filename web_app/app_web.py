import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import numpy as np
from werkzeug.utils import secure_filename

# Get the parent directory (project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_APP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(WEB_APP_DIR, 'templates')
STATIC_DIR = os.path.join(WEB_APP_DIR, 'static')
UPLOADS_DIR = os.path.join(WEB_APP_DIR, 'uploads')

# Create Flask app with correct paths
app = Flask(__name__, 
            template_folder=TEMPLATES_DIR,
            static_folder=STATIC_DIR,
            static_url_path='/static')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['TEMPLATES_AUTO_RELOAD'] = True
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'wav', 'mp3', 'ogg'}

# Define emotion data
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_EMOJIS = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😊',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😲'
}

# Try to load models
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import cv2
    model_loaded = False
    face_cascade = None
    
    try:
        model_path = os.path.join(PROJECT_ROOT, 'models', 'best_emotion_model.h5')
        if os.path.exists(model_path):
            model = load_model(model_path)
            model_loaded = True
        else:
            print(f"Model not found at {model_path}")
            model_loaded = False
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False
    
    # Load face cascade classifier
    cascade_path = os.path.join(WEB_APP_DIR, 'haarcascade_frontalface_default.xml')
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)

except ImportError as e:
    print(f"Warning: TensorFlow/OpenCV not fully available: {e}")
    model_loaded = False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==================== ROUTES ====================

@app.route('/')
def home():
    """Serve home page"""
    return render_template('index.html')


@app.route('/camera')
def camera():
    """Serve camera page"""
    return render_template('camera.html')


@app.route('/photo')
def photo():
    """Serve photo page"""
    return render_template('photo.html')


@app.route('/text')
def text():
    """Serve text page"""
    return render_template('text.html')


@app.route('/audio')
def audio():
    """Serve audio page"""
    return render_template('audio.html')

@app.route('/lie-detector')
def lie_detector():
    """Serve lie detector page"""
    return render_template('lie_detector.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)


# ==================== API ENDPOINTS ====================

@app.route('/api/status', methods=['GET'])
def status():
    """Get system status"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'face_detector_loaded': face_cascade is not None,
        'emotions': EMOTIONS,
        'version': '2.0'
    })


@app.route('/api/recognize-text', methods=['POST'])
def recognize_text():
    """Analyze text emotion"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        language = data.get('language', 'auto')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Simulate emotion detection for now
        # In production, use actual text emotion model
        emotions = {emotion: np.random.uniform(0, 100) for emotion in EMOTIONS}
        max_emotion = max(emotions, key=emotions.get)
        
        # Normalize to percentages
        total = sum(emotions.values())
        emotions = {k: (v/total)*100 for k, v in emotions.items()}
        
        return jsonify({
            'status': 'success',
            'emotion': max_emotion,
            'emoji': EMOTION_EMOJIS[max_emotion],
            'confidence': emotions[max_emotion],
            'scores': emotions,
            'text_length': len(text),
            'language': language
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize-face', methods=['POST'])
def recognize_face():
    """Analyze face emotion from image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        from PIL import Image
        import io
        img = Image.open(file.stream)
        img_array = np.array(img.convert('RGB'))
        
        faces = []
        num_faces = 0
        if face_cascade is not None and model_loaded:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            num_faces = len(detected_faces)
            
            for i, (x, y, w, h) in enumerate(detected_faces):
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype('float32') / 255
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)
                
                prediction = model.predict(roi, verbose=0)
                emotions = {EMOTIONS[j]: float(prediction[0][j] * 100) for j in range(len(EMOTIONS))}
                
                max_emotion = max(emotions, key=emotions.get)
                total = sum(emotions.values())
                if total > 0:
                    emotions = {k: (v/total)*100 for k, v in emotions.items()}
                
                faces.append({
                    'face_id': i + 1,
                    'emotion': max_emotion,
                    'emoji': EMOTION_EMOJIS[max_emotion],
                    'confidence': emotions[max_emotion],
                    'scores': emotions
                })
        
        return jsonify({
            'status': 'success',
            'num_faces': num_faces,
            'faces': faces
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize-audio', methods=['POST'])
def recognize_audio():
    """Analyze audio emotion"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Simulate audio emotion recognition
        # In production, use actual audio analysis model
        emotions = {emotion: np.random.uniform(0, 100) for emotion in EMOTIONS}
        max_emotion = max(emotions, key=emotions.get)
        total = sum(emotions.values())
        emotions = {k: (v/total)*100 for k, v in emotions.items()}
        
        return jsonify({
            'status': 'success',
            'emotion': max_emotion,
            'emoji': EMOTION_EMOJIS[max_emotion],
            'confidence': emotions[max_emotion],
            'scores': emotions,
            'duration': '0:30',
            'tone': max_emotion
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize-frame', methods=['POST'])
def recognize_frame():
    """Recognize emotion from video frame"""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        import base64
        try:
            image_data = base64.b64decode(data['image'].split(',')[1])
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(image_data))
            img_array = np.array(img.convert('RGB'))
        except Exception as e:
            return jsonify({'error': f'Invalid image format: {str(e)}'}), 400
        
        # Check if face exists
        has_face = False
        if face_cascade is not None:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            has_face = len(faces) > 0
        
        if not has_face:
            return jsonify({
                'status': 'success',
                'face_detected': False,
                'emotion': 'None',
                'emoji': '-',
                'confidence': 0,
                'scores': {emotion: 0 for emotion in EMOTIONS},

            })
        
        # Try to predict emotion using the model
        emotions_dict = {emotion: 0.0 for emotion in EMOTIONS}
        
        # If model is loaded, use it
        if model_loaded:
            try:
                # Prepare image for model
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    roi = gray[y:y+h, x:x+w]
                    roi = cv2.resize(roi, (48, 48))
                    roi = roi.astype('float32') / 255
                    roi = np.expand_dims(roi, axis=0)
                    roi = np.expand_dims(roi, axis=-1)
                    
                    # Predict
                    prediction = model.predict(roi, verbose=0)
                    emotions_dict = {EMOTIONS[i]: float(prediction[0][i] * 100) for i in range(len(EMOTIONS))}
            except Exception as e:
                # Fallback to default
                print(f"Model prediction error: {e}")
        else:
            print("Model not loaded, cannot predict emotions properly.")
        
        # Normalize scores
        total = sum(emotions_dict.values())
        if total > 0:
            emotions_dict = {k: (v/total)*100 for k, v in emotions_dict.items()}
        
        max_emotion = max(emotions_dict, key=emotions_dict.get)
        

        
        return jsonify({
            'status': 'success',
            'face_detected': True,
            'emotion': max_emotion,
            'emoji': EMOTION_EMOJIS[max_emotion],
            'confidence': emotions_dict[max_emotion],
            'scores': emotions_dict,

        })
    
    except Exception as e:
        print(f"Error in recognize_frame: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize-combined', methods=['POST'])
def recognize_combined():
    """Combined analysis of text, face, and audio"""
    try:
        results = {
            'text': None,
            'face': None,
            'audio': None
        }
        
        # Text analysis
        if 'text' in request.form:
            text = request.form.get('text', '').strip()
            if text:
                emotions = {emotion: np.random.uniform(0, 100) for emotion in EMOTIONS}
                max_emotion = max(emotions, key=emotions.get)
                total = sum(emotions.values())
                emotions = {k: (v/total)*100 for k, v in emotions.items()}
                results['text'] = {
                    'emotion': max_emotion,
                    'emoji': EMOTION_EMOJIS[max_emotion],
                    'confidence': emotions[max_emotion]
                }
        
        # Face analysis
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '' and allowed_file(file.filename):
                emotions = {emotion: np.random.uniform(0, 100) for emotion in EMOTIONS}
                max_emotion = max(emotions, key=emotions.get)
                total = sum(emotions.values())
                emotions = {k: (v/total)*100 for k, v in emotions.items()}
                results['face'] = {
                    'emotion': max_emotion,
                    'emoji': EMOTION_EMOJIS[max_emotion],
                    'confidence': emotions[max_emotion]
                }
        
        # Audio analysis
        if 'audio' in request.files:
            file = request.files['audio']
            if file.filename != '' and allowed_file(file.filename):
                emotions = {emotion: np.random.uniform(0, 100) for emotion in EMOTIONS}
                max_emotion = max(emotions, key=emotions.get)
                total = sum(emotions.values())
                emotions = {k: (v/total)*100 for k, v in emotions.items()}
                results['audio'] = {
                    'emotion': max_emotion,
                    'emoji': EMOTION_EMOJIS[max_emotion],
                    'confidence': emotions[max_emotion]
                }
        
        return jsonify({
            'status': 'success',
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ==================== MAIN ====================

try:
    lie_model_path = os.path.join(PROJECT_ROOT, 'models', 'lie_detector_model.h5')
    if os.path.exists(lie_model_path):
        lie_detector_model = load_model(lie_model_path)
        lie_model_loaded = True
        print("✓ Multimodal Lie Detector model loaded")
    else:
        lie_model_loaded = False
except Exception as e:
    print(f"Error loading lie detector model: {e}")
    lie_model_loaded = False

@app.route('/api/detect-lie', methods=['POST'])
def detect_lie():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Декодирование видео (лица)
        import base64
        import io
        from PIL import Image
        import numpy as np
        import cv2

        image_data = base64.b64decode(data['image'].split(',')[1])
        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img.convert('RGB'))
        
        # Получение аудио-фич (40 bins от клиента)
        audio_features = np.array(data.get('audio_features', [0]*40), dtype='float32')
        if audio_features.shape[0] != 40:
            audio_features = np.resize(audio_features, (40,))
            
        has_face = False
        face_stress = 0.0
        audio_stress = float(np.mean(audio_features) * 100) * 1.5 # простой аудио-стресс
        lie_probability = 0.0
        
        if face_cascade is not None:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            has_face = len(faces) > 0
            
            if has_face and lie_model_loaded:
                (x, y, w, h) = faces[0]
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype('float32') / 255
                roi = np.expand_dims(roi, axis=-1)   # (48, 48, 1)
                video_input = np.expand_dims(roi, axis=0) # (1, 48, 48, 1)
                
                audio_input = np.expand_dims(audio_features, axis=0) # (1, 40)
                
                # Мультимодальное предсказание
                prediction = lie_detector_model.predict({'video_input': video_input, 'audio_input': audio_input}, verbose=0)
                lie_probability = float(prediction[0][0] * 100)
                
                # Симулируем показатели для интерфейса, так как модель демо
                face_stress = float(lie_probability * 0.8 + np.random.uniform(0, 10))
                audio_stress = float(np.mean(audio_features) * 100 * 2 + lie_probability * 0.2)
            else:
                # Fallback эвристика, если модель не загружена
                lie_probability = float(audio_stress + np.random.uniform(10, 30))
                face_stress = float(lie_probability * 0.6)

        return jsonify({
            'status': 'success',
            'lie_probability': float(max(0, min(100, lie_probability))),
            'face_stress': float(max(0, min(100, face_stress))),
            'audio_stress': float(max(0, min(100, audio_stress)))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Emotion Recognition AI Server")
    print("=" * 60)
    print(f"✓ Model loaded: {model_loaded}")
    print(f"✓ Face detector loaded: {face_cascade is not None}")
    print(f"✓ Emotions: {', '.join(EMOTIONS)}")
    print("=" * 60)
    print("📍 Website: http://localhost:8888")
    print("📍 API Base: http://localhost:8888/api/")
    print("=" * 60)
    
    app.run(debug=False, host='127.0.0.1', port=8888, use_reloader=False)






