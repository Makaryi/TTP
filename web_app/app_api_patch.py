import sys

patch = """

try:
    lie_model_path = os.path.join('models', 'lie_detector_model.h5')
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
"""

with open('projective/app_web.py', 'r') as f:
    text = f.read()

import re
# Remove the old endpoint
cleaned = re.sub(
r"""@app.route\('/api/detect-lie', methods=\['POST'\]\).*?return jsonify\(\{'error': str\(e\)\}\), 500""",
"",
text, flags=re.DOTALL
)

with open('projective/app_web.py', 'w') as f:
    f.write(cleaned + patch)

print("Updated app_web.py to use multimodal prediction!")
