import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate
import os

# Генерируем тестовые данные (X_video, X_audio => Y_labels)
X_video = np.random.rand(100, 48, 48, 1)
X_audio = np.random.rand(100, 40)
Y_labels = np.random.randint(2, size=(100, 1))

# Ветка видео
video_input = Input(shape=(48, 48, 1), name='video_input')
x = Conv2D(16, (3, 3), activation='relu')(video_input)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
video_out = Dense(32, activation='relu')(x)

# Ветка аудио 
audio_input = Input(shape=(40,), name='audio_input')
y = Dense(32, activation='relu')(audio_input)
audio_out = Dense(16, activation='relu')(y)

# Объединение двух веток
merged = concatenate([video_out, audio_out])
z = Dense(32, activation='relu')(merged)
output = Dense(1, activation='sigmoid', name='lie_output')(z)

model = Model(inputs=[video_input, audio_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Обучение демо-модели (Мультимодальная: Камера + Голос)...")
model.fit({'video_input': X_video, 'audio_input': X_audio}, Y_labels, epochs=5, batch_size=16)

os.makedirs('models', exist_ok=True)
model_path = os.path.join('models', 'lie_detector_model.h5')
model.save(model_path)
print(f"Модель сохранена в {model_path}")
