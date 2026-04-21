let webcamActive = false;
let webcamInterval = null;
let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;

// Tab switching
function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
    
    if (tabName !== 'image' && webcamActive) {
        stopWebcam();
    }
}

function switchSubtab(subtabName) {
    const parent = event.target.closest('.tabs, .subtabs, [data-subtabs]').parentElement;
    const container = event.target.closest('.tab-content, [data-tab]');
    
    container.querySelectorAll('.subtab-content').forEach(el => el.classList.remove('active'));
    container.querySelectorAll('.subtab-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(subtabName).classList.add('active');
    event.target.classList.add('active');
    
    if (subtabName !== 'webcam' && webcamActive) {
        stopWebcam();
    }
}

// File Upload
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary-color)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--border-color)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--border-color)';
    if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

function handleFileUpload(file) {
    if (!file.type.startsWith('image/')) {
        showStatus('uploadStatus', 'error', 'Пожалуйста, выберите изображение');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    showStatus('uploadStatus', 'loading', 'Обработка изображения...');

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayUploadResults(data);
            showStatus('uploadStatus', 'success', `✓ Обработано! Лиц: ${data.faces_detected}`);
        } else {
            showStatus('uploadStatus', 'error', `Ошибка: ${data.error}`);
        }
    })
    .catch(error => {
        showStatus('uploadStatus', 'error', `Ошибка: ${error.message}`);
    });
}

function displayUploadResults(data) {
    const html = `
        <h3>Результаты анализа</h3>
        ${data.faces_detected > 0 ? `
            <p style="color: #10b981; margin-bottom: 15px;">✓ Обнаружено ${data.faces_detected} лиц(а)</p>
            ${data.results.map(r => `
                <div class="result-item">
                    <div class="emotion-result">
                        <div class="emotion-name" style="color: ${r.color}">${r.emotion}</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${r.confidence}%; background-color: ${r.color};"></div>
                        </div>
                        <div class="confidence-text">${r.confidence.toFixed(1)}%</div>
                    </div>
                </div>
            `).join('')}
        ` : '<p style="color: #f59e0b;">⚠ Лица не обнаружены</p>'}
        <div class="result-image">
            <img src="${data.image}" alt="Результат">
        </div>
    `;
    
    document.getElementById('uploadResults').innerHTML = html;
    document.getElementById('uploadResults').style.display = 'block';
}

// Webcam
async function startWebcam() {
    try {
        showStatus('webcamStatus', 'loading', 'Запуск камеры...');
        
        const constraints = {
            video: { width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        document.getElementById('webcamVideo').srcObject = stream;

        document.getElementById('startBtn').style.display = 'none';
        document.getElementById('stopBtn').style.display = 'inline-flex';
        webcamActive = true;

        showStatus('webcamStatus', 'success', '✓ Камера включена. Анализирую эмоции...');

        const video = document.getElementById('webcamVideo');
        const canvas = document.getElementById('webcamCanvas');
        const ctx = canvas.getContext('2d');

        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            webcamInterval = setInterval(() => {
                if (webcamActive && video.readyState === video.HAVE_ENOUGH_DATA) {
                    ctx.drawImage(video, 0, 0);
                    const imageData = canvas.toDataURL('image/jpeg', 0.8);

                    fetch('/detect-webcam', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: imageData })
                    })
                    .then(r => r.json())
                    .then(data => {
                        if (data.success && data.results.length > 0) {
                            displayWebcamResults(data);
                        }
                    })
                    .catch(e => console.error(e));
                }
            }, 333);
        };
    } catch (error) {
        showStatus('webcamStatus', 'error', `Ошибка камеры: ${error.message}`);
    }
}

function stopWebcam() {
    if (document.getElementById('webcamVideo').srcObject) {
        document.getElementById('webcamVideo').srcObject.getTracks().forEach(t => t.stop());
    }
    if (webcamInterval) clearInterval(webcamInterval);
    
    document.getElementById('startBtn').style.display = 'inline-flex';
    document.getElementById('stopBtn').style.display = 'none';
    webcamActive = false;
    document.getElementById('webcamResults').style.display = 'none';
}

function displayWebcamResults(data) {
    const html = `
        <h3>Эмоции в реальном времени</h3>
        <p style="color: #10b981;">✓ ${data.faces_detected} лиц(а)</p>
        ${data.results.map(r => `
            <div class="result-item">
                <div class="emotion-result">
                    <div class="emotion-name" style="color: ${r.color}">${r.emotion}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${r.confidence}%; background-color: ${r.color};"></div>
                    </div>
                    <div class="confidence-text">${r.confidence.toFixed(1)}%</div>
                </div>
            </div>
        `).join('')}
    `;
    
    document.getElementById('webcamResults').innerHTML = html;
    document.getElementById('webcamResults').style.display = 'block';
}

// Text Analysis
function analyzeText() {
    const text = document.getElementById('textInput').value.trim();
    
    if (!text) {
        showStatus('textStatus', 'error', 'Пожалуйста, введите текст');
        return;
    }

    showStatus('textStatus', 'loading', 'Анализирую текст...');

    fetch('/analyze-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            displayTextResults(data.result);
            showStatus('textStatus', 'success', '✓ Анализ завершен');
        } else {
            showStatus('textStatus', 'error', `Ошибка: ${data.error}`);
        }
    })
    .catch(e => showStatus('textStatus', 'error', `Ошибка: ${e.message}`));
}

function displayTextResults(result) {
    const html = `
        <div class="sentiment-result">
            <div class="sentiment-emoji">${result.emoji}</div>
            <div class="sentiment-name" style="color: ${result.color}">${result.sentiment}</div>
            <div class="confidence-bar" style="max-width: 300px; margin: 15px auto;">
                <div class="confidence-fill" style="width: ${result.confidence}%; background-color: ${result.color};"></div>
            </div>
            <div class="confidence-text">${result.confidence.toFixed(1)}% уверенности</div>
        </div>
    `;
    
    document.getElementById('textResults').innerHTML = html;
    document.getElementById('textResults').style.display = 'block';
}

// Audio Recording
async function startRecording() {
    try {
        audioChunks = [];
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioStream = stream;
        
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
        mediaRecorder.start();

        document.getElementById('recordBtn').style.display = 'none';
        document.getElementById('stopRecordBtn').style.display = 'inline-flex';
        
        showStatus('audioStatus', 'loading', '🎙️ Запись... Говорите что-нибудь');
        
        startAudioVisualization(stream);
    } catch (error) {
        showStatus('audioStatus', 'error', `Ошибка микрофона: ${error.message}`);
    }
}

function stopRecording() {
    mediaRecorder.stop();
    audioStream.getTracks().forEach(t => t.stop());

    document.getElementById('recordBtn').style.display = 'inline-flex';
    document.getElementById('stopRecordBtn').style.display = 'none';
    document.getElementById('analyzeAudioBtn').style.display = 'inline-flex';
    
    showStatus('audioStatus', 'success', '✓ Запись завершена. Нажмите "Анализировать"');
}

function analyzeAudio() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const reader = new FileReader();
    
    reader.onload = (e) => {
        const audioData = e.target.result.substring(e.target.result.indexOf(',') + 1);
        
        showStatus('audioStatus', 'loading', 'Анализирую речь...');

        fetch('/analyze-audio', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audio: audioData })
        })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                displayAudioResults(data.result);
                showStatus('audioStatus', 'success', '✓ Анализ завершен');
            } else {
                showStatus('audioStatus', 'error', `Ошибка: ${data.error}`);
            }
        })
        .catch(e => showStatus('audioStatus', 'error', `Ошибка: ${e.message}`));
    };
    
    reader.readAsDataURL(audioBlob);
}

function displayAudioResults(result) {
    const html = `
        <div class="result-item">
            <div class="emotion-result">
                <div class="emotion-name">${result.emotion}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence}%;"></div>
                </div>
                <div class="confidence-text">${result.confidence.toFixed(1)}%</div>
            </div>
        </div>
        <p style="margin-top: 15px; color: var(--text-secondary);">
            <strong>Энергия:</strong> ${result.energy.toFixed(1)} | 
            <strong>Спектральный центр:</strong> ${result.spectral_centroid.toFixed(0)} Hz
        </p>
    `;
    
    document.getElementById('audioResults').innerHTML = html;
    document.getElementById('audioResults').style.display = 'block';
}

function startAudioVisualization(stream) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    const microphone = audioContext.createMediaStreamSource(stream);
    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    microphone.connect(analyser);

    const canvas = document.getElementById('audioCanvas');
    const canvasCtx = canvas.getContext('2d');

    function draw() {
        if (!mediaRecorder || mediaRecorder.state === 'inactive') return;

        requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArray);

        canvasCtx.fillStyle = 'rgb(15, 23, 42)';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = 'rgb(99, 102, 241)';
        canvasCtx.beginPath();

        const sliceWidth = canvas.width / dataArray.length;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;

            if (i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        canvasCtx.lineTo(canvas.width, canvas.height / 2);
        canvasCtx.stroke();
    }

    draw();
}

function showStatus(elementId, type, message) {
    const el = document.getElementById(elementId);
    el.textContent = message;
    el.className = `status-${type}`;
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelector('.tab-content').classList.add('active');
    document.querySelector('.tab-btn').classList.add('active');
    document.querySelector('.subtab-content').classList.add('active');
    document.querySelector('.subtab-btn').classList.add('active');
});
